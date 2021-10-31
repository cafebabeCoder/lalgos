import os
import torch
import io
import logging
import numpy as np
import pandas as pd
import random
from torch import nn
import torch.nn.functional as F 
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import Counter
from gensim.models import Word2Vec

from torchtext.vocab import Vocab
from torchvision.datasets import MNIST
from torchtext.utils import unicode_csv_reader
from torch.utils.data import DataLoader, random_split
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from  torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer, BertModel, BertConfig, get_cosine_schedule_with_warmup
from utils.alphabet import Alphabet
from utils.lstm_net_arg_core import data_path, UNK_CLS, LABEL_IDX, LABEL_NAME_IDX, model_params
from utils.file_utils import Print_log, get_log_file
from utils.functions import f1measure, _build_labels_from_dataset, build_alphabet_from_model,load_w2v_pretrain_emb
from pytorch_lightning.callbacks import ModelCheckpoint

logging.basicConfig(level=logging.DEBUG)
global_step = 0
log_file = get_log_file(model_params, over_write=True)
print_log = Print_log(log_file, mode='a', visible=False)
print_log(str(model_params))

class TrainAppCateDataset(Dataset):
    def __init__(self, data_dir, ngrams, min_freq, label_idx, cls_sample='weighted_sampling', yield_cls=True):
        self.columns = ["appid", "c1", "nc1","c2", "nc2",  "c3", "nc3", "c4", "nc4", "nickname", "intro", 'contractor_name', 'seg']
        self.data_columns = dict(zip(self.columns, range(len(self.columns))))
        self.label_idx = label_idx
        self.ngram_columns = ["nickname", "intro", 'contractor_name']
        self.ngrams = ngrams
        self.data_dir = data_dir
        self.yield_cls = yield_cls

        if cls_sample == 'under_sampling':   # 在数量较多的类别随机采样一些
            self.data = pd.read_csv(self.data_dir, header=None, names=self.columns) 
            self.data = self.under_sampling(500)
        elif cls_sample == 'weighted_sampling':   #给每个类别权重
            self.data = pd.read_csv(self.data_dir, header=None, names=self.columns) 
            self.data = self.weighted_sampling()
        else:
            self.data = pd.read_csv(self.data_dir, header=None, names=self.columns) 

    def weighted_sampling(self):
        # 计算weighted
        data_len = self.data.index.size
        sampled_counter = dict(zip(self.data.iloc[:, self.label_idx].unique(), [0] * data_len))  # 采样计数器
        label_counter = Counter(self.data.iloc[:, self.label_idx])  # 类别计数
        labels = len(label_counter.keys())  #有多少类别数
        logging.info("Label count:{}".format(labels))
        logging.info("Label counter:{}".format(label_counter))
        weight_counter = dict([(k, min(max(0.5/labels * data_len, v), 0.1 * data_len) ) for k, v in label_counter.items()]) #每个类别的权重
        logging.info("Weighted sampling: {}".format(", ".join(["{}:{}".format(k, int(v)) for k, v in weight_counter.items()])))

        # 先遍历一遍
        sampled_idx = [] 
        for sid in range(0, data_len):
            scat = self.data.iloc[sid, self.label_idx]
            if sampled_counter.get(scat, 0) < weight_counter[scat]:
                sampled_idx.append(sid)
                sampled_counter[scat] = sampled_counter[scat] + 1
            else:
                pass
        
        # 再补充over-sampling的部分
        flag = False
        while not flag:
            sid = random.randint(0, data_len-1) 
            scat = self.data.iloc[sid, self.label_idx]
            if sampled_counter.get(scat, 0) < weight_counter[scat]:
                sampled_idx.append(sid)
                sampled_counter[scat] = sampled_counter[scat] + 1
            else:
                pass

            flag = True
            for k, v in sampled_counter.items():
                flag = flag & (v >= sampled_counter[k])

        return self.data.iloc[sampled_idx, :].reset_index(drop=True)
   
    def under_sampling(self, num_sample):
        # 计算weighted
        data_len = self.data.index.size
        sampled_idx = [] 
        sampled_counter = dict(zip(self.data.iloc[:, self.label_idx].unique(), [0] * data_len)) 
        flag = False 
        for sid in range(0, data_len):
            scat = self.data.iloc[sid, self.label_idx]
            if sampled_counter.get(scat, 0) < num_sample:
                sampled_idx.append(sid)
                sampled_counter[scat] = sampled_counter[scat] + 1
            else:
                pass
        return self.data.iloc[sampled_idx, :].reset_index(drop=True)
    
    def _build_vocab_from_text(self, min_freq, ngrams):
        counter = Counter()
        with tqdm(unit_scale=0, unit='lines') as t:
            with io.open(self.data_dir, encoding="utf8") as f:
                reader = unicode_csv_reader(f)
                for row in reader:
                    tokens = "".join([row[self.data_columns[i]] for i in self.ngram_columns]).replace(" ", "")
                    counter.update(ngrams_iterator(tokens, ngrams))
                    t.update(1)
            vocab = Vocab(counter, min_freq=min_freq)
        return vocab

    def __len__(self):
        return self.data.index.size

    def __getitem__(self, i):
        row = self.data.iloc[i]
        # bert 部分
        tokens = []
        sent = "".join([str(row[idx]) for idx in self.ngram_columns]).replace(" ", "")
        tokens = tokens + list(ngrams_iterator(sent, self.ngrams))
        mask = [1] * len(tokens)

        # w2v 部分
        terms = row[self.data_columns['seg']].split("+")
        
        if self.yield_cls:
            cls = row[self.label_idx] 
            cls = int(cls)
            return {'label' :cls, 'bert_tokens' : tokens, 'mask':mask, 'w2v_tokens':terms, 'nk': row[9]} 
        else:
            sentence = ",".join([str(t) for t in list(row)])
            return {'sentence' :sentence, 'bert_tokens' : tokens, 'mask':mask, 'w2v_tokens':terms, 'nk':row[9]} 


class LSTM_NET(pl.LightningModule):
    # 字向量， 词向量
    def __init__(self, w2v_alphabet, pretrain_w2v_embedding, pretrain_w2v_dim , hidden_dim, num_layers, num_class, labeldict, dropout=0.5, fix_embedding=True, training=False):
        super(LSTM_NET, self).__init__()
        self.labeldict = labeldict
        self.pretrain_w2v_dim = int(pretrain_w2v_dim)
        # 词向量
        self.w2v_embedding = nn.Embedding(w2v_alphabet.size(), self.pretrain_w2v_dim)
        # 将预训练词向量载入self.word_embeddings中
        if pretrain_w2v_embedding is not None:
            self.w2v_embedding.weight.data.copy_(torch.from_numpy(pretrain_w2v_embedding))
        else:
            initrange = 0.5
            self.w2v_embedding.weight.data.uniform_(-initrange, initrange)
            self.w2v_embedding.apply(self.init_weight)

        # 载入bert中
        self.bert = BertModel(BertConfig())
        self.bert.from_pretrained("hfl/chinese-bert-wwm")

        if fix_embedding:  #固定， w2v , bert所有层都不学习
            self.w2v_embedding.weight.requires_grad = False # w2v
            for name, param in self.named_parameters():   # bert
                if "bert" in name:
                    param.requires_grad = False 
        else:   #不固定， bert只有最上层学习， 其他不学习; w2v学习
            for name, param in self.named_parameters():
                if ('bert' in name) and not name.startswith('bert.encoder.layer.10') and \
                    not name.startswith('bert.encoder.layer.11') and \
                    not name.startswith('bert.pooler'):
                    param.requires_grad = False

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bert_linear = nn.Linear(int(768), 256)
        self.bert_drop = nn.Dropout(0.2)
        # self.lstm = nn.LSTM(self.pretrain_w2v_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)

        # 上层分类器
        self.ffn = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(int(506), int(506/2)),
            nn.ReLU(),
            nn.Linear(int(506/2), num_class),
        )
        self.ffn.apply(self.init_weight)
        # self.output = nn.Softmax(dim=1)

    def init_weight(self, m):
        initrange = 0.5
        if type(m) == nn.Linear:
            m.weight.data.uniform_(-initrange, initrange)

    def forward_2(self, batch):
        # w2v
        # w2v_out = self.w2v_embedding(batch['w2v_tokens'])
        x, _ = self.lstm(w2v_out, None)
        x = x[:, -1, :]  # lstm output最后一个
        w2v_emb = x
        # bert
        bert_output = self.bert(batch['bert_tokens'], batch['mask'])
        bert_emb = torch.mean(bert_output['last_hidden_state'], axis=1)
        # logging.debug("term_emb size: {}\t word_emb size: {}".format(term_emb.shape, word_emb.shape))
        concat_emb = torch.cat((w2v_emb, bert_emb), -1)
        x = self.ffn(concat_emb)
        out = self.output(x)
        return out 

    def forward(self, batch):
        # w2v
        w2v_out = self.w2v_embedding(batch['w2v_tokens'])
        mask = batch['w2v_mask'].unsqueeze(dim=2)
        masked_out = mask * w2v_out
        d = w2v_out.shape[1]
        w2v_emb = torch.nn.functional.avg_pool2d(masked_out, (d, 1), stride=1, divisor_override=1) # divisor_override=1==sum_pooling
        # w2v_emb = torch.squeeze(nn.functional.avg_pool2d(w2v_out, (d, 1)))
        w2v_emb = torch.squeeze(w2v_emb)
        # bert
        bert_output = self.bert(batch['bert_tokens'], batch['mask'])
        bert_emb = torch.mean(bert_output['last_hidden_state'], axis=1)
        bert_emb = self.bert_linear(self.bert_drop(bert_emb))
        # logging.debug("term_emb size: {}\t word_emb size: {}".format(term_emb.shape, word_emb.shape))
        concat_emb = torch.cat((w2v_emb, bert_emb), -1)
        out = self.ffn(concat_emb)
        # out = self.output(x * 10)
        return out 

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        logits = self(batch)
        _real_label = batch['label'].to(logits.device)
        _pred_label = logits.argmax(1)

        loss = F.cross_entropy(logits, _real_label)
        metrics = f1measure(_real_label.cpu(), _pred_label.cpu(), list(self.labeldict.values()))

        for k, v in metrics.items():
            self.log('M/tra_'+k, metrics[k], on_step=False, on_epoch=True, prog_bar=True, logger=False)
        if self.global_step % 500 == 0:
            metrics['loss'] = loss 
            metrics['epoch'] = self.current_epoch
            print_log.log_metrics("Train", metrics)

        self.log('Loss/train', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return loss 

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch) 
        _pred_label = logits.argmax(1)
        _real_label = batch['label']
        # logging.debug("real:%s\npred:%s" %(_real_label.shape, _pred_label.shape))
        # print('[validation_step] pred_label', _pred_label)
        metrics = f1measure(_real_label.cpu(), _pred_label.cpu(), list(self.labeldict.values()))
        # for k, v in metrics.items():
            # if not type(metrics[k]) == torch.Tensor:
                # metrics[k] = torch.tensor(v)
        return metrics 
    
    # def test_epoch_endkk

    def validation_epoch_end(self, outputs):
        # logging.debug("%s\npred:%s" %(_real_label.shape, _pred_label.shape))
        res = dict()
        for key in outputs[0].keys():
            res[key] = torch.mean(torch.stack([output[key] for output in outputs])).item()
        self.log('M/val_'+key, res[key], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        print_log("-"*20)
        res['epoch'] = self.current_epoch
        print_log.log_metrics("Valid", res)
        print_log("-"*20)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=model_params['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2000, gamma=0.1, verbose=True)
        return {
            "optimizer": optimizer,
            "scheduler": scheduler
        }

def main():
    BATCH_SIZE = model_params['batch_size'] 
    TRAIN_PATH= model_params['train_path']
    MONITER = 'M/val_w_f1'

    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')    # bert用的token
    w2v_embedding_matrix, w2v_emb_dim, w2v_word_alphabet = load_w2v_pretrain_emb(model_params['w2v_mode_path'])

    train_iter = TrainAppCateDataset(
        data_dir=TRAIN_PATH, 
        ngrams=model_params['ngrams'], 
        min_freq=model_params['min_freq'], 
        label_idx=LABEL_IDX, 
        cls_sample= 'weighted_sampling',
        yield_cls=True)
    train_dataset = list(train_iter)
    logging.debug("Train dataset: %s" % train_dataset[0])

    labels_counter = _build_labels_from_dataset(dataset=train_iter, label_dir=os.path.join(data_path, "toc_tag_flatten.csv"), label_idx=LABEL_IDX) 
    num_class = len(labels_counter)
    labeldict = dict(zip(labels_counter.keys(), np.array(range(num_class))))
    logging.info("Label map: %s\nNum class: %d\nVocab w2v size: %s\nLabel dict: %s\n" %(str(labels_counter), num_class, w2v_word_alphabet.size(), labeldict))

    text_pipeline = lambda x:[w2v_word_alphabet.get_index(token) for token in x]
    label_pipeline = lambda x: labeldict[x]
    def collate_batch(batch):
        label_list, bert_list, w2v_list, mask_list, w2v_mask_list, nk_list = [], [], [], [], [], []
        for _item in batch:
            label_list.append(label_pipeline(_item['label']))
            bert_list.append(torch.tensor(tokenizer.convert_tokens_to_ids(_item['bert_tokens'])[:64]))
            mask_list.append(torch.tensor(_item['mask'][:64]))
            nk_list.append(_item['nk'])
            w2v = text_pipeline(_item['w2v_tokens'])[:64]
            w2v_list.append(torch.tensor(w2v))
            w2v_mask_list.append(torch.tensor(np.array([not v in [0, 1, 2] for v in w2v], dtype=np.int16)))

        label_list = torch.tensor(label_list, dtype=torch.int64)
        pad_bert = pad_sequence(bert_list, batch_first=True)
        pad_w2v = pad_sequence(w2v_list, batch_first=True)
        pad_w2v_mask = pad_sequence(w2v_mask_list, batch_first=True)
        pad_mask = pad_sequence(mask_list, batch_first=True)
        
        _d = {'label' : label_list, 'bert_tokens':pad_bert, 'w2v_tokens':pad_w2v, 'mask':pad_mask, 'w2v_mask':pad_w2v_mask, 'nk':nk_list} 
        return _d 

    test_iter = TrainAppCateDataset(
        data_dir=model_params['test_path'], 
        ngrams=model_params['ngrams'], 
        min_freq=model_params['min_freq'], 
        label_idx=LABEL_IDX, 
        cls_sample=None,
        yield_cls=True)
    test_dataset = list(test_iter)
    logging.debug("Test dataset: %s" % test_dataset[1])

    num_train = int(len(train_dataset) * 0.9)
    split_train, split_valid = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
    train_dataloader = DataLoader(split_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch, num_workers=4)
    valid_dataloader = DataLoader(split_valid, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch, num_workers=4)

    # logging.debug("Train dataloader: %s" % train_dataloader[0])
    id_token = tokenizer.ids_to_tokens
    for dtmp in train_dataloader:
        # logging.info(i)
        logging.debug("------Token------")
        # dtmp = train_dataloader[i]
        j = 0
        res = {}
        res['bert_line'] = [id_token.get(t, "") for t in dtmp['bert_tokens'][j].numpy() if t!=0]
        res['w2v_line'] = [w2v_word_alphabet.get_instance(t) for t in dtmp['w2v_tokens'][j].numpy() if t!=0]
        res['nk'] = dtmp['nk'][j]
        res['w2v_mask'] = dtmp['w2v_mask'][j]
        res['w2v_tokens'] = dtmp['w2v_tokens']
        for k, v in res.items():
            logging.debug("{}:\t{}".format(k, v))
        break

    model = LSTM_NET(
        w2v_alphabet=w2v_word_alphabet, 
        pretrain_w2v_embedding=w2v_embedding_matrix, 
        pretrain_w2v_dim=w2v_emb_dim, 
        fix_embedding=False,
        hidden_dim=50, num_layers=1, num_class=num_class, labeldict=labeldict, dropout=0.2)

    for n, p in model.named_parameters():
        if p.requires_grad:
            logging.info("model params: %s" %n)

    tb_log_file = get_log_file(model_params, 'tf_') 
    logging.info("Tensorflowboard file %s" %tb_log_file)
    logger = loggers.TensorBoardLogger(tb_log_file)
    es = EarlyStopping(monitor=MONITER,verbose=True, mode='max')
    checkpointing = ModelCheckpoint(dirpath=tb_log_file,
        filename='{epoch:02d}-{weighted_f1:.4f}', 
        save_top_k=1, 
        verbose=False, 
        monitor=MONITER, 
        mode='max')

    trainer = pl.Trainer(
        gpus=1, 
        # precision=16, 
        logger=logger,
        # val_check_interval=0.1, 
        # accelerator="dp", 
        gradient_clip_val = 1,
        log_every_n_steps=1000,
        min_epochs=model_params['min_epochs'], 
        max_epochs=model_params['max_epochs'],
        callbacks=[es, checkpointing]
        )

    trainer.fit(model, train_dataloader, valid_dataloader)
    trainer.test(model, test_dataloaders=test_dataloader)



if __name__ == '__main__':
    main()