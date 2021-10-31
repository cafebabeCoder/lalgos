import torch
import torch.utils.data as data
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import os
import json
import glob
import cv2
# import onnxruntime as ort
import numpy as np
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import hashlib
from transformers import BertTokenizer, BertModel, BertConfig
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import get_cosine_schedule_with_warmup

class TextImageDataset(Dataset):
    def __init__(self, train_data_path="data/train", val_data_path="data/val", mode="train"):
        super().__init__()
        self.mode = mode
        self.image_dir = "data/images"
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.data = []

        self.image_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.tokenizer = BertTokenizer.from_pretrained('RoBERTa_zh_L12_PyTorch')
        if self.mode == "train":
            with open(self.train_data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    obj = json.loads(line.strip())
                    self.data.append(obj)
        
        if self.mode == "val":
            with open(self.val_data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    obj = json.loads(line.strip())
                    self.data.append(obj)

    def __getitem__(self, index):
        obj = self.data[index]
        # image
        image_file_name = obj['image_name']
        image = self.image_preprocess(Image.open('data/images/' + image_file_name))

        # text
        title_tokens = self.tokenizer.tokenize(obj['title'])
        if type(obj['mainbody']) is str:
            mainbody_tokens = self.tokenizer.tokenize(obj['mainbody'])
        elif type(obj['mainbody']) is list and len(obj['mainbody'][0].get('answer_content', '')) > 0:
            mainbody_tokens = self.tokenizer.tokenize(obj['mainbody'][0].get('answer_content', ''))
        else:
            mainbody_tokens = self.tokenizer.tokenize('')

        if len(title_tokens) + len(mainbody_tokens) + 3 <= 384:
            text_tokens = ['[CLS]'] + title_tokens + ['[SEP]'] + mainbody_tokens + ['[SEP]']
            text_idx = self.tokenizer.convert_tokens_to_ids(text_tokens)
            text_segment = [0] + [0] * len(title_tokens) + [0] + [1] * len(mainbody_tokens) + [1]
            text_mask = [1] * len(text_idx)
        else:
            text_tokens = ['[CLS]'] + title_tokens + ['[SEP]'] + mainbody_tokens[:384-3-len(title_tokens)] + ['[SEP]']
            text_idx = self.tokenizer.convert_tokens_to_ids(text_tokens)
            text_segment = [0] + [0] * len(title_tokens) + [0] + [1] * len(mainbody_tokens[:384-3-len(title_tokens)]) + [1]
            text_mask = [1] * len(text_idx)
                
        return {'image': image, 
                'text_idx': text_idx, 
                'text_segment': text_segment, 
                'text_mask': text_mask}    

    def __len__(self):
        return len(self.data)

def collate_train(batch_data):
    def pad(input_list, pad_list, dir="left"):
        if dir == "left":
            return input_list + pad_list
        else:
            return pad_list + input_list
    
    max_text_len = max([len(e['text_idx']) for e in batch_data])
    input_images = torch.cat([e['image'].unsqueeze(0) for e in batch_data])
    input_text_idx = []
    input_text_segment = []
    input_text_mask = []
    for data in batch_data:
        input_text_idx.append(pad(data['text_idx'], [0] * (max_text_len - len(data['text_idx']))))
        input_text_segment.append(pad(data['text_segment'], [0] * (max_text_len - len(data['text_segment']))))
        input_text_mask.append(pad(data['text_mask'], [0] * (max_text_len - len(data['text_mask']))))
    
    # target = torch.tensor(np.diag([1] * len(batch_data)), dtype=torch.float32)

    return  input_images, \
            torch.tensor(input_text_idx), \
            torch.tensor(input_text_mask, dtype=torch.bool), \
            torch.tensor(input_text_segment, dtype=torch.long)

train_dataset = TextImageDataset(mode="train")
val_dataset = TextImageDataset(mode="val")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError('Reduction {} not implemented.'.format(reduction))
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, target):
        p_t = torch.where(target == 1, x, 1-x)
        fl = - 1 * (1 - p_t) ** self.gamma * torch.log(p_t)
        fl = torch.where(target == 1, fl * self.alpha, fl)
        return self._reduce(fl)

    def _reduce(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x

def recall_at_k_score(score, target, k, threshold=0.5):
    # print("[recall_at_k_score] score shape: {}, target shape: {}".format(score.shape, target.shape))
    assert score.shape == target.shape
    if k >= score.shape[1]:
        return 1
    else:
        res = 0.0
        for score, label in zip(score, target):
            score_list = list(enumerate(score.tolist()))
            label_list = list(enumerate(label.tolist()))
            score_list.sort(key=lambda tup: tup[1], reverse=True)
            # label_list.sort(key=lambda tup: tup[1], reversed=True)
            # label大于threshold的为正样本
            label_list = list(filter(lambda tup: tup[1] > threshold, label_list))
            score_topk_set = set([e[0] for e in score_list[:k]])
            label_topk_set = set([e[0] for e in label_list])
            recall_at_k = len(score_topk_set & label_topk_set) / len(label_topk_set)
            # print("recall_at_k:", recall_at_k)
            res += recall_at_k
        return res / score.shape[0]

class TextImageDSSM(pl.LightningModule):
    def __init__(self, batch_size=256):
        super(TextImageDSSM, self).__init__()
        # dataloader
        self.epochs = 10
        self.batch_size = batch_size

        self.bert = BertModel(BertConfig())
        self.bert.from_pretrained("RoBERTa_zh_L12_PyTorch")
        self.resnet50 = torch.hub.load('pytorch/vision:v0.8.0', 'resnet50', pretrained=True)
        self.image_encoder = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.text_fc1 = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(inplace=True),
        )
        self.image_fc1 = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True)
        )

        # self.target = torch.tensor(np.diag([1] * self.batch_size), dtype=torch.float32).cuda()
        # self.target = torch.tensor(np.diag([1] * self.batch_size), dtype=torch.float32)
        self.i2t_softmax = nn.Softmax(dim=0)
        self.t2i_softmax = nn.Softmax(dim=1)
        self.focal_loss = FocalLoss(reduction="sum")

        for name, param in self.named_parameters():
            if not name.startswith('bert.encoder.layer.9') and \
                not name.startswith('bert.encoder.layer.10') and \
                not name.startswith('bert.encoder.layer.11') and \
                not name.startswith('bert.pooler') and \
                not name.startswith('resnet50.layer3') and \
                not name.startswith('resnet50.layer4') and \
                not name.startswith('resnet50.fc') and \
                not name.startswith('text_fc1') and \
                not name.startswith('image_fc1'):
                param.requires_grad = False

    def forward(self, batch_data):
        input_image, input_text_idx, input_text_mask, input_text_segment = batch_data
        image_embedding = self.image_encoder(input_image)
        image_embedding = image_embedding.squeeze(-1).squeeze(-1)
        image_embedding = self.image_fc1(image_embedding)
        image_embedding = F.normalize(image_embedding, p=2, dim=1)

        bert_output = self.bert(input_text_idx, input_text_mask, input_text_segment)
        text_embedding = torch.mean(bert_output['last_hidden_state'], axis=1)
        text_embedding = self.text_fc1(text_embedding)
        text_embedding = F.normalize(text_embedding, p=2, dim=1)

        sim = image_embedding.mm(text_embedding.T)
        i2t_score = self.i2t_softmax(sim)
        t2i_score = self.t2i_softmax(sim)
        return i2t_score, t2i_score
        
    def training_step(self, batch, batch_nb):
        i2t_score, t2i_score = self(batch)
        # print(i2t_score.shape, t2i_score.shape, self.target.shape)
        # target = self.target.to(i2t_score.device)
        target = torch.eye(i2t_score.shape[0]).to(i2t_score.device)
        i2t_loss = self.focal_loss(i2t_score, target)
        t2i_loss = self.focal_loss(t2i_score, target)
        loss = i2t_loss + t2i_loss
        tensorboard_logs = {'train_loss': loss, 'i2t_loss': i2t_loss, 't2i_loss': t2i_loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        i2t_score, t2i_score = self(batch)
        # target = self.target.to(i2t_score.device)
        target = torch.eye(i2t_score.shape[0]).to(i2t_score.device)       
        i2t_recall_at_1 = recall_at_k_score(i2t_score, target, k=1)
        i2t_recall_at_3 = recall_at_k_score(i2t_score, target, k=3)
        i2t_recall_at_10 = recall_at_k_score(i2t_score, target, k=10)
        it2_recall_at_20 = recall_at_k_score(i2t_score, target, k=20)
        t2i_recall_at_1 = recall_at_k_score(t2i_score, target, k=1)
        t2i_recall_at_3 = recall_at_k_score(t2i_score, target, k=3)
        t2i_recall_at_10 = recall_at_k_score(t2i_score, target, k=10)
        t2i_recall_at_20 = recall_at_k_score(t2i_score, target, k=20)
        res = {
            'i2t_recall_at_1'   : i2t_recall_at_1,
            'i2t_recall_at_3'   : i2t_recall_at_3,
            'i2t_recall_at_10'  : i2t_recall_at_10,
            'it2_recall_at_20'  : it2_recall_at_20,
            't2i_recall_at_1'   : t2i_recall_at_1,
            't2i_recall_at_3'   : t2i_recall_at_3,
            't2i_recall_at_10'  : t2i_recall_at_10,
            't2i_recall_at_20'  : t2i_recall_at_20
        }
        # print('[validation_step] ', res)
        return res

    def validation_epoch_end(self, outputs):
        
        res = dict()
        res['i2t_recall_at_1'] = torch.mean(torch.cat([output['i2t_recall_at_1'] for output in outputs])).item()
        res['i2t_recall_at_3'] = torch.mean(torch.cat([output['i2t_recall_at_3'] for output in outputs])).item()
        res['i2t_recall_at_10'] = torch.mean(torch.cat([output['i2t_recall_at_10'] for output in outputs])).item()
        res['it2_recall_at_20'] = torch.mean(torch.cat([output['it2_recall_at_20'] for output in outputs])).item()
        res['t2i_recall_at_1'] = torch.mean(torch.cat([output['t2i_recall_at_1'] for output in outputs])).item()
        res['t2i_recall_at_3'] = torch.mean(torch.cat([output['t2i_recall_at_3'] for output in outputs])).item()
        res['t2i_recall_at_10'] = torch.mean(torch.cat([output['t2i_recall_at_10'] for output in outputs])).item()
        res['t2i_recall_at_20'] = torch.mean(torch.cat([output['t2i_recall_at_20'] for output in outputs])).item()
        print("[validation_epoch_end]", res)
        return {
            'val_matric': res['i2t_recall_at_3'],
            'log': res
        }

    def train_dataloader(self):
        return DataLoader(train_dataset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=64,
                        pin_memory=True,
                        drop_last=True,
                        collate_fn=collate_train)

    def val_dataloader(self):
        return DataLoader(val_dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=64,
                        pin_memory=True,
                        drop_last=True,
                        collate_fn=collate_train)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-05, eps=1e-4)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                            len(train_dataset.data) // self.batch_size, 
                                            self.epochs * len(train_dataset.data) // self.batch_size)
        return {
            "optimizer": optimizer,
            "scheduler": scheduler
        }


if __name__ == '__main__':
    model = TextImageDSSM(batch_size=1024)
    trainer = pl.Trainer(gpus=8, precision=16, val_check_interval=0.1, accelerator="dp", min_epochs=model.epochs, max_epochs=model.epochs,
                         checkpoint_callback=ModelCheckpoint(dirpath='./checkpoint/', filename='{epoch:02d}-{val_matric:.4f}', save_top_k=1,
                                                             verbose=True, monitor='val_matric', mode='max'))
    trainer.fit(model)

    # train_dataset = TextImageDataset(mode="train")
    # train_dataloader = DataLoader(train_dataset,
    #                      batch_size=256,
    #                      shuffle=True,
    #                      num_workers=64,
    #                      pin_memory=True,
    #                      drop_last=True,
    #                      collate_fn=collate_train)

    # for idx, data in enumerate(train_dataloader):
    #     # print(data[0].shape, data[1].shape, data[2].shape, data[3].shape)
    #     a = model.training_step(data, idx)
    #     print(idx, a)
    #     # print(torch.mean(a['last_hidden_state'], axis=1))
    #     # print(a['pooler_output'])
    #     # break
    #     # print(a.shape, b.shape)
    #     # sim_matrix = model(data)
    #     # print(sim_matrix)
