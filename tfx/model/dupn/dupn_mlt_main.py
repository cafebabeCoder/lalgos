#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2020/08/20 
# @Author  : lorineluo
# @File    : dupn_mtl_main.py

import sys
import tensorflow as tf
import os
import argparse
# 注册路径， 否则 找不到models
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../..")))
sys.path.insert(0, "/apdcephfs/private_lorineluo/python_data_analysis/src/models")
sys.path.insert(0, "/mnt/wfs/mmcommwfssz/user_lorineluo/dupn/src/models") 
# 设置 gpu， 否则 train_step error
gpus = tf.config.list_physical_devices('GPU')
print("-------")
if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("gpu: ", gpus)

print("path: ", os.path.abspath(os.path.join(os.getcwd(), "../..")))
print("tf version: ", tf.__version__)
print("-------")

import numpy as np
from model.dupn import dupn 
from model.dupn.dupn_mlt_args_core import run_params, model_params, dataset_params
from datasets import data_pipeline_dupn_mlt 
from common.optimizers import CustomSchedule
from layers.attention import point_wise_feed_forward_network, BahdanauAttention, MyBias
from common.file_utils import Print_log 

parser = argparse.ArgumentParser()
parser.add_argument( '--ds', default='', type=str)
parser.add_argument( '--mode', default='', type=str)
parser.add_argument( '--data_path', default='', type=str)
args, _ = parser.parse_known_args()

# tf.keras.backend.set_floatx('float16')

if args.ds != "":
    dataset_params['date'] = args.ds
if args.mode !="":
    run_params['mode'] = args.mode
if args.data_path != "":
    dataset_params['data_path'] = args.data_path

if __name__ == '__main__':
    log_file = os.path.join(dataset_params['log_path'], "%s_%s") % (dataset_params['model_version'], dataset_params['date'])
    print_log = Print_log(log_file) 
    print_log("\n".join(["\n", str(model_params), str(run_params), str(dataset_params), "\n"]))

    VOCAB_SIZE = data_pipeline_dupn_mlt.VOCAB_SIZE 

    SAVE_USER = True 
    SAVE_ITEM = False 
    d_model = model_params['model_dim']
    input_size = VOCAB_SIZE
    output_size = VOCAB_SIZE
    user_size= model_params['user_hash_size']
    user_emb_dim=model_params['user_dim']
    item_emb_dim= model_params['appuin_dim']
    enc_units= model_params['enc_units']
    att_units=model_params['att_units']
    basic_d_model=model_params['basic_dim']
    d_model=model_params['model_dim']
    batch_size=run_params['batch_size']
    max_length = dataset_params['padding_max_length']

    #行为序列中： 
    #   item feature: his_seq_appid, toc_catg_f, toc_catg_second, toc_tag, visit
    #   behavior feature: his_seq_ts, his_seq_tl,
    #用户基础特征： useruin, star, gender, region_code, language, platform, device, age, grade, city_level, install, install_ecc
    inputs={
        'useruin': tf.keras.Input(name='useruin', shape=(), dtype='string'),
        'star': tf.keras.Input(name='star', shape=(max_length, ), dtype='int32'),
        'his_seq_appid': tf.keras.Input(name='his_seq_appid', shape=(max_length, ), dtype='int32'),
        'toc_catg_f': tf.keras.Input(name='toc_catg_f', shape=(max_length, ), dtype='int32'),
        'toc_catg_second': tf.keras.Input(name='toc_catg_second', shape=(max_length, ), dtype='int32'),
        'toc_tag': tf.keras.Input(name='toc_tag', shape=(max_length, ), dtype='int32'),
        'gender': tf.keras.Input(name='gender', shape=(), dtype='int32'),
        'region_code': tf.keras.Input(name='region_code', shape=(), dtype='int32'),
        'language': tf.keras.Input(name='language', shape=(), dtype='int32'),
        'platform': tf.keras.Input(name='platform', shape=(), dtype='int32'),
        'device': tf.keras.Input(name='device', shape=(), dtype='int32'),
        'age': tf.keras.Input(name='age', shape=(), dtype='int32'),
        'grade': tf.keras.Input(name='grade', shape=(), dtype='int32'),
        'city_level': tf.keras.Input(name='city_level', shape=(), dtype='int32'),
        'uin': tf.keras.Input(name='uin', shape=(), dtype='int32'),
    }

    app_embedding_table = tf.keras.layers.Embedding(input_size + 1, item_emb_dim, name='app_emb')
    user_emb = tf.keras.layers.Embedding(user_size, user_emb_dim, name='user_emb')(inputs['uin'])
    star_emb = app_embedding_table(inputs['star'])
    star_emb = tf.keras.layers.GlobalAveragePooling1D()(star_emb)
    gender_emb = tf.keras.layers.Embedding(model_params['gender_size'], model_params['gender_dim'], name='gender_emb')(inputs['gender'])
    region_emb = tf.keras.layers.Embedding(model_params['region_size'], model_params['region_dim'], name='region_emb')(inputs['region_code'])
    language_emb = tf.keras.layers.Embedding(model_params['language_size'], model_params['language_dim'], name='language_emb')(inputs['language'])
    platform_emb = tf.keras.layers.Embedding(model_params['platform_size'], model_params['platform_dim'], name='platform_emb')(inputs['platform'])
    device_emb = tf.keras.layers.Embedding(model_params['device_size'], model_params['device_dim'], name='device_emb')(inputs['device'])
    age_emb = tf.keras.layers.Embedding(model_params['age_size'], model_params['age_dim'], name='age_emb')(inputs['age'])
    grade_emb = tf.keras.layers.Embedding(model_params['grade_size'], model_params['grade_dim'], name='grade_emb')(inputs['grade'])
    city_level_emb = tf.keras.layers.Embedding(model_params['city_level_size'], model_params['city_level_dim'], name='city_level_emb')(inputs['city_level'])

    basic_dense = tf.keras.layers.Concatenate(axis=-1, name='basic')([user_emb, star_emb, gender_emb, \
            region_emb, language_emb, platform_emb, device_emb, age_emb, grade_emb, city_level_emb])
    basic_emb = point_wise_feed_forward_network(basic_d_model, basic_d_model * 2)(basic_dense)

    his_seq_appid_emb = app_embedding_table(inputs['his_seq_appid'])
    # wxapp_type_emb = tf.keras.layers.Embedding(model_params['wxapp_type_size'], model_params['wxapp_type_dim'], name='wxapp_type_size')(inputs['wxapp_type'])
    toc_catg_f_emb = tf.keras.layers.Embedding(model_params['toc_catg_f_size'], model_params['toc_catg_f_dim'], name='toc_catg_f_dim')(inputs['toc_catg_f'])
    toc_catg_second_emb = tf.keras.layers.Embedding(model_params['toc_catg_second_size'], model_params['toc_catg_second_dim'], name='toc_catg_second_dim')(inputs['toc_catg_second'])
    toc_tag_emb = tf.keras.layers.Embedding(model_params['toc_tag_size'], model_params['toc_tag_dim'], name='toc_tag_dim')(inputs['toc_tag'])
    behavior_emb = tf.keras.layers.Concatenate(axis=-1, name='item')([his_seq_appid_emb, toc_catg_f_emb, toc_catg_second_emb, toc_tag_emb])
    enc_outputs = tf.keras.layers.GlobalAveragePooling1D()(behavior_emb)
    
    # rnn = tf.keras.layers.LSTM(enc_units, 
    #                         return_sequences=True, 
    #                         return_state=True, 
    #                         recurrent_initializer='glorot_uniform')
    # behavior_output, state_h, state_c = rnn(behavior_emb) 
    
    context_vector, attention_weights = BahdanauAttention(att_units)(basic_emb, enc_outputs)
    rep = tf.concat([basic_emb, context_vector], axis=-1)

    output_emb = tf.keras.layers.Dense(VOCAB_SIZE + 1)(rep)

    # bias_layer = MyBias(VOCAB_SIZE)

    model = tf.keras.Model(inputs = inputs, outputs = output_emb)

    # Instantiate an optimizer.
    # learning_rate = CustomSchedule(model_params['model_dim'], dtype=tf.float32) 
    learning_rate = 0.01
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # metrics
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(dtype=tf.float32)
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(dtype=tf.float32)
    train_topk_metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(k = i, dtype=tf.float32) for i in run_params['topk_acc']]
    val_topk_metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(k = i, dtype=tf.float32) for i in run_params['topk_acc']]

    # checkpoint
    checkpoint_path = os.path.join(dataset_params['model_save_path'], "ckp")
    ckpt = tf.train.Checkpoint(step=tf.Variable(0, name="step", dtype='int64'), model=model, 
        optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=run_params['max_to_keep'])

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print_log('Latest checkpoint restored!!')

    # tensorboard
    if run_params['is_tensorboard']:
        train_tb_path = os.path.join(dataset_params['model_save_path'], "tb/train")
        val_tb_path = os.path.join(dataset_params['model_save_path'], "tb/val")
        train_summary_writer = tf.summary.create_file_writer(train_tb_path)
        val_summary_writer = tf.summary.create_file_writer(val_tb_path)
    

    # 获取直播场景appuin
    live_appuin_name_map = data_pipeline_dupn_mlt.get_live_uin_name_dict(voc_dir=dataset_params['live_appuin_file'], uin_idx=0) 
    live_app_ids = data_pipeline_dupn_mlt.appTokenizer.encode(list(live_appuin_name_map.keys())) 
    live_app_ids = np.squeeze(np.array([i for i in live_app_ids if len(i) == 1]))
    live_app_len = len(live_app_ids)


    # loss 梯度剪切
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    def loss_fn(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    # metrics
    train_topk_metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(k = i, name="top{}_acc".format(i)) for i in [1, 10, 30]]

    def acc_fn(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        acc_ = train_topk_metrics[0](real, pred)
        mask = tf.cast(mask, dtype=acc_.dtype)
        acc_ *= mask

        return tf.reduce_sum(acc_)/tf.reduce_sum(mask)

    def acc10_fn(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = train_topk_metrics[1](real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    def acc30_fn(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = train_topk_metrics[2](real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

            # # dataset
    dataset = data_pipeline_dupn_mlt.train_input_fn( 
        dataset_params['data_path'], 
        dataset_params['date'], 
        dataset_params['duration'], 
        run_params['batch_size'])
    
        # print log
    class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if(batch % run_params['log_step_count_steps'] == 0):
                log = "[training step {}]\t".format(batch)
                log = log + "\t".join(["{}:{:.4f}".format(i, j) for i,j in logs.items()])
                print_log(log)

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss = loss_fn,
                  metrics = [acc_fn, acc10_fn, acc30_fn]) 

    model.fit(
        dataset, 
        epochs=run_params['epochs'],
        callbacks = [LossAndErrorPrintingCallback()]) 
