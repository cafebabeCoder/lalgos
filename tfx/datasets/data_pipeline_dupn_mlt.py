#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2020/08/20 
# @Author  : lorineluo
# @File    : data_pipepline.py

import os
from datetime import datetime, timedelta
import tensorflow as tf

from datasets.AppTokenizer import AppTokenizer 
from model.dupn.dupn_mlt_args_core import dataset_params, model_params
from common.file_utils import get_file_list


_CSV_COLUMN_NAMES = ['ds', 'useruin', 'star', 'his_seq_appid', 'his_seq_ts','his_seq_tl','wxapp_type', 'toc_catg_f', 'toc_catg_second', 'toc_tag',
    'visit', 'gender', 'region_code', 'language', 'platform', 'device', 'age', 'grade', 'city_level', 'install', 
    'install_ecc','live_appuin', 'room_id', 'appuin_roomid']
_CSV_COLUMN_DEFAULTS = ['', '', '', '', '', '', '', '', '', '', 
     '', 0, '', 0, 0, '', 0, 0, 0, '', '', '', '', '']

_CSV_COLUMN_TYPES = [tf.constant([''], dtype=tf.string) if i =='' else tf.int32 for i in _CSV_COLUMN_DEFAULTS]

USER_HASH_BUCKET_SIZE = model_params['user_hash_size'] 

appPath = dataset_params['appuin_file']
appTokenizer = AppTokenizer(appPath)
VOCAB_SIZE= appTokenizer.vocab_size

# liveAppPath = dataset_params['live_appuin_file']
# liveAppTokenizer = AppTokenizer(liveAppPath)
# LIVE_VOCAB_SIZE= liveAppTokenizer.vocab_size

with open("/apdcephfs/private_lorineluo/python_data_analysis/data/dupn/common/app_category", 'r', encoding='utf8') as f:
    app_cates = f.readlines()
app_cate_1 = app_cates[0].strip().split(",")
app_cate_1_map = {'':0}
app_cate_1_map.update(dict(zip(app_cate_1, range(1, len(app_cate_1)))))
app_cate_2 = app_cates[1].strip().split(",")
app_cate_2_map = {'':0}
app_cate_2_map.update(dict(zip(app_cate_2, range(1, len(app_cate_2)))))
app_tag = app_cates[2].strip().split(",")
app_tag_map = {'':0}
app_tag_map.update(dict(zip(app_tag, range(1, len(app_tag)))))

# 获取仅直播的app
def get_live_uin_name_dict(voc_dir=dataset_params['live_appuin_file'], uin_idx=0):
    dictionary = dict()
    file = open(voc_dir, 'r', encoding='utf-8')
    dictionary["UNK"] = "UNK"
    for line in file.readlines():
        line = line.strip()
        k = line.split(',')[1]
        v = line.split(',')[uin_idx]
        dictionary[k] = v
    dictionary['STAR'] = "STAR"
    dictionary['END'] = "END"

    # 注意关闭文件
    file.close()

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return reversed_dictionary


def sample_parse(visit, star, appuin, toc_catg_f, toc_catg_second, toc_tag):
    visit = appTokenizer.tf_encode(visit)
    star = appTokenizer.tf_encode(star)
    appuin = appTokenizer.tf_encode(appuin)
    appuin = tf.squeeze(appuin, axis=0)
    if tf.size(appuin) == 0:
        appuin = [0]
    pvisit = tf.keras.preprocessing.sequence.pad_sequences(visit, maxlen=dataset_params['padding_max_length'], padding="post", truncating='pre')[0]
    pstar = tf.keras.preprocessing.sequence.pad_sequences(star, maxlen=dataset_params['padding_max_length'], padding="post", truncating='pre')[0]
    
    toc_catg_f = [app_cate_1_map.get(i, 0) for i in toc_catg_f.numpy().decode().split("|")]
    toc_catg_f = tf.keras.preprocessing.sequence.pad_sequences([toc_catg_f], maxlen=dataset_params['padding_max_length'], padding="post", truncating='pre')[0]
    toc_catg_second = [app_cate_2_map.get(i, 0) for i in toc_catg_second.numpy().decode().split("|")]
    toc_catg_second = tf.keras.preprocessing.sequence.pad_sequences([toc_catg_second], maxlen=dataset_params['padding_max_length'], padding="post", truncating='pre')[0]
    toc_tag = [app_tag_map.get(i, 0) for i in toc_tag.numpy().decode().split("|")]
    toc_tag = tf.keras.preprocessing.sequence.pad_sequences([toc_tag], maxlen=dataset_params['padding_max_length'], padding="post", truncating='pre')[0]
    return pvisit, pstar, appuin, toc_catg_f, toc_catg_second, toc_tag

def tf_sample_parse(*items):
    visit, star, appuin, toc_catg_f, toc_catg_second, toc_tag = tf.py_function(sample_parse, 
        [items[3], items[2], items[21], items[7], items[8], items[9]], [tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32])
    visit.set_shape([None])
    star.set_shape([None])
    # appuin.set_shape([None])

    return items, appuin, visit, star, toc_catg_f, toc_catg_second, toc_tag

def _parse_line(*fields):
    _CSV_COLUMN_NAMES_ADD = ["live_appuin", "his_seq_appid", "star", "toc_catg_f", "toc_catg_second", "toc_tag"]
    # 将结果打包成字典
    features = dict(zip(_CSV_COLUMN_NAMES,fields[0]))
    features.update(dict(zip(_CSV_COLUMN_NAMES_ADD, fields[1:])))
    
    features['uin'] = tf.strings.to_hash_bucket_strong(
        features['useruin'], USER_HASH_BUCKET_SIZE, name="uin", key=[1,0])
    features['region_code'] = tf.strings.to_hash_bucket_strong(features['region_code'], 100, name='region_code', key=[1,0])
    features['device'] = tf.strings.to_hash_bucket_strong(features['device'], 100, name='device', key=[1,0])

    # 扔掉不需要的数据，将标签从特征中分离
    pop_keys = ['ds', 'his_seq_ts', 'his_seq_tl', 'wxapp_type', 'visit', 'install', 'install_ecc', 'room_id', 'appuin_roomid']
    [features.pop(k) for k in pop_keys]
    label = tf.squeeze(features.pop('live_appuin'))

    return features, label


def train_input_fn(data_path, start_time, days, batch_size):
    file_list = get_file_list(data_path, start_time, days)

    print("input file list:\n{}".format("\n".join(file_list)))

        # .shuffle(4096) \
    dataset = tf.data.experimental.CsvDataset(file_list, _CSV_COLUMN_TYPES, header=False)\
        .map(tf_sample_parse) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(tf.data.experimental.AUTOTUNE) \
        .map(_parse_line)

    return dataset

def predict_input_fn(data_path, start_time, days, batch_size):
    file_list = get_file_list(data_path, start_time, days)

    print("input file list:\n{}".format("\n".join(file_list)))

    dataset = tf.data.experimental.CsvDataset(file_list, _CSV_COLUMN_TYPES, header=False)\
        .map(tf_sample_parse) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(tf.data.experimental.AUTOTUNE) \
    .map(_parse_line)

    return dataset