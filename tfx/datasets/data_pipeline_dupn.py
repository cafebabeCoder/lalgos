#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2020/08/20 
# @Author  : lorineluo
# @File    : dupn_main.py

import os
from datetime import datetime, timedelta
import tensorflow as tf

from datasets.AppTokenizer import *
from model.dupn.dupn_args_core import dataset_params, model_params
from common.file_utils import *

_CSV_COLUMN_NAMES = ['ds', 'useruin', 'star', 'visit', 'gender', 'region_code',
    'language', 'platform', 'device', 'age', 'grade', 'city_level', 
    'install', 'install_ecc','live_appuin', 'room_id', 'appuin_roomid']
_CSV_COLUMN_DEFAULTS = ['', '', '', '', 0, '', 0, 0, '', 0, 0, 0, '', '', '', '', '']
_CSV_COLUMN_TYPES = [tf.string, tf.string, tf.constant([''], dtype=tf.string), tf.constant([''], dtype=tf.string), tf.int32, 
    tf.string, tf.int32, tf.int32, tf.string, tf.int32, 
    tf.int32, tf.int32, tf.constant([''], dtype=tf.string), tf.constant([''], dtype=tf.string),tf.constant([''], dtype=tf.string), 
    tf.constant([''], dtype=tf.string),tf.constant([''], dtype=tf.string)]

USER_HASH_BUCKET_SIZE = model_params['user_hash_size'] 
# APP_HASH_BUCKET_SIZE = dataset_params['appuin_hash_size']

appPath = dataset_params['appuin_file']
appTokenizer = AppTokenizer(appPath)
VOCAB_SIZE= appTokenizer.vocab_size

liveAppPath = dataset_params['live_appuin_file']
liveAppTokenizer = AppTokenizer(liveAppPath)
LIVE_VOCAB_SIZE= liveAppTokenizer.vocab_size

def sample_parse(visit, star, appuin):
    visit = appTokenizer.tf_encode(visit)
    star = appTokenizer.tf_encode(star)
    appuin = liveAppTokenizer.tf_encode(appuin)
    appuin = tf.squeeze(appuin, axis=0)
    if tf.size(appuin) == 0:
        appuin = [0]
    pvisit = tf.keras.preprocessing.sequence.pad_sequences(visit, maxlen=dataset_params['padding_max_length'], padding="post", truncating='pre')[0]
    pstar = tf.keras.preprocessing.sequence.pad_sequences(star, maxlen=dataset_params['padding_max_length'], padding="post", truncating='pre')[0]

    return pvisit, pstar, appuin

def tf_sample_parse(*items):
    visit, star, appuin = tf.py_function(sample_parse, [items[3], items[2], items[14]], [tf.int32, tf.int32, tf.int32])
    visit.set_shape([None])
    star.set_shape([None])
    appuin.set_shape([None])

    return items, appuin, visit, star

def _parse_line(*fields):
    _CSV_COLUMN_NAMES_ADD = ["live_appuin", "visit", "star"]
    # 将结果打包成字
    features = dict(zip(_CSV_COLUMN_NAMES,fields[0]))
    features.update(dict(zip(_CSV_COLUMN_NAMES_ADD, fields[1:])))

    # features['appuin'] = tf.strings.to_hash_bucket_strong(
        # features['appuin'], APP_HASH_BUCKET_SIZE, name="appuin", key=[2,3])
    features['uin'] = tf.strings.to_hash_bucket_strong(
        features['useruin'], USER_HASH_BUCKET_SIZE, name="uin", key=[1,0])
    features['region_code'] = tf.strings.to_hash_bucket_strong(features['region_code'], 100, name='region_code', key=[1,0])
    features['device'] = tf.strings.to_hash_bucket_strong(features['device'], 100, name='device', key=[1,0])

    # 将标签从特征中分离
    label = features['live_appuin']

    return features, label


def train_input_fn(data_path, start_time, days, batch_size):
    file_list = get_file_list(data_path, start_time, days)

    print("input file list:\n{}".format("\n".join(file_list)))

    dataset = tf.data.experimental.CsvDataset(file_list, _CSV_COLUMN_TYPES, header=False)\
        .map(tf_sample_parse) \
        .shuffle(5000) \
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