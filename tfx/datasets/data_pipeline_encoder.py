#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2020/08/20 
# @Author  : lorineluo
# @File    : data_pipeline_encoder.py

import os
import tensorflow as tf

from datasets.AppTokenizerMap import AppTokenizer
from model.encoder.encoder_args_core import dataset_params, model_params
from common.file_utils import get_file_list
from common.date_utils import tf_date_trans

# 样本列
_SAMPLE_NAMES = ['uin', 'his_seq', 'his_ts', 'his_timelong', 'today_seq', 'today_ts', 'today_timelong']
_CSV_COLUMN_DEFAULTS = ['' for _ in range(len(_SAMPLE_NAMES))]
_CSV_COLUMN_TYPES = [tf.constant([''], dtype=tf.string) for _ in range(len(_SAMPLE_NAMES))]
_CSV_COLUMN_NAMES = ['his_seq', 'today_seq','his_week', 'today_week', 'his_hour', 'today_hour', 'enc_padding_mask']

# 定义参数
padding_input_max_length = dataset_params['padding_input_max_length']
appPath = dataset_params['appuin_file']

# 初始化appTokenizer
appTokenizer = AppTokenizer(appPath, uin_idx=-1) # 使用appid
VOCAB_SIZE= appTokenizer.vocab_size

# 解析点击序列， inp = his_seq, outp = today_seq
def sample_parse_seq(inp, outp):
    inp = appTokenizer.tf_encode(inp)
    outp = [appTokenizer.tf_encode(outp)[0]]
    pad_inp = tf.keras.preprocessing.sequence.pad_sequences([inp], maxlen=padding_input_max_length, padding="post", truncating='pre')[0]

    return pad_inp, outp

# 解析时间戳
def sample_parse_timestamp(inp, outp):
    # trans his_seq 
    his_week, his_hour = tf_date_trans(tf.strings.split(inp, "|").numpy())
    pad_his_week = tf.keras.preprocessing.sequence.pad_sequences([his_week], maxlen=padding_input_max_length, padding="post", truncating='pre')[0]
    pad_his_hour = tf.keras.preprocessing.sequence.pad_sequences([his_hour], maxlen=padding_input_max_length, padding="post", truncating='pre')[0]

    # trans today_seq
    today_week, today_hour = tf_date_trans(tf.strings.split(outp, "|").numpy())
    today_week = today_week[0]
    today_hour = today_hour[0]

    return pad_his_week, today_week, pad_his_hour, today_hour

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[tf.newaxis, tf.newaxis, :]

def tf_sample_parse(*items):
    # 解析点击序列
    his_seq, today_click = tf.py_function(sample_parse_seq, [items[1], items[4]], [tf.int32, tf.int32])
    his_seq.set_shape([padding_input_max_length])
    today_click.set_shape([1])

    # 解析时间戳
    his_week, today_week, his_hour, today_hour = tf.py_function(sample_parse_timestamp, [items[2], items[5]], [tf.int32, tf.int32, tf.int32, tf.int32])
    his_week.set_shape([padding_input_max_length])
    his_hour.set_shape([padding_input_max_length])
    today_week.set_shape([])
    today_hour.set_shape([])

    # 把padding的0部分mask掉，不参与attention计算
    enc_padding_mask = create_padding_mask(his_seq)

    return his_seq, today_click, his_week, today_week, his_hour, today_hour, enc_padding_mask

# 最后一层包装，这里是包装成map
def _parse_line(*fields):
    features = dict(zip(_CSV_COLUMN_NAMES,fields))
    labels = features.pop('today_seq')

    return features, labels

def train_input_fn(data_path, start_time, days, batch_size):
    file_list = get_file_list(data_path, start_time, days)
    print("input file list:\n{}".format("\n".join(file_list)))

    dataset = tf.data.experimental.CsvDataset(file_list, _CSV_COLUMN_TYPES, header=False, field_delim=',') \
        .map(tf_sample_parse) \
        .shuffle(600) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(tf.data.experimental.AUTOTUNE) \
        .map(_parse_line)

    return dataset