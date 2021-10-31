#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2020/08/20 
# @Author  : lorineluo
# @File    : dupn_main.py

import os
from datetime import datetime, timedelta
import tensorflow as tf

from datasets.AppTokenizerMap import AppTokenizer
from model.transformer.transformer_arg_core import dataset_params, model_params
from common.file_utils import get_file_list
from common.date_utils import tf_date_trans

# 样本列： uin, his_seq, his_ts, his_timelong, today_seq, today_ts, today_timelong

_CSV_COLUMN_NAMES = ['useruin', 'his_seq', 'his_ts','his_tl', 'today_seq', 'today_ts', 'today_tl']
_CSV_COLUMN_DEFAULTS = ['' for _ in range(len(_CSV_COLUMN_NAMES))] #['', '', '', '', '']
_CSV_COLUMN_TYPES = [tf.string, tf.string, tf.string, tf.string, tf.string, tf.string, tf.string]

input_max_length = dataset_params['padding_input_max_length']
output_max_length = dataset_params['padding_output_max_length']
appPath = dataset_params['appuin_file']

appTokenizer = AppTokenizer(appPath, uin_idx=-1)
VOCAB_SIZE= appTokenizer.vocab_size

def sample_parse_seq(inp, outp):
    # V-2 = START, V-1 = END
    inp = [VOCAB_SIZE - 2] + appTokenizer.tf_encode(inp) + [VOCAB_SIZE - 1]
    outp = [VOCAB_SIZE - 2] + appTokenizer.tf_encode(outp) + [VOCAB_SIZE - 1]
    pinp = tf.keras.preprocessing.sequence.pad_sequences([inp], maxlen=input_max_length, padding="post", truncating='pre')[0]
    poutp = tf.keras.preprocessing.sequence.pad_sequences([outp], maxlen=output_max_length, padding="post", truncating='pre')[0]

    return pinp, poutp 

def sample_parse_timestamp(inp, outp):
    his_week, his_hour = tf_date_trans(tf.strings.split(inp, "|").numpy())
    pad_his_week = tf.keras.preprocessing.sequence.pad_sequences([[0] + his_week + [0]], maxlen=input_max_length, padding="post", truncating='pre')[0]
    pad_his_hour = tf.keras.preprocessing.sequence.pad_sequences([[0] + his_hour + [0]], maxlen=input_max_length, padding="post", truncating='pre')[0]

    today_week, today_hour = tf_date_trans(tf.strings.split(outp, "|").numpy())
    pad_today_week = tf.keras.preprocessing.sequence.pad_sequences([[0] + today_week + [0]], maxlen=output_max_length, padding="post", truncating='pre')[0]
    pad_today_hour = tf.keras.preprocessing.sequence.pad_sequences([[0] + today_hour + [0]], maxlen=output_max_length, padding="post", truncating='pre')[0]

    return pad_his_week, pad_today_week, pad_his_hour, pad_today_hour

def tf_sample_parse(*items):
    his_seq, today_seq = tf.py_function(sample_parse_seq, [items[1], items[4]], [tf.int32, tf.int32])
    his_seq.set_shape([input_max_length])
    today_seq.set_shape([output_max_length])

    his_week, today_week, his_hour, today_hour = tf.py_function(sample_parse_timestamp, [items[2], items[5]], [tf.int32, tf.int32, tf.int32, tf.int32])
    his_week.set_shape([input_max_length])
    today_week.set_shape([output_max_length])
    his_hour.set_shape([input_max_length])
    today_hour.set_shape([output_max_length])
    return his_seq, today_seq, his_week, today_week, his_hour, today_hour

def _parse_line(*fields):
    _CSV_COLUMN_NAMES = ['his_seq', 'today_seq','his_week', 'today_week', 'his_hour', 'today_hour']
    features = dict(zip(_CSV_COLUMN_NAMES,fields))

    return features

def filter_max_length(x, y, x_max_length = input_max_length, y_max_length = output_max_length):
    return tf.logical_and(tf.size(x) <= x_max_length, tf.size(y) <= y_max_length)

def train_input_fn(data_path, start_time, days, batch_size):
    file_list = get_file_list(data_path, start_time, days)
    print("input file list:\n{}".format("\n".join(file_list)))

    dataset = tf.data.experimental.CsvDataset(file_list, _CSV_COLUMN_TYPES, header=False, field_delim=',') \
        .map(tf_sample_parse) \
        .shuffle(100) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(tf.data.experimental.AUTOTUNE) \
        .cache()\
        .map(_parse_line)

    return dataset