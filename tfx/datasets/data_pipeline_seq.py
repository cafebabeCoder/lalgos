'''
Created on May, 2020
Processing datasets.

@author: lorineluo
'''

import os
from datetime import datetime, timedelta
import tensorflow as tf
import tensorflow_datasets as tfds

from datasets.AppTokenizer import *
from common.file_utils import *

_CSV_COLUMN_NAMES = ["ds", "useruin", "visit", "star", "appuin"]
_CSV_COLUMN_TYPES = [tf.string, tf.string, tf.constant([''], dtype=tf.string),tf.constant([''], dtype=tf.string), tf.string]

BUFFER_SIZE = 1000
appPath='/home/lorineluo/python_data_analysis/data/common/appuininfo_test'
MAX_LENGTH = 64

appTokenizer = AppTokenizer(appPath)

def encode(lang1, lang2):
    lang1 = appTokenizer.encoder.encode(
        lang1.numpy())

    lang2 = appTokenizer.encoder.encode(
        lang2.numpy())

    return lang1, lang2

def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en


def encode_single(lang1):
    lang1 = appTokenizer.encoder.encode(
        lang1.numpy())

    return lang1


def tf_encode_single(pt):
    result_pt = tf.py_function(encode_single, [pt], tf.int64)
    result_pt.set_shape([])

    return result_pt


def filter_max_length(x, y):
    return tf.logical_and(tf.size(x) <= MAX_LENGTH,
                        tf.size(y) <= MAX_LENGTH)


def train_input_fn(data_path, start_time, days, batch_size):
    """Load and return dataset of batched examples for use during training."""
    file_list = get_file_list(data_path, start_time, days)

    dataset = tf.data.experimental.CsvDataset(file_list, _CSV_COLUMN_TYPES, header=True).map(
        lambda *items: (items[2], items[4])) \
        .map(tf_encode) \
        .filter(filter_max_length) \
        .cache() \
        .shuffle(BUFFER_SIZE) \
        .padded_batch(batch_size, padded_shapes=([MAX_LENGTH], [1])) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def eval_input_fn(data_path, start_time, days, batch_size):
    """Load and return dataset of batched examples for use during eval."""
    file_list = get_file_list(data_path, start_time, days)

    print("input file list:\n{}".format("\n".join(file_list)))

    val_dataset = tf.data.experimental.CsvDataset(file_list, _CSV_COLUMN_TYPES, header=True).map(
        lambda *items: (items[2])) \
        .map(tf_encode_single) \
        .filter(filter_max_length) \
        .cache() \
        .padded_batch(batch_size, padded_shapes=([MAX_LENGTH])) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    
    return val_datas