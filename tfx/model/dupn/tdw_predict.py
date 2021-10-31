#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2020/08/20 
# @Author  : lorineluo
# @File    : tdw_predict.py

import argparse
import sys
import os
sys.path.insert(0, "/apdcephfs/private_lorineluo/python_data_analysis/src/models")
import tensorflow as tf

from pyspark import SparkContext, SQLContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, LongType
from pytoolkit import TDWSQLProvider, TDWUtil
from sparkfuel.common import util

from utils import tdw_utils, save_utils
from model.dupn import dupn 
from model.dupn.args_core import dataset_params, model_params, run_params
from datasets import data_pipeline_basic_seq

parser = argparse.ArgumentParser()
parser.add_argument( '--ds', default='20201103', type=str, help='YYYYMMDD')
parser.add_argument( '--data_path', default='hdfs://ss-wxg-3-v2/stage/outface/WXG/g_wxg_wxplat_wxbiz_offline_datamining/lorineluo/models/dupn/sample_predict/', type=str)
parser.add_argument( '--model_path', default='hdfs://ss-wxg-3-v2/stage/outface/WXG/g_wxg_wxplat_wxbiz_offline_datamining/lorineluo/models/dupn/model_save/', type=str)
parser.add_argument( '--result_path', default='hdfs://ss-wxg-3-v2/stage/outface/WXG/g_wxg_wxplat_wxbiz_offline_datamining/lorineluo/models/dupn/result/', type=str)
parser.add_argument( '--dest_db', default='wxg_wxa_offline_data', type=str, help='')
parser.add_argument( '--dest_table', default='dwmid_daily_wxapp_ecc_emb', type=str, help='')
parser.add_argument( '--user', default='tdw_lorineluo', type=str, help='')
parser.add_argument( '--passwd', default='lorineluo', type=str, help='')
parser.add_argument( '--partitions', default=1000, type=int, help='')
# parser.add_argument( '--ds', default='20201103', type=str, help='YYYYMMDD')

args, _ = parser.parse_known_args()
if args.ds != "":
    dataset_params['date'] = args.ds
if args.data_path != "":
    dataset_params['data_path'] = args.data_path

data_path=os.path.join(args.data_path, args.ds)
print("data path:\t", data_path)

DEST_DB = args.dest_db 
DEST_TABLE = args.dest_table
USER = args.user
PASSWD = args.passwd 

def main():
    conf = SparkConf().setAppName('user_emb_predict')
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    wxa_util = TDWUtil(user=USER, passwd=PASSWD, dbName=DEST_DB)
    wxa_tdw = TDWSQLProvider(spark, user=USER, passwd=PASSWD, db=DEST_DB)

    # def load_data():
    #     with open("/apdcephfs/private_lorineluo/python_data_analysis/data/dupn/sample_seq/20200817/seq2seq_0817", "r", encoding="utf8") as file:
    #         lines = []
    #         for line in file:
    #             lines.append(line)
    #         return lines 

    rdd = sc.textFile(data_path)
    print(rdd.count())

    def predict(it):
        util.setup_hadoop_classpath()
        # hdfs_path: **/model_save.zip  具体到文件
        # local_file_name="model_zip" copy到本地的文件名， 默认即可
        # checkfile="checkpoint" 检查是否上传成功的文件
        # model_path="./model/" 本地解压缩的目录， 等于之后load的目录， 必须保留，因为要检查这个dir
        hdfs_model_path = os.path.join(args.model_path, args.ds, "model_save.tar.gz")
        print(hdfs_model_path)
        model_file_path = "./hdfs_model"
        util.copy_to_local_file(hdfs_model_path, model_file_path=model_file_path)

        def rdd_generator():
            while True:
                try:
                    line = next(it)
                    yield line#.replace("\n", "")
                except StopIteration:
                    return

        def _parse_record(line):
            return tf.io.decode_csv(line, record_defaults=data_pipeline_basic_seq._CSV_COLUMN_DEFAULTS, field_delim=",")

        # dataset
        dataset = tf.data.TextLineDataset.from_generator(rdd_generator, output_types=tf.string)\
            .map(_parse_record, num_parallel_calls=20) \
            .map(data_pipeline_basic_seq.tf_sample_parse) \
            .batch(run_params['batch_size'], drop_remainder=True) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .map(data_pipeline_basic_seq._parse_line)

        # load model
        VOCAB_SIZE = data_pipeline_basic_seq.VOCAB_SIZE 
        LIVE_VOCAB_SIZE = data_pipeline_basic_seq.LIVE_VOCAB_SIZE 
        result = []

        model = dupn.DUPN( VOCAB_SIZE,LIVE_VOCAB_SIZE, 
            model_params['user_hash_size'], 
            model_params['user_dim'] , 
            model_params['appuin_dim'], 
            model_params['enc_units'], 
            model_params['att_units'], 
            model_params['basic_dim'], 
            model_params['model_dim'], 
            run_params['batch_size'])
        
        checkpoint = tf.train.Checkpoint(myModel=model)             # 实例化Checkpoint，指定恢复对象为model
        checkpoint.restore(tf.train.latest_checkpoint(model_file_path))    # 从文件恢复模型参数

        # predict
        for step, (x, y) in enumerate(dataset):
            logits, _ = model(x, from_logits=True)
            for idx, logit in enumerate(logits.numpy()):
                useruin = x['useruin'][idx].numpy().decode()
                emb =  ",".join([str(round(f, 4)) for f in logit])
                result.append([useruin, emb])
 
        return result

    user_emb = rdd.repartition(args.partitions).mapPartitions(predict)
    # print(user_emb.count())

    # for i in user_emb.take(5):
    #     print(i[0])
    #     print(i[1])
    #     break

    # 写入表
    saved_data = user_emb.map(lambda x: (args.ds, int(x[0]), x[1]))
    schema_list = [
        ('ds', StringType()),
        ('useruin', LongType()),
        ('emb', StringType())
    ]
    save_utils.save_table(sc, DEST_DB, DEST_TABLE, schema_list, saved_data, args.ds, None, over_write=True)

    # 写hdfs
    user_emb.saveAsTextFile(os.path.join(args.result_path, args.ds))

if __name__ == "__main__":
    main()