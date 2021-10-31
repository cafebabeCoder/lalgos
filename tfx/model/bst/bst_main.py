#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2020/08/20 
# @Author  : lorineluo
# @File    : encoder_main.py
# 

import sys
sys.path.insert(0, "/apdcephfs/private_lorineluo/python_data_analysis/src/models")
sys.path.insert(0, '/home/lorineluo/python_data_analysis/src/models')

import os
import tensorflow as tf
import numpy as np
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

from model.bst.bst import BST 
from model.bst.bst_args_core import dataset_params, model_params, run_params 
from common.optimizers import CustomSchedule
from datasets import data_pipeline_bst
from common.file_utils import Print_log 

if __name__ == '__main__':
    log_file = os.path.join(dataset_params['log_path'], "%s_%s") % (dataset_params['model_version'], dataset_params['date'])
    model_save_path = os.path.join(dataset_params['model_save_path'], "%s_%s") %(dataset_params['model_version'], dataset_params['date'])
    vocab_size = data_pipeline_bst.VOCAB_SIZE # 输入输出是一致的, 必须要+2， 输入padding的时候多了2个字符

    print_log = Print_log(log_file) 
    print_log("\n".join(["\n", str(model_params), str(run_params), str(dataset_params), "\n"]))

    model = BST(vocab_size)

    # lr
    learning_rate = 0.001 # CustomSchedule(model_params['d_model'], warmup_steps=500) 

    # loss 梯度剪切
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    def loss_fn(real, pred):
        # app的编号没出现为1， padding才是0， 而read没有padding， 只有1,需要mask掉
        mask = tf.squeeze(tf.math.logical_not(tf.math.equal(tf.subtract(real, 1), 0)))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    # metrics, 每一次求acc 都是在之前的结果上累积， 所以一个epoch之后，应该归0, 用 sample_weight 来mask 不计算的那部分
    topk_acc_objects = [tf.keras.metrics.SparseTopKCategoricalAccuracy(k = i, name="top{}_acc".format(i)) for i in run_params['topk_acc']]

    def acc_fn0(real, pred):
        # real = 1说明是没有appid的， mask掉， 不计算
        mask = tf.cast(tf.math.logical_not(tf.math.equal(real, 1)), dtype=pred.dtype)
        topk_acc_objects[0].update_state(real, pred, sample_weight = mask)
        # 访问结果：acc_objects.result().numpy()
        return topk_acc_objects[0].result()

    def acc_fn1(real, pred):
        mask = tf.cast(tf.math.logical_not(tf.math.equal(real, 1)), dtype=pred.dtype)
        topk_acc_objects[1].update_state(real, pred, sample_weight = mask)
        return topk_acc_objects[1].result()

    def acc_fn2(real, pred):
        mask = tf.cast(tf.math.logical_not(tf.math.equal(real, 1)), dtype=pred.dtype)
        topk_acc_objects[2].update_state(real, pred, sample_weight = mask)
        return topk_acc_objects[2].result()

    def acc_fn3(real, pred):
        mask = tf.cast(tf.math.logical_not(tf.math.equal(real, 1)), dtype=pred.dtype)
        topk_acc_objects[3].update_state(real, pred, sample_weight = mask)
        return topk_acc_objects[3].result()

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss = loss_fn,
                  metrics = [acc_fn0, acc_fn1, acc_fn2, acc_fn3]) 
    
    # dataset
    dataset = data_pipeline_bst.train_input_fn(
        data_path=dataset_params['data_path'], 
        start_time=dataset_params['date'], 
        days=dataset_params['duration'], 
        batch_size=run_params['batch_size']
    )

    # tensorboard
    train_tb_path = os.path.join(model_save_path, "tb")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=train_tb_path, 
        update_freq=run_params['save_summary_steps'], 
        histogram_freq=run_params['save_summary_steps'])

    # checkpoint
    # save_weights_only = True, 会以ckp的形式保存， 加载方式： model.load_weights(train_ckp_path)， 注意不同于.train.CheckpointManager的方式。
    # save_weights_only = False, 会以pb+variable的形式保存， 加载方式：tf.keras.models.load_model(path）
    ckp_path = os.path.join(model_save_path, "ckp")
    # restore model
    try:
        model.load_weights(ckp_path)
        print_log("restore model {} success!".format(ckp_path))
    except:
        print_log("no model exist {}!".format(ckp_path))

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckp_path,
        save_weights_only=True,
        # monitor='val_acc',
        # mode='max',
        # save_best_only=True,
        save_freq=run_params['save_checkpoints_steps']
        )

    # print log
    class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if(batch % run_params['log_step_count_steps'] == 0):
                log = "[training step {}]\t".format(batch)
                log = log + "\t".join(["{}:{:.4f}".format(i, j) for i,j in logs.items()])
                print_log(log)

    #fit
    # 如果用fit的话， 需要注意dataset一定要和模型内部的顺序、个数一致。
    model.fit(
        dataset, 
        epochs=run_params['epochs'], 
        callbacks=[tensorboard_callback, model_checkpoint_callback, LossAndErrorPrintingCallback()])

    # save
    pb_path = os.path.join(model_save_path, "pb")
    model.save(pb_path)

    # save onnx
    # import keras2onnx
    # output_model_path = "encoder.onnx"
    # onnx_model = keras2onnx.convert_keras(model, model.name)
    # keras2onnx.save_model(onnx_model, output_model_path)
