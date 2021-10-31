#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2020/08/20 
# @Author  : lorineluo
# @File    : dupn_main.py

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

from model.dupn import dupn 
from model.dupn.dupn_args_core import run_params, model_params, dataset_params
from datasets import data_pipeline_dupn
from common.optimizers import CustomSchedule
from common.file_utils import Print_log 

parser = argparse.ArgumentParser()
parser.add_argument( '--ds', default='', type=str)
parser.add_argument( '--mode', default='', type=str)
parser.add_argument( '--data_path', default='', type=str)
args, _ = parser.parse_known_args()

if args.ds != "":
    dataset_params['date'] = args.ds
if args.mode !="":
    run_params['mode'] = args.mode
if args.data_path != "":
    dataset_params['data_path'] = args.data_path

if __name__ == '__main__':
    log_file = os.path.join(dataset_params['log_path'], "%s_%s") % (dataset_params['model_version'], dataset_params['date'])
    model_save_path = os.path.join(dataset_params['model_save_path'], "%s_%s") %(dataset_params['model_version'], dataset_params['date'])

    print_log = Print_log(log_file) 
    print_log("\n".join(["\n", str(model_params), str(run_params), str(dataset_params), "\n"]))

    VOCAB_SIZE = data_pipeline_dupn.VOCAB_SIZE 
    LIVE_VOCAB_SIZE = data_pipeline_dupn.LIVE_VOCAB_SIZE 

    model = dupn.DUPN( VOCAB_SIZE,LIVE_VOCAB_SIZE, 
        model_params['user_hash_size'], 
        model_params['user_dim'] , 
        model_params['appuin_dim'], 
        model_params['enc_units'], 
        model_params['att_units'], 
        model_params['basic_dim'], 
        model_params['model_dim'], 
        run_params['batch_size'])

    # Instantiate an optimizer.
    learning_rate = CustomSchedule(model_params['model_dim']) 
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # loss
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # metrics
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    train_topk_metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(k = i) for i in run_params['topk_acc']]
    val_topk_metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(k = i) for i in run_params['topk_acc']]

    # checkpoint
    checkpoint_path = os.path.join(model_save_path, "ckp")
    ckpt = tf.train.Checkpoint(step=tf.Variable(0, name="step", dtype='int64'), model=model, 
        optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=run_params['max_to_keep'])

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')

    # dataset
    dataset = data_pipeline_dupn.train_input_fn( 
        dataset_params['data_path'], 
        dataset_params['date'], 
        dataset_params['duration'], 
        run_params['batch_size'])

    # tensorboard
    if run_params['is_tensorboard']:
        train_tb_path = os.path.join(model_save_path, "tb/train")
        val_tb_path = os.path.join(model_save_path, "tb/val")
        train_summary_writer = tf.summary.create_file_writer(train_tb_path)
        val_summary_writer = tf.summary.create_file_writer(val_tb_path)

    @tf.function()
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits, att = model(x)  # Logits for this minibatch
            # Loss value for this minibatch
            loss_value = loss_fn(y, logits)
            # Add extra losses created during this forward pass:  其他层可能有loss
            loss_value += sum(model.losses)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        #更新 metrics 
        train_acc_metric.update_state(y, logits)
        for train_topk_metric in train_topk_metrics:    
            train_topk_metric.update_state(y, logits)
        return loss_value

    @tf.function()
    def test_step(x, y):
        val_logits, att = model(x)
        loss_value = loss_fn(y, val_logits)
        loss_value += sum(model.losses)
        #更新acc
        val_acc_metric.update_state(y, val_logits)
        for val_topk_metric in val_topk_metrics:    
            val_topk_metric.update_state(y, val_logits)

        return loss_value
    
    # Visualize weights & biases as histogram in Tensorboard.
    def summarize_weights(step):
        variables = model.variables[-14:-2]
        for w in variables:
            tf.summary.histogram(w.name, w.numpy(), step=step)

    # predict
    if 'predict' in run_params['mode']:
        print_log('----predict----')
        if not os.path.exists(dataset_params['result_path']):
            os.makedirs(dataset_params['result_path'])
        file_path = os.path.join(dataset_params['result_path'], dataset_params['date'])
        with open(file_path, "w+") as iwf:
            for step, (x_batch, y_batch) in enumerate(dataset):
                logits, _ = model(x_batch, from_logits=True)
                for idx, logit in enumerate(logits.numpy()):
                    iwf.write(x_batch['useruin'][idx].numpy().decode() + "\t")
                    iwf.write(",".join([str(round(f, 4)) for f in logit]))
                    iwf.write("\n")
            iwf.close()

    if 'train' in run_params['mode']:
        train_topks = []
        print_log('----train----')
        # Iterate over the batches of a dataset.
        for epoch in range(run_params['epochs']):
            # train
            for _, (x_batch_train, y_batch_train) in enumerate(dataset): 
                ckpt.step.assign_add(1)
                step = ckpt.step
                loss_value = train_step(x_batch_train, y_batch_train)          
                train_acc = train_acc_metric.result()
                train_topks = [train_topk_metric.result() for train_topk_metric in train_topk_metrics]                

                if step % run_params['log_step_count_steps'] == 0: 
                    tops = "\t".join(["top"+str(k)+":"+str(round(v.numpy(), 4)) for k, v in dict(zip(run_params['topk_acc'], train_topks)).items()])
                    print_log("training step %d\tloss: %.4f\tacc: %.4f\t%s" %(step, float(loss_value), float(train_acc), tops))
                if run_params['is_tensorboard'] & (step % run_params['save_summary_steps'] == 0):
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss_value, step=step)    
                
                    with train_summary_writer.as_default():
                        tf.summary.scalar('acc', train_acc, step=step)
                        for i, train_topk in enumerate(train_topks):
                            tf.summary.scalar('topk/'+str(run_params['topk_acc'][i]), train_topk, step=step)
                        # plot variables
                        summarize_weights(epoch)
                # if step > 100:
                    # break

            tops = "\t".join(["top"+str(k)+":"+str(round(v.numpy(), 4)) for k, v in dict(zip(run_params['topk_acc'], train_topks)).items()])
            print_log("training epoch %d\tloss: %.4f\tacc: %.4f\t%s" %(epoch, float(loss_value), float(train_acc), tops))
            train_acc_metric.reset_states()
            for train_topk_metric in train_topk_metrics:    
                train_topk_metric.reset_states()

            # save ckp
            # if epoch % run_params['save_checkpoints_steps'] == 0:
                # ckpt_path = ckpt_manager.save()
                # print_log('saving checkpoint epoch %d: %s' %(epoch, ckpt_path))

            # val 只在这个epoch结束之后计算
            if 'evaluate' in run_params['mode']:
                for step, (x_batch_val, y_batch_val) in enumerate(dataset):
                    loss_value = test_step(x_batch_val, y_batch_val)
                    break
                val_acc = val_acc_metric.result()
                val_topks = [val_topk_metric.result() for val_topk_metric in val_topk_metrics]

                if run_params['is_tensorboard']:
                    with val_summary_writer.as_default():
                        tf.summary.scalar('loss', loss_value, step=ckpt.step)   

                    with val_summary_writer.as_default():
                        tf.summary.scalar('acc', val_acc, step=ckpt.step)
                        for i, val_topk in enumerate(val_topks):
                            tf.summary.scalar('topk/'+str(run_params['topk_acc'][i]), val_topk, step=ckpt.step)

                tops = "\t".join(["top"+str(k)+":"+str(v.numpy()) for k, v in dict(zip(run_params['topk_acc'], val_topks)).items()])
                print_log("val epoch %d\t acc: %.4f, topk: %.4f\t%s" %(epoch, float(val_acc), float(val_topk), tops))
                val_acc_metric.reset_states()
                for val_topk_metric in val_topk_metrics:    
                    val_topk_metric.reset_states()

        # save h5        
        # h5_path = os.path.join(dataset_params['model_save_path'], "h5")
        # model.save(h5_path)

