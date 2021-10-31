#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2020/08/20 
# @Author  : lorineluo
# @File    : transformer_main.py

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

from model.transformer.transformer import Transformer
from model.transformer.transformer_arg_core import dataset_params, model_params, run_params 
from layers.attention import create_padding_mask, create_look_ahead_mask 
from common.optimizers import CustomSchedule
from datasets import data_pipeline_transformer
from common.file_utils import Print_log 

if __name__ == '__main__':
    log_file = os.path.join(dataset_params['log_path'], "%s_%s") % (dataset_params['model_version'], dataset_params['date'])
    model_save_path = os.path.join(dataset_params['model_save_path'], "%s_%s") %(dataset_params['model_version'], dataset_params['date'])
    print_log = Print_log(log_file)
    print_log("\n".join(["\n", str(model_params), str(run_params), str(dataset_params), "\n"]))

    vocab_size = data_pipeline_transformer.VOCAB_SIZE # 输入输出是一致的

    model = Transformer(num_layers=model_params['num_layers'], 
                        d_model=model_params['d_model'], 
                        num_heads=model_params['num_heads'], 
                        dff=model_params['dff'],
                        input_vocab_size=vocab_size, 
                        target_vocab_size=vocab_size, 
                        pe_input=dataset_params['padding_input_max_length'], 
                        pe_target=dataset_params['padding_output_max_length'])
    
    # Instantiate an optimizer.
    learning_rate = CustomSchedule(model_params['d_model']) 
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    def loss_fn(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    # metrics
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    def acc_fn(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        acc_ = tf.keras.metrics.sparse_categorical_accuracy(real, pred)

        mask = tf.cast(mask, dtype=acc_.dtype)
        acc_ *= mask

        return tf.reduce_sum(acc_)/tf.reduce_sum(mask)

    # prediction的长度与tar_real一致，计算指标的时候，会把prediction的整个序列和tar_real合并到一起算。 但是在预测的时候，是循环k次，每次都取prediction最后一个. 
    def topk_acc_fn(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        topk_accs = []
        for k in run_params['topk_acc']:
            topk_acc_ = tf.keras.metrics.sparse_top_k_categorical_accuracy(real, pred, k=k)
            mask = tf.cast(mask, dtype=topk_acc_.dtype)
            topk_acc_ *= mask

            topk_accs.append(tf.reduce_sum(topk_acc_)/tf.reduce_sum(mask))
        return topk_accs

    # ckp
    ckp_path = os.path.join(model_save_path, 'ckp')
    ckpt = tf.train.Checkpoint(step=tf.Variable(0, name="step", dtype='int64'), model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckp_path, max_to_keep=run_params['max_to_keep'])

    # restore the latest checkpoint
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print_log(('latest checkpoint restored!'))

    # dataset
    dataset = data_pipeline_transformer.train_input_fn(
        data_path=dataset_params['data_path'], 
        start_time=dataset_params['date'], 
        days=dataset_params['duration'], 
        batch_size=run_params['batch_size']
    )

    # tensorboard
    if run_params['is_tensorboard']:
        train_tb_path = os.path.join(model_save_path, "tb/train")
        val_tb_path = os.path.join(model_save_path, "tb/val")
        train_summary_writer = tf.summary.create_file_writer(train_tb_path)
        val_summary_writer = tf.summary.create_file_writer(val_tb_path)

    def create_masks(inp, tar):
        # for encoding
        enc_padding_mask = create_padding_mask(inp)
        # for decoding
        dec_padding_mask = create_padding_mask(inp)

        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32)
    ]
    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar, his_week, his_hour, today_week, today_hour):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        with tf.GradientTape() as tape: 
            predictions, _, _ = model({
                "inp": inp, 
                'tar':tar_inp, 
                'training':True, 
                'enc_padding_mask':enc_padding_mask, 
                'look_ahead_mask':combined_mask, 
                'dec_padding_mask':dec_padding_mask, 
                'inp_week':his_week, 
                'inp_hour':his_hour, 
                'tar_week':today_week, 
                'tar_hour':today_hour}) 
            loss = loss_fn(tar_real, predictions)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        train_loss_metric(loss)
        acc = acc_fn(tar_real, predictions)
        # 只能读二维数组，所以要从(batch_szie, seq, dim ) -> (batch_size * seq, dim)
        reshape_tar_real = tf.reshape(tar_real, [-1])
        reshape_predictions = tf.reshape(predictions, [-1, vocab_size])
        topk_accs = topk_acc_fn(reshape_tar_real, reshape_predictions)

        return loss, acc, topk_accs

    @tf.function(input_signature=train_step_signature)
    def test_step(inp, tar, his_week, his_hour, today_week, today_hour):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        predictions, _, _ = model({
            "inp": inp, 
            'tar':tar_inp, 
            'training':True, 
            'enc_padding_mask':enc_padding_mask, 
            'look_ahead_mask':combined_mask, 
            'dec_padding_mask':dec_padding_mask, 
            'inp_week':his_week, 
            'inp_hour':his_hour, 
            'tar_week':today_week, 
            'tar_hour':today_hour}) 
        loss = loss_fn(tar_real, predictions)

        return loss 

    # Visualize weights & biases as histogram in Tensorboard.
    def summarize_weights(step):
        draw_var = {}
        for w in model.variables:
            if 'mhAttention' in w.name:
                draw_var[w.name] = w
        for w in draw_var.values():
            tf.summary.histogram(w.name, w.numpy(), step=step)

    if 'train' in run_params['mode']:
        print_log('----train----')
        # Iterate over the batches of a dataset.
        for epoch in range(run_params['epochs']):
            # train
            try:
                for step, samples in enumerate(dataset): 
                    ckpt.step.assign_add(1)
                    step = ckpt.step
                    loss_value, train_acc, train_topks = train_step(samples['his_seq'], samples['today_seq'], samples['his_week'], samples['his_hour'], samples['today_week'], samples['today_hour'])          
                    train_loss = train_loss_metric.result()

                    if step % run_params['log_step_count_steps'] == 0:
                        tops = "\t".join(["top"+str(k)+":"+str(v.numpy()) for k, v in dict(zip(run_params['topk_acc'], train_topks)).items()])
                        print_log("[training step %d]\tacc:%.4f\tloss:%.4f\tno_mask_loss:%.4f\t%s" %(step, float(train_acc), float(loss_value), float(train_loss), tops))

                    if run_params['is_tensorboard'] & (step % run_params['save_summary_steps'] == 0):
                        with train_summary_writer.as_default():
                            tf.summary.scalar('loss', train_loss, step=ckpt.step)    
                    
                        with train_summary_writer.as_default():
                            tf.summary.scalar('acc', train_acc, step=ckpt.step)
                            for i, train_topk in enumerate(train_topks):
                                tf.summary.scalar('topk/'+str(run_params['topk_acc'][i]), train_topk, step=ckpt.step)
                            # plot variables
                            summarize_weights(epoch)

                    # save ckp
                    if step % run_params['save_checkpoints_steps'] == 0:

                        # model(tf.constant([[0],[1]], dtype=tf.int32), tf.constant(np.zeros(shape=(2, 0)), dtype=tf.int32), 
                        # False, tf.constant([[[[1.]]],[[[0.]]]]), tf.constant(np.zeros(shape=(2, 1, 0, 0)), dtype=tf.float32), 
                        # tf.constant([[[[1.]]],[[[0.]]]]), tf.constant([[0], [1]]),tf.constant([[0],[1]]), tf.constant([[0],[1]]), tf.constant([[0],[1]]))
                        ckpt_path = ckpt_manager.save()
                        print_log('saving checkpoint step %d: %s' %(step, ckpt_path))
                    # break

                # 保存pb
                # model.save(os.path.join(model_save_path, 'pb2'))
                # tf.saved_model.save(model, os.path.join(model_save_path, 'pb'))

            except tf.errors.InvalidArgumentError:
                print('line error')
            # train_acc_metric.reset_states()
            # train_loss_metric.reset_states()
            # for train_topk_metric in train_topk_metrics:    
                # train_topk_metric.reset_states()

            # val 只在这个epoch结束之后计算
            # if 'evaluate' in run_params['mode']:
            #     for step, (x_batch_val, y_batch_val) in enumerate(dataset):
            #         loss_value = test_step(x_batch_val, y_batch_val)
            #         break
            #     val_acc = val_acc_metric.result()
            #     val_loss = val_loss_metric.result() 
            #     val_topks = [val_topk_metric.result() for val_topk_metric in val_topk_metrics]

            #     if run_params['is_tensorboard']:
            #         with val_summary_writer.as_default():
            #             tf.summary.scalar('loss', val_loss, step=ckpt.step)   

            #         with val_summary_writer.as_default():
            #             tf.summary.scalar('acc', val_acc, step=ckpt.step)
            #             for i, val_topk in enumerate(val_topks):
            #                 tf.summary.scalar('topk/'+str(run_params['topk_acc'][i]), val_topk, step=ckpt.step)

            #     tops = "\t".join(["top"+str(k)+":"+str(v.numpy()) for k, v in dict(zip(run_params['topk_acc'], val_topks)).items()])
            #     print_log(log_file, True, "val epoch %d\tacc: %.4f\tloss: %.4f" %(epoch, float(val_acc), float(val_loss)))
            #     val_acc_metric.reset_states()
            #     val_loss_metric.reset_states()
            #     for val_topk_metric in val_topk_metrics:    
            #         val_topk_metric.reset_states()




