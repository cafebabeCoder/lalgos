#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2020/08/20 
# @Author  : lorineluo
# @File    : arg_core.py

data_path = "/apdcephfs/private_lorineluo/python_data_analysis/data/transformer"

model_params = {
    # emb dim
    'num_layers' : 4, # encoder & decoder layers
    'd_model' : 32, #appid emb dim, 能整除num_heads
    'num_heads' : 2,  # attention multi heads
    'dff' : 512,
}

dataset_params = {
    'model_version' : 'transformer',
    'appuin_file' : data_path + '/common/recomm_appuininfo' ,
    'data_path' : data_path + '/sample' ,
    'model_save_path' : data_path,
    'log_path' : data_path + "/log",
    'date' : '20201120',
    'duration' : 1,  # 训练几天的数据
    'padding_input_max_length' : 32,  # 输入长度， 为his_seq长度
    'padding_output_max_length': 16 # 输出长度， today_seq长度
}

run_params = {
    'mode': 'train',
    'batch_size': 32, 
    'epochs': 1,
    'topk_acc': [30, 100],
    'max_to_keep' : 3, # ckpt
    'save_summary_steps': 200,
    'log_step_count_steps': 100,
    'save_checkpoints_steps': 1000,
    'is_tensorboard': True,
}