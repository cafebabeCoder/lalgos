#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2020/08/20 
# @Author  : lorineluo
# @File    : arg_core.py

data_path = "/apdcephfs/private_lorineluo/python_data_analysis/data/encoder"

model_params = {
    'num_layers' : 4, # encoder layer的block数
    'd_model' : 32, #appid emb的维度，time/position encoding的维度, 整除num_heads
    'num_heads' : 2,  # encoder attention的multi head数
    'dff' : 512, # encoder最后一层的embedding维度
}

dataset_params = {
    'model_version' : 'gru_noposition_notime_pad32',
    'model_save_path' : data_path,
    'appuin_file' : '/apdcephfs/private_lorineluo/python_data_analysis/data/common/recomm_appuininfo',
    'data_path' : data_path + '/sample' ,
    'log_path' : data_path + "/log",
    'date' : '20201012', # 训练数据日期
    'duration' : 1,  # 训练几天的数据
    'padding_input_max_length': 32,  # 输入的his_seq长度
}

run_params = {
    'mode': 'train',
    'batch_size': 512, 
    'epochs': 2,
    'topk_acc': [2, 4, 8, 10],
    'save_summary_steps': 50,
    'log_step_count_steps': 100,
    'save_checkpoints_steps': 500,
    'is_tensorboard': True,
}