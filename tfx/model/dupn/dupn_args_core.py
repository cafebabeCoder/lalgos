#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2020/08/20 
# @Author  : lorineluo
# @File    : arg_core.py

data_path = "/apdcephfs/private_lorineluo/python_data_analysis/data/dupn"

model_params = {
    # emb dim
    # 'appuin_hash_size' : 1000000,
    'appuin_dim' : 16,
    
    'user_hash_size' : 10000,
    'user_dim' : 16,

    'toc_catg_f_size':10,
    'toc_catg_f_dim':8,
    'toc_catg_second_size':22,
    'toc_catg_second_dim':8,
    'toc_tag_size':330,
    'toc_tag_dim':8,

    'gender_size' : 3,
    'gender_dim' : 2,
    'region_size' : 100,
    'region_dim' : 4,
    'language_size' : 8,
    'language_dim' : 2,
    'platform_size' : 6,
    'platform_dim' : 2,
    'device_size' : 1024,
    'device_dim' : 4,
    'age_size' : 100,
    'age_dim' : 4,
    'grade_size' : 10 ,
    'grade_dim' : 4,
    'city_level_size' : 10,
    'city_level_dim' : 4,
    'wxapp_type_emb' : 4,

    # basic
    'basic_dim' : 32,

    # behivor
    'enc_units' : 16,

    # attention
    'att_units' : 16,

    # concat 
    'model_dim' : 16,

}

# local 
dataset_params = {
   'model_version' : 'base',
   'appuin_file' : data_path + '/common/recomm_appuininfo' ,
   'live_appuin_file' : data_path + '/common/liveappuininfo' ,
   'data_path' : data_path + '/sample_seq' ,
   'log_path' : data_path + '/log' ,
   'date' : '20200817',
   'duration' : 1,  # 训练几天的数据
   'predict_path' : '',
   'padding_max_length': 32,
   'result_path': data_path + '/result',
   'model_save_path' : data_path,
}

# tdw 
# dataset_params = {
#     'appuin_file' : 'appuininfo_5' ,
#     'live_appuin_file' : 'liveappuininfo' ,
#     'data_path' : data_path + '/predict_seq' ,
#     'predict_path' : '',
#     'padding_max_length': 64,
#     'result': data_path + '/result',
# }

run_params = {
    'mode': 'train',
    'batch_size': 64, 
    'neg_sample': 4, 
    'epochs': 10,
    'topk_acc': [30, 50, 100],
    'learning_rate': 0.001,
    'optimizer': 'adam',  # TODO:增加opt
    'max_to_keep' : 3, # ckpt
    'save_summary_steps': 200,
    'log_step_count_steps': 100,
    'save_checkpoints_steps': 100,
    'is_tensorboard': False,
}
