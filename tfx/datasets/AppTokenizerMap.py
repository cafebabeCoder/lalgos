#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2020/08/20 
# @Author  : lorineluo
# @File    : AppTokenizer.py

import tensorflow as tf

class AppTokenizer(object):
    def __init__(self, voc_dir="", split='|', uin_idx=0):
        self.voc_dir = voc_dir
        self.uin_name_dict, self.uin_idx_dict = self.get_uin_name_dict(uin_idx)
        self.idx_uin_dict = {value : key for (key, value) in self.uin_idx_dict.items()}
        self.split = split

        self.vocab_size = len(self.uin_name_dict) + 1
        print('voc_dic: {} \t path: {} \n'.format(self.vocab_size, self.voc_dir))

    # 获取id 到name, uin_idx=0用于appuin编码， uin_idx=-1用于appid编码 
    def get_uin_name_dict(self, uin_idx=0):
        uin_name_dict = dict()
        uin_idx_dict = dict()
        file = open(self.voc_dir, 'r', encoding='utf-8')
        uin_name_dict["UNK"] = "UNK"
        uin_idx_dict[1] = "UNK"
        idx = 2
        for line in file.readlines():
            line = line.strip()
            name = line.split(',')[1]
            uin = line.split(',')[uin_idx]
            uin_name_dict[uin] = name 
            uin_idx_dict[uin] = idx
            idx = idx + 1
        uin_name_dict['STAR'] = "STAR"
        uin_idx_dict['STAR'] = idx
        uin_name_dict['END'] = "END"
        uin_idx_dict['END'] = idx + 1

        # 注意关闭文件
        file.close()

        # reversed_uin_name_dict = dict(zip(uin_name_dict.values(), uin_name_dict.keys()))
        return uin_name_dict, uin_idx_dict 

    # wx2f7fda52d8d031ee|wx2f7fda52d8d031ee => [1, 2] , 没出现的词用1来补， padding部分用0
    def encode(self, texts):
        """
        输入为 str or list 
        """
        if type(texts) == list:
            ids = [self.uin_idx_dict.get(i, 1) for i in texts]
        elif type(texts) == str:
            ids = [self.uin_idx_dict.get(i, 1) for i in texts.split(self.split)]
        else:
            ids = []
        return ids 

    # wx2f7fda52d8d031ee|wx2f7fda52d8d031ee => [1, 2] 
    def tf_encode(self, texts):
        """
          在 tf_function中使用。 
        """
        ids = [self.uin_idx_dict.get(i, 1) for i in texts.numpy().decode().split(self.split)]
        return ids 

    # 把 模型的id 映射为appid
    def decode(self, ids):
        """
        输入为list
        """
        texts = [self.idx_uin_dict.get(i, 'UNK') for i in ids]
        return texts 

    # 把 模型id直接映射为appname
    def transDecode(self, indices):
        """
            预测的结果转换成appname
            
            Args:
                indices: 一维数组， 预测结果，为appuin的index
        """
        result=self.decode(indices)
        predicted_sentence = [self.uin_name_dict.get(i, "UNK") for i in result]  
        return predicted_sentence