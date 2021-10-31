#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2020/08/20 
# @Author  : lorineluo
# @File    : AppTokenizer.py

import tensorflow as tf

class AppTokenizer(object):
    def __init__(self, voc_dir="", split='|', uin_idx=0):
        self.voc_dir = voc_dir
        self.voc_dic = self.uin_name_dict(uin_idx)
        self.split = split

        #method1 tfds.features.text.Tokenizer(), 需要 pip install tensorflow_datasets
        # tokenizer = tfds.features.text.Tokenizer()
        # vocabulary_set = set()
        # for text_tensor in labeled_dataset:
        #     some_tokens = tokenizer.tokenize(text_tensor.numpy())
        #     vocabulary_set.update(some_tokens)

        #method2 用tf.keras.preprocessing.text.Tokenizer()
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', split=split)
        self.tokenizer.fit_on_texts(self.voc_dic.keys())
        
        self.vocab_size = len(self.voc_dic) + 5
        print('voc_dic: {} \t path: {} \n'.format(self.vocab_size, self.voc_dir))

        # 获取id 到name, uin_idx=0用于appuin编码， uin_idx=-1用于appid编码 
    def uin_name_dict(self, uin_idx=0):
        dictionary = dict()
        file = open(self.voc_dir, 'r', encoding='utf-8')
        dictionary["UNK"] = "UNK"
        for line in file.readlines():
            line = line.strip()
            k = line.split(',')[1]
            v = line.split(',')[uin_idx]
            dictionary[k] = v
        dictionary['STAR'] = "STAR"
        dictionary['END'] = "END"

        # 注意关闭文件
        file.close()

        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return reversed_dictionary

    def encode(self, texts):
        """
        输入为 list
        """
        # self.tokenizer.fit_on_texts(texts) 耗时太高
        ids = self.tokenizer.texts_to_sequences(texts)
        return ids 

    def tf_encode(self, texts):
        """
          在 tf_function中使用。 
        """
        # self.tokenizer.fit_on_texts([texts.numpy().decode()])
        ids = self.tokenizer.texts_to_sequences([texts.numpy().decode()])
        return ids 

    # tf_encoder 对于序列的encoder可能不对，例如：'wx2f7fda52d8d031ee|wx2f7fda52d8d031ee' => [[]], 正确的结果是： [[], []]
    def tf_encode2(self, texts):
        """
          在 tf_function中使用。 
        """
        # self.tokenizer.fit_on_texts([texts.numpy().decode()])
        ids = self.tokenizer.texts_to_sequences(texts.numpy().decode().split(self.split))
        return ids 

    def decode(self, ids):
        """
        输入为list
        """
        texts = [self.tokenizer.index_word[i] for i in ids if i in self.tokenizer.index_word]
        return texts 

    def transDecode(self, indices):
        """
            预测的结果转换成appname
            
            Args:
                indices: 一维数组， 预测结果，为appuin的index
        """
        result=self.decode(indices)
        predicted_sentence = [self.voc_dic.get(i, "UNK") for i in result]  
        return predicted_sentence