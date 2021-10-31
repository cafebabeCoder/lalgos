#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2020/08/20 
# @Author  : lorineluo
# @File    : dupn.py

import numpy as np
import tensorflow as tf

from layers.attention import *
from layers.attention import point_wise_feed_forward_network, BahdanauAttention 
from model.dupn.dupn_args_core import model_params

class BehaviorGRUEncoder(tf.keras.layers.Layer):
    """
    行为序列建模部分, 
    用户行为当作序列输入LSTM, 得到的hidden和output

    Args:
        vocab_size: 用户点击item， item需要embedding, vocab_size为词表大小
        embedding: item embedding的维度, 也是输出的embedding维度
        units: gru units
    """

    def __init__(self, vocab_size, embedding_dim, units, name="behaviorEncoder", **kwargs):
        super(BehaviorGRUEncoder, self).__init__(name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)        
        self.rnn = tf.keras.layers.GRU(self.units, 
                                       return_sequences=True, 
                                       return_state=True, 
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.rnn(x, initial_state=hidden)
        return output, state

    @tf.function
    def initialize_hidden_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

class BehaviorLSTMEncoder(tf.keras.layers.Layer):
    """
    行为序列建模部分, 
    用户行为当作序列输入LSTM, 得到的hidden和output

    Args:
        vocab_size: 用户点击item， item需要embedding, vocab_size为词表大小
        embedding: item embedding的维度, 也是输出的embedding维度
        units: gru units
    """

    def __init__(self, vocab_size, embedding_dim, units, name="behaviorEncoder", **kwargs):
        super(BehaviorLSTMEncoder, self).__init__(name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)        
        self.rnn = tf.keras.layers.LSTM(self.units, 
                                       return_sequences=True, 
                                       return_state=True, 
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        # return_sequence=True, ouput就是全部时间步的结果；否则就是最后一个时间步的结果
        # return_state 返回的h, c都是最后一个时间步的h,c
        output, state_h, state_c = self.rnn(x) 
        return output, state_h

    @tf.function
    def initialize_hidden_state(self, batch_size):
        return tf.zeros([self.units, self.units])


class BasicInfoEncoder(tf.keras.layers.Layer):
    """
    用户基础信息建模部分
    先把各个基础信息embedding 再经过一个2层网络结构, 得到d_model维输出

    Args:
        user_size: user 需要embedding, user_size为user size(user 需要hash, user_size 就是User_hash_bucket_size)
        embedding_dim: user embedding dim, 模型输入user embedding维度
        d_model: 模型输出维度

    """

    def __init__(self, user_size, embedding_dim, d_model, name="basicInfoEncoder", **kwargs):
        super(BasicInfoEncoder, self).__init__(name=name, **kwargs)

        #网络部分
        self.d_model = d_model
        self.ffn = point_wise_feed_forward_network(d_model * 2, d_model)

        #emb部分
        self.user_emb = tf.keras.layers.Embedding(user_size, embedding_dim, name='user_emb')
        self.gender_emb = tf.keras.layers.Embedding(model_params['gender_size'], model_params['gender_dim'], name='gender_emb')
        self.region_emb = tf.keras.layers.Embedding(model_params['region_size'], model_params['region_dim'], name='region_emb')
        self.language_emb = tf.keras.layers.Embedding(model_params['language_size'], model_params['language_dim'], name='language_emb')
        self.platform_emb = tf.keras.layers.Embedding(model_params['platform_size'], model_params['platform_dim'], name='platform_emb')
        self.device_emb = tf.keras.layers.Embedding(model_params['device_size'], model_params['device_dim'], name='device_emb')
        self.age_emb = tf.keras.layers.Embedding(model_params['age_size'], model_params['age_dim'], name='age_emb')
        self.grade_emb = tf.keras.layers.Embedding(model_params['grade_size'], model_params['grade_dim'], name='grade_emb')
        self.city_level_emb = tf.keras.layers.Embedding(model_params['city_level_size'], model_params['city_level_dim'], name='city_level_emb')


    def call(self, inputs):

        user_emb = self.user_emb(inputs['uin'])        
        gender_emb = self.gender_emb(inputs['gender'])        
        region_emb = self.region_emb(inputs['region_code'])        
        language_emb = self.language_emb(inputs['language'])        
        platform_emb = self.platform_emb(inputs['platform'])        
        device_emb = self.device_emb(inputs['device'])        
        age_emb = self.age_emb(inputs['age'])        
        grade_emb = self.grade_emb(inputs['grade'])        
        city_level_emb = self.city_level_emb(inputs['city_level'])        

        basic_dense = tf.keras.layers.Concatenate(axis=-1)([user_emb, gender_emb, region_emb,\
            language_emb, platform_emb, device_emb, age_emb, grade_emb, city_level_emb])
        basic_emb = self.ffn(basic_dense)

        return basic_emb


class DUPN(tf.keras.Model):
    """
        DUPN 模型

        Args:
            vocab_size: item size
            user_size:  user size, user通常需要hash, 为USER_HASH_BUCKET_SIZE
            user_emb_dim: user emb 的维度
            item_emb_dim: item emb 的维度
            enc_units: behavior 序列建模时，RNN的units
            att_units: attention为BahdanauAttention，att_units 为网络第一层参数
            basic_d_model: basicEncoder输出的维度
            d_model: 模型输出的维度
            batch_size: BS
    """
    def __init__(self, input_size, output_size, user_size, user_emb_dim, item_emb_dim, enc_units, att_units, basic_d_model, d_model, batch_size, name="dupn", **kwargs):
        super(DUPN, self).__init__(name=name, **kwargs)
    
        # 网络部分
        self.batch_size = batch_size
        self.basicEncoder = BasicInfoEncoder(user_size, user_emb_dim, d_model)
        self.behaviorEncoder = BehaviorLSTMEncoder(input_size, item_emb_dim, enc_units)
        self.attention = BahdanauAttention(att_units)
        self.fc = tf.keras.layers.Dense(d_model, name='fc')
        self.softmax = tf.keras.layers.Dense(output_size, name="predict")


    def call(self, inputs, from_logits=False):
        # basic
        basic_emb = self.basicEncoder(inputs)

        init_hidden = self.behaviorEncoder.initialize_hidden_state(self.batch_size)
        behavior_output, behavior_hidden = self.behaviorEncoder(inputs['visit'], init_hidden)
        
        context_vector, attention_weights = self.attention(basic_emb, behavior_output)

        output = tf.concat([basic_emb, context_vector], axis=-1)
        output = self.fc(output)
        if from_logits:
            return output, attention_weights 
        else:
            output = self.softmax(output)
            return output, attention_weights