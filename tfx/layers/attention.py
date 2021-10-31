#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2020/08/20 
# @Author  : lorineluo
# @File    : attention.py

import numpy as np
import tensorflow as tf
import datetime

class HourEncoding(tf.keras.layers.Layer):
    """
    hour encoding
    对当天小时编码
    
    Args:   
        d_model: word embedding的长度，等于position embedding的长度
    """
    def __init__(self, d_model=512, name="TE", **kwargs):
        super(HourEncoding, self).__init__(name=name, **kwargs)
        self.d_model = d_model

    def get_angles(self, pos, i, d_model):
        angle_rate = 1 / np.power(2, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rate    

    def init_time_encoding(self, input_shape, dtype=tf.float32):
        angle_rads = self.get_angles(np.arange(25)[:, np.newaxis], np.arange(self.d_model)[np.newaxis, :], self.d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        return tf.convert_to_tensor(angle_rads, dtype=tf.float32)

    def build(self, input_shape):
        self.HOURE = self.add_weight(name='hour_pe', 
                                  shape=(25, self.d_model),
                                  initializer=self.init_time_encoding,
                                  trainable=False,
                                  dtype=tf.float32)
    def get_config(self):
        return {'d_model': self.d_model}

    def call(self, x):
        return tf.nn.embedding_lookup(self.HOURE, x)


class WeekEncoding(tf.keras.layers.Layer):
    """
    hour encoding
    对星期几编码
    
    Args:   
        d_model: word embedding的长度，等于position embedding的长度
    """
    def __init__(self, d_model=512, name="TE", **kwargs):
        super(WeekEncoding, self).__init__(name=name, **kwargs)
        self.d_model = d_model

    def get_angles(self, pos, i, d_model):
        angle_rate = 1 / np.power(2, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rate    

    def init_time_encoding(self, input_shape, dtype=tf.float32):
        angle_rads = self.get_angles(np.arange(8)[:, np.newaxis], np.arange(self.d_model)[np.newaxis, :], self.d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        return tf.convert_to_tensor(angle_rads, dtype=tf.float32)

    def build(self, input_shape):
        self.WEEKE = self.add_weight(name='week_pe', 
                                  shape=(8, self.d_model),
                                  initializer=self.init_time_encoding,
                                  trainable=False,
                                  dtype=tf.float32)
    def get_config(self):
        return {'d_model': self.d_model}
                                                                 
    def call(self, x):
        return tf.nn.embedding_lookup(self.WEEKE, x)

class PositionalEncoding(tf.keras.layers.Layer):
    """
    position encoding
    
    Args:   
        max_position: seq的最大长度
        d_model: word embedding的长度，等于position embedding的长度
    """
    def __init__(self, max_position=100, d_model=512, name="PE", **kwargs):
        super(PositionalEncoding, self).__init__(name=name, **kwargs)
        self.max_position = max_position
        self.d_model = d_model
         
    def get_angles(self, pos, i, d_model):
        angle_rate = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rate

    def init_positional_encoding(self, input_shape, dtype=tf.float32):
        angle_rads = self.get_angles(np.arange(self.max_position)[:, np.newaxis], np.arange(self.d_model)[np.newaxis, :], self.d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        return tf.convert_to_tensor(angle_rads, dtype=tf.float32)

    def build(self, input_shape):
        self.PE = self.add_weight(name='w_pe', 
                                  shape=(self.max_position, self.d_model),
                                  initializer=self.init_positional_encoding,
                                  trainable=False,
                                  dtype=tf.float32)

    def get_config(self):
        return {'d_model': self.d_model, 'max_position': self.max_position}

    def call(self, x, mask=None):
        # pos_encoding = self.PE[tf.newaxis, :tf.shape(x)[1], :]
        pos_encoding = tf.expand_dims(self.PE, 0)
        pos_encoding = pos_encoding[:, :tf.shape(x)[1], :]

        return x + pos_encoding 


class BahdanauAttention(tf.keras.layers.Layer):
    """
        BahdanauAttention

        Args:
            units: query,values需要过一个一层网络， units为网络参数
    """
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    def get_config(self):
        return {'units': self.units}


class ScaledDotProductAttention(tf.keras.layers.Layer):
    """
    scalaedDotProductAttention
    计算attention weight
    第一个维度batch_size, 三者都需要一致；
    倒数第二个维度seq_len k,v需要保持一致， ie.: seq_len_k = seq_len_v
    最后一个维度， q,k 需要保持一致.
    mask: 可以是look ahead 或者 padding.

    Args:
        q: shape=(..., seq_len_q, depth)
        k: shape=(..., seq_len_k, depth)
        v: shape=(..., seq_len_v, depth_v)

    Returns:
        output, attention_weights
    """
    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def call(self, q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        #scaled
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (..., seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, seq_len_v)
        return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    MultiHeadAttention
    
    Args:
        d_model: attention concat的总维度
        num_heads: 多头attention的 head数
    """
    def __init__(self, d_model, num_heads, name="mhAttention", **kwargs):
        super(MultiHeadAttention, self).__init__(name=name, **kwargs)
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0

        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.scaled_dot_product_attention = ScaledDotProductAttention()
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1 , self.num_heads, self.depth))
        # transpose  result to (batch_size, num_heads, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def get_config(self):
        return {'d_model': self.d_model, 'num_heads': self.num_heads}

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (..., seq_len_q, d_model)
        k = self.wk(k)  # (..., seq_len_k, d_model)
        v = self.wv(v)  # (..., seq_len_v, d_model) 

        q = self.split_heads(q, batch_size) # (..., num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size) # (..., num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size) # (..., num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) # (..., seq_len, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (..., seq_len_q, d_model)

        output = self.dense(concat_attention)   # (..., seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]
    
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

class EncoderLayer(tf.keras.layers.Layer):
    """
        EncoderLayer
        由一个multi-head attention, add&norm, ffn, add&norm组成

        Args:
            d_model: 输出embedding长度， num_heads的整数倍
            num_heads: 多头的个数
            dff: 2层前馈网络
            rate:dropout

    """
    def __init__(self, d_model, num_heads, dff, rate=0.1, name="encoder_layer", **kwargs):
        super(EncoderLayer, self).__init__(name=name, **kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        att_out, att_weighs = self.mha(x, x, x, mask)
        att_out = self.dropout1(att_out, training=training)
        out1 = self.layernorm1(x + att_out)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2, att_weighs

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate,
        }
        return config

class DecoderLayer(tf.keras.layers.Layer):
    '''
        DecoderLayer
        由 masked multi-head attention, add&norm, multi-head attention, add&norm, ffn, add&norm 组成

        Args:
            d_model: 输出embedding长度
            num_heads: 多头个数
            dff: 2层前馈网络
    '''
    def __init__(self, d_model, num_heads, dff, rate=0.1, name="decoder_layer", **kwargs):
        super(DecoderLayer, self).__init__(name=name, **kwargs)

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        att1, att_weight_block1 = self.mha1(x, x, x, look_ahead_mask)
        att1 = self.dropout1(att1, training=training)
        out1 = self.layernorm1(att1 + x)

        att2, att_weight_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        att2 = self.dropout2(att2, training=training)
        out2 = self.layernorm2(att2 + out1)

        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)

        out3 = self.layernorm3(ffn_out + out2)

        return out3, att_weight_block1, att_weight_block2


class Encoder(tf.keras.layers.Layer):
    """
        Encoder

        Args:
            num_layers: 有多少encoder_layer
            d_model: 模型向量的维度
            num_heads: 多头的个数
            dff: 2层前馈网络中的第一层， 第二层维度为d_model
            input_vocab_size: 输入词表的长度
            maximum_position_encoding: pe编码中序列的长度
            rate: dropout
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, self.d_model) #_positional_encoding(maximum_position_encoding, self.d_model)
        self.week_encoding = WeekEncoding(d_model)
        self.hour_encoding = HourEncoding(d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    # @tf.function(input_signature=[tf.TensorSpec([None, None], np.int32), tf.TensorSpec([None, None], np.int32),tf.TensorSpec([None, None], np.int32),tf.TensorSpec([], bool),
            # tf.TensorSpec([None, None, None, None], np.float32)])
    def call(self, x, week, hour, training, mask):
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  #论文中的放缩
        # x += self.pos_encoding[:, :seq_len, :]
        x = self.pos_encoding(x)

        weeke = self.week_encoding(week)[:, :tf.shape(x)[1], :] # 只取与seq相同长度的time 
        houre = self.hour_encoding(hour)[:, :tf.shape(x)[1], :] 
        x = x + weeke + houre

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block = self.enc_layers[i](x, training, mask)
            attention_weights['encoder_layer{}'.format(i+1)] = block

        return x, attention_weights


class Decoder(tf.keras.layers.Layer):
    """
        Encoder

        Args:
            num_layers: 有多少encoder_layer
            d_model: 模型向量的维度
            num_heads: 多头的个数
            dff: 2层前馈网络中的第一层， 第二层维度为d_model
            target_vocab_size: 输出词表的长度
            maximum_position_encoding: pe编码中序列的长度
            rate: dropout
    """
    
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, self.d_model) # positional_encoding(maximum_position_encoding, self.d_model)
        self.week_encoding = WeekEncoding(d_model)
        self.hour_encoding = HourEncoding(d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

#    @tf.function(input_signature=[tf.TensorSpec([None, None], np.int32), tf.TensorSpec([None, None, None], np.float32),tf.TensorSpec([None, None], np.int32),tf.TensorSpec([None, None], np.int32), tf.TensorSpec([], bool),
            # .TensorSpec([None, None, None, None], np.float32), tf.TensorSpec([None, None, None, None], np.float32)])
    def call(self, x, enc_output, week, hour, training, look_ahead_mask, padding_mask):
        # seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x += self.pos_encoding[:, :seq_len, :]
        x = self.pos_encoding(x)

        weeke = self.week_encoding(week)[:, :tf.shape(x)[1], :] # 只取与seq相同长度的time 
        houre = self.hour_encoding(hour)[:, :tf.shape(x)[1], :] 
        x = x + weeke + houre

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape = (batch_size, tar_seq_len, d_model)
        return x, attention_weights


class MyBias(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(MyBias, self).__init__()
        self.w = self.add_weight(
            shape=(units, 1), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w)