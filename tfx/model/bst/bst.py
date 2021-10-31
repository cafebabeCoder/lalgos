import numpy as np
import tensorflow as tf

from layers.attention import WeekEncoding, HourEncoding, PositionalEncoding, EncoderLayer, Encoder, Decoder 
from layers.attention import point_wise_feed_forward_network, BahdanauAttention 
from model.bst.bst_args_core import model_params


class BasicInfoEncoder(tf.keras.layers.Layer):
    """
    用户基础信息建模部分
    先把各个基础信息embedding 再经过一个2层网络结构, 得到d_model维输出

    Args:
        user_size: user 需要embedding, user_size为user size(user 需要hash, user_size 就是User_hash_bucket_size)
        embedding_dim: user embedding dim, 模型输入user embedding维度
        d_model: 模型输出维度

    """

    def __init__(self, d_model, name="basicInfoEncoder", **kwargs):
        super(BasicInfoEncoder, self).__init__(name=name, **kwargs)

        #网络部分
        self.d_model = d_model
        self.ffn = point_wise_feed_forward_network(d_model * 2, d_model)

        #emb部分
        self.user_emb_layer = tf.keras.layers.Embedding(model_params['user_hash_size'], model_params['user_dim'], name='user_emb_layer')
        self.gender_emb_layer = tf.keras.layers.Embedding(model_params['gender_size'], model_params['gender_dim'], name='gender_emb_layer')
        self.region_emb_layer = tf.keras.layers.Embedding(model_params['region_size'], model_params['region_dim'], name='region_emb_layer')
        self.language_emb_layer = tf.keras.layers.Embedding(model_params['language_size'], model_params['language_dim'], name='language_emb_layer')
        self.platform_emb_layer = tf.keras.layers.Embedding(model_params['platform_size'], model_params['platform_dim'], name='platform_emb_layer')
        self.device_emb_layer = tf.keras.layers.Embedding(model_params['device_size'], model_params['device_dim'], name='device_emb_layer')
        self.age_emb_layer = tf.keras.layers.Embedding(model_params['age_size'], model_params['age_dim'], name='age_emb_layer')
        self.grade_emb_layer = tf.keras.layers.Embedding(model_params['grade_size'], model_params['grade_dim'], name='grade_emb_layer')
        self.city_level_emb_layer = tf.keras.layers.Embedding(model_params['city_level_size'], model_params['city_level_dim'], name='city_level_emb_layer')


    def call(self, inputs):
        user_emb = self.user_emb_layer(inputs['useruin'])
        gender_emb = self.gender_emb_layer(inputs['gender'])      

        region_emb = self.region_emb_layer(inputs['region_code'])        
        language_emb = self.language_emb_layer(inputs['language'])        
        platform_emb = self.platform_emb_layer(inputs['platform'])        
        device_emb = self.device_emb_layer(inputs['device'])        
        age_emb = self.age_emb_layer(inputs['age'])        
        grade_emb = self.grade_emb_layer(inputs['grade'])        
        city_level_emb = self.city_level_emb_layer(inputs['city_level'])        

        basic_dense = tf.keras.layers.Concatenate(axis=-1)([user_emb, gender_emb, region_emb,\
            language_emb, platform_emb, device_emb, age_emb, grade_emb, city_level_emb])
        basic_emb = self.ffn(basic_dense)
        basic_emb = tf.squeeze(basic_emb)

        return basic_emb


class InstallAppEncoder(tf.keras.layers.Layer):
    """
    用户安装app部分
    先把各个基础息embedding, 然后pooling 

    Args:
        app_size: app hash数量 
        app_dim: app dim
    """

    def __init__(self, app_size, app_dim, name="InstallAppEncoder", **kwargs):
        super(InstallAppEncoder, self).__init__(name=name, **kwargs)

        #网络部分
        self.pooling_layer = tf.keras.layers.GlobalAveragePooling1D(name='app_pooling') 
        self.concat_layer = tf.keras.layers.Concatenate(axis = -1, name='app_eccapp_concat') 

        #emb部分
        self.app_emb = tf.keras.layers.Embedding(app_size, app_dim, name='app_emb')

    def call(self, inputs):

        install_app_emb = self.app_emb(inputs['install'])        
        install_ecc_app_emb = self.app_emb(inputs['install_ecc'])        

        poolinged_app = self.pooling_layer(install_app_emb)
        poolinged_ecc_app = self.pooling_layer(install_ecc_app_emb)

        app_dense = self.concat_layer([poolinged_app, poolinged_ecc_app])

        return app_dense 

class BST(tf.keras.Model):
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
    def __init__(self, output_size, name="bst", **kwargs):
        super(BST, self).__init__(name=name, **kwargs) # 网络部分 
        self.basicEncoder = BasicInfoEncoder(model_params['basic_d_model'])
        self.installAppEncoder = InstallAppEncoder(model_params['install_app_size'], model_params['install_app_dim'])
        self.concat_layer = tf.keras.layers.Concatenate(axis = -1, name='app_eccapp_concat') 
        self.fc = tf.keras.layers.Dense(model_params['d_model'], name='fc')
        self.softmax = tf.keras.layers.Dense(output_size, name="predict")


    def call(self, inputs, from_logits=False):
        # basic
        basic_emb = self.basicEncoder(inputs)
        install_app_emb = self.installAppEncoder(inputs)
        
        output = self.concat_layer([basic_emb, install_app_emb])

        output = self.fc(output)
        if from_logits:
            return output#, attention_weights 
        else:
            output = self.softmax(output)
            return output#, attention_weights


def get_bst_model_raw(max_length, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, user_size, training=True, rate=0.1):
    inputs = {
        'his_seq': tf.keras.Input(name='his_seq', shape=(max_length), dtype='int32'),
        'his_week': tf.keras.Input(name='his_week', shape=(max_length), dtype='int32'),
        'his_hour' : tf.keras.Input(name='his_hour', shape=(max_length), dtype='int32'),
        'today_hour' : tf.keras.Input(name='today_hour', shape=(), dtype='int32'),
        'today_week' : tf.keras.Input(name='today_week', shape=(), dtype='int32'),
        'enc_padding_mask': tf.keras.Input(name='enc_padding_mask', shape=(1, 1, max_length), dtype='float32'),
        'gender' : tf.keras.Input(name='gender', shape=(), dtype='int32'),
        'region_code' : tf.keras.Input(name='region_code', shape=(), dtype='int32'),
        'language' : tf.keras.Input(name='language', shape=(), dtype='int32'),
        'platform' : tf.keras.Input(name='platform', shape=(), dtype='int32'),
        'device' : tf.keras.Input(name='device', shape=(), dtype='int32'),
        'age' : tf.keras.Input(name='age', shape=(), dtype='int32'),
        'grade' : tf.keras.Input(name='grade', shape=(), dtype='int32'),
        'city_level' : tf.keras.Input(name='city_level', shape=(), dtype='int32'),
        'install' : tf.keras.Input(name='install', shape=(), dtype='int32'),
        'install_ecc' : tf.keras.Input(name='install_ecc', shape=(), dtype='int32'),
        'useruin' : tf.keras.Input(name='useruin', shape=(), dtype='int32'),
    }
    basicKeys = ['gender', 'region_code', 'language', 'platform', 'device', 'age', 'grade', 'city_level', 'install', 'install_ecc', 'useruin']
    basicLayer = BasicInfoEncoder(d_model = d_model)({k:v for k, v in basicKeys})

    attention_weights = {}
    
    x = tf.keras.layers.Embedding(input_vocab_size, d_model, name='id_emb')(inputs['his_seq'])
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))  #论文中的放缩
    # x = PositionalEncoding(max_length, d_model, name='pe')(x)
    weeke = WeekEncoding(d_model, name='week_encoding')(inputs['his_week'])[:, :tf.shape(x)[1], :] # 只取与seq相同长度的time 
    houre = HourEncoding(d_model, name='hour_encoding')(inputs['his_hour'])[:, :tf.shape(x)[1], :] 
    x = x + weeke + houre

    x = tf.keras.layers.Dropout(rate, name='drop_out')(x, training=training)

    for i in range(num_layers):
        x, block = EncoderLayer(d_model, num_heads, dff, rate, name='encoder_layer_{}'.format(i))(x, training, inputs['enc_padding_mask'])
        attention_weights['encoder_layer_{}'.format(i+1)] = block

    enc_output = tf.reshape(x, [-1, d_model * max_length])    # [batch_size, -1]
    
    twe = WeekEncoding(d_model, name="today_week_encoding")(tf.expand_dims(inputs['today_week'], 0))
    the = HourEncoding(d_model, name="today_hour_encoding")(tf.expand_dims(inputs['today_hour'], 0))
    twe = tf.reshape(twe, [-1, d_model])
    the = tf.reshape(the, [-1, d_model])
    enc_outputs = tf.concat([enc_output, twe, the], axis=1)
    
    ffn = tf.keras.layers.Dense(d_model, name='ffn')(enc_outputs)

    final_output = tf.keras.layers.Dense(target_vocab_size, name='output')(ffn)  # (batch_size, target_vocab_size)

    model = tf.keras.Model(inputs, final_output)

    return model
