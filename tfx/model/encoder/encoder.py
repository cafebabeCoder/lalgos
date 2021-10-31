import numpy as np
import tensorflow as tf

from layers.attention import WeekEncoding, HourEncoding, PositionalEncoding, EncoderLayer, Encoder, Decoder 

def get_encoder_model_raw(max_length, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, training=True, rate=0.1):
    inputs = {
        'his_seq': tf.keras.Input(name='his_seq', shape=(max_length), dtype='int32'),
        'his_week': tf.keras.Input(name='his_week', shape=(max_length), dtype='int32'),
        'today_week' : tf.keras.Input(name='today_week', shape=(), dtype='int32'),
        'his_hour' : tf.keras.Input(name='his_hour', shape=(max_length), dtype='int32'),
        'today_hour' : tf.keras.Input(name='today_hour', shape=(), dtype='int32'),
        'enc_padding_mask': tf.keras.Input(name='enc_padding_mask', shape=(1, 1, max_length), dtype='float32'),
    }
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
