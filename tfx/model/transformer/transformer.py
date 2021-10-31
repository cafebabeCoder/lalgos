import numpy as np
import tensorflow as tf

from layers.attention import WeekEncoding, HourEncoding, PositionalEncoding, EncoderLayer, Encoder, Decoder 

class Transformer(tf.keras.Model):
    '''
        Transformer

        Args:
            num_layers: encoder和decoder包含多少层
            d_model: 模型embedding参数
            num_heads: 多头的个数
            dff: 2层前馈网络中的第1层， 第二层为d_model
            input_vocab_size: 输入词典长度
            target_vocab_size: 输出词典长度
            pe_input: 输入时，PE的长度; 为input_maximum_position_encoding
            pe_target: 输出时， PE的长度； 为target_maximun_position_encoding
            rate: dropout
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1, name="transformer", **kwargs):
        super(Transformer, self).__init__(name=name, **kwargs)

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_layers, dff, target_vocab_size, pe_target, rate)
        self.ffc = tf.keras.layers.Dense(dff)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        # inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask, inp_week, inp_hour, tar_week, tar_hour
        enc_output, encoder_weights = self.encoder(inputs['inp'], inputs['inp_week'], inputs['inp_hour'], inputs['training'], inputs['enc_padding_mask'])

        dec_outputs, decoder_weights = self.decoder(inputs['tar'], enc_output, inputs['tar_week'], inputs['tar_hour'], inputs['training'], inputs['look_ahead_mask'], inputs['dec_padding_mask'])

        out = self.ffc(dec_outputs)
        final_out = self.final_layer(out)

        return final_out, encoder_weights, decoder_weights 
