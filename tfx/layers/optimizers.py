#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2020/10/20 
# @Author  : lorineluo
# @File    : optimizers.py

import numpy as np
import tensorflow as tf

# https://arxiv.org/abs/1706.03762 attention is all you need的学习率衰减
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000, dtype=tf.float32):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.dtype = dtype

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
  
    d_model = tf.cast(self.d_model, self.dtype)

    return tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2)

  def get_config(self):
    return {"d_model": self.d_model, "warmup_steps":self.warmup_steps}