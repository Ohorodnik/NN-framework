#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:33:13 2020

@author: vitalii
"""

import tensorflow as tf
from tensorflow import keras
from nn import NeuralNet

# %%
X = tf.random.normal(shape=(100, 3))

# %%
tf.random.set_seed(42)

inputs = keras.layers.Input(shape=(3, ))
xx = keras.layers.Dense(5, activation='relu')(inputs)
xx = keras.layers.Dense(3, activation='relu')(xx)
xx = keras.layers.Dense(4, activation='relu')(xx)
outputs = keras.layers.Dense(3, activation='softmax')(xx)

model_tf = keras.Model(inputs=inputs, outputs=outputs)

pred_tf = model_tf(X)

# %%
tf.random.set_seed(42)
lrs = [
       keras.layers.Dense(5, activation='relu'),
       keras.layers.Dense(3, activation='relu'),
       keras.layers.Dense(4, activation='relu'),
       keras.layers.Dense(3, activation='softmax')
]

model_my = NeuralNet(lrs)

pred_my = model_my(X)

# %%

(pred_my == pred_tf).numpy().all()