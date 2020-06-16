#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:42:24 2020

@author: vitalii
"""

import numpy as np
import tensorflow as tf
from nn.activations import ReLU, SoftMax, Sigmoid
from tensorflow import keras
from tensorflow.random import uniform, normal
from tensorflow.keras import losses

# %%

my_activations = [
        ReLU(),
        Sigmoid()
        ]
tf_activations = [
        keras.activations.relu,
        keras.activations.sigmoid
        ]

for i, (my_activation, tf_activation) in enumerate(zip(my_activations, tf_activations)):
    X = tf.Variable(tf.random.normal(shape=(100, 5)) * i**(1/2))
    
    my_A = my_activation(X)
    
    with tf.GradientTape() as tape:
        tf_A = tf_activation(X)
    
    assert np.allclose(my_A, tf_A)
    
    tf_dZ = tape.gradient(target=tf_A, sources=[X])[0]
    
    my_dZ = tf.reshape(
        tf.ones(shape=(1, 5)) @ my_activation.get_jacobian(X),
        shape=X.shape
    )
    
    assert np.allclose(my_dZ, tf_dZ)
    
# %%
## testting jacobians
X = tf.Variable(tf.random.normal(shape=(100, 5)))

softmax = SoftMax()
my_A = softmax(X)

with tf.GradientTape() as tape:
    tf_A = keras.activations.softmax(X)

assert np.allclose(my_A, tf_A)

tf_dZ = tape.batch_jacobian(target=tf_A, source=X)

my_dZ = softmax.get_jacobian(X)

assert np.allclose(my_dZ, tf_dZ)

# %%
shape = (4, 2)
y_true = tf.math.round(softmax(normal(shape)))
logits = normal(shape)