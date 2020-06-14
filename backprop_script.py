#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:42:24 2020

@author: vitalii
"""

import tensorflow as tf
from activations import ReLU
from tensorflow import keras

# %%

my_activations = [
        ReLU(),
        ]
tf_activations = [
        keras.activations.relu,
        ]

for i, (my_activation, tf_activation) in enumerate(zip(my_activations, tf_activations)):
    X = tf.Variable(tf.random.normal(shape=(100, 5)) * i**(1/2))
    
    my_A = my_activation(X)
    
    with tf.GradientTape() as tape:
        tf_A = tf_activation(X)
    
    assert (my_A == tf_A).numpy().all()
    
    tf_dZ = tape.gradient(target=tf_A, sources=[X])[0]
    
    my_dZ = tf.reshape(
        tf.ones(shape=(1, 5)) @ my_activation.get_jacobian(X),
        shape=X.shape
    )
    
    assert (my_dZ == tf_dZ).numpy().all()