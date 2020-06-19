#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:42:24 2020

@author: vitalii
"""

import numpy as np
import tensorflow as tf
from nn.activations import ReLU, SoftMax, Sigmoid
from nn.losses import CategoricalCrossentropy, BinaryCrossentropy, MeanSquaredError
from tensorflow import keras
from tensorflow.random import uniform, normal
from tensorflow.keras import losses

# %%
# Elementwice activations
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
# Softmax activations
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
# Categorical loss with logits output.

shape = (100, 4)
y_true = tf.math.round(softmax(normal(shape)))
logits = normal(shape)

my_loss = CategoricalCrossentropy(from_logits=True)
tf_loss = losses.CategoricalCrossentropy(from_logits=True)

with tf.GradientTape(persistent=True) as tape:
    tape.watch(logits)
    loss_tf = tf_loss(y_true=y_true, y_pred=logits)
    
loss_my = my_loss(y_true, y_pred=logits)

assert np.allclose(loss_my, loss_tf)
    
dz_tf = tape.gradient(loss_tf, [logits])[0]
dz_my = my_loss.get_gradients(y_true, logits)

assert np.allclose(dz_tf, dz_my)

# %%
# Categorical loss with probability output.

shape = (100, 5)
y_true = tf.math.round(softmax(normal(shape)))
logits = normal(shape)

my_loss = CategoricalCrossentropy(from_logits=False)
tf_loss = losses.KLDivergence()

with tf.GradientTape(persistent=True) as tape:
    tape.watch(logits)
    pred = softmax(logits)
    loss_tf = tf_loss(y_true=y_true, y_pred=pred)
    
loss_my = my_loss(y_true, y_pred=pred)

# float point comparison + numerical stability issues
#assert np.allclose(loss_my, loss_tf, rtol=1e-05, atol=1e-08)
    
dz_tf, da_tf = tape.gradient(loss_tf, [logits, pred])
da_my = my_loss.get_gradients(y_true, pred)

assert np.allclose(da_my, da_tf, rtol=1e-05, atol=1e-08)

# %%
# Binary loss with probability output.

shape = (500, 1)
y_true = tf.math.round(softmax(normal(shape)))
logits = normal(shape)

my_loss = BinaryCrossentropy()
tf_loss = losses.BinaryCrossentropy()

with tf.GradientTape(persistent=True) as tape:
    tape.watch(logits)
    pred = keras.activations.sigmoid(logits)
    loss_tf = tf_loss(y_true=y_true, y_pred=pred)
    
loss_my = my_loss(y_true, y_pred=pred)

assert np.allclose(loss_my, loss_tf, rtol=1e-05, atol=1e-08)
    
dz_tf, da_tf = tape.gradient(loss_tf, [logits, pred])
da_my = my_loss.get_gradients(y_true, pred)

assert np.allclose(da_my, da_tf, rtol=1e-05, atol=1e-08)

# %%
# MSE losss

shape = (500, 1)
y_true = normal(shape)
y_pred = normal(shape)

my_loss = MeanSquaredError()
tf_loss = losses.MeanSquaredError()

with tf.GradientTape() as tape:
    tape.watch(y_pred)
    loss_tf = tf_loss(y_true, y_pred)

loss_my = my_loss(y_true, y_pred)

assert np.allclose(loss_my, loss_tf)
    
da_tf = tape.gradient(loss_tf, [y_pred])[0]
da_my = my_loss.get_gradients(y_true, y_pred)

assert np.allclose(da_tf, da_my)