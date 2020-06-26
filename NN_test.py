#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:06:09 2020

@author: vitalii
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.random import normal
from tensorflow import keras
from nn import NN, activations

# %%
def test_coupling(
        my_activation,
        tf_activation,
        loss,
        inputs,
        y_true,
        units
        ):
    tf.random.set_seed(42)
    tf_layer = Dense(units, activation=tf_activation)
    tf_layer.build(inputs.shape)
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([inputs, *tf_layer.trainable_weights])
        pred_tf = tf_layer(inputs)
        loss = loss(y_true, pred_tf)
    
    *grads_tf, dY = tape.gradient(loss, [inputs, *tf_layer.trainable_weights, pred_tf])
    
    tf.random.set_seed(42)
    my_layer = NN.Layer(units, my_activation)
    my_layer.build(inputs.shape)
    
    pred_my = my_layer(inputs)
    
    dX, [dW, dB] = my_layer.backprop(dY)
    grads_my = [dX, dW, dB]
    
    assert np.allclose(pred_my, pred_tf)

    assert all(np.allclose(grad_my, grad_tf) for grad_my, grad_tf in zip(grads_my, grads_tf))
    
    return [pred_my, pred_tf, grads_my, grads_tf]
    
# %%
# softmax + KLD (probabilities)

def test_softmax_kld_probs():
    units, N = 4, 100
    shape = (N, 6)
    inputs = normal(shape)
    y_true = tf.math.round(keras.activations.softmax(normal((N, units))))
    
    test_coupling(
        activations.SoftMax(),
        'softmax',
        keras.losses.KLD,
        inputs,
        y_true,
        units
        )

# %%

# linear + Categorical loss with logits

def test_linear_categorical_loss_logits():
    units, N = 4, 100
    shape = (N, 6)
    inputs = normal(shape)
    y_true = tf.math.round(keras.activations.softmax(normal((N, units))))
    
    test_coupling(
        activations.Linear(),
        'linear',
        keras.losses.CategoricalCrossentropy(from_logits=True),
        inputs,
        y_true,
        units
        )

# %% 

# sigmoid

def test_sigmoid_binary_loss():
    N = 400
    shape = (N, 6)
    inputs = normal(shape)
    y_true = tf.math.round(keras.activations.sigmoid(normal((N, 1))))
    
    test_coupling(
        activations.Sigmoid(),
        'sigmoid',
        keras.losses.BinaryCrossentropy(),
        inputs,
        y_true,
        1
        )


# %% 

# MSE

def test_relu_mse():
    N = 400
    shape = (N, 6)
    inputs = normal(shape) * 10
    y_true = normal((N, 1)) * 10
    
    test_coupling(
        activations.ReLU(),
        'relu',
        keras.losses.MSE,
        inputs,
        y_true,
        1
        )

# %%

if __name__ ==  '__main__':
    test_relu_mse()
    test_linear_categorical_loss_logits()
    test_sigmoid_binary_loss()
    test_softmax_kld_probs()
    print('All ok')
