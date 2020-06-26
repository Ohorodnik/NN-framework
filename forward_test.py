#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:33:13 2020

@author: vitalii
"""

import tensorflow as tf
from tensorflow import keras
from nn.NN import NeuralNet, Layer

# %%
X = tf.random.normal(shape=(100, 3))

# %%

def test_forward_NeuralNet():
    tf.random.set_seed(42)
    
    inputs = keras.layers.Input(shape=(3, ))
    xx = keras.layers.Dense(5, activation='relu')(inputs)
    xx = keras.layers.Dense(3, activation='relu')(xx)
    xx = keras.layers.Dense(4, activation='relu')(xx)
    outputs = keras.layers.Dense(3, activation='softmax')(xx)
    
    model_tf = keras.Model(inputs=inputs, outputs=outputs)
    
    pred_tf = model_tf(X)
    
    tf.random.set_seed(42)
    lrs = [
           keras.layers.Dense(5, activation='relu'),
           keras.layers.Dense(3, activation='relu'),
           keras.layers.Dense(4, activation='relu'),
           keras.layers.Dense(3, activation='softmax')
    ]
    
    model_my = NeuralNet(lrs)
    
    pred_my = model_my(X)
    
    
    (pred_my == pred_tf).numpy().all()


# %%


def test_initialization():
    input_shape = (1, 3)
    
    tf.random.set_seed(42)
    
    l_tf = keras.layers.Dense(3)
    l_tf.build(input_shape)
    
    
    tf.random.set_seed(42)
    
    l_my = Layer(3)
    l_my.build(input_shape)
    
    
    (l_my.kernel == l_tf.kernel).numpy().all()
    (l_my.bias == l_tf.bias).numpy().all()

# %%

def test_activations():
    input_shape = (1, 3)
    activations = [
        keras.activations.relu,
        keras.activations.softmax,
        keras.activations.tanh,
        keras.activations.sigmoid,
        keras.activations.linear
        ]
    
    widths = list(range(2, 7))
    
    for activation, units in zip(activations, widths):
        tf.random.set_seed(units)
    
        l_tf = keras.layers.Dense(units, activation=activation)
        l_tf.build(input_shape)
        
        tf.random.set_seed(units)
        
        l_my = Layer(units, activation=activation)
        l_my.build(input_shape)
        
        assert ((l_my.kernel == l_tf.kernel).numpy().all()
                and (l_my.bias == l_tf.bias).numpy().all())
        
        X = tf.random.normal(shape=(100, 3))
        
        assert (l_my(X) == l_tf(X)).numpy().all()

# %%

def test_forward_layer_net():
    tf.random.set_seed(42)
    
    inputs = keras.layers.Input(shape=(3, ))
    xx = keras.layers.Dense(5, activation='relu')(inputs)
    xx = keras.layers.Dense(3, activation='relu')(xx)
    xx = keras.layers.Dense(4, activation='relu')(xx)
    outputs = keras.layers.Dense(3, activation='softmax')(xx)
    
    model_tf = keras.Model(inputs=inputs, outputs=outputs)
    
    pred_tf = model_tf(X)
    
    
    tf.random.set_seed(42)
    lrs = [
           Layer(5, activation=keras.activations.relu),
           Layer(3, activation=keras.activations.relu),
           Layer(4, activation=keras.activations.relu),
           Layer(3, activation=keras.activations.softmax)
    ]
    
    model_my = NeuralNet(lrs)
    
    pred_my = model_my(X)
    
    
    
    (pred_my == pred_tf).numpy().all()
    
# %%

if __name__ == '__main__':
    test_activations()
    test_forward_NeuralNet()
    test_forward_layer_net()
    test_initialization()
    print('All ok')