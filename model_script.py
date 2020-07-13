#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:51:04 2020

@author: vitalii
"""

import tensorflow as tf
from sklearn import datasets

from nn import NN, activations, losses, optimization
from nn.utils import plot_decision, train

# %%
X, Y = datasets.make_moons(n_samples=6_000, noise=0.3, random_state=0)

# %%

tf.random.set_seed(42)

loss = losses.BinaryCrossentropy()

dataset = (
    tf.data.Dataset.from_tensor_slices(
        (tf.cast(X, dtype=tf.float32), tf.cast(Y, dtype=tf.int32))
    )
    .shuffle(X.shape[0])
    .batch(64)
)


clf = NN.NeuralNet()
clf.add(NN.Layer(units=3, activation=activations.ReLU()))
clf.add(NN.Layer(units=3, activation=activations.ReLU()))
clf.add(NN.Layer(units=3, activation=activations.ReLU()))
clf.add(NN.Layer(units=1, activation=activations.Sigmoid()))

train(
    NN=clf,
    dataset=dataset,
    loss=loss,
    optimizer=optimization.Adam(),
    epochs=50,
    use_tape=False,
)

plot_decision(clf, X, Y)
