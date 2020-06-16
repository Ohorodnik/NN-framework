#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:17:55 2020

@author: vitalii
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from nn.utils import train, plot_decision
from nn import optimization
import datetime

tf.random.set_seed(42)

# %%

X, Y = datasets.make_moons(n_samples=6_000, noise=0.3, random_state=0)

# %%

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(X, dtype=tf.float32), tf.cast(Y, dtype=tf.int32))
    ).shuffle(X.shape[0]).batch(64)

# %%
plt.scatter(X[:, 0], X[:, 1], c=Y.ravel())


# %%

log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
    )

# %%

inputs = tf.keras.Input(shape=(2,))
x = tf.keras.layers.Dense(3, activation='relu')(inputs)
x = tf.keras.layers.Dense(3, activation='relu')(x)
x = tf.keras.layers.Dense(3, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

clf = tf.keras.Model(inputs=inputs, outputs=outputs)
clf.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
    )
history = clf.fit(
    x=X,
    y=Y,
    epochs=50,
    callbacks=[tensorboard_callback]
    )

# %%
tf.random.set_seed(42)

loss = tf.keras.losses.BinaryCrossentropy()

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(X, dtype=tf.float32), tf.cast(Y, dtype=tf.int32))
    ).shuffle(X.shape[0]).batch(64)

inputs = tf.keras.Input(shape=(2,))
x = tf.keras.layers.Dense(3, activation='relu')(inputs)
x = tf.keras.layers.Dense(3, activation='relu')(x)
x = tf.keras.layers.Dense(3, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

clf = tf.keras.Model(inputs=inputs, outputs=outputs)

train(
    NN=clf,
    dataset=dataset,
    loss=loss,
    optimizer=optimization.Adam(),
    epochs=50
)

plot_decision(clf, X, Y)
