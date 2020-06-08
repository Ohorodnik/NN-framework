#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:17:55 2020

@author: vitalii
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from optimization import GradientDescent
from utils import train, plot_decision
import datetime

tf.random.set_seed(42)

# %%
lb =  LabelBinarizer()

X, Y = datasets.make_classification(
    n_samples=10_000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    class_sep=2,
    hypercube=True,
    random_state=42
)

# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=1
)
Y_train_ = Y_train
Y_train = lb.fit_transform(Y_train)

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(X_train, dtype=tf.float32), tf.cast(Y_train, dtype=tf.int32))
    ).shuffle(X_train.shape[0]).batch(64)

# %%
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train_.ravel())


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
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

clf = tf.keras.Model(inputs=inputs, outputs=outputs)
clf.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
    )
history = clf.fit(
    x=X_train,
    y=Y_train,
    epochs=50,
    callbacks=[tensorboard_callback]
    )

# %%
loss = tf.keras.losses.CategoricalCrossentropy()
gradient_descent = GradientDescent(learning_rate=0.01)

inputs = tf.keras.Input(shape=(2,))
x = tf.keras.layers.Dense(3, activation='relu')(inputs)
x = tf.keras.layers.Dense(3, activation='relu')(x)
x = tf.keras.layers.Dense(3, activation='relu')(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

clf = tf.keras.Model(inputs=inputs, outputs=outputs)

train(
    NN=clf,
    dataset=dataset,
    loss=loss,
    optimizer=gradient_descent,
    epochs=50
)

plot_decision(clf, X_train, Y_train_)
