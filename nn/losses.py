#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:22:29 2020

@author: vitalii
"""

import tensorflow as tf


# %%
class BinaryCrossentropy(object):
    '''
    Cross-entropy loss for binary classification
    loss = -mean(Y * ln(A) + (1 - Y) * ln(1 - A))

    Methods
    -------
    get_gradient(A) -> gradinet
        gradients of the loss
    '''

    def __call__(self, y_true, y_pred):
        """
        Calculate estimation of loss over mini-batch.

        Parameters
        ----------
        y_true : tf.Tensor
            true labels.
        y_pred : tf.Tensor
            predicted labels.

        Returns
        -------
        loss : float
            estimation of loss.

        """

        y_true = tf.reshape(tf.cast(y_true, tf.float32), (-1, 1))

        assert y_true.shape == y_pred.shape

        return - tf.math.reduce_mean(
            y_true * tf.math.log(y_pred)
            + (1 - y_true) * tf.math.log(1 - y_pred)
        )

    def get_gradient(self, y_true, y_pred):
        """
        Per-sample gradients of loss with respect to predicted labels.

        Parameters
        ----------
        y_true : tf.Tensor
            true labels.
        y_pred : tf.Tensor
            predicted lables.

        Returns
        -------
        gradients : tf.Tensor
            gradients of loss evaluated at predicted labels. (rows)

        """

        y_true = tf.reshape(tf.cast(y_true, tf.float32), (-1, 1))

        assert y_true.shape == y_pred.shape

        return (y_pred - y_true) / (y_pred - y_pred**2) / y_true.shape[0]


# %%
class CategoricalCrossentropy(object):
    '''
    Cross-entropy loss for multi-class clasification

    get_gradient(A, use_logits) -> gradinet
        get gradient
    '''

    def __init__(self, from_logits=False):
        """

        Parameters
        ----------
        from_logits : bool, optional
            Whether predictions are logits or probabilities. The default is False.

        Returns
        -------
        None.

        """

        self.from_logits = from_logits

    def __call__(self, y_true, y_pred):
        """
        Calculate estimation of loss over mini-batch.


        Parameters
        ----------
        y_true : tf.Tensor
            true labels.
        y_pred : tf.Tensor
            predicted labels.

        Returns
        -------
        loss : float
            estimation of loss.

        """

        assert y_true.shape == y_pred.shape

        y_true = tf.cast(y_true, tf.float32)

        if self.from_logits:
            A = tf.keras.activations.softmax(y_pred)
        else:
            A = y_pred

        return - tf.math.reduce_sum(
            1 / y_true.shape[0]
            * (y_true + 1e-07) * tf.math.log(A)
        )

    def get_gradient(self, y_true, y_pred):
        """
        Per-sample gradients of the loss with respect to predicted labels.

        Parameters
        ----------
        y_true : tf.Tensor
            true labels.
        y_pred : tf.Tensor
            predicted lables. Logist if self.from_logits=True, probavilities otherwise.

        Returns
        -------
        gradients : tf.Tensor
            gradients of the loss evaluated at predicted labels. (rows)

        """

        assert y_true.shape == y_pred.shape

        y_true = tf.cast(y_true, tf.float32)

        if self.from_logits:
            return (tf.keras.activations.softmax(y_pred) - y_true) / y_true.shape[0]
        else:
            return -1 / y_true.shape[0] * (y_true + 1e-07) / y_pred


# %%
class MeanSquaredError(object):
    '''
    Loss for regression.

    get_gradient(A) -> gradinet
        get gradient
    '''

    def __call__(self, y_true, y_pred):
        """
        Calcaulate estimate of loss over mini-bathc.

        Parameters
        ----------
        y_true : tf.Tensor
            true values.
        y_pred : tf.Tensor
            estimated values.

        Returns
        -------
        loss : float
            loss over mini-batch.

        """

        y_true = tf.reshape(y_true, (-1, 1))

        assert y_true.shape == y_pred.shape

        return tf.math.reduce_mean(
            (y_true - y_pred)**2
        )

    def get_gradient(self, y_true, y_pred):
        """
        Per-sample gradients of MSE loss with respect to y_pred.

        Parameters
        ----------
        y_true : tf.Tensor
            true values.
        y_per : tf.Tensor
            estimated values.

        Returns
        -------
        gradients : tf.Tensor
            gradients of the loss with respect to y_pred. (rows)

        """

        y_true = tf.reshape(y_true, (-1, 1))

        assert y_true.shape == y_pred.shape

        return - 2 / (y_true.shape[0]) * (y_true - y_pred)
