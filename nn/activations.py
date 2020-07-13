#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:00:53 2020.

@author: vitalii
"""

import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.linalg import diag


# %%
class ReLU(object):
    """
    Rectified linear unit activation.

    A = max(Z, 0)

    Methods
    -------
    get_jacobians(X) -> jacobians
        get batch jacobions with respect to preacivation Z, evaluated at X.
    """

    def __call__(self, Z):
        """
        Applies elementwise ReLu activation.

        Parameters
        ----------
        Z : tf.Tensor
            inputs.

        Returns
        -------
        A : tf.Tensor
            inputs thransformed by ReLu activation function.

        """

        return tf.math.maximum(Z, 0)

    def get_jacobian(self, Z):
        """
        Computes batch jacobians of ReLu activation function with respect to input (Z),
        evaluated at Z.

        Parameters
        ----------
        Z : tf.Tensor
            compute jocobian at Z.

        Returns
        -------
        jacobian : tf.Tensor
            3D tensor of batch jacobians.

        """

        partials = tf.math.maximum(tf.math.sign(Z), 0)

        return tf.linalg.diag(partials)


# %%
class SoftMax(object):
    """
    Softmax activation.
    A_i = e^(Z_i) / sum(e^Z)

    Methods
    -------
    get_jacobian(Z) -> jacobian
        get jacobion with respect to inputs Z, evaluated at Z.
    """

    def __call__(self, Z):
        """
        Applies softmax activation funcion.

        Parameters
        ----------
        Z : tf.Tensor
            inputs (logits).

        Returns
        -------
        A : tf.Tensor
            inputs transformed by softmax activation function (probabilities).
        """

        return activations.softmax(Z)

    def get_jacobian(self, Z):
        """
        computes batch jacobians of sofmax acvivation fuction, evaluated at Z.

        Parameters
        ----------
        Z : tf.Tensor
            inputs.

        Returns
        -------
        jacobian : tf.Tesnor
            3D tensor of batch jacobians.

        """

        A = self(Z)

        return tf.linalg.diag(A) - tf.reshape(A, (-1, Z.shape[1], 1)) @ tf.reshape(
            A, (-1, 1, Z.shape[1])
        )


# %%
class Sigmoid(object):
    """
    Sigmoid activation
    A = 1 / (1 + e^-Z)

    Methods
    -------
    get_jacobian(X) -> jacobian
        get jacobion with respect to preacivation Z, evaluated at X.
    """

    def __call__(self, Z):
        """
        Applies elementwise sigmoid activation function.

        Parameters
        ----------
        Z : tf.Tensor
            inputs.

        Returns
        -------
        A : tf.Tensor
            inputs transformed by sigmoid activation function.

        """

        return activations.sigmoid(Z)

    def get_jacobian(self, Z):
        """
        computes batch jacobians of sigmoid acvivation fuction, evaluated at Z.

        Parameters
        ----------
        Z : tf.Tensor
            inputs.

        Returns
        -------
        jacobian : tf.Tesnor
            3D tensor of batch jacobians.

        """

        A = self(Z)

        return diag(A * (1 - A))


# %%
class Linear(object):
    """
    Linear activation function

    Methods
    -------
    get_jacobian(Z) -> jacobian
        computes bath jacobians.

    """

    def __call__(self, Z):

        return Z

    def get_jacobian(self, Z):

        return tf.eye(Z.shape[1], batch_shape=Z.shape[:1])
