#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:22:29 2020

@author: vitalii
"""

import tensorflow as tf
from tensorflow import keras


# %%
class BinaryCrossentropy(object):
    '''
    Cross-entropy loss for binary classification
    loss = -mean(Y * ln(A) + (1 - Y) * ln(1 - A))
    
    Methods
    -------
    get_gradient(A) -> gradinet
        get gradient
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
        
        return - tf.math.reduce_mean(
            y_true * tf.math.log(y_pred)
            + (1 - y_true) * tf.math.log(1 - y_pred)
        )
    
    
    def get_gradient(self, y_true, y_pred):
        """
        Gradient of loss with respect to predicted labels.

        Parameters
        ----------
        y_true : tf.Tensor
            true labels.
        y_pred : tf.Tensor
            predicted lables.

        Returns
        -------
        gradient : tf.Tensor
            gradient of loss evaluated at predicted labels.

        """
        
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
        Gradient of loss with respect to predicted labels.

        Parameters
        ----------
        y_true : tf.Tensor
            true labels.
        y_pred : tf.Tensor
            predicted lables. Logist if self.from_logits=True, probavilities otherwise.

        Returns
        -------
        gradient : tf.Tensor
            gradient of loss evaluated at predicted labels.

        """
        
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
    pass