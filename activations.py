#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:00:53 2020

@author: vitalii
"""

import tensorflow as tf

# %%
class ReLU(object):
    '''
    Rectified linear unit activation.
    A = max(Z, 0)
    
    Methods
    -------
    get_jacobians(X) -> jacobians
        get batch jacobions with respect to preacivation Z, evaluated at X.
        WARNING: 3D tensor.
    '''
    
    def __call__(self, Z):
        """
        Applies ReLu activation to preactivations.

        Parameters
        ----------
        Z : tf.Tensor
            inputs.

        Returns
        -------
        activations : tf.Tensor
            inputs thransformed by ReLu activation function.

        """
        
        return tf.math.maximum(Z, 0)
    
    
    def get_jacobian(self, Z):
        """
        Compute jacobian of ReLu activation function with respect to input (Z),
        evaluated at Z.

        Parameters
        ----------
        Z : tf.Tensor
            compute jocobian at Z.

        Returns
        -------
        jacobian : tf.Tensor
            jacobian ReLu computed at Z.

        """
        
        partials = tf.math.maximum(tf.math.sign(Z), 0)
        
        return tf.linalg.diag(partials)
    

# %%
class SoftMax(object):
    '''
    Softmax activation.
    A_i = e^(Z_i) / sum(e^Z)
    
    Methods
    -------
    get_jacobian(X) -> jacobian
        get jacobion with respect to preacivation Z, evaluated at X.
    '''
    pass


# %%
class Sigmoid(object):
    '''
    Sigmoid activation
    A = 1 / (1 + e^Z)
    
    Methods
    -------
    get_jacobian(X) -> jacobian
        get jacobion with respect to preacivation Z, evaluated at X.
    '''
    pass


# %%

