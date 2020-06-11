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
    get_jacobian(X) -> jacobian
        get jacobion with respect to preacivation Z, evaluated at X.
    '''
    pass
    

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
class BinaryCrossentropy(object):
    '''
    Cross-entropy loss for binary classification
    loss = -mean(Y * ln(A) + (1 - Y) * ln(1 - A))
    
    Methods
    -------
    get_gradient(A) -> gradinet
        get gradient
    '''
    pass


# %%
class CategoricalCrossentropy(object):
    '''
    Cross-entropy loss for multi-class clasification
    
    get_gradient(A, use_logits) -> gradinet
        get gradient
    '''
    pass


# %%
class MeanSquaredError(object):
    '''
    Loss for regression.
    
    get_gradient(A) -> gradinet
        get gradient
    '''
    pass

