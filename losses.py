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