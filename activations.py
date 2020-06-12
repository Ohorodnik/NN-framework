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

