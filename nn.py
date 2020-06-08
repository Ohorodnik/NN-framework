import tensorflow as tf
import numpy as np
from itertools import count


class NeuralNet:
    """
    A neural network model

    Attributes
    ----------
    trainable_params : itarable
        contains all trainable parameters.
    layers : iterable
        ordered collection of network layers.
    """


    def __init__(self, layers=[]):
        """
        Parameters
        ----------
        layers : iterable
            layers of the network
        """

        trainable_weights = []
        for layer in layers:
            trainalble_weights += layer.trainable_weights

        self.layers = layers
        self.trainable_weights = trainable_weights


    def __call__(self, inputs):
        """
        Given inputs predict correct outputs

        Parameters
        ----------
        inputs : tf.Tensor
            inupt features. shape=(sample size, number of features)

        Returns
        -------
        activations : tf.Tensor
            activations of last layer on network.
            shape=(sample size, output dimmentions)
        """
        
        activations = inputs
        for layer in self.layers:
            assert inputs.shape[1] == layer.W.shape[0]
            activations = layer(activations)

        return activations


    def add(self, layer):
        """
        Apend leayer to network

        Layer input dimmentions must be same as output dimmentions
        of network.

        Parameters
        ----------
        layer : Layer
            instance of Layer.

        Returns
        -------
        self
            network with added layer 
        """

        if self.layers:
            assert self.layers[-1].W.shape[1] == layer.W.shape[0]
        
        self.layers.append(layer)
        self.trainable_weights += layer.trainavle_weights

        return self


class Layer:
    """
    A single layer of neural network.

    Attributes
    ---------
    W : tf.Variable
        Weights associated with layer inputs
        shape=(input, output)
    B : tf.Variable
        Biases associated with layer inputs
        shape=(1, outputs)
    activation : tensorflow function
        activation of the layer
    """
    
    _id = count(0)

    def __init__(self, units_num, input_dim, activation, initialization):
        """
        Parameters
        ----------
        units_num : int
            number of units in layer
        input_dim : int
            number of unit inputs
        activation : func
            activation function
        initialization: func
            function to initialize W
            func(shape, dtype) -> tf.constant
        """

        

    def __call__(self, inputs):
        """
        Given inpust compute actiovations of the layer.

        Parameters
        ----------
        inputs : tf.Tensor
            inputs to layer
            shape=(sample size, input dimention)

        Returns
        -------
        activation : tf.Tensor
            activations of the layer
            shape=(sample size, number of units in the layer)
        """


        return acivations



class Dropout:
    """
    Dropout regulariozation layer.

    Attributes
    ----------
    keep_prob : float
        probability of retraining unit
        0 < keep_prob < 1
    """

    def __init__(self, keep_prob):
        pass



    def __call__(self, activations):
        """
        Apply dropout regularization to the activataions.
        Rescales activations to have uncahnged expectaions.

        Parameters
        ----------
        activations : tf.Tensor
            activations of the layer

        Returns
        -------
        reguralized_activation : tf.Tensor
            reguralized activations of the layer
        """

        reguralized_activations = None

        return reguralized_activations
