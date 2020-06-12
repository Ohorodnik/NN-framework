import tensorflow as tf
import numpy as np
from itertools import count


# %%
class NeuralNet(object):
    """
    A neural network model

    Attributes
    ----------
    trainable_params : itarable
        contains all trainable parameters.
    layers : iterable
        ordered collection of network layers.
        
    Methods
    -------
    add(layer) -> self
        append layer to network.
    backprop() -> gradinents
        Backpropagete gdatient of the loss through the network.
    """


    def __init__(self, layers=[]):
        """
        Parameters
        ----------
        layers : iterable
            layers of the network
        """

        self.layers = layers
        self.trainable_weights = []
        self._is_bilded = False


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
            assert activations.shape[1] == layer.W.shape[0]
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

        return self
    
    
    def build(self,input_shape):
        """
        Create variables for all layers in the network.

        Parameters
        ----------
        input_shape : Collection
            shape of a single input.

        Returns
        -------
        None.

        """
        
        input_dim = input_shape
        for layer in self.layers:
            layer.build(input_dim)
            input_dim = (layer.units, )
            
            self.trainable_weights += layer.trainable_weights
    
    
    def backprop(self, dY):
        """
        Backpropagete gdatient of the loss through the network.

        Parameters
        ----------
        dY : tf.Tensor
            gradien of the loss with respect to output of the network.

        Returns
        -------
        gradients : iterable
            collection of the gradients of the loss with respect to all trainable
            parametrs. (Same order as trainable_weights)

        """
        
        gradients = None
        
        return gradients


# %%
class Layer(object):
    """
    A single generic layer of neural network.

    Attributes
    ---------
    trainable_weights : iterable
        trainable weights of layer:
        W - weights associated with layer inputs. shape=(input, output)
        B - biases associated with layer inputs. shape=(1, outputs) (if use_bias=True)
    activation : function
        activation of the layer.
    units : int
        number of unint is the layer.
    input_shape: int
        number of inputs to a unit.
    l2_regularization : float
        constant controling amount of regularization applied to weight matrix.
        
    Methods
    -------
    backprop(dA) -> gradinents
        Compute backpropagation step.
    build(input_shape) -> None
        Create variables for the layer.
    """
    
    _id = count(0)

    def __init__(
            self, units, activation, kernel_initializer, bias_initializer,
             input_shape=None, use_bias=True, l2_regularizatoin=0.0
        ):
        """

        Parameters
        ----------
        units_num : int
            number of unint is the layer.
        activation : func
            ativation function for layer.
        kernel_initializer : func
            given shape return tensor inintialized acording to initialization scheme.
        bias_initializer : func
            given shape return tensor inintialized acording to initialization scheme.
        input_num : int, optional
            number of inputs to a unit. The default is None
        use_bias : bool, optional
            wheter to use bias term when computing preactivation. The default is True.
        l2_regularizatoin : float, optional
            constant controling amount of regularization applied to weight matrix.
            The default is 0.0.

        Returns
        -------
        None.

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

        activations = None

        return activations

    def backprop(self, dA):
        '''
        Compute backpropagation step.

        Parameters
        ----------
        dA : tf.Tensor
            gradient of  loss with respect to activations of the layer

        Returns
        -------
        gradients : iterable
            gradients of the loss with respect to trainable weights of layer

        '''
        
        gradients = None
        
        return gradients
    
    
    def build(self, input_shape):
        """
        Create variables for the layer.

        Parameters
        ----------
        input_shape : Collection
            shape of a single input.

        Returns
        -------
        None.

        """


# %%
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


# %%
class BatchNormalization(object):
    """
    TODO
    """
    