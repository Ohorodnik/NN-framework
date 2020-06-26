import tensorflow as tf
import numpy as np

from tensorflow import keras
from nn import activations


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
        layers_built = all([layer.built for layer in layers])
        
        if layers_built:
            for layer_l, layer_r in zip(layers[:-1], layers[1:]):
                assert layer_l.units == layer_r.kernel.shape[0]
        else:
            pass
                
        
        self.layers = layers[:]
        self.trainable_weights = []
        self.built = layers_built


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
        
        if not self.built:
            self.build(inputs.shape)
        else:
            pass
        
        activations = inputs
        for layer in self.layers:
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
        
        if layer.built:
            # layer.kernel only exisits after layer.buid() call.
            assert self.layers[-1].units == layer.kernel.shape[0]
        else:
            self.built = False

        self.layers.append(layer)

        return self
    
    
    def build(self, input_shape):
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
        
        input_dim = input_shape[1]
        for layer in self.layers:
            if not layer.built:
                layer.build(input_shape=(1, input_dim))
            else:
                pass
            
            input_dim = layer.units
            
            self.trainable_weights += layer.trainable_weights
            
        self.built = True
    
    
    def backprop(self, dY):
        """
        Backpropagete gratient of the loss through the network.

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
        
        gradients = []
        dA = dY
        
        for layer in reversed(self.layers):
            dA, trainable = layer.backprop(dA)
            gradients = [*trainable, *gradients]
        
        return gradients


# %%
class Layer(object):
    """
    A single generic layer of neural network.

    Attributes
    ---------
    kernel : tf.Tensor
        weihts matrix. Availbale only after Layer.bulid() was called.
    bias : tf.Tensor
        bias vector. Available only after Layer.build() and only if Layer.use_bias is
        set to True.
    trainable_weights : iterable
        trainable weights of layer:
        W - weights associated with layer inputs. shape=(input, output)
        B - biases associated with layer inputs. shape=(1, outputs) (if use_bias=True)
    use_bias : bool
        indicator of whether to use bias term for computing pre-activations.
    activation : function
        activation of the layer.
    built : bool
        indicator of whether layer variables have been initialized.
    units : int
        number of unint is the layer.
    input_shape: int
        number of inputs to a unit.
    kernel_reguralizer : func
        returns loss term for weight regularization.
        
    Methods
    -------
    backprop(dA) -> gradinents
        Compute backpropagation step.
    build(input_shape) -> None
        Create variables for the layer.
    """

    def __init__(self,
            units,
            activation=activations.Linear(),
            kernel_initializer=keras.initializers.GlorotUniform(),
            bias_initializer=tf.zeros,
            input_shape=None,
            use_bias=True,
            kernel_reguralizer=None
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
        kernel_reguralizer : func
            returns loss term for weight regularization.

        Returns
        -------
        None.

        """
        
        self.units = units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.input_shape = input_shape
        self.use_bias = use_bias
        self.kernel_reguralizer = kernel_reguralizer
        self.built = False


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
        
        if not self.built:
            self.built(inputs.shape)
        else:
            pass
        
        self.inputs = inputs
        
        Z = inputs @ self.kernel
        if self.use_bias:
            Z += self.bias
        else:
            pass
        self.Z = Z

        activations = self.activation(Z)

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
            collection of the form:
                [dA, [d-trainable]]
            where:
            dA - gradient with respect to inputs.
            [d-trainable] - gradients with respect to all trainable parameters.

        '''
        
        dZ = tf.reshape(
            dA[:, tf.newaxis, :] @ self.activation.get_jacobian(self.Z),
            shape=dA.shape
        )
        
        dB = tf.math.reduce_sum(dZ, axis=0, keepdims=True)
        
        dW = tf.matmul(self.inputs, dZ, transpose_a=True)
        
        dX = tf.matmul(dZ, self.kernel, transpose_b=True)
        
        return [dX, [dW, dB]]
    
    
    def build(self, input_shape):
        """
        Create variables for the layer.

        Parameters
        ----------
        input_shape : Collection
            shape of an input.

        Returns
        -------
        None.

        """
        input_dim = input_shape[1]
        trainable_weights = []
        self.kernel = tf.Variable(
            self.kernel_initializer(shape=(input_dim, self.units))
            )
        trainable_weights.append(self.kernel)
        
        if self.use_bias:
            self.bias = tf.Variable(self.bias_initializer(shape=(1, self.units)))
            trainable_weights.append(self.bias)
        else:
            pass
        
        self.built = True
        self.trainable_weights = trainable_weights


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
    