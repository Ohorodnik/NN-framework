import tensorflow as tf
import numpy as np


# %%
class BaseGradientDescent(object):
    """
    A class represinting optimization algorithm.

    Atributes
    ---------
    learning_rate : float
        learnin rate or step size
    momentum : float [0, 1]
        cosntant used in velocity unpdate:
            velocity = momentum * velocity - learning_rate * gradient
            parameter = parameter + velocity
            
    Methods
    -------
    update_parameters(NN, gradients)
        Applay parameters update.
    """


    def __init__(self, learning_rate=0.01):
        """
        Parameters
        ----------
        learning_rate : float
            learning rate or step 
        momentum : float [0, 1]
            cosntant used in velocity unpdate:
        """
        
        self.learning_rate = learning_rate


    def apply_gradients(self, grads_and_vars):
        """
        Parameters
        ----------
        grads_and_vars : iterable
            list of (gradient, vatiable) pairs
        """

        for gradient, variable in grads_and_vars:
            update = self._get_update(var=variable, grad=gradient)
            variable.assign_add(update)

    
    def _get_update(self, var, grad):
        """
        Compute updater for given parameter.
        """
    
        return -self.learning_rate * grad


# %%
class GradientDescent(BaseGradientDescent):
    """
    A class represinting optimization algorithm.

    Atributes
    ---------
    learning_rate : float
        learnin rate or step size
    momentum : float [0, 1]
        cosntant used in velocity unpdate:
            velocity = momentum * velocity - learning_rate * gradient
            parameter = parameter + velocity
            
    Methods
    -------
    update_parameters(NN, gradients)
        Applay parameters update.
    """


    def __init__(self, learning_rate=0.01, momentum=0):
        """
        Parameters
        ----------
        learning_rate : float
            learning rate or step 
        momentum : float [0, 1]
            cosntant used in velocity unpdate:
        """
        
        super().__init__(learning_rate)
        self.momentum = momentum
        self._velocities = dict()

    
    def _get_update(self, var, grad):
        """
        Compute updater for given parameter.
        
        WARNING: implementation rely on tf.Variable.name. Each
        paramater must have unique and constant name.
        """

        assert var.shape == grad.shape

        if self.momentum != 0:
            velocity = self._velocities.setdefault(var.name, tf.zeros_like(grad))
        
            assert velocity.shape == grad.shape
            
            new_velocity = self.momentum * velocity - self.learning_rate * grad
            self._velocities[var.name] = new_velocity
        else:
            new_velocity = -self.learning_rate * grad
    
        return new_velocity



#%%
class Adam(BaseGradientDescent):
    """
    Adam optimization algorithm. See: https://arxiv.org/abs/1412.6980.
    
    Attributes
    ----------
    learning_rate : float
        step size
    first_moment_rate : float [0, 1]
        decay rate for first moment estimation
    second_moment_rate : foat [0, 1]
        decay rate for second moment estimation
    num_stability : float
        small constant for numerical stability
        
    Methods
    -------
    update_parameters(NN, gradients)
        Applay parameters update.
    """
    
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
        pass
    
    
    def _get_update(self, var, grad):
        pass
    


# %%
def sample_mini_batches(X, Y, mini_batch_size, random_seed=None):
    """
    Iterator over mini-batches.
    Each mini-batch contain a tuple of the form (sub-samle of X, sub-sample of Y).


    Each sub-sample, except the last, has shape same size.

    Last sub-sample may be smaller if mini-bath size is not
    factor of sample size (sample_size % mini_batch_size != 0)

    Sampling done in a way that preserves correspondance between
    X and Y.

    Parameters
    ----------
    X : tf.Tensor
        matrix of input features
        shape=(sample size, number of features)
    Y : tf.tensor
        matrix of correct labels
        shape=(sample size, label dimmentions)
    mini_batch_size : int
        size of mini-bathes

    Returns
    -------
    """
    
    # if random_seed:
    #     tf.random.set_random_seed(random_seed)
    # else:
    #     pass
    
    # shuffled_index = tf.random.shuffle(np.arange(start=0, stop=X.shape[0], step=1))

    # for i in range(np.math.ceil(X.shape[0]/mini_batch_size)):
    #     start = i * mini_batch_size
    #     end = start + mini_batch_size
    #     yield (X[start:end, :], Y[start:end, :])
