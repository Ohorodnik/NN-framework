import tensorflow as tf
import numpy as np


# %%
class GradientDescent:
    """
    A class represinting optimization algorithm.

    Atributes
    ---------
    learning_rate : float
        learnin rate or step size

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
            learning rate or step size
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
            update = self.learning_rate * self.__get_update(
                var=variable,
                grad=gradient
                )
            variable.assign_sub(update)

    
    def __get_update(self, var, grad):
        """
        Compute updater for given parameter.
        
        WARNING: implementation rely on tf.Variable.name. Each
        paramater must have unique and constant name.
        """

        return grad



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
