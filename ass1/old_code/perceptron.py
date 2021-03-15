import random

# Third-party libraries
import numpy as np

# defining the perceptron

class Perceptron:

    def __init__(self, learning_rate, n_iters, val, X, y):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None
        self.target = None

        # X = data, y = "key" for data, val = target number for the perceptron 
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # init weights are random values from -0.5 to +0.5
        self.weights=  np.random.rand(n_features)-0.5
        # init bias is set to 0
        self.bias = 0

        # set target array y_ based on whether each value in the "key" matches the target number
        self.target = np.array([1 if i == val else 0 for i in y])
    
    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)