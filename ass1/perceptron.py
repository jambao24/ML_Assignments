# Assign_1__mnist_perc.py
# James Bao (PSU ID: 934097519)

import pandas as pd
import numpy as np


# defining the perceptron

class Perceptron:

    # n_samples = rows of X, n_inputs = cols of X
    def __init__(self, learning_rate, val, weight, bias):
        '''
        Perceptron has 1 target value, 1 bias value, and 1 'row' of weights for each sample

        learning_rate = provided by user
        X = input data (2D array)
        y = "answer key" (1D array)
        val = the value that the input is being tested for

        # of samples = X.shape[0] (rows)
        # of features = X.shape[1] (columns)
        self.weights = 2D array of weights (w_1 to w_784) for all inputs and every sample
        self.bias = 1D array of biases (w_0) for every sample
        '''
        self.lr = learning_rate
        # init weights to 1D array of 784 random values from -0.05 to +0.05
        self.weights=  weight
        # init bias to 1 random value between -0.05 to +0.05
        self.bias = bias
        # store target value 
        self.val = val
        # activation func is just step_func
        self.activation_func = self._scalar_step_func    

    # 1D array of binary values representing whether each sample matches val 
    # y = 'target' t_k, target = 'answer key' from unprocessed data
    def compute_target(self, target):
        return np.array([1 if i == self.val else 0 for i in target])
  
    # this scalar_step_func is only applied to a single value (NOT an array)
    def _scalar_step_func(self, x):
       if (x >= 0):
            return 1
       else:
            return 0

    # iterate the perceptron through just 1 sample at a time
    def _iterate_single_(self, data_row, idx, y_t):
        '''
        data_row = x_i_k
        self.weights[idx, :] = w_k
        self.bias[idx] = w_0
        y_t = t_k
        linear_output = y_k

        w_k = w_k + LR*(t_k - y_k)*x_i_k
        w_0 = w_0 + LR*(t_k - y_k)*1
        '''

        ## 1D array of binary values representing whether each sample matches val 
        ## y = 'target' t_k, target = 'answer key' from unprocessed data
        #y = np.array([1 if i == self.val else 0 for i in target])   

        linear_output = np.dot(data_row, self.weights) + self.bias
        y_ = self.activation_func(linear_output)
        update = self.lr * (y_t[idx] - y_)
        self.weights += update * data_row
        self.bias += update   
        return linear_output   

    # calculate the linear_output for the perceptron through just 1 sample at a time
    def _calculate_single_(self, data_row, idx):
        return np.dot(data_row, self.weights) + self.bias 
