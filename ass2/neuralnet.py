# Assign_2__mnist_NNcle.py
# James Bao (PSU ID: 934097519)

import pandas as pd
import numpy as np

from math import e

# defining the neurons

class HiddenNeur:

    # n_samples = rows of X, n_inputs = cols of X
    def __init__(self, learning_rate, alpha, val, n):
        '''
        Neuron has 1 target value, 1 bias value, and 1 'row' of weights for each sample

        learning_rate = provided by user
        val = the value that the input is being tested for
        '''
        self.lr = learning_rate
        self.al = alpha
        self.n_ = n
        mw = np.random.uniform(-0.05, 0.05, (self.n_))
        self.weights = mw[1:]
        self.bias = mw[0]   
        # store target value 
        self.val = val

    '''
    # 1D array of binary values representing whether each sample matches val 
    # y = 'target' t_k, target = 'answer key' from unprocessed data
    def compute_target(self, target):
        return np.array([0.9 if i == self.val else 0.1 for i in target])
    '''

    # sigmoid function applied to single value (NOT an array)
    def _sigmoid_func(self,x):
        return 1/(1 + e**(-x))

    # sigmoid function applied to an array
    def _sigmoid_func_arr(self,x):
        return np.reciprocal(1 + np.exp(-x))
    '''
    # iterate the neuron through just 1 sample at a time
    def _iterate_single_(self, data_row, idx):
        
        not sure what to put here
        
        hidden_dot = np.dot(self.weight, data_row) + self.bias
        return self._sigmoid_func_arr(hidden_dot)  
    '''
    # calculate the output value for the hidden neuron, for just 1 sample
    def _calculate_single_(self, data_row, idx):
        hidden_dot = np.dot(self.weights, data_row) + self.bias
        return self._sigmoid_func(hidden_dot) 

    # calculate the error for this hidden neuron
    '''
    hidden = hidden value
    out_weights = w_k (col) from hidden-outer matrix, corresponding to the jth hidden neuron
    out_error = output errors for del_k
    '''
    def _calculate_error_(self, hidden, out_weights, out_error):
        t1 = 1 - hidden
        t2 = np.dot(out_weights, out_error)
        t = np.multiply(hidden, t1)
        return np.multiply(t,t2)

    # calculate the changes for the row of i->h weights corresponding to this hidden neuron
    '''
    data_row = input [1D array]
    error_val = hidden error for this neuron [scalar]
    prev = row from previous weight change matrix [1D array]
    '''
    def _calculate_del_w_(self, data_row, error_val, prev):
        retarr = np.empty(self.n_)
        retarr[0] = np.add(self.lr * error_val, self.al * prev[0])
        retarr[1:] = np.add(self.lr * np.outer(error_val, data_row), self.al * prev[1:])
        return retarr

    # update weights

    def _update_(self, changes):
        self.bias += changes[0]
        self.weights += changes[1:]
        return


class OuterNeur:

    # n = # of hidden nodes
    def __init__(self, learning_rate, alpha, val, n):
        '''
        Neuron has 1 target value, 1 bias value, and 1 'row' of weights for each sample

        learning_rate = provided by user
        val = the value that the input is being tested for
        '''
        self.lr = learning_rate
        self.al = alpha
        self.n_ = n
        mw = np.random.uniform(-0.05, 0.05, (self.n_))
        self.weights = mw[1:]
        self.bias = mw[0] 
        # store target value 
        self.val = val   

    '''
    # 1D array of binary values representing whether each sample matches val 
    # y = 'target' t_k, target = 'answer key' from unprocessed data
    def compute_target(self, target):
        return np.array([0.9 if i == self.val else 0.1 for i in target])
    '''

    # sigmoid function applied to single value (NOT an array)
    def _sigmoid_func(self,x):
        return 1/(1 + e**(-x))

    # sigmoid function applied to an array
    def _sigmoid_func_arr(self,x):
        return np.reciprocal(1 + np.exp(-x))
    '''
    # iterate the perceptron through just 1 sample at a time
    def _iterate_single_(self, input, idx):
        output_dot = np.dot(self.weights, input) + self.bias
        return self._sigmoid_func_arr(output_dot)  
    '''

    
    # calculate the output value for the outer neuron, for just 1 sample
    def _calculate_single_(self, input, idx):
        output_dot = np.dot(self.weights, input) + self.bias
        return self._sigmoid_func(output_dot)  


    # calculate the error for 1 output neuron
    def _calculate_error_(self, target, output):
        t1 = 1.0 - output 
        t2 = np.subtract(target, output)
        t = np.multiply(output, t1)
        return np.multiply(t, t2)


    # calculate the changes for the row of h->o weights corresponding to this hidden neuron
    '''
    h->o matrix dimensions = 10 * num_hidden
    each output neuron is responsible for generating num_hidden+1 values of the matrix

    hidden_vals = hidden neurons output [1D array]
    error_val = output error for this neuron [scalar]
    prev = row from previous weight change matrix [1D array]
    '''
    def _calculate_del_w_(self, hidden_vals, error_val, prev):
        retarr = np.empty(self.n_)
        retarr[0] = np.add(self.lr * error_val, self.al * prev[0])
        retarr[1:] = np.add(self.lr * np.outer(error_val, hidden_vals), self.al * prev[1:])
        return retarr

    # return h->o matrix row (not including bias weight)
    # output row dimensions = num_hidden elements
    def _get_weights_(self):
        return self.weights

    # update weights

    def _update_(self, changes):
        self.bias += changes[0]
        self.weights += changes[1:]
        return
