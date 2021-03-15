# Assign_1__mnist_perc.py
# James Bao (PSU ID: 934097519)


### Libraries
# Standard Library
import random
import csv

# Third-party libraries
import pandas as pd
import numpy as np
#import sklearn as skl 


'''
import matplotlib.pyplot as plt 
'''

# preprocessing

raw = pd.read_csv('mnist_train.csv', header = None)
m_array = np.array(raw)
np.random.shuffle(m_array) # shuffle the dataset before processing
m_target = m_array[:,0]
m_data = np.divide(m_array[:,1:], 255.0)

train = pd.read_csv('mnist_validation.csv', header = None)
valid_array = np.array(train)
np.random.shuffle(valid_array) # shuffle the dataset before processing
v_target = valid_array[:,0]
v_data = np.divide(valid_array[:,1:], 255.0)

'''
#prep = pd.read_csv('preprocess.csv', header = None)
prep = pd.read_csv('test.csv', header = None)
p_array = np.array(prep)
np.random.shuffle(p_array) # shuffle the dataset before processing
p_target = p_array[:,0]
p_data = np.divide(p_array[:,1:], 255.0)


print("target size: ")
print(p_target.shape)
print()
print("data size: ") 
print(p_data.shape)
print()
'''


# defining the perceptron

class Perceptron:

    # n_samples = rows of X, n_features = cols of X
    def __init__(self, learning_rate, val, X, y):
        '''
        Perceptron has 1 target value, 1 bias value, and 1 'row' of weights for each sample

        # of samples = X.shape[0] (rows)
        # of features = X.shape[1] (columns)

        '''
        self.lr = learning_rate
        # init weights to 2D array of random values from -0.05 to +0.05
        self.weights=  0.1*np.random.rand(X.shape[0], X.shape[1])-0.05
        # init bias to 1D array of ones
        self.bias = np.ones(X.shape[0])
        # out = record of linear_output for each sample
        self.out = np.zeros(X.shape[0])
        # set target array y_ based on whether each value in the "key" matches the target number
        self.target = np.array([1 if i == val else 0 for i in y])
        # activation func is just step_func
        self.activation_func = self._scalar_step_func
        # same as above but for an entire array
        self.activ_func = self._unit_step_func
  
    # this scalar_step_func is only applied to a single value (NOT an array)
    def _scalar_step_func(self, x):
       if (x >= 0):
            return 1
       else:
            return 0

    # this unit_step_func is applied to an array
    def _unit_step_func(self, x):
       return np.where(x>0, 1, 0)

    '''
    # iterate the perceptron through all the samples 1 time
    def _iterate_sam_(self, X):
        for idx, x_i in enumerate(X):
            lin_output = np.dot(x_i, self.weights[idx]) + self.bias[idx]
            self.out[idx] = lin_output
            y_ = self.activ_func(lin_output)
            update = self.lr * (self.target[idx] - y_)
            self.weights[idx] += update * x_i
            self.bias[idx] += update
    '''

    # iterate the perceptron through just 1 sample at a time
    def _iterate_single_(self, data_row, idx):
        linear_output = np.dot(data_row, self.weights[idx, :]) + self.bias[idx]
        self.out[idx] = linear_output
        y_ = self.activation_func(linear_output)
        update = self.lr * (self.target[idx] - y_)
        self.weights[idx, :] += update * data_row
        self.bias[idx] += update   
        return linear_output     


class Perceptron_Manager:
    '''
    Manages 10 perceptrons and iterates them all through a specified number of epochs 
    '''
    # initialize all the perceptrons to be managed in this instance
    def __init__(self, n_iters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, target, acc):
        #self.lr = learning_rate
        self.n_it = n_iters
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6
        self.p7 = p7
        self.p8 = p8
        self.p9 = p9
        self.targets = target
        self.a_vals = acc

    # iterate through all weights 
    # compute accuracy for each epoch
    def __iterate__(self, X):

        # iterate through n_it epochs
        for i in range(self.n_it):
            
            # array of classifications
            y_temp = np.empty(X.shape[0])

            # iterate through all samples in the dataset 
            for idx, x_i in enumerate(X):
                v_arr = np.empty(10)
                # iterate 1 sample through all 10 perceptrons
                v_arr[0] = self.p0._iterate_single_(x_i, idx)
                v_arr[1] = self.p1._iterate_single_(x_i, idx)
                v_arr[2] = self.p2._iterate_single_(x_i, idx)
                v_arr[3] = self.p3._iterate_single_(x_i, idx)
                v_arr[4] = self.p4._iterate_single_(x_i, idx)
                v_arr[5] = self.p5._iterate_single_(x_i, idx)
                v_arr[6] = self.p6._iterate_single_(x_i, idx)
                v_arr[7] = self.p7._iterate_single_(x_i, idx)
                v_arr[8] = self.p8._iterate_single_(x_i, idx)
                v_arr[9] = self.p9._iterate_single_(x_i, idx)
                # y_temp[idx] = np.amax(v_arr) - wrong numpy function

                # store 'best fit' as classification in array of classifiations
                y_temp[idx] = np.argmax(v_arr)
            
            # compare classifications (y_temp) with target values (self.targets)
            accu_tally = np.where(y_temp == self.targets, 1, 0)
            self.a_vals[i] = np.average(accu_tally)
            


'''

per0 = Perceptron(0.00001, 0, p_data, p_target)
per1 = Perceptron(0.00001, 1, p_data, p_target)
per2 = Perceptron(0.00001, 2, p_data, p_target)
per3 = Perceptron(0.00001, 3, p_data, p_target)
per4 = Perceptron(0.00001, 4, p_data, p_target)
per5 = Perceptron(0.00001, 5, p_data, p_target)
per6 = Perceptron(0.00001, 6, p_data, p_target)
per7 = Perceptron(0.00001, 7, p_data, p_target)
per8 = Perceptron(0.00001, 8, p_data, p_target)
per9 = Perceptron(0.00001, 9, p_data, p_target)

per10 = Perceptron(0.001, 0, p_data, p_target)
per11 = Perceptron(0.001, 1, p_data, p_target)
per12 = Perceptron(0.001, 2, p_data, p_target)
per13 = Perceptron(0.001, 3, p_data, p_target)
per14 = Perceptron(0.001, 4, p_data, p_target)
per15 = Perceptron(0.001, 5, p_data, p_target)
per16 = Perceptron(0.001, 6, p_data, p_target)
per17 = Perceptron(0.001, 7, p_data, p_target)
per18 = Perceptron(0.001, 8, p_data, p_target)
per19 = Perceptron(0.001, 9, p_data, p_target)

per20 = Perceptron(0.1, 0, p_data, p_target)
per21 = Perceptron(0.1, 1, p_data, p_target)
per22 = Perceptron(0.1, 2, p_data, p_target)
per23 = Perceptron(0.1, 3, p_data, p_target)
per24 = Perceptron(0.1, 4, p_data, p_target)
per25 = Perceptron(0.1, 5, p_data, p_target)
per26 = Perceptron(0.1, 6, p_data, p_target)
per27 = Perceptron(0.1, 7, p_data, p_target)
per28 = Perceptron(0.1, 8, p_data, p_target)
per29 = Perceptron(0.1, 9, p_data, p_target)


Accuracy_vals1 = np.empty(50)
Accuracy_vals2 = np.empty(50)
Accuracy_vals3 = np.empty(50)

# initialize constructors for managing perceptrons
test_per1 = Perceptron_Manager(50, per0, per1, per2, per3, per4, per5, per6, per7, per8, per9, p_target, Accuracy_vals1)
test_per2 = Perceptron_Manager(50, per10, per11, per12, per13, per14, per15, per16, per17, per18, per19, p_target, Accuracy_vals2)
test_per3 = Perceptron_Manager(50, per20, per21, per22, per23, per24, per25, per26, per27, per28, per29, p_target, Accuracy_vals3)

# iterate through data through 50 epochs and get accuracy rates
print("Testing 0.00001")
print()
test_per1.__iterate__(p_data)
print("Accuracy values per Epoch (0.00001): ")
print(Accuracy_vals1)
print()
print("Testing 0.001")
print()
test_per2.__iterate__(p_data)
print("Accuracy values per Epoch (0.001): ")
print(Accuracy_vals2)
print()
print("Testing 0.1")
print()
test_per3.__iterate__(p_data)
print("Accuracy values per Epoch (0.1): ")
print(Accuracy_vals3)
print()

'''