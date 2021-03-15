# Assign_1__mnist_perc.py
# James Bao (PSU ID: 934097519)


### Libraries
# Standard Library
import random
import csv

# Third-party libraries
import pandas as pd
import numpy as np


# preprocessing

raw = pd.read_csv('mnist_train.csv', header = None)
m_array = np.array(raw)
np.random.shuffle(m_array) # shuffle the dataset before processing
m_target = m_array[:,0]
m_data = np.divide(m_array[:,1:], 255.0)

valid = pd.read_csv('mnist_validation.csv', header = None)
valid_array = np.array(valid)
np.random.shuffle(valid_array) # shuffle the dataset before processing
v_target = valid_array[:,0]
v_data = np.divide(valid_array[:,1:], 255.0)


# defining the perceptron

class Perceptron:

    # n_samples = rows of X, n_inputs = cols of X
    def __init__(self, learning_rate, val, samples, inputs, y):
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
        self.target = 1D array of binary values representing whether each sample matches val

        '''
        self.lr = learning_rate
        # init weights to 2D array of random values from -0.05 to +0.05
        self.weights=  0.1*np.random.uniform(samples, inputs)-0.05
        # init bias to 1D array of ones multipled by a random weight between -0.05 to +0.05
            #self.bias = np.ones(samples)
        self.bias = 0.1*np.random.rand(samples)-0.05
            ## out = record of linear_output for each sample
            #self.out = np.zeros(samples)
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

    # iterate the perceptron through just 1 sample at a time
    def _iterate_single_(self, data_row, idx):
        '''
        data_row = x_i_k
        self.weights[idx, :] = w_k
        self.bias[idx] = w_0
        self.target = t_k
        linear_output = y_k

        w_k = w_k + LR*(t_k - y_k)*x_i_k
        w_0 = w_0 + LR*(t_k - y_k)*1
        '''
        linear_output = np.dot(data_row, self.weights[idx, :]) + self.bias[idx]
        #self.out[idx] = linear_output
        y_ = self.activation_func(linear_output)
        update = self.lr * (self.target[idx] - y_)
        self.weights[idx, :] += update * data_row
        self.bias[idx] += update   
        return linear_output     


# handling the perceptrons to run 10 perceptrons simultaneously on the same dataset

class Perceptron_Manager:
    '''
    Manages 10 perceptrons and iterates them all through a specified number of epochs 
    '''
    # initialize all the perceptrons to be managed in this instance
    def __init__(self, learning_rate, n_iters, data, target, acc, willhe):
        self.lr = learning_rate
        self.n_it = n_iters
        self.p0 = Perceptron(self.lr, 0, data.shape[0], data.shape[1], m_target)
        self.p1 = Perceptron(self.lr, 1, data.shape[0], data.shape[1], m_target)
        self.p2 = Perceptron(self.lr, 2, data.shape[0], data.shape[1], m_target)
        self.p3 = Perceptron(self.lr, 3, data.shape[0], data.shape[1], m_target)
        self.p4 = Perceptron(self.lr, 4, data.shape[0], data.shape[1], m_target)
        self.p5 = Perceptron(self.lr, 5, data.shape[0], data.shape[1], m_target)
        self.p6 = Perceptron(self.lr, 6, data.shape[0], data.shape[1], m_target)
        self.p7 = Perceptron(self.lr, 7, data.shape[0], data.shape[1], m_target)
        self.p8 = Perceptron(self.lr, 8, data.shape[0], data.shape[1], m_target)
        self.p9 = Perceptron(self.lr, 9, data.shape[0], data.shape[1], m_target)
        self.targets = target
        self.a_vals = acc
        self.yesno = willhe

    # iterate through all weights 
    # compute accuracy for each epoch
    def __iterate__(self, X, mat):

        # iterate through n_it epochs
        for i in range(self.n_it):
            
            # read entire dataset through all 10 perceptrons
            y_temp = self._iterate_helper_(X)
            
            # compare classifications (y_temp) with target values (self.targets)
            accu_tally = np.where(y_temp == self.targets, 1, 0)
            self.a_vals[i] = np.average(accu_tally)
        
        print()
        # create confusion matrix 
        if (self.yesno > 0):
            # pseudocode- for validation data, run the code again on 51st run to generate a new set of actual outputs
            # y_final = actual output (classification) -> rows
            # self.targets = expected output (target) -> columns
            y_final = self._iterate_helper(X) 
            for idx in enumerate(y_final):
                mat[y_final[idx], self.targets[idx]] += 1
            return

    # helper function for reading data through all 10 perceptrons
    def _iterate_helper_(self, data):
        # array of classifications
        y_temp = np.empty(data.shape[0])

        # iterate through all the samples in the dataset 
        for idx, x_i in enumerate(data):
            # create array of outputs from all 10 perceptrons
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

            # store 'best fit' as classification for this sample, in array of classifiations for all samples
            y_temp[idx] = np.argmax(v_arr)

        return y_temp

        



# Initialize data collection arrays
print()
epochs = np.arange(50)
Accuracy_train1 = np.empty(50)
Accuracy_train2 = np.empty(50)
Accuracy_train3 = np.empty(50)
Accuracy_train4 = np.empty(50)
Accuracy_train5 = np.empty(50)
Accuracy_train6 = np.empty(50)


# initialize constructors for managing perceptrons
# for training data, last parameter = 0 [no confusion matrix]
# for validation data, last parameter = 1 [confusion matrix is populated]
test_per1 = Perceptron_Manager(0.00001, 50, m_data, m_target, Accuracy_train1, 0)
test_per2 = Perceptron_Manager(0.001, 50, m_data, m_target, Accuracy_train2, 0)
test_per3 = Perceptron_Manager(0.01, 50, m_data, m_target, Accuracy_train3, 0)

test_per4 = Perceptron_Manager(0.00001, 50, v_data, v_target, Accuracy_train4, 1)
test_per5 = Perceptron_Manager(0.001, 50, v_data, v_target, Accuracy_train5, 1)
test_per6 = Perceptron_Manager(0.1, 50, v_data, v_target, Accuracy_train6, 1)


# initialize confusion matrices for different learning rates
confusion_1 = np.zeros((10, 10))
confusion_2 = np.zeros((10, 10))
confusion_3 = np.zeros((10, 10))

# Run training/validation data through 50 epochs and get accuracy rates
'''
print("Training 0.00001")
print()
test_per1.__iterate__(m_data, confusion_1)
print("Accuracy values per Epoch (0.00001): ")
print(Accuracy_train1)
print()
'''
print("Training 0.001")
print()
test_per2.__iterate__(m_data, confusion_2)
print("Accuracy values per Epoch (0.001): ")
print(Accuracy_train2)
print()
'''
print("Training 0.1")
print()
test_per3.__iterate__(m_data, confusion_3)
print("Accuracy values per Epoch (0.1): ")
print(Accuracy_train3)
print()


print("Validation 0.00001")
print()
test_per4.__iterate__(v_data, confusion_1)
print("Accuracy values per Epoch (0.00001): ")
print(Accuracy_train4)
print()

print("Validation 0.001")
print()
test_per5.__iterate__(v_data, confusion_2)
print("Accuracy values per Epoch (0.001): ")
print(Accuracy_train5)
print()

print("Validation 0.1")
print()
test_per6.__iterate__(v_data, confusion_3)
print("Accuracy values per Epoch (0.1): ")
print(Accuracy_train6)
print()
'''