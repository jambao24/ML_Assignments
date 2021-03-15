# Assign_1__mnist_perc.py
# James Bao (PSU ID: 934097519)


### Libraries
# Standard Library
import random
import csv

# Third-party libraries
import pandas as pd
#import pandas.plotting._matplotlib as plt
import numpy as np


# preprocessing

raw = pd.read_csv('mnist_train.csv', header = None)
m_array = np.array(raw)
np.random.shuffle(m_array) # shuffle the dataset before processing
m_target = m_array[:,0]
m_data = np.divide(m_array[:,1:], 255.0)

valid = pd.read_csv('mnist_validation.csv', header = None)
#valid = pd.read_csv('ass1/test.csv', header = None) # smaller test file of 420 samples
valid_array = np.array(valid)
#np.random.shuffle(valid_array) # shuffle the dataset before processing
v_target = valid_array[:,0]
v_data = np.divide(valid_array[:,1:], 255.0)


# defining the perceptron

class Perceptron:

    # n_samples = rows of X, n_inputs = cols of X
    def __init__(self, learning_rate, val, weight, bias, y):
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
        # init weights to 1D array of 784 random values from -0.05 to +0.05
        self.weights=  weight
        # init bias to 1 random value between -0.05 to +0.05
        self.bias = bias
        # set target array y_ based on whether each value in the "key" matches the target number
        self.target = np.array([1 if i == val else 0 for i in y])
        # activation func is just step_func
        self.activation_func = self._scalar_step_func
  
    # this scalar_step_func is only applied to a single value (NOT an array)
    def _scalar_step_func(self, x):
       if (x >= 0):
            return 1
       else:
            return 0

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
        linear_output = np.dot(data_row, self.weights) + self.bias
        y_ = self.activation_func(linear_output)
        update = self.lr * (self.target[idx] - y_)
        self.weights += update * data_row
        self.bias += update   
        return linear_output   

    # calculate the linear_output for the perceptron through just 1 sample at a time
    def _calculate_single_(self, data_row, idx):
        return np.dot(data_row, self.weights) + self.bias 


# handling the perceptrons to run 10 perceptrons simultaneously on the same dataset

class Perceptron_Manager:
    '''
    Manages 10 perceptrons and iterates them all through a specified number of epochs 
    '''
    # initialize all the perceptrons to be managed in this instance
    def __init__(self, learning_rate, n_iters, target, acc):
        self.lr = learning_rate
        self.n_it = n_iters
        self.master_weights = np.random.uniform(-0.05, 0.05, (10, 785))
        self.p0 = Perceptron(self.lr, 0, self.master_weights[0, 1:], self.master_weights[0,0], m_target)
        self.p1 = Perceptron(self.lr, 1, self.master_weights[1, 1:], self.master_weights[1,0], m_target)
        self.p2 = Perceptron(self.lr, 2, self.master_weights[2, 1:], self.master_weights[2,0], m_target)
        self.p3 = Perceptron(self.lr, 3, self.master_weights[3, 1:], self.master_weights[3,0], m_target)
        self.p4 = Perceptron(self.lr, 4, self.master_weights[4, 1:], self.master_weights[4,0], m_target)
        self.p5 = Perceptron(self.lr, 5, self.master_weights[5, 1:], self.master_weights[5,0], m_target)
        self.p6 = Perceptron(self.lr, 6, self.master_weights[6, 1:], self.master_weights[6,0], m_target)
        self.p7 = Perceptron(self.lr, 7, self.master_weights[7, 1:], self.master_weights[7,0], m_target)
        self.p8 = Perceptron(self.lr, 8, self.master_weights[8, 1:], self.master_weights[8,0], m_target)
        self.p9 = Perceptron(self.lr, 9, self.master_weights[9, 1:], self.master_weights[9,0], m_target)
        self.targets = target
        self.a_vals = acc


    # iterate through all weights 
    # compute accuracy for each epoch
    def __iterate__(self, X):
        
        # calculate accuracy of initial weights
        #print("Calculating initial accuracy...")
        #print()
        y_0 = self._calculate_helper_(X)
        accu_0 = np.where(y_0 == self.targets, 1, 0)
        # 0th index = accuracy @ 'epoch 0'
        self.a_vals[0] = 100*np.average(accu_0)

        # iterate through n_it epochs
        for i in range(self.n_it):
            
            # read entire dataset through all 10 perceptrons
            y_temp = self._iterate_helper_(X)
            
            # compare classifications (y_temp) with target values (self.targets)
            accu_tally = np.where(y_temp == self.targets, 1, 0)
            self.a_vals[i+1] = 100*np.average(accu_tally)
        

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

        # need to make sure output data is of same data type as self.targets
        return y_temp.astype(int)

    # helper function for calculating accuracy of initial weights
    def _calculate_helper_(self, data):
        # array of classifications
        y_temp0 = np.empty(data.shape[0])

        # iterate through all the samples in the dataset 
        for idx, x_i in enumerate(data):
            # create array of outputs from all 10 perceptrons
            v_arr = np.empty(10)
            # iterate 1 sample through all 10 perceptrons
            v_arr[0] = self.p0._calculate_single_(x_i, idx)
            v_arr[1] = self.p1._calculate_single_(x_i, idx)
            v_arr[2] = self.p2._calculate_single_(x_i, idx)
            v_arr[3] = self.p3._calculate_single_(x_i, idx)
            v_arr[4] = self.p4._calculate_single_(x_i, idx)
            v_arr[5] = self.p5._calculate_single_(x_i, idx)
            v_arr[6] = self.p6._calculate_single_(x_i, idx)
            v_arr[7] = self.p7._calculate_single_(x_i, idx)
            v_arr[8] = self.p8._calculate_single_(x_i, idx)
            v_arr[9] = self.p9._calculate_single_(x_i, idx)

            # store 'best fit' as classification for this sample, in array of classifiations for all samples
            y_temp0[idx] = np.argmax(v_arr)

        # need to make sure output data is of same data type as self.targets
        return y_temp0.astype(int)
    
    def _confusion_matrix_(self, X, mat):
        '''
        for validation data, run the code again on 51st run to generate a new set of outputs
        y_final = actual output (classification) -> rows
        self.targets = expected output (target) -> columns
        '''
        y_final = self._iterate_helper_(X) 
        
        for i in range(self.targets.size):
            actual = int(y_final[i])
            xpec = self.targets[i]
            mat[actual, xpec] += 1
        '''
        #actual = pd.Series(y_final, index = [0-10000], dtype = 'int32')
        #expected = pd.Series(self.targets, index = [0-10000], dtype = 'int64')
        mat = pd.crosstab(y_final, self.targets)
        '''
        return



# Initialize data collection arrays
print()
epochs = np.arange(51)
Accuracy_train1 = np.empty(51)
Accuracy_train2 = np.empty(51)
Accuracy_train3 = np.empty(51)
Accuracy_va1 = np.empty(51)
Accuracy_va2 = np.empty(51)
Accuracy_va3 = np.empty(51)


# initialize constructors for managing perceptrons
    # for training data, last parameter = 0 [no confusion matrix]
    # for validation data, last parameter = 1 [confusion matrix is populated]
test_per1 = Perceptron_Manager(0.00001, 50, m_target, Accuracy_train1)
test_per2 = Perceptron_Manager(0.001, 50, m_target, Accuracy_train2)
test_per3 = Perceptron_Manager(0.01, 50, m_target, Accuracy_train3)

test_per4 = Perceptron_Manager(0.00001, 50, v_target, Accuracy_va1)
test_per5 = Perceptron_Manager(0.001, 50, v_target, Accuracy_va2)
test_per6 = Perceptron_Manager(0.1, 50, v_target, Accuracy_va3)


# initialize confusion matrices for different learning rates
confusion_1 = np.zeros((10, 10))
confusion_2 = np.zeros((10, 10))
confusion_3 = np.zeros((10, 10))

# Run training/validation data through 50 epochs and get accuracy rates
'''
test_per1.__iterate__(m_data)
print("Training Accuracy values per Epoch (0.00001): ")
print(Accuracy_train1)
print()

test_per2.__iterate__(m_data)
print("Training Accuracy values per Epoch (0.001): ")
print(Accuracy_train2)
print()

test_per3.__iterate__(m_data)
print("Training Accuracy values per Epoch (0.1): ")
print(Accuracy_train3)
print()
'''
'''
np.savetxt('acc_tr1.csv', [Accuracy_train1], delimiter=',')
np.savetxt('acc_tr2.csv', [Accuracy_train2], delimiter=',')
np.savetxt('acc_tr3.csv', [Accuracy_train3], delimiter=',')
'''

test_per4.__iterate__(v_data)
print("Validation Accuracy values per Epoch (0.00001): ")
print(Accuracy_va1)
print()
'''
test_per5.__iterate__(v_data)
print("Validation Accuracy values per Epoch (0.001): ")
print(Accuracy_va2)
print()

test_per6.__iterate__(v_data)
print("Validation Accuracy values per Epoch (0.1): ")
print(Accuracy_va3)
print()
''''''
np.savetxt('acc_va1.csv', [Accuracy_va1], delimiter=',')
np.savetxt('acc_va2.csv', [Accuracy_va2], delimiter=',')
np.savetxt('acc_va3.csv', [Accuracy_va3], delimiter=',')
'''

test_per4._confusion_matrix_(v_data, confusion_1)
print("Confusion Matrix 0.00001")
print(confusion_1)
print()
'''
test_per5._confusion_matrix_(v_data, confusion_2)
print("Confusion Matrix 0.001")
print(confusion_2)
print()
test_per6._confusion_matrix_(v_data, confusion_3)
print("Confusion Matrix 0.1")
print(confusion_3)
print()
'''

'''
#np.savetxt('confu1.csv', [confusion_1], delimiter=',')
pd.DataFrame(confusion_1).to_csv("confu1.csv", header=None, index=None)
pd.DataFrame(confusion_2).to_csv("confu2.csv", header=None, index=None)
pd.DataFrame(confusion_3).to_csv("confu3.csv", header=None, index=None)
'''

pd.plotly.plot(Accuracy_train1,'b', Accuracy_va1, 'r')
pd.plotly.plot(Accuracy_train2,'b', Accuracy_va2, 'r')
pd.plotly.plot(Accuracy_train3,'b', Accuracy_va3, 'r')


