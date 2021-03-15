# Assign_2__mnist_NN.py
# James Bao (PSU ID: 934097519)



from math import e
import pandas as pd
import numpy as np


''' 
this version exists solely to test testing0.py, which is the 2-output NN example from the slides
'''

class NN_Manager:
    ''' 
    Manages master weights for input->hidden and hidden->output, as well as hidden/output neuron values and error.
    Takes a fixed 1) learning rate, 2) number of iterations, 3) number of hidden neurons, and 4) momentum value- as parameters. 

    '''
    # initialize all the perceptrons to be managed in this instance
    def __init__(self, learning_rate, n_iters, n_inputs, num_hidden, momentum):
        self.lr = learning_rate
        self.n_it = n_iters
        self.n_in = n_inputs
        self.num = num_hidden
        self.alpha = momentum

        # self.num = 2 (2 hidden) + 1 bias
        # 2 outputs
        self.output_mw = np.array([0.1,-0.2,0.1,0.4,-0.1,0.1]).reshape(2,3)
        # n_in = 2 inputs + 1 bias = 3
        # self.num = 2 (2 hidden)
        self.hidden_mw = np.array([-0.4,0.2,0.1,-0.2,0.4,-0.1]).reshape(2,3)

        # arrays for storing values in hidden and output nodes
        self.hidden_nodes = np.zeros(self.num)
        self.output_nodes = np.zeros(2)
        # arrays for storing error values in hidden and output nodes
        self.hidden_error = np.zeros(self.num)
        self.output_error = np.zeros(2)

        # change in weights (for current epoch)
        self.o_change = np.zeros((2, self.num + 1))
        self.h_change = np.zeros((self.num, self.n_in + 1))
        # change in weights (from previous epoch)
        self.o_change_t = np.zeros((2, (self.num + 1)))
        self.h_change_t = np.zeros((self.num, self.n_in + 1))


    # sigmoid function applied to single value (NOT an array)
    def _sigmoid_func(self,x):
        return 1/(1 + e**(-1*x))  

    # sigmoid function applied to an array
    def _sigmoid_func_arr(self,x):
        return 1/(1 + np.exp(-x))


    # iterate through epochs 
    # compute accuracy for each epoch
    def __iterate__(self, train_X, train_Y, acc_train):
        '''
        same basic procedure from Ass1, but with added steps in _iterate_helper_()- 
        1) calculate initial weights before starting epoch runs
        2) run training data 1x during epoch run to update weights
        3) run training data and validation data on updated weights to get accuracy data

        difference- target values are now 0.9 and 0.1 insead of 1.0 and 0.0
        '''
        # calculate accuracy of initial weights [training]
        y_0 = self._calculate_helper_(train_X)
        # train_Y = [0.9, 0.1]
        # therefore, we should expect the highest value of the output to be at index 0
            #accu_y0 = np.where(y_0 == train_Y, 1, 0)
        accu_y0 = np.where(y_0 == 0, 1, 0)
        # 0th index = accuracy @ 'epoch 0'
        acc_train[0] = 100*np.average(accu_y0)

        print(self.output_nodes, acc_train[0])

        # iterate through n_it epochs
        for i in range(self.n_it):
            
            # read entire dataset through neural network to update weights
            self._iterate_helper_(train_X, train_Y)

            # get output FOR TRAINING DATA after update
            y_0temp = self._calculate_helper_(train_X)
            # compare classifications (y_temp) with target values (self.targets) FOR TRAINING DATA
                #accu_y0 = np.where(y_0temp == train_Y, 1, 0)
            accu_y0 = np.where(y_0temp == 0, 1, 0)
            acc_train[i+1] = 100*np.average(accu_y0)
            print(self.output_nodes, acc_train[i+1])

    # helper function for reading data through neural network
    def _iterate_helper_(self, data, target):
        '''
        initialize binary arrays of correct/incorrect classifications for each output
        t_k = 2D array of target.shape[0] rows and n columns
        in this case, target.shape[0] = 1 and n = 2
        in the for loop, we want to make sure each iteration is reading in 1 row of t_k,
        so that the resulting output_error array is a 1D array
        this eliminates confusion with array dims when computing the hidden_error
        '''
        t_k = target


        # iterate through all the samples in the dataset 
        for idx, x_i in enumerate(data):
            # compute values for hidden and output nodes- forward propagation
            # same as in _calculate_helper_
            hidden_dot = np.add(np.dot(self.hidden_mw[:,1:], x_i), self.hidden_mw[:,0])
            self.hidden_nodes = self._sigmoid_func_arr(hidden_dot)

            output_dot = np.add(np.dot(self.output_mw[:,1:], self.hidden_nodes), self.output_mw[:,0])
            self.output_nodes = self._sigmoid_func_arr(output_dot)

            # compute error values- back propagation
            '''
            output_error del_k = o_k * (1 - o_k) * (t_k - o_k)

            for this sample, target only has 1 sample, and t_k is a 2D array of 1 row and 2 cols
            '''
            o_te = np.subtract(np.ones(2), self.output_nodes)
            o_te_diff = np.subtract(t_k[0,:],self.output_nodes)
            o_temp = np.multiply(self.output_nodes, o_te)
            self.output_error = np.multiply(o_temp, o_te_diff)
            '''
            hidden_error del_j = h_j * (1 - h_j) * sum__k(w_kj . del_k)

            del_1 = h_1 * (1 - h_1) * sum__k(w_k1 . del_k)
            del_2 = h_2 * (1 - h_2) * sum__k(w_k2 . del_k)
            '''
            h_diff = np.subtract(np.ones(self.num), self.hidden_nodes)
            h_diffsum = np.dot(np.transpose(self.output_mw[:,1:]), self.output_error)
                #arg1 = np.transpose(self.output_mw[:,1:])
                ##arg2 = np.transpose(self.output_error)
                #h_diffsum = np.dot(arg1, self.output_error)
            #h_diffsum.reshape(h_diff.shape[0])
            h_temp = np.multiply(self.hidden_nodes, h_diff)
            self.hidden_error = np.multiply(h_temp, h_diffsum)

            # compute change in weights for current epoch
            '''
            self.o_change = np.zeros(2, self.num + 1)
            self.h_change = np.zeros(self.num, self.n_in)

            del__w_kj = LR * del_k * h_j + alpha * del_w_kj_0
            del__w_ji = LR * del_j * x_i + alpha * del_w_ji_0

            '''
            self.o_change[:,0] = np.add(self.lr * self.output_error, self.alpha * self.o_change_t[:,0])
            self.h_change[:,0] = np.add(self.lr * self.hidden_error, self.alpha * self.h_change_t[:,0])

            self.o_change[:,1:] = np.add(self.lr * np.outer(self.output_error, self.hidden_nodes), self.alpha * self.o_change_t[:,1:])
            self.h_change[:,1:] = np.add(self.lr * np.outer(self.hidden_error, x_i), self.alpha * self.h_change_t[:,1:])

            # update weights
            self.hidden_mw = np.add(self.hidden_mw, self.h_change)
            self.output_mw = np.add(self.output_mw, self.o_change)

            # update change in weights for previous epoch
            self.h_change_t = self.h_change
            self.o_change_t = self.o_change

        return


    # helper function for calculating accuracy of initial weights
    def _calculate_helper_(self, data):
        # array of classifications
        y_temp0 = np.empty(data.shape[0])

        # iterate through all the samples in the dataset 
        for idx, x_i in enumerate(data):
            # iterate 1 sample through the network

            # the equivalent of "_calculate_single_" needs to have access to all the values of the hidden neurons
            # this means the inner->hidden calculations need to be done first

            hidden_dot = np.add(np.dot(self.hidden_mw[:,1:], x_i), self.hidden_mw[:,0])
            self.hidden_nodes = self._sigmoid_func_arr(hidden_dot)

            output_dot = np.add(np.dot(self.output_mw[:,1:], self.hidden_nodes), self.output_mw[:,0])
            self.output_nodes = self._sigmoid_func_arr(output_dot)
           
            # store 'best fit' as classification for this sample, in array of classifiations for all samples
            y_temp0[idx] = np.argmax(self.output_nodes)

        # need to make sure output data is of same data type as self.targets
        return y_temp0.astype(int)

    
    def _confusion_matrix_(self, X, target, mat):
        '''
        for validation data, run the code again on 51st run to generate a new set of outputs
        y_final = actual output (classification) -> rows
        self.targets = expected output (target) -> columns
        '''
        y_final = self._calculate_helper_(X) 
        
        for i in range(target.size):
            actual = int(y_final[i])
            xpec = target[i]
            mat[actual, xpec] += 1

        return