# Assign_2__mnist_NN.py
# James Bao (PSU ID: 934097519)

from neuralnet import OuterNeur
from neuralnet import HiddenNeur
from math import e
import pandas as pd
import numpy as np



# handling the perceptrons to run 10 perceptrons simultaneously on the same dataset

''' 
    Hidden neuron set master weights matrix will have dimensions of (20/50/100, 785)
    Output neuron set master weights matrix will have dimensions of (10, 20/50/100)

    self.output_mw = weights for hidden->output (10 rows, num_hidden+1 columns)
    self.hidden_mw = weights for input->hidden (num_hidden rows, 785 columns)

        "output bias" = self.output_mw[:, 0]
        "hidden bias" = self.hidden_mw[:, 0]

    self.hidden_nodes = 1D array of (num_hidden)
    self.output_nodes = 1D array of (10)
    self.hidden_error = 1D array of (num_hidden) -> output_error must be calculated first!
    self.output_error = 1D array of (10)

    self.o_change = 2D array of (10 rows, num_hidden+1 columns)
    self.h_change = 2D array of (num_hidden rows, 785 columns)
    self.o_change_t = 2D array of (10 rows, num_hidden+1 columns)
    self.h_change_t = 2D array of (num_hidden rows, 785 columns

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
        self.num = num_hidden
        self.alpha = momentum
        self.n_in = n_inputs

        #self.output_mw = np.random.uniform(-0.05, 0.05, (10, self.num+1))
        #self.output_mw_cross = self.output_mw[:,1:]
        #self.output_mw_bias = self.output_mw[:,0]
        # NeuralNet parameters- learning rate, corresponding value from target array [digits 0-9], weights for inputs, weight for bias
        #self.output_s = [OuterNeur(self.lr, self.alpha, i, self.output_mw[i,1:], self.output_mw[i,0]) for i in range(10)]
        self.output_s = [OuterNeur(self.lr, self.alpha, i, self.num+1) for i in range(10)]

        #self.hidden_mw = np.random.uniform(-0.05, 0.05, (self.num, self.n_in))
        #self.hidden_mw_cross = self.hidden_mw[:,1:]
        #self.hidden_mw_bias = self.hidden_mw[:,0]
        # NeuralNet parameters- learning rate, corresponding value from target array [digits 0-9], weights for inputs, weight for bias
        #self.hidden_s = [HiddenNeur(self.lr, self.alpha, i, self.hidden_mw[i,1:], self.hidden_mw[i,0]) for i in range(self.num)]
        self.hidden_s = [HiddenNeur(self.lr, self.alpha, i, self.n_in) for i in range(self.num)]

        # arrays for storing values in hidden and output nodes
        self.hidden_nodes = np.zeros(self.num)
        self.output_nodes = np.zeros(10)
        # arrays for storing error values in hidden and output nodes
        self.hidden_error = np.zeros(self.num)
        self.output_error = np.zeros(10)

        # change in weights (for current epoch)
        self.o_change = np.zeros((10, (self.num + 1)))
        self.h_change = np.zeros((self.num, self.n_in))
        # change in weights (from previous epoch)
        self.o_change_t = np.zeros((10, (self.num + 1)))
        self.h_change_t = np.zeros((self.num, self.n_in))


    # sigmoid function applied to single value (NOT an array)
    def _sigmoid_func(self,x):
        return 1/(1 + e**(-1*x))  

    # sigmoid function applied to an array
    def _sigmoid_func_arr(self,x):
        '''return 1/(1 + np.exp(-x))'''
        return np.reciprocal(1 + np.exp(-x))


    # iterate through epochs 
    # compute accuracy for each epoch
    def __iterate__(self, train_X, train_Y, valid_X, valid_Y, acc_train, acc_valid):
        '''
        same basic procedure from Ass1, but with added steps in _iterate_helper_()- 
        1) calculate initial weights before starting epoch runs
        2) run training data 1x during epoch run to update weights
        3) run training data and validation data on updated weights to get accuracy data

        difference- target values are now 0.9 and 0.1 insead of 1.0 and 0.0
        '''
        # calculate accuracy of initial weights [training]
        y_0 = self._calculate_helper_(train_X)
        accu_y0 = np.where(y_0 == train_Y, 1, 0)
        # 0th index = accuracy @ 'epoch 0'
        acc_train[0] = 100*np.average(accu_y0)

        # calculate accuracy of initial weights [valid]
        y_v = self._calculate_helper_(valid_X)
        accu_v0 = np.where(y_v == valid_Y, 1, 0)
        # 0th index = accuracy @ 'epoch 0'
        acc_valid[0] = 100*np.average(accu_v0)

        print(acc_train[0], acc_valid[0])

        # iterate through n_it epochs
        for i in range(self.n_it):

            # read entire dataset through neural network to update weights
            self._iterate_helper_(train_X, train_Y)

            # get output FOR TRAINING DATA after update
            y_0temp = self._calculate_helper_(train_X)
            # compare classifications (y_temp) with target values (self.targets) FOR TRAINING DATA
            accu_tally1 = np.where(y_0temp == train_Y, 1, 0)
            acc_train[i+1] = 100*np.average(accu_tally1)

            # get output FOR VALIDATION DATA after update
            y_vtemp = self._calculate_helper_(valid_X)
            # compare classifications (y_temp) with target values (self.targets) FOR TRAINING DATA
            accu_tally2 = np.where(y_vtemp == valid_Y, 1, 0)
            acc_valid[i+1] = 100*np.average(accu_tally2)

            print(acc_train[i+1], acc_valid[i+1])

            
        print()
        

    # helper function for reading data through neural network
    def _iterate_helper_(self, data, target):
        # initialize binary arrays of correct/incorrect classifications for each output
        # t_k = 2D array of target.shape[0] rows and 10 columns
        t_k = np.zeros((target.shape[0], 10))
        t_k[:,0] = np.array([0.9 if i == 0 else 0.1 for i in target])
        t_k[:,1] = np.array([0.9 if i == 1 else 0.1 for i in target])
        t_k[:,2] = np.array([0.9 if i == 2 else 0.1 for i in target])
        t_k[:,3] = np.array([0.9 if i == 3 else 0.1 for i in target])
        t_k[:,4] = np.array([0.9 if i == 4 else 0.1 for i in target])
        t_k[:,5] = np.array([0.9 if i == 5 else 0.1 for i in target])
        t_k[:,6] = np.array([0.9 if i == 6 else 0.1 for i in target])
        t_k[:,7] = np.array([0.9 if i == 7 else 0.1 for i in target])
        t_k[:,8] = np.array([0.9 if i == 8 else 0.1 for i in target])
        t_k[:,9] = np.array([0.9 if i == 9 else 0.1 for i in target])

        # iterate through all the samples in the dataset 
        for idx, x_i in enumerate(data):
            # compute values for hidden and output nodes- forward propagation
            # same as in _calculate_helper_
            # the equivalent of "_calculate_single_" needs to have access to all the values of the hidden neurons
            # this means the inner->hidden calculations need to be done first
            self.hidden_nodes = [self.hidden_s[i]._calculate_single_(x_i, idx) for i in range(self.num)]
            self.output_nodes = [self.output_s[j]._calculate_single_(self.hidden_nodes, idx) for j in range(10)]
            h_t = np.asarray(self.hidden_nodes)
            self.hidden_nodes = h_t
            o_t = np.asarray(self.output_nodes)
            self.output_nodes = o_t
                
            # compute error values- back propagation
            '''
            output_error del_k = o_k * (1 - o_k) * (t_k - o_k)
            '''
            self.output_error = [self.output_s[k]._calculate_error_(t_k[idx,k], self.output_nodes[k]) for k in range(10)]
            o_temp = np.asarray(self.output_error)
            self.output_error = o_temp

            '''
            hidden_error del_j = h_j * (1 - h_j) * sum__k(w_kj . del_k)
            '''
            # input from hidden-output weights array can't include the bias weights (which in this case is at index [:,0]). 
            # function output includes columns 1 to 100
            temp = [self.output_s[k]._get_weights_() for k in range(10)]
            s_temp = np.asarray(temp)
            self.hidden_error = [self.hidden_s[l]._calculate_error_(self.hidden_nodes[l], s_temp[:,l], self.output_error) for l in range(self.num)]
            h_temp = np.asarray(self.hidden_error)
            self.hidden_error = h_temp

            # compute change in weights for current epoch
            '''
            del__w_kj = LR * del_k * h_j + alpha * del_w_kj_0
            del__w_ji = LR * del_j * x_i + alpha * del_w_ji_0
            '''
            #self.o_change[:,0] = np.add(self.lr * self.output_error, self.alpha * self.o_change_t[:,0])
            #self.o_change[:,1:] = np.add(self.lr * np.outer(self.output_error, self.hidden_nodes), self.alpha * self.o_change_t[:,1:])
            #self.h_change[:,0] = np.add(self.lr * self.hidden_error, self.alpha * self.h_change_t[:,0])
            #self.h_change[:,1:] = np.add(self.lr * np.outer(self.hidden_error, x_i), self.alpha * self.h_change_t[:,1:])
            
            self.o_change = [self.output_s[l]._calculate_del_w_(self.hidden_nodes, self.output_error[l], self.o_change_t[l,:]) for l in range(10)]
            self.h_change = [self.hidden_s[l]._calculate_del_w_(x_i, self.hidden_error[l], self.h_change_t[l,:]) for l in range(self.num)]
            o_c = np.asarray(self.o_change)
            self.o_change = o_c
            h_c = np.asarray(self.h_change)
            self.h_change = h_c

            # update weights
                #self.hidden_mw = np.add(self.hidden_mw, self.h_change)
                #self.output_mw = np.add(self.output_mw, self.o_change)
            for i in range(self.num):
                self.hidden_s[i]._update_(self.h_change[i,:])
            for j in range(10):
                self.output_s[j]._update_(self.o_change[j,:])

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
            self.hidden_nodes = [self.hidden_s[i]._calculate_single_(x_i, idx) for i in range(self.num)]
            self.output_nodes = [self.output_s[j]._calculate_single_(self.hidden_nodes, idx) for j in range(10)]
            h_t = np.asarray(self.hidden_nodes)
            self.hidden_nodes = h_t
            o_t = np.asarray(self.output_nodes)
            self.output_nodes = o_t
           
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