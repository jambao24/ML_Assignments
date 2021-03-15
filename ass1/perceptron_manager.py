# Assign_1__mnist_perc.py
# James Bao (PSU ID: 934097519)

from perceptron import Perceptron
import pandas as pd
import numpy as np



# handling the perceptrons to run 10 perceptrons simultaneously on the same dataset

class Perceptron_Manager:
    '''
    Manages 10 perceptrons and iterates them all through a specified number of epochs 
    Each perceptron is initialized with a predetermined learning rate, target value, and weights
    '''
    # initialize all the perceptrons to be managed in this instance
    def __init__(self, learning_rate, n_iters):
        self.lr = learning_rate
        self.n_it = n_iters
        self.master_weights = np.random.uniform(-0.05, 0.05, (10, 785))
        self.p0 = Perceptron(self.lr, 0, self.master_weights[0, 1:], self.master_weights[0,0])
        self.p1 = Perceptron(self.lr, 1, self.master_weights[1, 1:], self.master_weights[1,0])
        self.p2 = Perceptron(self.lr, 2, self.master_weights[2, 1:], self.master_weights[2,0])
        self.p3 = Perceptron(self.lr, 3, self.master_weights[3, 1:], self.master_weights[3,0])
        self.p4 = Perceptron(self.lr, 4, self.master_weights[4, 1:], self.master_weights[4,0])
        self.p5 = Perceptron(self.lr, 5, self.master_weights[5, 1:], self.master_weights[5,0])
        self.p6 = Perceptron(self.lr, 6, self.master_weights[6, 1:], self.master_weights[6,0])
        self.p7 = Perceptron(self.lr, 7, self.master_weights[7, 1:], self.master_weights[7,0])
        self.p8 = Perceptron(self.lr, 8, self.master_weights[8, 1:], self.master_weights[8,0])
        self.p9 = Perceptron(self.lr, 9, self.master_weights[9, 1:], self.master_weights[9,0])


    # iterate through all weights 
    # compute accuracy for each epoch
    def __iterate__(self, train_X, train_Y, valid_X, valid_Y, acc_train, acc_valid):
        '''
        revised procedure- 
        1) calculate initial weights before starting epoch runs
        2) run training data 1x during epoch run to update weights
        3) run training data and validation data on updated weights to get accuracy data
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

        # iterate through n_it epochs
        for i in range(self.n_it):
            
            # read entire dataset through all 10 perceptrons to update weights
            #y_temp = self._iterate_helper_(train_X, train_Y)
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
        

    # helper function for reading data through all 10 perceptrons
    def _iterate_helper_(self, data, target):
        # initialize binary arrays of correct/incorrect classifications for each perceptron
        t0 = self.p0.compute_target(target)
        t1 = self.p1.compute_target(target)
        t2 = self.p2.compute_target(target)
        t3 = self.p3.compute_target(target)
        t4 = self.p4.compute_target(target)
        t5 = self.p5.compute_target(target)
        t6 = self.p6.compute_target(target)
        t7 = self.p7.compute_target(target)
        t8 = self.p8.compute_target(target)
        t9 = self.p9.compute_target(target)

        # iterate through all the samples in the dataset 
        for idx, x_i in enumerate(data):
            # iterate 1 sample through all 10 perceptrons
            # the _iterate_single_ function needs the classification array as an input
            self.p0._iterate_single_(x_i, idx, t0)
            self.p1._iterate_single_(x_i, idx, t1)
            self.p2._iterate_single_(x_i, idx, t2)
            self.p3._iterate_single_(x_i, idx, t3)
            self.p4._iterate_single_(x_i, idx, t4)
            self.p5._iterate_single_(x_i, idx, t5)
            self.p6._iterate_single_(x_i, idx, t6)
            self.p7._iterate_single_(x_i, idx, t7)
            self.p8._iterate_single_(x_i, idx, t8)
            self.p9._iterate_single_(x_i, idx, t9)
        return


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