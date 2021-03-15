'''no idea what i'm doing so far'''

'''
### Libraries
# Standard Library
import random

# Third-party libraries
import numpy as np

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        # ensure all values in y are 0 or 1, formats them to np.array
        y_ = np.array([1 if i > 0 else 0 for i in y])


        # iteration
        # activation_func only used for 1 sample
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights, self.bias)
                y_predicted = self.activation_func(linear_output)

                update = self.lr + (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.biase += update


    # activation_func used for multiple samples
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted


    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)
        
        
 '''
 
 
'''
 Steve Braich To Everyone | 12:44:54 PM
Michael, I use PyCharm, you can download it free. First make sure you install python 3.6 or higher. Then install Pycharm, create a project using a venv, and then add libraries like numpy and panda. 
'''