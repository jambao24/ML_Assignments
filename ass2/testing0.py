# Assign_2__mnist_NN.py
# James Bao (PSU ID: 934097519)


### Libraries
# Third-party libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from neuralnet_manager_0 import NN_Manager

t_data = np.array([1,0]).reshape(1,2)
t_target = np.array([0.9,0.1]).reshape(1,2)

epochs = np.arange(51)
acc = np.zeros(51)

# learning rate = 0.1, 50 iters, 2 inputs, 2 hidden, momentum = 0.9
test = NN_Manager(0.1, 50, 2, 2, 0.9)
# the below function will print out 1) the output values and 2) the accuracy, for the toy dataset, once per epoch

test.__iterate__(t_data, t_target, acc)

print()

