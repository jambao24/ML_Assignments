# Assign_1__mnist_perc.py
# James Bao (PSU ID: 934097519)


### Libraries
# Third-party libraries
import pandas as pd
import numpy as np

import pandas.plotting._matplotlib as plt

from perceptron_manager import Perceptron_Manager

# preprocessing

raw = pd.read_csv('mnist_train.csv', header = None)
m_array = np.array(raw)
np.random.shuffle(m_array) # shuffle the dataset before processing
m_target = m_array[:,0]
m_data = np.divide(m_array[:,1:], 255.0)

valid = pd.read_csv('mnist_validation.csv', header = None)
#valid = pd.read_csv('ass1/test.csv', header = None) # smaller test file of 420 samples
valid_array = np.array(valid)
np.random.shuffle(valid_array) # shuffle the dataset before processing
v_target = valid_array[:,0]
v_data = np.divide(valid_array[:,1:], 255.0)





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
test_per1 = Perceptron_Manager(0.00001, 50)
test_per2 = Perceptron_Manager(0.001, 50)
test_per3 = Perceptron_Manager(0.01, 50)



# initialize confusion matrices for different learning rates
confusion_1 = np.zeros((10, 10))
confusion_2 = np.zeros((10, 10))
confusion_3 = np.zeros((10, 10))

# Run training/validation data through 50 epochs and get accuracy rates

test_per1.__iterate__(m_data, m_target, v_data, v_target, Accuracy_train1, Accuracy_va1)
print("Training Accuracy values per Epoch (0.00001): ")
print(Accuracy_train1)
print()
print("Validat Accuracy values per Epoch (0.00001): ")
print(Accuracy_va1)
print()

test_per2.__iterate__(m_data, m_target, v_data, v_target, Accuracy_train2, Accuracy_va2)
print("Training Accuracy values per Epoch (0.001): ")
print(Accuracy_train2)
print()
print("Validat Accuracy values per Epoch (0.001): ")
print(Accuracy_va2)
print()

test_per3.__iterate__(m_data, m_target, v_data, v_target, Accuracy_train3, Accuracy_va3)
print("Training Accuracy values per Epoch (0.1): ")
print(Accuracy_train3)
print()
print("Validat Accuracy values per Epoch (0.1): ")
print(Accuracy_va3)
print()


np.savetxt('acc_tr1.csv', [Accuracy_train1], delimiter=',')
np.savetxt('acc_tr2.csv', [Accuracy_train2], delimiter=',')
np.savetxt('acc_tr3.csv', [Accuracy_train3], delimiter=',')
np.savetxt('acc_va1.csv', [Accuracy_va1], delimiter=',')
np.savetxt('acc_va2.csv', [Accuracy_va2], delimiter=',')
np.savetxt('acc_va3.csv', [Accuracy_va3], delimiter=',')


# compute confusion matrices from validation data

test_per1._confusion_matrix_(v_data, v_target, confusion_1)
print("Confusion Matrix 0.00001")
print(confusion_1)
print()

test_per2._confusion_matrix_(v_data, v_target, confusion_2)
print("Confusion Matrix 0.001")
print(confusion_2)
print()
test_per3._confusion_matrix_(v_data, v_target, confusion_3)
print("Confusion Matrix 0.1")
print(confusion_3)
print()


#np.savetxt('confu1.csv', [confusion_1], delimiter=',')
pd.DataFrame(confusion_1).to_csv("confu1.csv", header=None, index=None)
pd.DataFrame(confusion_2).to_csv("confu2.csv", header=None, index=None)
pd.DataFrame(confusion_3).to_csv("confu3.csv", header=None, index=None)



'''
plt.figure(220)
plt.title("0.00001 learning rate") 
plt.scatter(epochs,Accuracy_train1, c='b', marker='x', label='1')
plt.scatter(epochs, Accuracy_va1, c='r', marker='s', label='-1')
plt.legend(loc='upper left')
plt.show()

plt.figure(230)
plt.title("0.001 learning rate") 
plt.scatter(epochs,Accuracy_train2, c='b', marker='x', label='1')
plt.scatter(epochs, Accuracy_va2, c='r', marker='s', label='-1')
plt.legend(loc='upper left')
plt.show()

plt.figure(240)
plt.title("0.1 learning rate") 
plt.scatter(epochs,Accuracy_train3, c='b', marker='x', label='1')
plt.scatter(epochs, Accuracy_va3, c='r', marker='s', label='-1')
plt.legend(loc='upper left')
plt.show()
'''

