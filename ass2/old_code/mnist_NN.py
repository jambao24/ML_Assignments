# Assign_2__mnist_NN.py
# James Bao (PSU ID: 934097519)


### Libraries
# Third-party libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from neuralnet_manager import NN_Manager
from neuralnet_manager_ import NN_Manager

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

#test = pd.read_csv('test0.csv', header = None) # smaller test file of 420 samples
test = pd.read_csv('test00.csv', header = None) # smaller test file of 4200 samples
t_array = np.array(test)
np.random.shuffle(t_array) # shuffle the dataset before processing
t_target = t_array[:,0]
t_data = np.divide(t_array[:,1:], 255.0)


# for part 2, create new test datasets that are 1/2 and 1/4 the size of the original
'''
#m_tally = np.zeros(10)
m_tally2 = np.zeros(10)
m_tally4 = np.zeros(10)
#for idx, x_i in enumerate(m_array):
#    val = x_i[0]
#    m_tally[val] += 1
m_tally = np.array([5923,6742,5958,6131,5842,5421,5918,6265,5851,5949])


np.random.shuffle(m_array)
m_array2 = m_array[0:30000]
np.random.shuffle(m_array2) # shuffle the dataset before processing

np.random.shuffle(m_array)
m_array4 = m_array[0:15000]
np.random.shuffle(m_array4) # shuffle the dataset before processing

# verify that training dataset was divided into 1/2 and 1/4 correctly
unique, counts = np.unique(m_array[:,0], return_counts=True)
print(dict(zip(unique, counts)))
unique2, counts2 = np.unique(m_array2[:,0], return_counts=True)
print(dict(zip(unique2, counts2)))
unique4, counts4 = np.unique(m_array4[:,0], return_counts=True)
print(dict(zip(unique4, counts4)))
'''

np.random.shuffle(t_array)
hf = (t_array.shape[0]//2)
m_array2 = m_array[0:hf]

np.random.shuffle(t_array)
qt = t_array.shape[0]//4
m_array4 = m_array[0:qt]

# process m_array2 and m_array4 the same way as the original training dataset
np.random.shuffle(m_array2) # shuffle the dataset before processing
m_target2 = m_array2[:,0]
m_data2 = np.divide(m_array2[:,1:], 255.0)

np.random.shuffle(m_array4) # shuffle the dataset before processing
m_target4 = m_array4[:,0]
m_data4 = np.divide(m_array4[:,1:], 255.0)




# Initialize data collection arrays
print()
# Part 1- vary # of hidden units
Accuracy_train1_20 = np.empty(51)
Accuracy_train1_50 = np.empty(51)
Accuracy_train1_100 = np.empty(51)
Accuracy_va1_20 = np.empty(51)
Accuracy_va1_50 = np.empty(51)
Accuracy_va1_100 = np.empty(51)
# Part 2- vary size of training dataset
Accuracy_train2_half = np.empty(51)
Accuracy_train2_qrt = np.empty(51)
Accuracy_va2_half = np.empty(51)
Accuracy_va2_qrt = np.empty(51)
# Part 3- vary momentum values from nonzero
Accuracy_train3_25 = np.empty(51)
Accuracy_train3_50 = np.empty(51)
Accuracy_train3_95 = np.empty(51)
Accuracy_va3_25 = np.empty(51)
Accuracy_va3_50 = np.empty(51)
Accuracy_va3_95 = np.empty(51)


# initialize confusion matrices for different learning rates
epochs = np.arange(51)
confusion_train1_20 = np.zeros((10, 10))
confusion_train1_50 = np.zeros((10, 10))
confusion_train1_100 = np.zeros((10, 10))
confusion_train2_half = np.zeros((10, 10))
confusion_train2_qrt = np.zeros((10, 10))
confusion_train3_25 = np.zeros((10, 10))
confusion_train3_50 = np.zeros((10, 10))
confusion_train3_95 = np.zeros((10, 10))


# Run training/validation data through 50 epochs and get accuracy rates

'''
acc1 = np.empty(51)
acc2 = np.empty(51)
cmat = np.zeros((10, 10))
# learning rate = 0.1, n_iters = 50, n_inputs = 785, num_hidden = 100, momentum = 0
test_samp = NN_Manager(0.1, 50, 785, 100, 0)
test_samp.__iterate__(t_data, t_target, v_data, v_target, acc1, acc2)

plt.figure(88)
#plt.title("test (n = 420)") 
plt.title("test (n = 4200)") 
plt.grid(b=True, which='minor', axis='both')
plt.scatter(epochs, acc1, c='b', marker='x', label='training')
plt.scatter(epochs, acc2, c='r', marker='s', label='validation')
plt.legend(loc='upper left')
plt.show()

test_samp._confusion_matrix_(v_data, v_target, cmat)
print("Confusion Matrix")
print(cmat)
print()
'''


pt1_1 = NN_Manager(0.1, 50, 785, 20, 0)
pt1_1.__iterate__(t_data, t_target, v_data, v_target, Accuracy_train1_20, Accuracy_va1_20)
pt1_2 = NN_Manager(0.1, 50, 785, 50, 0)
pt1_2.__iterate__(t_data, t_target, v_data, v_target, Accuracy_train1_50, Accuracy_va1_50)
pt1_3 = NN_Manager(0.1, 50, 785, 100, 0)
pt1_3.__iterate__(t_data, t_target, v_data, v_target, Accuracy_train1_100, Accuracy_va1_100)

pt2_1 = NN_Manager(0.1, 50, 785, 100, 0)
pt2_1.__iterate__(m_data2, t_target2, v_data, v_target, Accuracy_train2_half, Accuracy_va2_half)
pt2_2 = NN_Manager(0.1, 50, 785, 100, 0)
pt2_2.__iterate__(m_data4, m_target4, v_data, v_target, Accuracy_train2_qrt, Accuracy_va2_qrt)

pt3_1 = NN_Manager(0.1, 50, 785, 100, 0.25)
pt3_1.__iterate__(t_data, t_target, v_data, v_target, Accuracy_train3_25, Accuracy_va3_25)
pt3_2 = NN_Manager(0.1, 50, 785, 100, 0.50)
pt3_2.__iterate__(t_data, t_target, v_data, v_target, Accuracy_train3_50, Accuracy_va3_50)
pt3_3 = NN_Manager(0.1, 50, 785, 100, 0.95)
pt3_3.__iterate__(t_data, t_target, v_data, v_target, Accuracy_train3_95, Accuracy_va3_95)

# save accuracy values to CSV
np.savetxt('acc_tr1_20.csv', [Accuracy_train1_20], delimiter=',')
np.savetxt('acc_tr1_50.csv', [Accuracy_train1_50], delimiter=',')
np.savetxt('acc_tr1_100.csv', [Accuracy_train1_100], delimiter=',')

np.savetxt('acc_tr2_50.csv', [Accuracy_train2_half], delimiter=',')
np.savetxt('acc_tr2_25.csv', [Accuracy_train2_qrt], delimiter=',')

np.savetxt('acc_tr3_25.csv', [Accuracy_train3_25], delimiter=',')
np.savetxt('acc_tr3_50.csv', [Accuracy_train3_50], delimiter=',')
np.savetxt('acc_tr3_95.csv', [Accuracy_train3_95], delimiter=',')

np.savetxt('acc_va1_20.csv', [Accuracy_va1_20], delimiter=',')
np.savetxt('acc_va1_50.csv', [Accuracy_va1_50], delimiter=',')
np.savetxt('acc_va1_100.csv', [Accuracy_va1_100], delimiter=',')

np.savetxt('acc_va2_50.csv', [Accuracy_va2_half], delimiter=',')
np.savetxt('acc_va2_25.csv', [Accuracy_va2_qrt], delimiter=',')

np.savetxt('acc_va3_25.csv', [Accuracy_va3_25], delimiter=',')
np.savetxt('acc_va3_50.csv', [Accuracy_va3_50], delimiter=',')
np.savetxt('acc_va3_95.csv', [Accuracy_va3_95], delimiter=',')

'''
pt1_1._confusion_matrix_(v_data, v_target, confusion_train1_20)
print("Confusion Matrix Pt1.1")
print(confusion_train1_20)
print()
pt1_2._confusion_matrix_(v_data, v_target, confusion_train1_50)
print("Confusion Matrix Pt1.2")
print(confusion_train1_50)
print()
pt1_3._confusion_matrix_(v_data, v_target, confusion_train1_100)
print("Confusion Matrix Pt1.3")
print(confusion_train1_100)
print()

pt2_1._confusion_matrix_(v_data, v_target, confusion_train2_half)
print("Confusion Matrix Pt2.1")
print(confusion_train2_half)
print()
pt2_2._confusion_matrix_(v_data, v_target, confusion_train2_qrt)
print("Confusion Matrix Pt2.2")
print(confusion_train2_qrt)
print()

pt3_1._confusion_matrix_(v_data, v_target, confusion_train3_25)
print("Confusion Matrix Pt3.1")
print(confusion_train3_25)
print()
pt3_2._confusion_matrix_(v_data, v_target, confusion_train3_50)
print("Confusion Matrix Pt3.2")
print(confusion_train3_50)
print()
pt3_3._confusion_matrix_(v_data, v_target, confusion_train3_95)
print("Confusion Matrix Pt3.3")
print(confusion_train3_95)
print()
'''

# compute confusion matrices from validation data

pd.DataFrame(confusion_train1_20).to_csv("confu1_20.csv", header=None, index=None)
pd.DataFrame(confusion_train1_50).to_csv("confu1_50.csv", header=None, index=None)
pd.DataFrame(confusion_train1_100).to_csv("confu1_100.csv", header=None, index=None)
pd.DataFrame(confusion_train2_half).to_csv("confu2_50.csv", header=None, index=None)
pd.DataFrame(confusion_train2_qrt).to_csv("confu2_25.csv", header=None, index=None)
pd.DataFrame(confusion_train3_25).to_csv("confu3_25.csv", header=None, index=None)
pd.DataFrame(confusion_train3_50).to_csv("confu3_50.csv", header=None, index=None)
pd.DataFrame(confusion_train3_95).to_csv("confu3_95.csv", header=None, index=None)


'''
2020/07/14- TIL when plotting multiple graphs in Python, you have to close the first popup window before the next one can load
'''

plt.figure(1)
plt.title("Part 1- 20 hidden neurons") 
plt.grid(b=True, which='both', axis='both')
plt.scatter(epochs, Accuracy_train1_20, c='b', marker='x', label='training')
plt.scatter(epochs, Accuracy_va1_20, c='r', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show()

plt.figure(2)
plt.title("Part 1- 50 hidden neurons") 
plt.grid(b=True, which='both', axis='both')
plt.scatter(epochs, Accuracy_train1_50, c='b', marker='x', label='training')
plt.scatter(epochs, Accuracy_va1_50, c='r', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show()

plt.figure(3)
plt.title("Part 1- 100 hidden neurons") 
plt.grid(b=True, which='both', axis='both')
plt.scatter(epochs, Accuracy_train1_100, c='b', marker='x', label='training')
plt.scatter(epochs, Accuracy_va1_100, c='r', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show()

plt.figure(4)
plt.title("Part 2- half set") 
plt.grid(b=True, which='both', axis='both')
plt.scatter(epochs, Accuracy_train2_half, c='b', marker='x', label='training')
plt.scatter(epochs, Accuracy_va2_half, c='r', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show()

plt.figure(5)
plt.title("Part 2- quarter set") 
plt.grid(b=True, which='both', axis='both')
plt.scatter(epochs, Accuracy_train2_qrt, c='b', marker='x', label='training')
plt.scatter(epochs, Accuracy_va2_qrt, c='r', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show()

plt.figure(6)
plt.title("Part 3- 0.25 momentum") 
plt.grid(b=True, which='both', axis='both')
plt.scatter(epochs, Accuracy_train3_25, c='b', marker='x', label='training')
plt.scatter(epochs, Accuracy_va3_25, c='r', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show()

plt.figure(7)
plt.title("Part 3- 0.50 momentum") 
plt.grid(b=True, which='both', axis='both')
plt.scatter(epochs, Accuracy_train3_50, c='b', marker='x', label='training')
plt.scatter(epochs, Accuracy_va3_50, c='r', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show()

plt.figure(8)
plt.title("Part 3- 0.95 momentum") 
plt.grid(b=True, which='both', axis='both')
plt.scatter(epochs, Accuracy_train3_95, c='b', marker='x', label='training')
plt.scatter(epochs, Accuracy_va3_95, c='r', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show()

