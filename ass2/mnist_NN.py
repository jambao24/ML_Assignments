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

#m_tally = np.zeros(10)
m_tally2 = np.zeros(10)
m_tally4 = np.zeros(10)
#for idx, x_i in enumerate(m_array):
#    val = x_i[0]
#    m_tally[val] += 1
m_tally = np.array([5923,6742,5958,6131,5842,5421,5918,6265,5851,5949])


np.random.shuffle(m_array)
hf = 30000
m_array2 = m_array[0:hf]

np.random.shuffle(m_array)
qt = 15000
m_array4 = m_array[0:qt]

# verify that training dataset was divided into 1/2 and 1/4 correctly
unique, counts = np.unique(m_array[:,0], return_counts=True)
print(dict(zip(unique, counts)))
unique2, counts2 = np.unique(m_array2[:,0], return_counts=True)
print(dict(zip(unique2, counts2)))
unique4, counts4 = np.unique(m_array4[:,0], return_counts=True)
print(dict(zip(unique4, counts4)))


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


pt1_1 = NN_Manager(0.1, 50, 785, 20, 0)
pt1_1.__iterate__(m_data, m_target, v_data, v_target, Accuracy_train1_20, Accuracy_va1_20)
np.savetxt('acc_tr1_20.csv', [Accuracy_train1_20], delimiter=',')
np.savetxt('acc_va1_20.csv', [Accuracy_va1_20], delimiter=',')
pt1_1._confusion_matrix_(v_data, v_target, confusion_train1_20)
pd.DataFrame(confusion_train1_20).to_csv("confu1_20.csv", header=None, index=None)

pt1_2 = NN_Manager(0.1, 50, 785, 50, 0)
pt1_2.__iterate__(m_data, m_target, v_data, v_target, Accuracy_train1_50, Accuracy_va1_50)
np.savetxt('acc_tr1_50.csv', [Accuracy_train1_50], delimiter=',')
np.savetxt('acc_va1_50.csv', [Accuracy_va1_50], delimiter=',')
pt1_1._confusion_matrix_(v_data, v_target, confusion_train1_50)
pd.DataFrame(confusion_train1_50).to_csv("confu1_50.csv", header=None, index=None)

pt1_3 = NN_Manager(0.1, 50, 785, 100, 0)
pt1_3.__iterate__(m_data, m_target, v_data, v_target, Accuracy_train1_100, Accuracy_va1_100)
np.savetxt('acc_tr1_100.csv', [Accuracy_train1_100], delimiter=',')
np.savetxt('acc_va1_100.csv', [Accuracy_va1_100], delimiter=',')
pt1_1._confusion_matrix_(v_data, v_target, confusion_train1_100)
pd.DataFrame(confusion_train1_100).to_csv("confu1_100.csv", header=None, index=None)

pt2_1 = NN_Manager(0.1, 50, 785, 100, 0)
pt2_1.__iterate__(m_data2, m_target2, v_data, v_target, Accuracy_train2_half, Accuracy_va2_half)
np.savetxt('acc_tr2_50.csv', [Accuracy_train2_half], delimiter=',')
np.savetxt('acc_va2_50.csv', [Accuracy_va2_half], delimiter=',')
pt2_1._confusion_matrix_(v_data, v_target, confusion_train2_half)
pd.DataFrame(confusion_train2_half).to_csv("confu2_half.csv", header=None, index=None)

pt2_2 = NN_Manager(0.1, 50, 785, 100, 0)
pt2_2.__iterate__(m_data4, m_target4, v_data, v_target, Accuracy_train2_qrt, Accuracy_va2_qrt)
np.savetxt('acc_tr2_25.csv', [Accuracy_train2_qrt], delimiter=',')
np.savetxt('acc_va2_25.csv', [Accuracy_va2_qrt], delimiter=',')
pt2_2._confusion_matrix_(v_data, v_target, confusion_train2_qrt)
pd.DataFrame(confusion_train2_qrt).to_csv("confu2_qrt.csv", header=None, index=None)

pt3_1 = NN_Manager(0.1, 50, 785, 100, 0.25)
pt3_1.__iterate__(m_data, m_target, v_data, v_target, Accuracy_train3_25, Accuracy_va3_25)
np.savetxt('acc_tr3_25.csv', [Accuracy_train3_25], delimiter=',')
np.savetxt('acc_va3_25.csv', [Accuracy_va3_25], delimiter=',')
pt3_1._confusion_matrix_(v_data, v_target, confusion_train3_25)
pd.DataFrame(confusion_train3_25).to_csv("confu3_25.csv", header=None, index=None)

pt3_2 = NN_Manager(0.1, 50, 785, 100, 0.50)
pt3_2.__iterate__(m_data, m_target, v_data, v_target, Accuracy_train3_50, Accuracy_va3_50)
np.savetxt('acc_tr3_50.csv', [Accuracy_train3_50], delimiter=',')
np.savetxt('acc_va3_50.csv', [Accuracy_va3_50], delimiter=',')
pt3_2._confusion_matrix_(v_data, v_target, confusion_train3_50)
pd.DataFrame(confusion_train3_50).to_csv("confu3_50.csv", header=None, index=None)

pt3_3 = NN_Manager(0.1, 50, 785, 100, 0.95)
pt3_3.__iterate__(m_data, m_target, v_data, v_target, Accuracy_train3_95, Accuracy_va3_95)
np.savetxt('acc_tr3_95.csv', [Accuracy_train3_95], delimiter=',')
np.savetxt('acc_va3_95.csv', [Accuracy_va3_95], delimiter=',')
pt3_3._confusion_matrix_(v_data, v_target, confusion_train3_95)
pd.DataFrame(confusion_train3_95).to_csv("confu3_95.csv", header=None, index=None)



'''
tr1a = pd.read_csv('acc_tr1_20.csv', header = None)
va1a = pd.read_csv('acc_va1_20.csv', header = None)
tr1b = pd.read_csv('acc_tr1_50.csv', header = None)
va1b = pd.read_csv('acc_va1_50.csv', header = None)
tr1c = pd.read_csv('acc_tr1_100.csv', header = None)
va1c = pd.read_csv('acc_va1_100.csv', header = None)
tr2a = pd.read_csv('acc_tr2_50.csv', header = None)
va2a = pd.read_csv('acc_va1_50.csv', header = None)
tr2b = pd.read_csv('acc_tr2_25.csv', header = None)
va2b = pd.read_csv('acc_va2_25.csv', header = None)
'''


plt.figure(1)
plt.title("Part 1- 20 hidden neurons") 
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xlim((0,50))
plt.ylim((0,100))
plt.grid(b=True, which='both', axis='both')
plt.scatter(epochs, tr1a, c='b', marker='x', label='training')
plt.scatter(epochs, va1a, c='r', marker='s', label='validation')
#.scatter(epochs, Accuracy_train1_20, c='b', marker='x', label='training')
#plt.scatter(epochs, Accuracy_va1_20, c='r', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show()

plt.figure(2)
plt.title("Part 1- 50 hidden neurons") 
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xlim((0,50))
plt.ylim((0,100))
plt.grid(b=True, which='both', axis='both')
plt.scatter(epochs, tr1b, c='b', marker='x', label='training')
plt.scatter(epochs, va1b, c='r', marker='s', label='validation')
#plt.scatter(epochs, Accuracy_train1_50, c='b', marker='x', label='training')
#plt.scatter(epochs, Accuracy_va1_50, c='r', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show()

plt.figure(3)
plt.title("Part 1- 100 hidden neurons") 
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xlim((0,50))
plt.ylim((0,100))
plt.grid(b=True, which='both', axis='both')
plt.scatter(epochs, tr1c, c='b', marker='x', label='training')
plt.scatter(epochs, va1c, c='r', marker='s', label='validation')
#plt.scatter(epochs, Accuracy_train1_100, c='b', marker='x', label='training')
#plt.scatter(epochs, Accuracy_va1_100, c='r', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show()

plt.figure(4)
plt.title("Part 2- half set") 
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xlim((0,50))
plt.ylim((0,100))
plt.grid(b=True, which='both', axis='both')
#plt.scatter(epochs, Accuracy_train2_half, c='b', marker='x', label='training')
#plt.scatter(epochs, Accuracy_va2_half, c='r', marker='s', label='validation')
plt.scatter(epochs, tr2a, c='b', marker='x', label='training')
plt.scatter(epochs, va2a, c='r', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show()

plt.figure(5)
plt.title("Part 2- quarter set") 
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xlim((0,50))
plt.ylim((0,100))
plt.grid(b=True, which='both', axis='both')
#plt.scatter(epochs, Accuracy_train2_qrt, c='b', marker='x', label='training')
#plt.scatter(epochs, Accuracy_va2_qrt, c='r', marker='s', label='validation')
plt.scatter(epochs, tr2b, c='b', marker='x', label='training')
plt.scatter(epochs, va2b, c='r', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show()

plt.figure(6)
plt.title("Part 3- 0.25 momentum") 
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xlim((0,50))
plt.ylim((0,100))
plt.grid(b=True, which='both', axis='both')
plt.scatter(epochs, Accuracy_train3_25, c='b', marker='x', label='training')
plt.scatter(epochs, Accuracy_va3_25, c='r', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show()

plt.figure(7)
plt.title("Part 3- 0.50 momentum") 
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xlim((0,50))
plt.ylim((0,100))
plt.grid(b=True, which='both', axis='both')
plt.scatter(epochs, Accuracy_train3_50, c='b', marker='x', label='training')
plt.scatter(epochs, Accuracy_va3_50, c='r', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show()

plt.figure(8)
plt.title("Part 3- 0.95 momentum") 
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xlim((0,50))
plt.ylim((0,100))
plt.grid(b=True, which='both', axis='both')
plt.scatter(epochs, Accuracy_train3_95, c='b', marker='x', label='training')
plt.scatter(epochs, Accuracy_va3_95, c='r', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show()

