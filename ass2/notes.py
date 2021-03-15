import pandas as pd
import numpy as np

raw = pd.read_csv('mnist_train.csv', header = None)
m_array = np.array(raw)
np.random.shuffle(m_array) # shuffle the dataset before processing
m_target = m_array[:,0]
m_data = np.divide(m_array[:,1:], 255.0)

# for part 2, create new test datasets that are 1/2 and 1/4 the size of the original
'''
tally for each 0-9 digit from the Assignment 1 confusion matrices. 
ct = pd.read_csv('confusion_matrix_tally.csv', header = None)
'''

#m_tally = np.zeros(10)
m_tally2 = np.zeros(10)
m_tally4 = np.zeros(10)
#for idx, x_i in enumerate(m_array):
#    val = x_i[0]
#    m_tally[val] += 1
m_tally = np.array([5923,6742,5958,6131,5842,5421,5918,6265,5851,5949])

print("m_tally: ")
print(m_tally)
print()

m_thresh2 = m_tally
m_thresh4 = m_tally
np.true_divide(m_thresh2,2)
np.true_divide(m_thresh4,4)

m_tally2 = np.zeros(10, dtype=int)
m_tally4 = np.zeros(10, dtype=int)

m_array2 = np.zeros((1,785), dtype=int)
m_array4 = np.zeros((1,785), dtype=int)

# keep adding training data samples to m_array4 until the 1/4 or 1/2 threshold for the digit is reached
# idea is to keep the ratio of digits the same in the 1/2 and 1/4 datasets as in the original
for idx, x_i in enumerate(m_array):
    # get first element in row (corresponds to 'answer key' for data sample)
    temp = x_i[0]
    # add row to m_array2 and m_array4 if # of samples threshold for digit hasn't been reached yet
    if ((m_tally4[temp] < m_thresh4[temp]) & (m_tally2[temp] < m_thresh2[temp])):
        np.concatenate(m_array2, x_i)
        m_tally2[temp] += 1
        np.concatenate(m_array4, x_i)
        m_tally4[temp] += 1
    # add row to m_array2 only, if # of samples threshold for digit has been reached for m_array4
    elif (m_tally2[temp] < m_thresh2[temp]):
        np.concatenate(m_array2, x_i)
        m_tally2[temp] += 1
    # stop when m_array2 has been filled and the # of samples has been reached for all digits
    # this occurs when the m_tally2 and m_thresh2 arrays are equivalent
    elif (np.array_equal(m_tally2, m_thresh2)):
        break
    
# process m_array2 and m_array4 the same way as the original training dataset
np.random.shuffle(m_array2) # shuffle the dataset before processing
m_target2 = m_array2[:,0]
m_data2 = np.divide(m_array2[:,1:], 255.0)

np.random.shuffle(m_array) # shuffle the dataset before processing
m_target4 = m_array4[:,0]
m_data4 = np.divide(m_array4[:,1:], 255.0)


print("m_tally2: ")
print(m_tally2)
print()
print("m_tally4: ")
print(m_tally4)
print()


--------------------------------------------------------------------------------------

'''
7-16 old code:

print("m_tally: ")
print(m_tally)
print()


m_thresh2 = m_tally
m_thresh4 = m_tally
np.true_divide(m_thresh2,2)
np.true_divide(m_thresh4,4)

m_tally2 = np.zeros(10, dtype=int)
m_tally4 = np.zeros(10, dtype=int)


#m_array2 = np.zeros((0,785), dtype=int)
#m_array4 = np.zeros((0,785), dtype=int)

# keep adding training data samples to m_array4 until the 1/4 or 1/2 threshold for the digit is reached
# idea is to keep the ratio of digits the same in the 1/2 and 1/4 datasets as in the original
for idx, x_i in enumerate(m_array):
    # get first element in row (corresponds to 'answer key' for data sample)
    temp = x_i[0]
    # add row to m_array2 and m_array4 if # of samples threshold for digit hasn't been reached yet
    if ((m_tally4[temp] < m_thresh4[temp]) & (m_tally2[temp] < m_thresh2[temp])):
        np.concatenate((m_array2, x_i[None,:]),axis=1)
        m_tally2[temp] += 1
        np.concatenate((m_array4, x_i[None,:]),axis=1)
        m_tally4[temp] += 1
    # add row to m_array2 only, if # of samples threshold for digit has been reached for m_array4
    elif (m_tally2[temp] < m_thresh2[temp]):
        np.concatenate((m_array2, x_i[None,:]),axis=1)
        m_tally2[temp] += 1
    # stop when m_array2 has been filled and the # of samples has been reached for all digits
    # this occurs when the m_tally2 and m_thresh2 arrays are equivalent
    elif (np.array_equal(m_tally2, m_thresh2)):
        break

'''


'''
2020/07/14- TIL when plotting multiple graphs in Python, you have to close the first popup window before the next one can load
'''


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


'''
#random scrap from testing smaller training subsets


acc1 = np.empty(51)
acc2 = np.empty(51)
cmat = np.zeros((10, 10))
# learning rate = 0.1, n_iters = 50, n_inputs = 785, num_hidden = 100, momentum = 0
test_samp = NN_Manager(0.1, 50, 785, 100, 0)
test_samp.__iterate__(t_data, t_target, v_data, v_target, acc1, acc2)

plt.figure(88)
#plt.title("test (n = 420)") 
plt.title("test (n = 4200)") 
plt.grid(b=True, which='both', axis='both')
plt.scatter(epochs, acc1, c='b', marker='x', label='training')
plt.scatter(epochs, acc2, c='r', marker='s', label='validation')
plt.legend(loc='upper left')
plt.show()

test_samp._confusion_matrix_(v_data, v_target, cmat)
print("Confusion Matrix")
print(cmat)
print()
'''

