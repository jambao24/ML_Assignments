# Assign3_naive_bayes_classifier
# James Bao (PSU ID: 934097519)


'''
Per Bayes theorem... 
class_MAP = argmax(class <- {+1, -1}) P(x_1,...,x_n | class)*P(class)/P(x_1,...,x_n)

In practice, we only need to calculate
argmax(class <- {+1, -1}) P(class) Π_i P(x_i | class)
because P(x_1,...,x_n) is constant w/ respect to i
'''


import numpy as np
import pandas as pd
import math


# process data files
'''
Datasets do not necessarily all have the same # of columns. 
Right-most column corresponds to the class label. 
All of the other columns correspond to an attribute in the data. 
Yeast example has 10 classes and 8 attributes.
'''

# preprocessing

y_train = pd.read_csv('yeast_training.txt', header = None, delim_whitespace= True).to_numpy().astype(float)
    #y_train = pd.read_csv('pendigits_training.txt', header = None, delim_whitespace= True).to_numpy().astype(float)
    #y_train = pd.read_csv('satellite_training.txt', header = None, delim_whitespace= True).to_numpy().astype(float)
label_idx = y_train.shape[1]-1
# sort by values in last column (class values) before splitting
ytr = y_train[y_train[:,-1].argsort()] 
train_label = ytr[:,-1]
train_feat = ytr[:,0:-1]

y_test = pd.read_csv('yeast_test.txt', header = None, delim_whitespace= True).to_numpy().astype(float)
    #y_test = pd.read_csv('pendigits_test.txt', header = None, delim_whitespace= True).to_numpy().astype(float)
    #y_test = pd.read_csv('satellite_test.txt', header = None, delim_whitespace= True).to_numpy().astype(float)
# sort by values in last column (class values) before splitting
np.random.shuffle(y_test)
yts = y_test[y_test[:,-1].argsort()] 
test_label = yts[:,-1]
test_feat = yts[:,0:-1]


# Retroactive verification of mean + stdev calculations on yeast data for Class 10 (contains 4 data samples)
#print(train_feat[-4:,:])
#print()
#print(train_label[-4:])
#print()

# get number of occurences of each class in the training data
unique, counts = np.unique(ytr[:,-1], return_counts=True)
print(dict(zip(unique, counts)))


# Obtain means and stdevs for Training Data
'''
Training Data Steps- 
1) separate out subset of data samples for each class [from 1 to 10]
2) create 2 matrices of dimensions 10 * (label_idx) -> label_idx = number of attributes [from 1 to 8]
3) compute the mean and stdev for each column across the subset of samples
3a) for the mean, just sum all the values and divide by the number of samples in the subset
3b) for the stdev, calculate stdev for each column in the subset
'''

# rows = classes, cols = attributes
# e.g. train_mean[4,5] = mean for Class 5, Attribute 6
train_means = np.zeros((counts.shape[0], label_idx), dtype=float)
train_stdevs = np.zeros((counts.shape[0], label_idx), dtype=float)

# index of sorted list of training samples
tally = 0
# array of actual class values found in training samples
class_n = unique.astype(int)

# iterate through sorted list of training samples
for i in range(counts.shape[0]):
    n_samples = counts[i]
    temp_idx = tally + n_samples
    # For i class, we want to look at samples tally to tally + n_samples - 1 in train data
    # train_means[i,j] = mean of jth attribute of ith class
    for j in range(label_idx):
        train_means[i,j] = np.mean(ytr[tally:temp_idx,j])
        train_stdevs[i,j] = np.std(ytr[tally:temp_idx,j])
        # adjust all 0 stdev values to 0.01
        if (train_stdevs[i,j] == 0.0):
            train_stdevs[i,j] = 0.01
        # print mean and stdev for each attribute for each class
        # make sure to access ith index of class_n (array of actual class values)
        print("Class ", class_n[i], ", attribute ", j+1, ", mean = ", "%.2f" % train_means[i,j], ", std = ", "%.2f" % train_stdevs[i,j])
    tally += n_samples
    #print(tally)
    print()

print()


'''
Testing Data steps-
1) For each sample in the testing data, calculate N(x) for each classifier and each attribute
2) N(x) = 1/(std*sqrt(2*pi))*exp(-(x-mean)^2/(2*std^2))
3) P(x_i | class) = N(x_i; mean_i,c; stdev_i,c)
4) class_NB(x) = argmax[class c- {+1, -1}] (ln(P(class)) + sum_i(ln_P(x_i | class)))

We want class_NB(x) to compare P(x_i | class) for all the classes, using the data for each attribute

Classes = rows of train_means/train_stdevs, Attributes = cols of train_means/train_stdevs


pseudocode

for <iterate through samples in test data>:
    * calculate 2D matrix test_sample_pdx for indiv sample using data from train_means and train_stdevs
            ** this step will use the ith row of test_feat
    * using test_sample_pdx, compute data from 4. for each class [each row] of test_sample_pdx
    * note the output from class_NB(x), the value that is calculated, and how many matching values there were
    * compare with the actual value (which is the ith element of test_label)
    * compute accuracyff
    print("ID = ", "%5d" % i+1, ", predicted = ", class_NB(x), ", probability = ", "%.2f" % train_means[i,j], ", std = ", "%.2f" % train_stdevs[i,j])
'''

# 1D array of classification probabilities
temp_cl = np.zeros(counts.shape[0], dtype=float)
# 2D array of N(x) values
temp_pdx = np.zeros((counts.shape[0], label_idx), dtype=float)
# probability that sample matches any 1 of the classes (assumed to be constant)
p_default = 1/counts.shape[0]
# 1D array of classification accuracies ofr all test samples
accurs = np.zeros(yts.shape[0])


# return value for probability density function on a single value input
# (x_v is a single scalar, mean_v and stdev_v are also scalars)
def get_pdx_(x_v, mean_v, stdev_v):
    val = (x_v-mean_v)^2/(2*stdev_v^2)
    return (1/(stdev_v*math.sqrt(2*math.pi)))*np.exp(val)

# return 2D matrix for probability density function on a single value input
# (x_arr is the array corresponding to the attribute values for a single test sample)
# mean and stdev are the 2D arrays generated during the training phase
def get_pdx(x_arr, mean, stdev):
    #temp = x_v*np.zeros((counts.shape[0], label_idx), dtype=float)
    temp = np.zeros((counts.shape[0], label_idx), dtype=float)
    for i in range(counts.shape[0]):
        temp[i,:] = x_arr

    vals = 0.5*np.square(temp - mean)/np.square(stdev)
    vals_temp = np.exp(-vals)/(stdev*np.sqrt(2*np.pi))
    return np.where(vals_temp <= 0.0000000000001, 0.0000000000001, vals_temp)


# calculate probability values for each class using the log value
# t_cl = 1D array to store class probabilities
# t_pdx = 2D pdx matrix
# p_cl = const prob for each class
def get_probs(t_cl, t_pdx, p_cl):
    for i in range(class_n.shape[0]):
        t_cl[i] = _get_probs_helper(t_pdx[i,:], np.log(p_cl))
        #t_cl[i] = _get_probs_helper_(t_pdx[i,:], p_cl)
    return t_cl

# helper function for above function
# calculate the log of each element in the input array (parameter)
# input array is 1 row of the 2D pdx matrix
# "p_const" is actually the log of p_cl from get_probs
def _get_probs_helper(p_attrs, p_const):
    logvals = np.log(p_attrs)
    retval = p_const
    for i in range(logvals.shape[0]):
        retval += logvals[i]
    return retval


# version of _get_probs_helper that doesn't use logs
# calculate the log of each element in the input array (parameter)
# input array is 1 row of the 2D pdx matrix
def _get_probs_helper_(p_attrs, p_const):
    retval = p_const
    for i in range(p_attrs.shape[0]):
        retval *= p_attrs[i]
    return retval

# get array of occurances of maxval
# we want the output array to contain the actual class values from training/test datasets
# these class values don't necessarily match the indices of the t_cl (class probabilities) array
# therefore, we need to make sure we're adding the actual class values from the class_n array
def get_maxvals(t_cl, maxval):
    class_n_idx = np.argmax(t_cl)
    retarr = np.array(class_n[class_n_idx])
    for i in range(class_n_idx, t_cl.shape[0]):
        if t_cl[i] == maxval:
            np.append(retarr, class_n[i])
    return retarr



# loop through all test samples
for i, x_i in enumerate(yts):
    #test_label[i] = actual classification
    #test_feat[i,:] = test sample attributes

    # compute N(x) matrix for data sample i
    temp_pdx = get_pdx(test_feat[i,:], train_means, train_stdevs)
    # compute probability for each class, using N(x) matrix Gaussian probabilities
    t_cl = get_probs(temp_cl, temp_pdx, p_default)

    # compute accuracy- get index/indices of max val in t_cl, number of occurances, and indices of occurances
    tmax = np.amax(t_cl)
    maxvals = get_maxvals(t_cl, tmax)

    # get index of prediction
    arr = np.where(maxvals == test_label[i], 1, 0)
    # get average value of predictions
    accurs[i] = np.average(arr)
    # if average = 1.0 or 0.0, that means there's only 1 class w/ max probability
    if (np.average(arr) == 1.0 or np.average(arr) == 0.0):
        prediction = maxvals
    else:
        # if there are multiple tied classifications, pick the first value
        prediction = np.argmax(maxvals == test_label[i])

    # ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n
    print("ID = ", i+1, ", predicted = ", prediction, ", probability = ", "%.4f" % tmax, ", true = ", "%d" % test_label[i], ", accuracy = ", "%4.2f" % accurs[i])


#array = np.array([0.5,0.2,0.8,0.8,0.3])
#test = get_probs_helper(array)
#print(test)

print()
print("classification accuracy =", "%6.4f" % np.average(accurs))
print()

