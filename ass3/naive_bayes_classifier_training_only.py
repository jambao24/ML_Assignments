# Assign3_naive_bayes_classifier
# James Bao (PSU ID: 934097519)

'''
class NB_dummy: 
    retval = 0;
    
    def calc(self, val1, val2, sum):
        return (val1 * val2 // sum);
'''

'''
s = pd.read_csv('sort.txt', header = None, delim_whitespace= True).to_numpy().astype(float)
#np.random.shuffle(y_test) # shuffle the dataset before processing
#s[s[:,-1].argsort()] # sort by values in last column (class values) before splitting
s_sorted = s[s[:,-1].argsort()] #https://thispointer.com/sorting-2d-numpy-array-by-column-or-row-in-python/
#np.take_along_axis(s, label_idx, axis=0)

s_label = s_sorted[:,-1]
s_feat = s_sorted[:,0:-1]

print(s_feat)
print()
print(s_label)
print()
'''

'''
Per Bayes theorem... 
class_MAP = argmax(class <- {+1, -1}) P(x_1,...,x_n | class)*P(class)/P(x_1,...,x_n)

In practice, we only need to calculate
argmax(class <- {+1, -1}) P(class) Π_i P(x_i | class)
because P(x_1,...,x_n) is constant w/ respect to i
'''



import numpy as np
import pandas as pd


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
#np.random.shuffle(y_train) # shuffle the dataset before processing
# sort by values in last column (class values) before splitting
ytr = y_train[y_train[:,-1].argsort()] 
train_label = ytr[:,-1]
train_feat = ytr[:,0:-1]

y_test = pd.read_csv('yeast_test.txt', header = None, delim_whitespace= True).to_numpy().astype(float)
    #y_test = pd.read_csv('pendigits_test.txt', header = None, delim_whitespace= True).to_numpy().astype(float)
    #y_test = pd.read_csv('satellite_test.txt', header = None, delim_whitespace= True).to_numpy().astype(float)
#np.random.shuffle(y_test) # shuffle the dataset before processing
# sort by values in last column (class values) before splitting
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
#print(dict(zip(unique, counts)))
#print(counts)


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

tally = 0
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
        print("Class ", i+1, ", attribute ", j+1, ", mean = ", "%.2f" % train_means[i,j], ", std = ", "%.2f" % train_stdevs[i,j])
    tally += n_samples
    #print(tally)
    print()

print()


# incomplete code for the test data portion, in class format
'''
class NaiveBayes:

    def _init_(self, classif, pdx_vals, p_const, mu, std):
        self.temp_cl = classif
        self.temp_pdx = pdx_vals
        self.p_default = p_const
        self.mean = mu
        self.stdev = std


    # return value for probability density function on a single value input
    # (x_v is a single scalar, mean_v and stdev_v are also scalars)
    def _get_pdx_(self, x_v, mean_v, stdev_v):
        val = (x_v-mean_v)^2/(2*stdev_v^2)
        return (1/(stdev_v*math.sqrt(2*math.pi)))*np.exp(val)

    # return 2D matrix for probability density function on a single value input
    # (x_arr is the array corresponding to the attribute values for a single test sample)
    # mean and stdev are the 2D arrays generated during the training phase
    def _get_pdx(self, x_arr):
        #temp = x_v*np.zeros((counts.shape[0], label_idx), dtype=float)
        temp = np.zeros((counts.shape[0], label_idx), dtype=float)
        for i in range(counts.shape[0]):
            temp[i,:] = x_arr
        vals = np.square(temp - self.mean)/(2 * np.square(self.stdev))
        return (1/(stdev*math.sqrt(2*math.pi)))*np.exp(vals)

    # calculate probability values for each class using the log value
    # t_cl = 1D array to store class probabilities
    # t_pdx = 2D pdx matrix
    # p_cl = const prob for each class
    def _get_probs(self):
        # 
        return 0

    # helper function for above function
    # calculate the log of each element in the input array (parameter)
    # input array is 1 row of the 2D pdx matrix
    def _get_probs_helper(self, p_attrs):
        logvals = np.log(p_attrs)
        retval = logvals[0]
        for i in range(logvals.shape[0]-1):
            retval *= logvals[i+1]
        return retval
'''

'''
np.square(x_v - mean_v)/(2 * np.square(stdev_v))
'''
