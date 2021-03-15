# Assign4_Kmeans_clustering
# James Bao (PSU ID: 934097519)


import sys
import numpy as np
import pandas as pd


# preprocessing
'''
y = pd.read_csv("optdigits.names", header = None, delim_whitespace= True)
print(y)
print()
'''
# read in possible classes (classifications) from info
classes = np.array([0,1,2,3,4,5,6,7,8,9])
n_class = 64

train = pd.read_csv("optdigits.train", header = None, delimiter= ',').to_numpy()
test = pd.read_csv("optdigits.test", header = None, delimiter= ',').to_numpy()

np.random.shuffle(train)
np.random.shuffle(test)

train_size = train.shape[0]
test_size = test.shape[0]

train_targ = train[:,-1]
train_data = train[:, 0:-1]
test_targ = test[:,-1]
test_data = test[:, 0:-1]

# k-means cluster assignment
K = 10
    # K = 30
# pick K number of cluster centers randomly
k_clusters = np.random.randint(low = 0, high = train_size-1, size = K)
# initialize clusters 
cluster_ctrs = np.zeros((K, n_class))
for h in range(K):
    t_val = k_clusters[h]
    cluster_ctrs[h] = train_data[t_val]

# create array to keep track of which cluster each training sample gets classified to
    #cluster_nums_tr = np.zeros(train_size)
    #cluster_nums_test = np.zeros(test_size)

'''
pseudocode- iterate through all samples n times until the cluster coordinates stop changing

'''


# function to compute the squared distance between 1 sample and a cluster center
def calculate_dist(sample, center):
    try:
        sample.shape == center.shape
    except:
        print('Error: sample and center different sizes')
    else: 
        n_dims = sample.shape[0]
        dist_sq = 0
        for i in range(n_dims):
            dist_sq += (sample[i] - center[i])**2
        
        return dist_sq


# get array of occurances of minval
# we want the output array to contain the actual class values from training/test datasets
# these class values don't necessarily match the indices of the d_sqs (distances squared) array
# therefore, we need to make sure we're adding the actual class values from the class_n array
def get_minvals(d_sqs, minval):
    class_n_idx = np.argmin(d_sqs)
    retarr = np.array(classes[class_n_idx])
    for i in range(np.argmax(d_sqs), d_sqs.shape[0]):
        if d_sqs[i] == minval:
            np.append(retarr, classes[i])
    return retarr


# compute new cluster center based on euclidian mean of data samples in cluster
# (as passed in using list format)
def compute_new_center(data_v):
    v_len = len(data_v)
    v_sum = np.zeros(n_class)
    for v in data_v:
        v_sum += v
    return np.divide(v_sum, v_len)

print()


di_keys = np.arange(K)
    #dict_data = dict.fromkeys(di_keys)
# _data = data in each cluster
# _classes = targets for data in each cluster,
# _idx = indices of data/targets in each cluster
dict_data = dict([(k, []) for k in di_keys])
dict_classes = dict([(k, []) for k in di_keys])
dict_idx = dict([(k, []) for k in di_keys])


for i, x_i in enumerate(train_data):
    distances = np.zeros(K)
    for j in range(K):
        # iterate through cluster_ctrs
        distances[j] = calculate_dist(x_i, cluster_ctrs[j])

    d_min = np.amin(distances)
    d_min_cl = np.argmin(distances)
        #minvals = get_minvals(distances, d_min)
    
    #cluster_nums_tr[i] = d_min_cl
    dict_data[d_min_cl].append(x_i)
    dict_classes[d_min_cl].append(train_targ[i])
    dict_idx[d_min_cl].append(i)


# create copy of old clusters
cluster_ctrs0 = cluster_ctrs

    #cl_names, cl_counts = np.unique(cluster_nums_tr, return_counts=True)
    #classif_tally = np.zeros(train_size, K)
for l in range(K):
    # get all values for entry l in dict_data
        # https://www.geeksforgeeks.org/python-get-values-of-particular-key-in-list-of-dictionaries/
    data_vals = [ k[l] for k in dict_data ]
    data_list = list(data_vals)
    # call compute_new_center function on classif_tally[:,l]
    cluster_ctrs[l] = compute_new_center(data_list)

print(cluster_ctrs)
print()

