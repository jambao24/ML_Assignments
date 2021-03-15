# Assign4_Kmeans_clustering
# James Bao (PSU ID: 934097519)


import sys
import numpy as np
import pandas as pd




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
# (as passed in using list format), assumes cluster is not empty
def compute_new_center(data_v):
    v_len = len(data_v)
    v_sum = np.zeros(n_class)
    for v in data_v:
        v_sum += v
    return np.divide(v_sum, v_len)


# calculate MSE for one cluster, assuming cluster is not empty
# data_v = samples in cluster passed in through list format
# cl_ctr = cluster_center
def calculate_mse(data_v, cl_ctr):
    v_len = len(data_v)
    v_sum = 0.0
    for v in data_v:
        # calculate distance squared between sample and cluster center
        v_sum += calculate_dist(v, cl_ctr)
    
    return np.divide(v_sum, v_len)


# calculate Mean Square Separation of clusters
# parameter clusters = array of cluster centers
def calculate_mss(clusters):
    denomin = K * (K-1) / 2
    numer = 0.0
    # for i = 0, j -> 1 to 9 [0 to 8]
    # for i = 1, j -> 2 to 9 [0 to 7]
    # etc.
    for i in range(K-1):
        for j in range(K-i-1):
            # apply calculate_dist function to pairs of cluster means
            numer += calculate_dist(clusters[i], clusters[i+j+1])
    return numer / denomin


def calculate_entropy_single(clusters):
    return 0.0


# preprocessing
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


di_keys = np.arange(K)
# _data = data in each cluster
# _classes = targets for data in each cluster,
# _idx = indices of data/targets in each cluster
dict_data = dict([(k, []) for k in di_keys])
dict_classes = dict([(k, []) for k in di_keys])
dict_idx = dict([(k, []) for k in di_keys])

# create 1D array for classifications
classif_tally = np.zeros(train_size)

cls_equal = False;
m = 0;


while (cls_equal == False):
#while (m == 0):
    print ("Iteration", m)

    # create temp dictionaries
    temp_data = dict([(k, []) for k in di_keys])
    temp_classes = dict([(k, []) for k in di_keys])
    temp_idx = dict([(k, []) for k in di_keys])

    for i, x_i in enumerate(train_data):
        distances = np.zeros(K)
        for j in range(K):
            # iterate through cluster_ctrs
            distances[j] = calculate_dist(x_i, cluster_ctrs[j])
        d_min_cl = np.argmin(distances)
        print(d_min_cl)

        # classification (cluster) = d_min_cl
        if (m == 0):
            dict_data[d_min_cl].append(x_i)
            dict_classes[d_min_cl].append(train_targ[i])
            dict_idx[d_min_cl].append(i)
        else: 
            temp_data[d_min_cl].append(x_i)
            temp_classes[d_min_cl].append(train_targ[i])
            temp_idx[d_min_cl].append(i)  

    #dic_sum1 = 0
    #dic_sum2 = 0
    #dic_sum3 = 0
    # len(list(filter(None, v)))
    '''
    for k, v in dict_idx.items():
        dic_sum1 += len(v)
        print(k, len(v))
    print()
    for k, v in dict_classes.items():
        dic_sum2 += len(v)
        print(k, len(v))
    print()
    
    for k, v in dict_data.items():
        dic_sum3 += len(v)
        print(k, len(v))
    print()
    print(dic_sum3, train_size)
    print()
    '''

    if (m > 0):
        dict_data = temp_data
        dict_classes = temp_classes
        dict_idx = temp_idx

    mse_vals = np.zeros((K))


    for l in range(K):
        # get all values for entry l in dict_data
        #data_vals = [ key[l] for key in dict_data ]
        data_vals = dict_data.get(l)
        data_list = list(data_vals)

        cluster_ctrs0 = np.zeros((K, n_class))
        # call compute_new_center function on classif_tally[:,l]
        if (len(data_list) == 0):
            cluster_ctrs0[l] = cluster_ctrs[l]
            mse_vals[l] = -1.0
        else: 
            cluster_ctrs0[l] = compute_new_center(data_list)
            mse_vals[l] = calculate_mse(data_list, cluster_ctrs0[l])


    print("Mean Squared Error: ", mse_vals)
    valid_mse = np.where(mse_vals >= 0.0, 1.0, 0.0)
    valid_mse_vals = np.where(mse_vals >= 0.0, mse_vals, 0.0)
    mse_num = np.sum(valid_mse_vals)
    mse_denom = np.sum(valid_mse)
    print ("Average Mean Squared Error: ", np.divide(mse_num, mse_denom))

    print("Mean Squared Separation: ", calculate_mss(cluster_ctrs0))
    print()

    cls_equal = np.array_equal(cluster_ctrs, cluster_ctrs0)
    m += 1;
    cluster_ctrs = cluster_ctrs0

print()
