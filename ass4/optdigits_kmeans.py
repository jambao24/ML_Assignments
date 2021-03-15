# Assign4_Kmeans_clustering
# James Bao (PSU ID: 934097519)


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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


# calculates the entropy of a cluster
# takes in a list of target classes that the samples in the cluster belong to as a parameter
def calculate_entropy(targets_list):
    '''
    formula- entropy_i = sum(classes) -(prob_i,j * log_2(prob_i,j))
    where prob_i,j = number of occurances of class j in cluster i
    '''
    targets = np.array(targets_list)
    types, counts = np.unique(targets, return_counts=True)
    count_sum = np.sum(counts)

    probs = np.divide(counts, count_sum)
    log2probs = np.log2(probs)
    to_sum = np.multiply(probs, log2probs)
    return -1*np.sum(to_sum)


## preprocessing
classes = np.array([0,1,2,3,4,5,6,7,8,9])
n_class = 64

train = pd.read_csv("optdigits.train", header = None, delimiter= ',').to_numpy()
test = pd.read_csv("optdigits.test", header = None, delimiter= ',').to_numpy()

np.random.shuffle(train)

train_size = train.shape[0]
test_size = test.shape[0]

train_targ = train[:,-1]
train_data = train[:, 0:-1]
test_targ = test[:,-1]
test_data = test[:, 0:-1]

## Training Section

# k-means cluster assignment
    #K = 10
K = 30
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

# keep iterating through loop until cluster coordinates stop changing
while (cls_equal == False):
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
        #print(d_min_cl)

        # classification (cluster) = d_min_cl
        if (m == 0):
            dict_data[d_min_cl].append(x_i)
            dict_classes[d_min_cl].append(train_targ[i])
            dict_idx[d_min_cl].append(i)
        else: 
            temp_data[d_min_cl].append(x_i)
            temp_classes[d_min_cl].append(train_targ[i])
            temp_idx[d_min_cl].append(i)  

    if (m > 0):
        dict_data = temp_data
        dict_classes = temp_classes
        dict_idx = temp_idx

    mse_vals = np.zeros((K))
    entropy_vals = np.zeros((K))
    cluster_ctrs0 = np.zeros((K, n_class))
    for l in range(K):
        # get all values for entry l in dict_data
        #data_vals = [ key[l] for key in dict_data ]
        data_vals = dict_data.get(l)
        data_list = list(data_vals)

        target_vals = dict_classes.get(l)
        targ_list = list(target_vals)

        # call compute_new_center function on classif_tally[:,l]
        if (len(data_list) == 0):
            cluster_ctrs0[l] = cluster_ctrs[l]
            mse_vals[l] = -1.0
            entropy_vals[l] = -1.0
        else: 
            cluster_ctrs0[l] = compute_new_center(data_list)
            mse_vals[l] = calculate_mse(data_list, cluster_ctrs0[l])
            entropy_vals[l] = calculate_entropy(targ_list)


    print("Mean Squared Error: ", mse_vals)
    valid_mse = np.where(mse_vals >= 0.0, 1.0, 0.0)
    valid_mse_vals = np.where(mse_vals >= 0.0, mse_vals, 0.0)
    mse_num = np.sum(valid_mse_vals)
    mse_denom = np.sum(valid_mse)
    print ("Average Mean Squared Error: ", np.divide(mse_num, mse_denom))

    print("Mean Squared Separation: ", calculate_mss(cluster_ctrs0))

    print("Entropy per Cluster: ", entropy_vals)
    valid_entr = np.where(entropy_vals >= 0.0, 1.0, 0.0)
    valid_entr_vals = np.where(entropy_vals >= 0.0, entropy_vals, 0.0)
    entr_num = np.sum(valid_entr_vals)
    entr_denom = np.sum(valid_entr)
    print("Average Entropy: ", np.divide(entr_num, entr_denom))
    print()

    cls_equal = np.array_equal(cluster_ctrs, cluster_ctrs0)
    m += 1;
    cluster_ctrs = cluster_ctrs0

# save cluster coordinates to csv
    #np.savetxt('cluster_coords_30.csv', cluster_ctrs, fmt='%f', delimiter=',')
np.savetxt('cluster_coords_30.csv', cluster_ctrs, fmt='%f', delimiter=',')
print()


## Testing Section

from PIL import Image

#cl_ctrs1 = pd.read_csv("cluster_coords_10_1.csv", header = None, delimiter= ',').to_numpy() # clusters correspond to all 10 classes
#cl_ctrs = pd.read_csv("cluster_coords_10_2.csv", header = None, delimiter= ',').to_numpy()
#cl_ctrs = pd.read_csv("cluster_coords_10_3.csv", header = None, delimiter= ',').to_numpy()
#cl_ctrs = pd.read_csv("cluster_coords_10_4.csv", header = None, delimiter= ',').to_numpy()
#cl_ctrs2 = pd.read_csv("cluster_coords_10_5.csv", header = None, delimiter= ',').to_numpy() # lowest Avg MSE

#clst1 = pd.read_csv("cluster_coords_30_1.csv", header = None, delimiter= ',').to_numpy()
#clst2 = pd.read_csv("cluster_coords_30_2.csv", header = None, delimiter= ',').to_numpy()
#clst3 = pd.read_csv("cluster_coords_30_3.csv", header = None, delimiter= ',').to_numpy()
#clst4 = pd.read_csv("cluster_coords_30_4.csv", header = None, delimiter= ',').to_numpy()
#clst5 = pd.read_csv("cluster_coords_30_5.csv", header = None, delimiter= ',').to_numpy()

'''
for i in range(K):
    cl_vis = cluster_ctrs[i].reshape((8,8))
    cl_vis = cl_ctrs[i].reshape((8,8))

    cl_vis = cl_vis / 16 * 255
    #file_name = f"K30_cluster5_{i}.png"
    file_name = f"K30_cluster1_{i}.png"
    ith_img = Image.fromarray(np.uint8(cl_vis), 'L')
    print(f"saving image- {file_name}")
    ith_img.save(file_name)
    #print()
'''

'''
    # K=10 run 1 (clusters equivalent to digit classes)
    #convert1 = np.array([4,0,6,7,3,1,8,9,5,2])
    # K=10 run 5 (clusters have lowest Average MSE)
    #convert2 = np.array([0,2,7,8,0,3,4,6,9,6])
k30_1 = np.array([2,8,9,5,8,3,9,4,2,3,1,9,4,3,6,7,5,7,3,8,1,6,7,3,4,5,0,7,1,2])
k30_2 = np.array([3,2,0,4,2,8,7,1,8,0,5,5,3,6,5,4,2,4,9,7,6,7,6,4,9,6,0,9,9,1])
k30_3 = np.array([3,2,7,6,7,6,8,2,0,7,3,1,8,1,4,2,3,6,0,6,9,4,0,4,5,6,9,5,2,4]) # lowest Avg MSE
k30_4 = np.array([4,7,9,9,9,1,3,5,7,0,6,1,3,7,0,8,1,0,7,8,9,3,4,4,5,1,5,6,2,2])
k30_5 = np.array([5,6,5,8,1,2,6,7,4,1,9,9,3,4,6,2,0,9,2,7,5,9,0,7,8,3,4,1,2,4]) # 2nd lowest Avg MSE

# create confusion matrices for classifications
# actual = rows, expected = cols
    #confu_k10_ctr1 = np.zeros((10, 10), dtype = int)
    #confu_k10_ctr2 = np.zeros((10, 10), dtype = int)
confu_k30_1 = np.zeros((10, 10), dtype = int)
confu_k30_2 = np.zeros((10, 10), dtype = int)
confu_k30_3 = np.zeros((10, 10), dtype = int)
confu_k30_4 = np.zeros((10, 10), dtype = int)
confu_k30_5 = np.zeros((10, 10), dtype = int)

count = 0
tally1 = 0
tally2 = 0
tally3 = 0
tally4 = 0
tally5 = 0

# iterate through test data
for i, x_i in enumerate(test_data):
    #samp_vis = x_i.reshape((8,8)) / 16 * 255
    #im = Image.fromarray(np.uint8(samp_vis), 'L')
    #plt.imshow(im)

    distances1 = np.zeros(K)
    distances2 = np.zeros(K)
    distances3 = np.zeros(K)
    distances4 = np.zeros(K)
    distances5 = np.zeros(K)
    for j in range(K):
        # iterate through cluster_ctrs
        distances1[j] = calculate_dist(x_i, clst1[j])
        distances2[j] = calculate_dist(x_i, clst2[j])
        distances3[j] = calculate_dist(x_i, clst3[j])
        distances4[j] = calculate_dist(x_i, clst4[j])
        distances5[j] = calculate_dist(x_i, clst5[j])
    dmin1 = np.argmin(distances1)
    dmin2 = np.argmin(distances2)
    dmin3 = np.argmin(distances3)
    dmin4 = np.argmin(distances4)
    dmin5 = np.argmin(distances5)

    # classification (cluster) = d_min_cl
    expected = test_targ[i]
    actual1 = k30_1[dmin1]
    actual2 = k30_2[dmin2]
    actual3 = k30_3[dmin3]
    actual4 = k30_4[dmin4]
    actual5 = k30_5[dmin5]

    #print(expected, actual1, actual2)
    #confu_k10_ctr1[actual1, expected] += 1
    #confu_k10_ctr2[actual2, expected] += 1
    confu_k30_1[actual1, expected] += 1
    confu_k30_2[actual2, expected] += 1
    confu_k30_3[actual3, expected] += 1
    confu_k30_4[actual4, expected] += 1
    confu_k30_5[actual5, expected] += 1

    if (actual1 == expected): tally1 += 1
    if (actual2 == expected): tally2 += 1
    if (actual3 == expected): tally3 += 1
    if (actual4 == expected): tally4 += 1
    if (actual5 == expected): tally5 += 1
    count += 1
print()

print("Confusion Matrix for Run 1:")
print(confu_k30_1)
print()
accur1 = np.divide(tally1, count)*100
print("Accuracy for Run 1:", accur1)
print()
print("Confusion Matrix for Run 2:")
print(confu_k30_2)
print()
accur2 = np.divide(tally2, count)*100
print("Accuracy for Run 2:", accur2)
print()
print("Confusion Matrix for Run 3:")
print(confu_k30_3)
print()
accur3 = np.divide(tally3, count)*100
print("Accuracy for Run 3:", accur3)
print()
print("Confusion Matrix for Run 4:")
print(confu_k30_4)
print()
accur4 = np.divide(tally4, count)*100
print("Accuracy for Run 4:", accur4)
print()
print("Confusion Matrix for Run 5:")
print(confu_k30_5)
print()
accur5 = np.divide(tally5, count)*100
print("Accuracy for Run 5:", accur5)
print()

    #pd.DataFrame(confu_k10_ctr1).to_csv("confu_k10_1.csv", header=None, index=None)
    #pd.DataFrame(confu_k10_ctr2).to_csv("confu_k10_2.csv", header=None, index=None)
pd.DataFrame(confu_k30_1).to_csv("confu_k30_1.csv", header=None, index=None)
pd.DataFrame(confu_k30_2).to_csv("confu_k30_2.csv", header=None, index=None)
pd.DataFrame(confu_k30_3).to_csv("confu_k30_3.csv", header=None, index=None)
pd.DataFrame(confu_k30_4).to_csv("confu_k30_4.csv", header=None, index=None)
pd.DataFrame(confu_k30_5).to_csv("confu_k30_5.csv", header=None, index=None)
'''
