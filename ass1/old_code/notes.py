# python tutorial- how to read/write csv file 


'''
import csv

with open('names.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    with open('new_names.csv', 'w') as new_file:



Chunking practice:

chunk_size = 10000
batch_no = 1

for chunk in pd.read_csv('b_v', chunksize = chunk_size):
    chunk.to_csv('b_v' + str(batch_no) + '.csv', index=False)
    batch_no += 1






https://code.visualstudio.com/docs/python/environments

ok wow I'm a dumbass. the reason i wasn't getting pandas and scikit-learn to import was because i didn't have the right path to the site-packages 

'''


'''
# numpy- [rows, columns]
mydata = pd.read_csv("preprocess.csv", header=None)
m_array = np.array(mydata)
mydata_target = m_array[:,0]
mydata_data = np.divide(m_array[:,1:], 255.0)


print("target: ")
print(mydata_target)
print()
print("data: ") 
print(mydata_data.shape)
print()
print("data values: ")
print(mydata_data[1,111:145])
print()
'''


#ok fuck using chunks, it's too hard to append the separated data and other people have said the full dataset can be read in one go if you use pandas
'''
chunksize = 10000
target = np.empty((1, 1))
data = np.empty((1, 784))
for chunk in pd.read_csv('mnist_train.csv', header = None, chunksize = chunksize):
    m_array = np.array(chunk)
    m_target = m_array[:,0]
    m_data = np.divide(m_array[:,1:], 255.0)
    np.concatenate((target, m_target), axis = 0)
    np.concatenate((data, m_data), axis = 0) 

'''


# three different learning rates: η = 0.00001, 0.001, and 0.1. 



'''
    # X = data, y = "key" for data, val = target number for the perceptron 
    def fit(self, X, y):
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # init weights are random values from -0.5 to +0.5
        self.weights=  np.random.rand(n_features)-0.5
        # init bias is set to 0
        self.bias = 0

        # set target array y_ based on whether each value in the "key" matches the target number
        y_ = np.array([1 if i == self.target else 0 for i in y])

        
        # iteration
        # activation_func only used for 1 sample
        ##j = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr + (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
                ##j += 1
                ##print('j = ' + str(j))
                ##print() 


IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed

^ got this error when debugging iteration

'''



