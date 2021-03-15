# ML_Assignments

CS455/545 Summer 2020- Machine Learning assignments


ASSIGNMENT 1- Perceptrons.
You will train 10 perceptrons that will, as a group, learn to classify the handwritten digits in the MNIST dataset (Training: https://drive.google.com/file/d/1d5R0VJz42RNdEkp-Iv_0bpyeap8RhlFU/view?usp=sharing, Validation: https://drive.google.com/file/d/1vX5PlDc9ASivqbBsXCTM1e8RaHOJXA0b/view?usp=sharing). See the class slides for details of the perceptron architecture and perceptron learning algorithm. Each perceptron will have 785 inputs and one output. Each perceptron’s target is one of the 10 digits, 0−9.


ASSIGNMENT 2- Neural Network.
For this homework you will implement a two-layer neural network (i.e, one hidden-layer)to perform the handwritten digit recognition task of Assignment 1. Please write your own neural network code; don’t use code written by others, though you can refer to other code  if you need help understanding the algorithm. You may use whatever programming  language you prefer. The dataset for this task is the MNIST dataset that you used in Assignment 1. 


ASSIGNMENT 3- Naive Bayes Classifiers.
In this task you will implement naive Bayes classifiers based on Gaussians. You must implement a Matlab function or a Python executable file called naive_bayes that learns a naive Bayes classifier for a classification problem, given some training data and some additional options. In particular, your function can be invoked as follows:
  naive_bayes(<training_file>, <test_file>)
If you use Python, just convert the Matlab function arguments to command-line arguments. The arguments provide to the function the following information:
•	The first argument is the path name of the training file, where the training data is stored. The path name can specify any file stored on the local computer.
•	The second argument is the path name of the test file, where the test data is stored. The path name can specify any file stored on the local computer.
Training and test files are taken from the UCI datasets directory. A description of the datasets and the file format can be found here: http://www.cs.pdx.edu/~doliotis/MachineLearningSummer2020/assignments/uci_datasets/dataset_description.html


ASSIGNMENT 4- K-means clustering algorithm.
In this homework you will experiment on using the K-means clustering algorithm to cluster and classify the OptDigits data, originally from the UCI ML repository. 
  Dataset: Download the data from our class website: http://web.cecs.pdx.edu/~doliotis/MachineLearningWinter2020/assignments/assignment04/optdigits.zip
