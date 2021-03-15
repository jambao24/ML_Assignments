import numpy as numpy

'''
sample NaiveBayes classifier to understand how to code concept

https://www.youtube.com/watch?v=BqUmKsfSWho
'''

class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # init mean, var, priors
        self._mean = np.zeros((n_classes, n_features), dtype = np.float64)
        self._var = np.zeros((n_classes, n_features), dtype = np.float64)
        self._priors = np.zeros((n_classes), dtype = np.float64)

        for c in self._classes:
            X_c = X[c==y]
            self._mean[c,:] = X_c.mean(axis=0)
            self._var[c,:] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0]


    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred


    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.log(self._pdf(idx, x))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        
        return self._classes[np.argmax(posteriors)]


    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var **2))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator // denominator 


'''
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from nb import NaiveBayes

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
 
X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print("Naive Bayes classif. accuracy", accuracy(y_test, predictions))
'''


'''
<img alt="Photo by Carolyn Qi on April 22, 2020. Image may contain: 1 person, tree, outdoor and nature" class="FFVAD" decoding="auto" sizes="478px" 
srcset="https://instagram.fhio2-2.fna.fbcdn.net/v/t51.2885-15/sh0.08/e35/p640x640/93881704_226955678576268_3234681138790581617_n.jpg?_nc_ht=instagram.fhio2-2.fna.fbcdn.net&amp;_nc_cat=100&amp;_nc_ohc=OxUphLZvtI0AX_G6Xq1&amp;oh=6eaa9e3c87c3b9d9960d3a2879767286&amp;oe=5F464633 640w,
https://instagram.fhio2-2.fna.fbcdn.net/v/t51.2885-15/sh0.08/e35/p750x750/93881704_226955678576268_3234681138790581617_n.jpg?_nc_ht=instagram.fhio2-2.fna.fbcdn.net&amp;_nc_cat=100&amp;_nc_ohc=OxUphLZvtI0AX_G6Xq1&amp;oh=f50a40cdcef51e41c9dd654af1226bab&amp;oe=5F4856B3 750w,
https://instagram.fhio2-2.fna.fbcdn.net/v/t51.2885-15/e35/p1080x1080/93881704_226955678576268_3234681138790581617_n.jpg?_nc_ht=instagram.fhio2-2.fna.fbcdn.net&amp;_nc_cat=100&amp;_nc_ohc=OxUphLZvtI0AX_G6Xq1&amp;oh=756101e92839d862bc74646ae8cbaaaf&amp;oe=5F464AD8 1080w" 
src="https://instagram.fhio2-2.fna.fbcdn.net/v/t51.2885-15/e35/p1080x1080/93881704_226955678576268_3234681138790581617_n.jpg?_nc_ht=instagram.fhio2-2.fna.fbcdn.net&amp;_nc_cat=100&amp;_nc_ohc=OxUphLZvtI0AX_G6Xq1&amp;oh=756101e92839d862bc74646ae8cbaaaf&amp;oe=5F464AD8" style="object-fit: cover;">
'''
