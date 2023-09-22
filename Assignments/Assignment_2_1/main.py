import numpy
import matplotlib.pyplot as plt
import numpy as np

from knn import KNN

# Fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

numpy.random.shuffle(dataset)

split_ratio = 0.8

X_train = dataset[:int(len(dataset) * split_ratio), 0:8]
X_val = dataset[int(len(dataset) * split_ratio):, 0:8]
Y_train = dataset[:int(len(dataset) * split_ratio), 8]
Y_val = dataset[int(len(dataset) * split_ratio):, 8]


classifier = KNN(k=8)

classifier.fit(X_train, Y_train)

result = classifier.predict(X_val)
print(result)

acc = np.sum(result == Y_val) / len(Y_val)
print(acc)

