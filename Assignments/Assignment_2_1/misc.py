import numpy
import numpy as np

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


def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance


def shortest_distance(x, x_rest, y_rest):
    shortest = euclidean_distance(x, x_rest[0])
    predicted = y_rest[0]

    for i in range(len(x_rest)):
        if euclidean_distance(x, x_rest[i]) <= shortest:
            shortest = euclidean_distance(x, x_rest[i])
            predicted = y_rest[i]
    return predicted, shortest


TP = 0
TN = 0
FP = 0
FN = 0

for i in range(len(X_val)):
    x = X_val[i]
    y = Y_val[i]
    pred, shortest = shortest_distance(x, X_train, Y_train)
    print("Y:", pred, "Y hat", y, "Distance:", shortest)

    if y == 1 and pred == 1:
        TP += 1

    if y == 0 and pred == 0:
        TN += 1

    if y == 1 and pred == 0:
        FN += 1

    if y == 0 and pred == 1:
        FP += 1

print("Accuracy:", (TP+TN) / (TP + TN + FP + FN))
print("Recall:", TP / (TP + FN))
print("Precision", TP / (TP + FP))
print("F1", (2*TP) / (2*TP + FP + FN))


from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X_train, Y_train)

print(neigh.predict([X_val[0]]))




class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]

    def _predict(self, x):
        # compute the distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]

        # get the closest K

        # majority vote

# Calculate the distance to all points

#

# Small K: Low bias, High variance, Overfitting
# High K:
