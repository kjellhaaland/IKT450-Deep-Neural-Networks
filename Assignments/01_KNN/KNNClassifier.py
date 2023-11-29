import numpy

from collections import Counter


class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.Y_train = None

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train

    def predict(self, x_predict, distance_type="euclidean", p=3):
        predictions = [self._single_prediction(x, distance_type, p) for x in x_predict]
        return predictions

    def _single_prediction(self, x, distance_type, p):
        # Compute the distances

        if distance_type == "euclidean":
            distances = [self._eucledian_distance(x, x_train) for x_train in self.X_train]
        elif distance_type == "manhattan":
            distances = [self._manhattan_distance(x, x_train) for x_train in self.X_train]
        else:
            distances = [self._minkowski_distance(x, x_train, p) for x_train in self.X_train]

        # Get the closest K
        indices = numpy.argsort(distances)[: self.k]
        labels = [self.Y_train[index] for index in indices]

        # Majority vote
        counter = Counter(labels)
        return counter.most_common()[0][0]

    # Euclidean Distance
    def _eucledian_distance(self, arr1, arr2):
        distance = numpy.sqrt(numpy.sum((numpy.power(arr1 - arr2, 2))))
        return distance

    def _manhattan_distance(self, arr1, arr2):
        distance = numpy.sum(numpy.abs(arr1 - arr2))
        return distance

    def _minkowski_distance(self, arr1, arr2, p):
        distance = numpy.power(numpy.sum(numpy.power(numpy.abs(arr1 - arr2), p)), 1 / p)
        return distance
