from collections import Counter

import numpy as np
import pandas as pd


class MyKNNClf:
    def __init__(self, k: int = 3, metric='euclidean'):
        self.k = k
        self.train_size = None
        self.metric = metric
        self.classes_num = 0

    def __repr__(self) -> str:
        return f'MyKNNClf class: k={self.k}'

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.train_size = X.shape
        self.classes_num = len(np.unique(y))

    def predict(self, test_X):
        return np.argmax(self.predict_proba(test_X), axis=1)

    def predict_proba(self, test_X):
        dists = self._get_distances(test_X)
        labels = []
        for i, obj in enumerate(dists):
            argsorted = np.argsort(obj)
            dists[i] = self._gaussian(dists[i] / dists[i][argsorted[self.k]])  # K(ro(u, x_u[i])/ ro(u, x_u[k+1]))
            dists[i] = dists[i][argsorted]
            labels.append(self.y[argsorted])

        probs = np.zeros((len(test_X), self.classes_num))

        for i, test_point_distances in enumerate(dists):
            for j, distance in enumerate(test_point_distances):
                probs[i][labels[i][j]] += distance
            probs[i] /= np.sum(probs[i])
        return probs

    def _get_dist(self, x, y):
        if self.metric == 'euclidean':
            return np.linalg.norm(x - y)
        if self.metric == 'chebyshev':
            return np.max(np.abs(x - y))
        if self.metric == 'manhattan':
            return np.sum(np.abs(x - y))
        if self.metric == 'cosine':
            return 1 - np.sum(np.multiply(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y))

    def _get_distances(self, test_X):
        dists = np.zeros((len(test_X), self.train_size[0]))
        for i_test, test_object in enumerate(test_X):
            for i_train, train_object in enumerate(self.X):
                dists[i_test][i_train] = self._get_dist(test_object, train_object)

        return dists

    @staticmethod
    def _gaussian(arr: np.array):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * arr ** 2)
