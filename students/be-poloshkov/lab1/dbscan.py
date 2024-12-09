from collections import deque

import numpy as np
import pandas as pd


class MyDBSCAN:
    def __init__(self, eps=3, min_samples=3, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def __repr__(self) -> str:
        return f'MyDBSCAN class: eps={self.eps}, min_samples={self.min_samples}'

    def pairwise_distances(self, X):
        n = len(X)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distances[i][j] = distances[j][i] = self._get_dist(
                    X.iloc[i].values,
                    X.iloc[j].values,
                )
        return distances

    def find_neighbours(self, point_idx: int, distances: np.ndarray):
        return [
            nb_idx
            for nb_idx, distance in enumerate(distances[point_idx])
            if nb_idx != point_idx and distance < self.eps
        ]

    def fit_predict(self, X: pd.DataFrame):
        n = len(X)
        clusters = [0] * n
        cluster_id = 1

        distances = self.pairwise_distances(X)

        for i in range(n):
            if clusters[i] != 0:
                continue

            if self.expand_cluster(distances, clusters, i, cluster_id):
                cluster_id += 1

        self.labels_ = clusters
        return self.labels_

    def expand_cluster(self, distances, clusters, i, cluster_id):
        neighbours = self.find_neighbours(i, distances)
        if len(neighbours) < self.min_samples:
            clusters[i] = -1
            return False

        clusters[i] = cluster_id
        for j in neighbours:
            clusters[j] = cluster_id

        neighbours = deque(neighbours)
        while neighbours:
            j_nbs = self.find_neighbours(neighbours.pop(), distances)
            if len(j_nbs) < self.min_samples:
                continue

            for k in j_nbs:
                if clusters[k] < 1:
                    if clusters[k] == 0:
                        neighbours.append(k)
                    clusters[k] = cluster_id

        return True

    def _get_dist(self, x, y):
        if self.metric == 'euclidean':
            return np.linalg.norm(x - y)
        if self.metric == 'chebyshev':
            return np.max(np.abs(x - y))
        if self.metric == 'manhattan':
            return np.sum(np.abs(x - y))
        if self.metric == 'cosine':
            return 1 - np.sum(np.multiply(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y))