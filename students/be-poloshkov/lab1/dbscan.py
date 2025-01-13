from collections import deque

import numpy as np
import pandas as pd
from pandas.core.array_algos.replace import compare_or_regex_search


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
            if nb_idx != point_idx and distance <= self.eps
        ]

    def fit_predict(self, X: pd.DataFrame):
        n = len(X)

        distances = self.pairwise_distances(X)
        neighbors = [self.find_neighbours(i, distances) for i in range(n)]
        core_points = {i for i in range(n) if len(neighbors[i]) >= self.min_samples}

        if len(core_points) == 0:
            return range(n)

        labels = [-1] * n
        i = 0

        while len(core_points) > 0:
            core_point = core_points.pop()
            neighbor_queue = deque([core_point])
            core_points.add(core_point)
            while neighbor_queue:
                neighbor = neighbor_queue.popleft()
                labels[neighbor] = i
                if neighbor in core_points:
                    neighbor_queue.extend(neighbors[neighbor])
                    core_points.remove(neighbor)
            i += 1

        self.labels = labels
        return labels


    def _get_dist(self, x, y):
        if self.metric == 'euclidean':
            return np.linalg.norm(x - y)
        if self.metric == 'chebyshev':
            return np.max(np.abs(x - y))
        if self.metric == 'manhattan':
            return np.sum(np.abs(x - y))
        if self.metric == 'cosine':
            return 1 - np.sum(np.multiply(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y))