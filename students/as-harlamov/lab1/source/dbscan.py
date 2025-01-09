from collections import deque

import numpy as np
import pandas as pd

from clustering import Clustering


class DBSCANClustering(Clustering):
    def __init__(self, eps: int = 3, min_samples: int = 3):
        self.eps = eps
        self.min_samples = min_samples
        super().__init__()

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
        nbs = self.find_neighbours(i, distances)
        if len(nbs) < self.min_samples:
            clusters[i] = -1
            return False

        clusters[i] = cluster_id
        for j in nbs:
            clusters[j] = cluster_id

        nbs = deque(nbs)
        while nbs:
            j_nbs = self.find_neighbours(nbs.pop(), distances)
            if len(j_nbs) < self.min_samples:
                continue

            for k in j_nbs:
                if clusters[k] < 1:
                    if clusters[k] == 0:
                        nbs.append(k)
                    clusters[k] = cluster_id

        return True
