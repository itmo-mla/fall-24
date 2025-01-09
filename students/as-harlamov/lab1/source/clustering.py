from collections import defaultdict

import numpy as np
import pandas as pd


class Clustering:
    def __init__(self):
        self.labels_: list | None = None

    def __str__(self):
        args = ', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())
        return f'{self.__class__.__name__}({args})'

    @classmethod
    def euclidean_distance(cls, x1: np.array, x2: np.array):
        return np.linalg.norm(x1 - x2)

    @classmethod
    def pairwise_distances(cls, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distances[i][j] = distances[j][i] = cls.euclidean_distance(
                    X.iloc[i].values,
                    X.iloc[j].values,
                )
        return distances

    @classmethod
    def centroid(cls, points: np.ndarray) -> np.array:
        return np.array([np.mean(feature) for feature in zip(*points)])

    @classmethod
    def mean_distance(cls, points: list[np.array]) -> np.floating:
        return np.mean([
            cls.euclidean_distance(points[i], points[j])
            for i in range(len(points))
            for j in range(i + 1, len(points))
        ] or [0])

    @classmethod
    def get_cluster_distances(cls, X: pd.DataFrame, labels: list[int]) -> tuple[np.floating, np.array]:
        n = len(X)
        cluster_points = defaultdict(list)
        for i in range(n):
            cluster_points[labels[i]].append(i)

        cluster_centroids = [
            Clustering.centroid(X.iloc[list(point_indices)].values)
            for point_indices in cluster_points.values()
        ]

        mean_extra_cluster_distance = cls.mean_distance(cluster_centroids)
        mean_intra_cluster_distance = np.array([
            cls.mean_distance(X.iloc[list(point_indices)].values)
            for point_indices in cluster_points.values()
        ])

        return mean_extra_cluster_distance, mean_intra_cluster_distance
