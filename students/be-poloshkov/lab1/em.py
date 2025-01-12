import random

import numpy as np
from scipy.stats import multivariate_normal


class MyEM:
    def __init__(self, n_clusters: int = 2, max_iter: int = 1000, weight_delta_threshold: float = 1e-3):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.weight_threshold = weight_delta_threshold
        self.means = None
        self.si = None
        self.weights = None

    def fit_predict(self, data):
        data = data.to_numpy()

        num_clusters = self.n_clusters
        num_features = data.shape[1]
        num_objects = len(data)

        centers = np.array([random.choice(data) for _ in range(num_clusters)])
        sigmas = np.empty((num_clusters, num_features))

        for k in range(num_clusters):
            diff = data - centers[k]
            sigmas[k] = np.diag(np.dot(diff.T, diff)) / num_objects

        weights = 1 / num_clusters * np.ones((num_clusters, 1))

        y_prev = None
        for _ in range(self.max_iter):
            probs = []
            for i in range(num_clusters):
                ro_sq = np.sum((data[:, :] - centers[i]) ** 2 / sigmas[i], axis=1)
                probs += [
                    np.power((2 * np.pi), -num_features / 2)
                    / np.prod(np.sqrt(sigmas[i]))
                    * np.exp(-ro_sq / 2)
                ]
            probs = np.array(probs, dtype=np.float32).T

            g = weights.T * probs
            g /= g.sum(axis=1)[:, np.newaxis]

            weights = 1 / num_objects * np.sum(g, axis=0)

            centers = (
                    g.T @ data / weights[:, np.newaxis] / num_objects
            )

            sigmas = np.empty((num_clusters, num_features))
            for k in range(num_clusters):
                sigmas[k] = g[:, k] @ (data - centers[k]) ** 2
            sigmas /= weights[:, np.newaxis]
            sigmas /= num_objects

            return np.argmax(g, axis=1)
