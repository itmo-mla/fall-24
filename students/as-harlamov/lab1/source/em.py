import numpy as np
from scipy.stats import multivariate_normal

from clustering import Clustering


class EMClustering(Clustering):
    def __init__(
        self,
        n_clusters: int = 2,
        max_iter: int = 100,
        tol: float = 1e-3,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.means = None
        self.covariances = None
        self.weights = None
        super().__init__()

    def fit_predict(self, X):
        X = X.values
        n_samples, n_features = X.shape

        self.means = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        self.covariances = [np.cov(X.T) for _ in range(self.n_clusters)]
        self.weights = np.ones(self.n_clusters) / self.n_clusters
        responsibilities = np.zeros((n_samples, self.n_clusters))

        for iteration in range(self.max_iter):
            for k in range(self.n_clusters):
                responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                    X,
                    mean=self.means[k],
                    cov=self.covariances[k],
                    allow_singular=True,
                )

            responsibilities /= responsibilities.sum(axis=1, keepdims=True)

            N_k = responsibilities.sum(axis=0)

            new_means = np.dot(responsibilities.T, X) / N_k[:, np.newaxis]
            new_covariances = []

            for k in range(self.n_clusters):
                if N_k[k] == 0:
                    new_covariances.append(np.eye(n_features))
                else:
                    diff = X - new_means[k]
                    new_covariances.append(np.dot(responsibilities[:, k] * diff.T, diff) / N_k[k])

            new_weights = N_k / n_samples

            if (np.abs(new_weights - self.weights) < self.tol).all():
                break

            self.means = new_means
            self.covariances = new_covariances
            self.weights = new_weights

        self.labels_ = np.argmax(responsibilities, axis=1)
        return self.labels_
