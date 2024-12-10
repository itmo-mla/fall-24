import numpy as np
from typing import Tuple

np.random.seed(0)


def init_cluster_centers_by_kmeans(samples: np.ndarray, cluster_count: int, ) -> np.ndarray:
    init_indices = np.random.choice(samples.shape[0], size=cluster_count, replace=False)
    centers = samples[init_indices]

    cluster_assignments = np.zeros(len(samples), dtype=int)
    for i, sample in enumerate(samples):
        distances = np.linalg.norm(sample - centers, axis=1) ** 2
        cluster_assignments[i] = np.argmin(distances)

    while True:
        new_centers = np.zeros_like(centers, dtype=np.float64)
        counts = np.zeros(cluster_count, dtype=np.float64)

        for i, cluster_idx in enumerate(cluster_assignments):
            new_centers[cluster_idx] += samples[i]
            counts[cluster_idx] += 1

        for j in range(cluster_count):
            if counts[j] > 0:
                new_centers[j] /= counts[j]

        if np.all(np.linalg.norm(centers - new_centers, axis=1) < 0.01):
            return new_centers

        for i, sample in enumerate(samples):
            distances = np.linalg.norm(sample - new_centers, axis=1) ** 2
            cluster_assignments[i] = np.argmin(distances)

        centers = new_centers


def p(
        X: np.ndarray,
        mean: np.ndarray,
        variance: np.ndarray
) -> np.ndarray:
    coeff = 1 / np.sqrt((2 * np.pi) ** X.shape[1] * np.prod(variance))
    exponent = -0.5 * np.sum(((X - mean) ** 2) / variance, axis=1)
    return coeff * np.exp(exponent)


def em_gmm(
        X: np.ndarray,
        cluster_count: int,
        max_iteration: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples, n_features = X.shape

    centers = init_cluster_centers_by_kmeans(X, cluster_count)
    sigma = np.full((cluster_count, n_features), np.var(X, axis=0))
    w = np.full(cluster_count, 1 / cluster_count)

    centers_old = np.zeros(centers.shape)
    g = np.zeros((n_samples, cluster_count))

    for iteration in range(max_iteration):
        # E-шаг:

        g = np.array([w[k] * p(X, centers[k], sigma[k]) for k in range(cluster_count)]).T
        g = g / np.sum(g, axis=1)[:, np.newaxis]

        # M-шаг
        w = np.sum(g, axis=0)

        centers = (g.T @ X) / w[:, np.newaxis]

        for k in range(cluster_count):
            sigma[k] = np.sum(g[:, k][:, np.newaxis] * (X - centers[k]) ** 2, axis=0) / w[k]

        w = w / n_samples

        if np.all(np.abs(np.linalg.norm(centers - centers_old, axis=1)) < 1e-6):
            break
        centers_old = centers
    return np.argmax(g, axis=1)
