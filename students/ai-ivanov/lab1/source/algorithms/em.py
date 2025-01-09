import random
import numpy as np


def em_clustering(
    objects: np.ndarray,
    num_features: int,
    num_clusters: int,
    max_step: int = 30,
) -> np.ndarray:
    """
    EM clustering algorithm.

    Returns:
        np.ndarray: shape (N, )
    """
    num_objects = len(objects)

    # Initialize cluster centers randomly
    centers = np.array([random.choice(objects) for _ in range(num_clusters)])
    sigmas = np.empty((num_clusters, num_features))

    # Initialize covariance matrices
    for k in range(num_clusters):
        diff = objects - centers[k]
        sigmas[k] = np.diag(np.dot(diff.T, diff)) / num_objects  # sigma**2

    # Initialize cluster weights uniformly
    weights = 1 / num_clusters * np.ones((num_clusters, 1))

    y_prev = None
    for _ in range(max_step):
        # Calculate probabilities for each object belonging to each cluster
        probs = []
        for i in range(num_clusters):
            # (x-μ)ᵀΣ⁻¹(x-μ)
            ro_sq = np.sum((objects[:, :] - centers[i]) ** 2 / sigmas[i], axis=1)
            probs += [
                # (2π)^(-d/2)
                np.power((2 * np.pi), -num_features / 2)
                # Σ⁻¹
                / np.prod(np.sqrt(sigmas[i]))
                # exp(-ro_sq / 2)
                * np.exp(-ro_sq / 2)
            ]
        probs = np.array(probs, dtype=np.float32).T  # (N, K)

        # Expectation step: Calculate responsibilities
        # gᵢⱼ = wⱼ * p(xᵢ | μⱼ, Σⱼ)
        g = weights.T * probs  # (K, N)
        # gᵢⱼ /= Σ_j gᵢⱼ
        g /= g.sum(axis=1)[:, np.newaxis]  # (K, N)

        # Maximization step: Update parameters
        # wⱼ = 1/N * Σ_i gᵢⱼ
        weights = 1 / num_objects * np.sum(g, axis=0)  # (K, )

        # μⱼ = Σ_i gᵢⱼ * xᵢ / Σ_i gᵢⱼ
        centers = (
            g.T @ objects / weights[:, np.newaxis] / num_objects
        )  # (K, N_features)

        # Σⱼ = Σ_i gᵢⱼ * (xᵢ - μⱼ)ᵀ(xᵢ - μⱼ) / Σ_i gᵢⱼ
        sigmas = np.empty((num_clusters, num_features))
        for k in range(num_clusters):
            sigmas[k] = g[:, k] @ (objects - centers[k]) ** 2
        sigmas /= weights[:, np.newaxis]
        sigmas /= num_objects

        # Assign objects to clusters
        # Assigns each data point to the cluster for which it has the highest responsibility (probability)
        y = np.argmax(g, axis=1)  # (N, )

        # Check for convergence
        if y_prev is not None and np.array_equal(y, y_prev):
            return y

        y_prev = y
    return y
