import numpy as np
from scipy.spatial.distance import cdist
import random


NOISE_ROLE = -1
CORE_POINT_ROLE = 1
BORDER_POINT_ROLE = 0


def dbscan_clustering(
    objects: np.ndarray,
    epsilon: float,
    min_samples: int,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform DBSCAN clustering on the input objects.

    Args:
        objects (np.ndarray): Input data points.
        epsilon (float): Maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): Minimum number of samples in a neighborhood for a point to be considered as a core point.
        max_iter (int): Maximum number of iterations for the algorithm.

    Returns:
        tuple[np.ndarray, np.ndarray]: Roles and cluster assignments for each object.
    """
    # Calculate pairwise distances between objects
    distances = cdist(objects, objects)
    np.fill_diagonal(distances, np.inf)

    N = len(objects)

    clusters = np.empty((N,))
    clusters[:] = np.nan
    roles = clusters.copy()

    current_cluster_id = 0
    iter_counter = 0

    idxs_range = np.arange(N, dtype=np.uint32)

    while np.any(np.isnan(roles)):
        unassigned_idxs = np.argwhere(np.isnan(clusters)).reshape((-1,))
        random_idx = random.choice(unassigned_idxs)

        neighbors = idxs_range[distances[random_idx] < epsilon]
        if len(neighbors) < min_samples:
            roles[random_idx] = NOISE_ROLE
            K = [random_idx]
        else:
            roles[random_idx] = CORE_POINT_ROLE
            clusters[random_idx] = current_cluster_id

            K = np.append(neighbors, random_idx)
            while np.any(np.isnan(clusters[K])):
                idx2check = K[np.isnan(clusters[K])]
                idx = random.choice(idx2check)
                k_n = idxs_range[distances[idx] < epsilon]
                if len(k_n) >= min_samples:
                    roles[idx] = CORE_POINT_ROLE
                    K = np.union1d(K, k_n)
                else:
                    roles[idx] = BORDER_POINT_ROLE
                clusters[idx] = current_cluster_id
            current_cluster_id += 1

        distances[K] = np.inf

        iter_counter += 1
        if iter_counter == max_iter:
            break

    clusters[np.isnan(clusters)] = -1
    return roles, clusters
