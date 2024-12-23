import numpy as np


def hierarchy(distances) -> tuple[dict, dict, int]:
    """
    Perform hierarchical clustering on a distance matrix.

    Args:
        distances (np.ndarray): A square matrix of pairwise distances between objects.

    Returns:
        tuple: A tuple containing:
            - pairwise_distances (dict): Distances between merged clusters.
            - clusters_power (dict): The size (power) of each cluster.
            - optimal_clusters (int): The optimal number of clusters.
    """
    num_objects = len(distances)

    # All clusters have only one object initially
    powers = list(np.ones(num_objects, dtype=np.int64))

    # Initialize cluster IDs
    cluster_ids = list(range(num_objects))

    # Store distances between merged clusters
    pairwise_distances = {}

    # Store cluster sizes
    clusters_power = {id: p for id, p in zip(cluster_ids, powers)}

    min_distances = []
    num_clusters_history = []

    while num_objects > 2:
        # Set diagonal to infinity to prevent self-clustering
        np.fill_diagonal(distances, np.inf)

        # Find the closest pair of clusters
        idx_ = np.argmin(distances)
        min_distance = np.min(distances)
        min_distances.append(min_distance)
        num_clusters_history.append(num_objects)
        idx_i, idx_j = np.unravel_index(idx_, distances.shape)

        if idx_i == idx_j:
            raise ValueError("idx_i == idx_j == {}".format(idx_i))

        # Get cluster powers
        i_p = powers[idx_i]
        j_p = powers[idx_j]

        # R(u, v): distance between clusters to be merged
        d = distances[idx_i, idx_j]
        pairwise_distances[(cluster_ids[idx_i], cluster_ids[idx_j])] = d

        # Remove merged clusters
        # Remember max_cluster_id to assign new cluster id in the end of the iteration
        max_cluster_id = max(cluster_ids)
        for idx in sorted([idx_i, idx_j], reverse=True):
            del powers[idx]
            del cluster_ids[idx]

        # Calculate coefficients for updating distances
        total_power = np.array(powers) + i_p + j_p  # (num_objects - 2)
        alpha_u = (np.array(powers) + i_p) / total_power  # (num_objects - 2)
        alpha_v = (np.array(powers) + j_p) / total_power  # (num_objects - 2)
        beta = -np.array(powers) / total_power  # (num_objects - 2)

        # Remove rows of merged clusters
        distances = np.delete(distances, [idx_i, idx_j], axis=0)

        # Calculate new distances using Ward's method formula:
        # d^2(u,v),k = α_u * d^2(i,k) + α_v * d^2(j,k) + β * d^2(i,j)
        # where:
        # - α_u, α_v are coefficients based on cluster sizes
        # - β is coefficient for original distance between merged clusters
        # - Square root is taken to get actual distance

        # new_cluster_distances is a vector of new distances to the newly created cluster
        new_cluster_distances = np.sqrt(
            alpha_u * np.square(distances[:, idx_i])
            + alpha_v * np.square(distances[:, idx_j])
            + beta * (d**2)
        )

        # Remove columns of merged clusters
        distances = np.delete(distances, [idx_i, idx_j], axis=1)

        # Create new distance matrix
        distances = np.pad(
            distances, ((0, 1), (0, 1)), mode="constant", constant_values=np.inf
        )
        distances[-1, :-1] = new_cluster_distances
        distances[:-1, -1] = new_cluster_distances

        # Update cluster information
        powers.append(i_p + j_p)
        cluster_ids.append(max_cluster_id + 1)
        clusters_power[max_cluster_id + 1] = i_p + j_p
        num_objects -= 1

    max_diff = 0
    for i in range(1, len(min_distances)):
        diff = min_distances[i] - min_distances[i-1]
        if diff > max_diff:
            max_diff = diff
            optimal_clusters = num_clusters_history[i-1]

    # Handle the last merge
    distances = np.min(distances).reshape((1, 1))
    pairwise_distances[(cluster_ids[0], cluster_ids[1])] = distances[0, 0]
    return pairwise_distances, clusters_power, optimal_clusters
