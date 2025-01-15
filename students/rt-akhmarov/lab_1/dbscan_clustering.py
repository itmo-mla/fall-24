import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import pairwise_distances

class DBSCAN:
    def __init__(self, epsilon: float, min_samples: int):
        self.eps = epsilon
        self.min_samples = min_samples

    def fit(self, X: np.ndarray):
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1, dtype=int)
        cluster_id = 0

        for idx in range(n_samples):
            if labels[idx] != -1: 
                continue

            distances = np.linalg.norm(X[idx] - X, axis=1)
            neighbors = np.where(distances <= self.eps)[0]

            if len(neighbors) < self.min_samples: 
                labels[idx] = -1
                continue

            cluster_id += 1
            labels[idx] = cluster_id
            queue = list(neighbors)

            while queue:
                neighbor_idx = queue.pop(0)

                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_id

                if labels[neighbor_idx] != 0:
                    continue

                labels[neighbor_idx] = cluster_id

                distances_neighbor = np.linalg.norm(X[neighbor_idx] - X, axis=1)
                neighbors_neighbor = np.where(distances_neighbor <= self.eps)[0]

                if len(neighbors_neighbor) >= self.min_samples:
                    queue.extend(neighbors_neighbor)

        clusters = [X[labels == cid] for cid in range(1, cluster_id + 1)]
        noise = X[labels == -1]
        return clusters, noise, labels
    

def visualize_clusters(data, clusters, noise, feature_names):
    num_features = len(feature_names)
    fig, axes = plt.subplots(num_features, num_features, figsize=(15, 15))

    for i in range(num_features):
        for j in range(num_features):
            if i == j:
                axes[i, j].set_visible(False)
                continue

            for cluster_idx, cluster in enumerate(clusters):
                cluster_points = pd.DataFrame(cluster, columns=feature_names)
                axes[i, j].scatter(cluster_points.iloc[:, j], cluster_points.iloc[:, i], label=f"Cluster {cluster_idx + 1}")

            noise_points = pd.DataFrame(noise, columns=feature_names)
            axes[i, j].scatter(noise_points.iloc[:, j], noise_points.iloc[:, i], color="black", label="Noise", alpha=0.6)

            axes[i, j].set_xlabel(feature_names[j])
            axes[i, j].set_ylabel(feature_names[i])

    plt.tight_layout()
    plt.legend()
    plt.show()


def compute_distances(X, labels):
    unique_clusters = np.unique(labels[labels != -1])
    intra_cluster_distances = []
    inter_cluster_distances = []

    for cluster in unique_clusters:
        cluster_points = X[labels == cluster]
        other_points = X[labels != cluster]

        if len(cluster_points) > 1:
            intra_dist = pairwise_distances(cluster_points).mean()
            intra_cluster_distances.append(intra_dist)

        if len(other_points) > 0:
            inter_dist = pairwise_distances(cluster_points, other_points).mean()
            inter_cluster_distances.append(inter_dist)

    return np.mean(intra_cluster_distances), np.mean(inter_cluster_distances)


