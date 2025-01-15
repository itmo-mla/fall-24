import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

def perform_dbscan(dataset, epsilon, min_points):
    data_array = dataset.values
    total_points = data_array.shape[0]
    cluster_labels = np.zeros(total_points, dtype=int)
    current_cluster = 0

    for idx in range(total_points):
        if cluster_labels[idx] != 0:
            continue

        point_distances = np.linalg.norm(data_array[idx] - data_array, axis=1)
        neighboring_points = np.where(point_distances <= epsilon)[0]

        if len(neighboring_points) < min_points:
            cluster_labels[idx] = -1
        else:
            current_cluster += 1
            cluster_labels[idx] = current_cluster

            neighbor_idx = 0
            while neighbor_idx < len(neighboring_points):
                current_point = neighboring_points[neighbor_idx]

                if cluster_labels[current_point] == -1:
                    cluster_labels[current_point] = current_cluster

                if cluster_labels[current_point] == 0:
                    cluster_labels[current_point] = current_cluster

                    new_distances = np.linalg.norm(data_array[current_point] - data_array, axis=1)
                    new_neighbors = np.where(new_distances <= epsilon)[0]

                    if len(new_neighbors) >= min_points:
                        neighboring_points = np.append(neighboring_points, new_neighbors)

                neighbor_idx += 1

    clusters = [[] for _ in range(current_cluster)]
    noise_points = []

    for idx in range(total_points):
        if cluster_labels[idx] == -1:
            noise_points.append(dataset.iloc[idx])
        else:
            clusters[cluster_labels[idx] - 1].append(dataset.iloc[idx])

    return clusters, noise_points

def visualize_clusters(clusters, axis):
    for cluster in clusters:
        cluster_data = np.array(cluster)
        axis.scatter(cluster_data[:, 0], cluster_data[:, 1])

def calculate_cluster_distances(clusters):
    intra_cluster_distances = []
    inter_cluster_distances = []

    for cluster in clusters:
        cluster_data = np.array(cluster)
        if len(cluster_data) > 1:
            distances = pairwise_distances(cluster_data)
            intra_cluster_distances.append(np.mean(distances))

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            distances = pairwise_distances(np.array(clusters[i]), np.array(clusters[j]))
            inter_cluster_distances.append(np.mean(distances))

    mean_intra = np.mean(intra_cluster_distances) if intra_cluster_distances else 0
    mean_inter = np.mean(inter_cluster_distances) if inter_cluster_distances else 0
    return mean_intra, mean_inter

if __name__ == "__main__":
    fig, axes = plt.subplots(2)
    fig.suptitle('DBSCAN Clustering Visualization')

    # Processing wine dataset
    wine_data = pd.read_csv("wine-clustering.csv")[['Alcohol', 'Proline']]

    # Custom DBSCAN
    start_time = time.time()
    wine_clusters, wine_noise = perform_dbscan(wine_data, epsilon=40, min_points=6)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Custom DBSCAN took {elapsed_time:.9f} seconds to complete.")

    wine_noise_array = np.array(wine_noise)
    mean_intra_custom, mean_inter_custom = calculate_cluster_distances(wine_clusters)
    print(f"Custom DBSCAN - Mean intra-cluster distance: {mean_intra_custom:.4f}, Mean inter-cluster distance: {mean_inter_custom:.4f}")

    axes[0].set_title('Wine Data (With Noise)')
    visualize_clusters(wine_clusters, axes[0])
    if wine_noise_array.shape[0] > 0:
        axes[0].scatter(wine_noise_array[:, 0], wine_noise_array[:, 1], color='black')

    axes[1].set_title('Wine Data (Without Noise)')
    visualize_clusters(wine_clusters, axes[1])

    # Library DBSCAN
    start_time = time.time()
    sklearn_dbscan = DBSCAN(eps=40, min_samples=6)
    sklearn_labels = sklearn_dbscan.fit_predict(wine_data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Library DBSCAN took {elapsed_time:.9f} seconds to complete.")

    sklearn_clusters = [[] for _ in range(max(sklearn_labels) + 1)]
    sklearn_noise = []

    for idx, label in enumerate(sklearn_labels):
        if label == -1:
            sklearn_noise.append(wine_data.iloc[idx])
        else:
            sklearn_clusters[label].append(wine_data.iloc[idx])

    mean_intra_lib, mean_inter_lib = calculate_cluster_distances(sklearn_clusters)
    print(f"Library DBSCAN - Mean intra-cluster distance: {mean_intra_lib:.4f}, Mean inter-cluster distance: {mean_inter_lib:.4f}")

    plt.tight_layout()
    plt.show()
