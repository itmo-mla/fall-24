import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
import time

def expectation_maximization(data, num_clusters, max_iterations=100):
    data_array = data.values

    cluster_weights = np.full(num_clusters, 1 / num_clusters)
    cluster_means = data_array[np.random.choice(len(data_array), num_clusters, replace=False)]
    cluster_variances = np.random.rand(num_clusters, data_array.shape[1])
    cluster_variances = np.maximum(cluster_variances, 1e-8)

    previous_labels = np.array([])

    for _ in range(max_iterations):
        # E-step: Calculate responsibilities
        responsibilities = np.zeros((len(data_array), num_clusters))
        for cluster_idx in range(num_clusters):
            likelihood = (
                np.prod(
                    (1 / np.sqrt(2 * np.pi * cluster_variances[cluster_idx])) *
                    np.exp(-0.5 * ((data_array - cluster_means[cluster_idx]) ** 2) / cluster_variances[cluster_idx]),
                    axis=1
                )
            )
            responsibilities[:, cluster_idx] = cluster_weights[cluster_idx] * likelihood

        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities_sum[responsibilities_sum == 0] = 1
        responsibilities /= responsibilities_sum

        # M-step: Update parameters
        cluster_weights = responsibilities.mean(axis=0)
        cluster_means = (responsibilities.T @ data_array) / responsibilities.sum(axis=0)[:, None]

        for cluster_idx in range(num_clusters):
            diff = data_array - cluster_means[cluster_idx]
            cluster_variances[cluster_idx] = (
                (responsibilities[:, cluster_idx] @ (diff ** 2)) /
                responsibilities[:, cluster_idx].sum()
            )

        cluster_variances = np.maximum(cluster_variances, 1e-8)

        # Check for convergence
        current_labels = np.argmax(responsibilities, axis=1)
        if np.array_equal(previous_labels, current_labels):
            break
        previous_labels = current_labels

    # Group data points by cluster
    clustered_data = [[] for _ in range(num_clusters)]
    for idx, cluster_idx in enumerate(current_labels):
        clustered_data[cluster_idx].append(data.iloc[idx])

    return clustered_data, current_labels

def visualize_clusters(clustered_data, subplot):
    for cluster in clustered_data:
        cluster_points = np.array(cluster)
        subplot.scatter(cluster_points[:, 0], cluster_points[:, 1])

def calculate_distances(data, labels, num_clusters):
    cluster_centers = np.array([data[labels == i].mean(axis=0) for i in range(num_clusters)])
    intra_distances = []
    for i in range(num_clusters):
        cluster_points = data[labels == i]
        distances = np.linalg.norm(cluster_points - cluster_centers[i], axis=1)
        intra_distances.append(distances.mean())
    
    # Compute pairwise distances between cluster centers
    inter_distances = cdist(cluster_centers, cluster_centers, metric='euclidean')
    np.fill_diagonal(inter_distances, np.nan)
    avg_inter_distance = np.nanmean(inter_distances)

    return np.mean(intra_distances), avg_inter_distance

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('EM Algorithm Clustering')

    wine_data = pd.read_csv("wine-clustering.csv")[['Alcohol', 'Proline']]

    # Custom EM Algorithm
    start_time = time.time()
    wine_clusters, custom_labels = expectation_maximization(wine_data, 3)
    custom_duration = time.time() - start_time

    ax.set_title('Wine Dataset - Custom EM')
    visualize_clusters(wine_clusters, ax)

    # Library EM Algorithm
    gmm = GaussianMixture(n_components=3, random_state=0)
    start_time = time.time()
    gmm.fit(wine_data)
    library_labels = gmm.predict(wine_data)
    library_duration = time.time() - start_time

    # Calculate distances
    wine_data_array = wine_data.values
    custom_intra, custom_inter = calculate_distances(wine_data_array, custom_labels, 3)
    library_intra, library_inter = calculate_distances(wine_data_array, library_labels, 3)

    # Print results
    print("Custom EM Algorithm:")
    print(f"Execution Time: {custom_duration:.4f} seconds")
    print(f"Average Intra-cluster Distance: {custom_intra:.4f}")
    print(f"Average Inter-cluster Distance: {custom_inter:.4f}")

    print("\nLibrary EM Algorithm:")
    print(f"Execution Time: {library_duration:.4f} seconds")
    print(f"Average Intra-cluster Distance: {library_intra:.4f}")
    print(f"Average Inter-cluster Distance: {library_inter:.4f}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
