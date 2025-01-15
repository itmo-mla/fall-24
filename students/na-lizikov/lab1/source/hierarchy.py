import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import linkage, fcluster
from time import time

def compute_new_distance(r_us, r_vs, r_uv, size_u, size_v, size_s, strategy='min'):
    coefficients = {
        'min': (0.5, 0.5, 0, -0.5),
        'max': (0.5, 0.5, 0, 0.5),
        'mean': (size_u / (size_u + size_v), size_v / (size_u + size_v), 0, 0),
        'center': (
            size_u / (size_u + size_v),
            size_v / (size_u + size_v),
            -(size_u / (size_u + size_v)) * (size_v / (size_u + size_v)),
            0
        ),
        'ward': (
            (size_s + size_u) / (size_s + size_u + size_v),
            (size_s + size_v) / (size_s + size_u + size_v),
            -size_s / (size_s + size_u + size_v),
            0
        )
    }
    if strategy not in coefficients:
        raise ValueError("Invalid strategy. Choose from: 'min', 'max', 'mean', 'center', 'ward'.")

    alpha_u, alpha_v, beta, gamma = coefficients[strategy]
    return alpha_u * r_us + alpha_v * r_vs + beta * r_uv + gamma * abs(r_us - r_vs)

def hierarchical_clustering(dataset, num_clusters, strategy='min', dendrogram=False):
    num_points = dataset.shape[0]
    clusters = [[i] for i in range(num_points)]
    active_clusters = [True] * num_points
    history_flags = []
    distance_matrix = np.zeros((num_points, num_points))
    linkage_data = []

    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = np.linalg.norm(dataset.iloc[i] - dataset.iloc[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    while active_clusters.count(True) > num_clusters:
        min_distance = np.inf
        closest_pair = (0, 0)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if active_clusters[i] and active_clusters[j]:
                    if distance_matrix[i, j] < min_distance:
                        min_distance = distance_matrix[i, j]
                        closest_pair = (i, j)

        cluster_a, cluster_b = closest_pair
        merged_cluster = clusters[cluster_a] + clusters[cluster_b]
        linkage_data.append([cluster_a, cluster_b, min_distance, len(merged_cluster)])

        active_clusters[cluster_a] = False
        active_clusters[cluster_b] = False
        active_clusters.append(True)
        history_flags.append(deepcopy(active_clusters))
        clusters.append(merged_cluster)

        distance_matrix = np.pad(distance_matrix, ((0, 1), (0, 1)), constant_values=0)

        for k in range(len(clusters) - 1):
            if k != cluster_a and k != cluster_b and active_clusters[k]:
                new_distance = compute_new_distance(
                    distance_matrix[cluster_a, k],
                    distance_matrix[cluster_b, k],
                    distance_matrix[cluster_a, cluster_b],
                    len(clusters[cluster_a]),
                    len(clusters[cluster_b]),
                    len(clusters[k]),
                    strategy
                )
                distance_matrix[k, -1] = new_distance
                distance_matrix[-1, k] = new_distance

    result_clusters = []
    for idx, cluster in enumerate(clusters):
        if active_clusters[idx]:
            result_clusters.append([dataset.iloc[point].to_numpy().tolist() for point in cluster])

    return (history_flags, linkage_data) if dendrogram else (result_clusters, linkage_data)

def visualize_clusters(cluster_groups, axis):
    for cluster in cluster_groups:
        points = np.array(cluster)
        axis.scatter(points[:, 0], points[:, 1])

def calculate_cluster_distances(dataset, cluster_labels):
    intra_distances = []
    inter_distances = []

    for cluster_id in np.unique(cluster_labels):
        cluster_points = dataset[cluster_labels == cluster_id]
        other_points = dataset[cluster_labels != cluster_id]

        if len(cluster_points) > 1:
            intra_distances.append(
                np.mean([np.linalg.norm(a - b) for i, a in enumerate(cluster_points) for b in cluster_points[i + 1:]])
            )

        if len(cluster_points) > 0 and len(other_points) > 0:
            inter_distances.append(
                np.mean([np.linalg.norm(a - b) for a in cluster_points for b in other_points])
            )

    mean_intra_distance = np.mean(intra_distances) if intra_distances else 0
    mean_inter_distance = np.mean(inter_distances) if inter_distances else 0
    return mean_intra_distance, mean_inter_distance


if __name__ == "__main__":
    wine_data = pd.read_csv("wine-clustering.csv")[['Alcohol', 'Proline']]
    start_time = time()
    wine_clusters = [
        hierarchical_clustering(wine_data, 3, method)[0] for method in ['min', 'max', 'mean', 'center', 'ward']
    ]
    custom_time = time() - start_time

    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    fig.suptitle('Hierarchical Clustering Visualization', fontsize=16)

    titles = [
        'Wine (min)', 'Wine (max)', 'Wine (mean)', 'Wine (center)', 'Wine (ward)',
    ]

    cluster_sets = wine_clusters 

    for idx, ax in enumerate(axes.flatten()):
        if idx < len(titles): 
            ax.set_title(titles[idx])
            visualize_clusters(cluster_sets[idx], ax)
        else:
            ax.axis('off') 

    start_time = time()
    linkage_matrix = linkage(wine_data, method='ward')
    library_labels = fcluster(linkage_matrix, t=3, criterion='maxclust')
    library_time = time() - start_time
    
    wine_data_np = wine_data.to_numpy()
    custom_labels = np.zeros(len(wine_data))
    for cluster_id, cluster in enumerate(wine_clusters):
        for point in cluster:
            for sub_point in point:
                idx = wine_data[
                    (wine_data['Alcohol'] == sub_point[0]) & 
                    (wine_data['Proline'] == sub_point[1])
                ].index[0]
                custom_labels[idx] = cluster_id

    custom_intra, custom_inter = calculate_cluster_distances(wine_data_np, custom_labels)
    library_intra, library_inter = calculate_cluster_distances(wine_data_np, library_labels)

    # Print results
    print(f"Custom implementation: time={custom_time:.4f}s, intra={custom_intra:.4f}, inter={custom_inter:.4f}")
    print(f"Library implementation: time={library_time:.4f}s, intra={library_intra:.4f}, inter={library_inter:.4f}")


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
