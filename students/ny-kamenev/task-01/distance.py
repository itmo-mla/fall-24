import numpy as np
from scipy.spatial.distance import pdist
def calculate_intra_cluster_distances_lib(data, labels):
    intra_distances = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_points = data[labels == label]
        if len(cluster_points) > 1:
            distances = pdist(cluster_points, metric='euclidean')
            intra_distances.append(np.mean(distances))
    return np.mean(intra_distances)


def calculate_inter_cluster_distances_lib(data, labels):
    unique_labels = np.unique(labels)
    valid_labels = unique_labels[unique_labels != -1]

    if len(valid_labels) < 2:
        return np.nan

    centroids = []
    for label in unique_labels:
        cluster_points = data[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)

    centroid_distances = pdist(centroids, metric='euclidean')
    return np.mean(centroid_distances)


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def calculate_intra_cluster_distances(data, clusters):
    intra_cluster_distances = []

    for cluster_id, elements in clusters.items():
        cluster_points = data[elements]
        num_points = len(cluster_points)

        if num_points > 1:
            total_distance = 0
            count = 0
            for i in range(num_points):
                for j in range(i + 1, num_points):
                    total_distance += euclidean_distance(cluster_points[i], cluster_points[j])
                    count += 1
            mean_distance = total_distance / count if count > 0 else 0
            intra_cluster_distances.append(mean_distance)

    return np.mean(intra_cluster_distances)


def calculate_inter_cluster_distances(data, clusters):
    cluster_centroids = []

    for elements in clusters.values():
        cluster_points = data[elements]
        centroid = np.mean(cluster_points, axis=0)
        cluster_centroids.append(centroid)

    total_distance = 0
    count = 0
    for i in range(len(cluster_centroids)):
        for j in range(i + 1, len(cluster_centroids)):
            total_distance += euclidean_distance(cluster_centroids[i], cluster_centroids[j])
            count += 1

    mean_inter_cluster_distance = total_distance / count if count > 0 else 0
    return mean_inter_cluster_distance
