import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import load_iris
import time
from clustering import show_metrics
from clustering import Clustering

def load_and_preprocess_data():
    iris = load_iris()
    data = iris.data
    labels = iris.target
    scaler = preprocessing.StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data, labels


def compute_initial_distance_matrix(data):
    num_points = data.shape[0]
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance = np.linalg.norm(data[i] - data[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix


def ward_distance(cluster1, cluster2, data):
    # Вычисляем центры кластеров
    center1 = np.mean(data[cluster1], axis=0)
    center2 = np.mean(data[cluster2], axis=0)
    # Расстояние между центрами
    distance = np.linalg.norm(center1 - center2)
    # Размеры кластеров
    size1 = len(cluster1)
    size2 = len(cluster2)
    # Формула метода Уорда
    ward_dist = (size1 * size2) / (size1 + size2) * distance ** 2
    return ward_dist


def hierarchical_clustering_ward(data):
    num_points = data.shape[0]
    clusters = {i: [i] for i in range(num_points)}
    current_cluster_id = num_points
    linkage_matrix = []

    centers = {i: data[i] for i in range(num_points)}

    distance_matrix = {}
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance = ward_distance(clusters[i], clusters[j], data)
            distance_matrix[(i, j)] = distance

    while len(clusters) > 1:
        (c1, c2), min_dist = min(distance_matrix.items(), key=lambda x: x[1])

        linkage_matrix.append([c1, c2, np.sqrt(min_dist), len(clusters[c1]) + len(clusters[c2])])

        clusters[current_cluster_id] = clusters[c1] + clusters[c2]
        del clusters[c1]
        del clusters[c2]

        keys_to_remove = [key for key in distance_matrix.keys() if c1 in key or c2 in key]
        for key in keys_to_remove:
            del distance_matrix[key]

        for cid in clusters.keys():
            if cid == current_cluster_id:
                continue
            key = tuple(sorted((cid, current_cluster_id)))
            distance = ward_distance(clusters[cid], clusters[current_cluster_id], data)
            distance_matrix[key] = distance

        current_cluster_id += 1

    return np.array(linkage_matrix)


def plot_dendrogram_custom(linkage_matrix, labels):
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, labels=labels, orientation='top', distance_sort='descending')
    plt.title('CUSTOM HIER')
    plt.xlabel('OBJ')
    plt.ylabel('DISTS')
    plt.show()


def plot_dendrogram_scipy(linkage_matrix, labels):
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, labels=labels, orientation='top', distance_sort='descending')
    plt.title('ETALON HIER')
    plt.xlabel('OBJ')
    plt.ylabel('DISTS')
    plt.show()

if __name__ == "__main__":
    data, true_labels = load_and_preprocess_data()
    num_points = data.shape[0]
    labels = [f"Point {i}" for i in range(num_points)]

    clustering = Clustering(name="Hierarchical Clustering")

    start_time = time.time()
    custom_linkage = hierarchical_clustering_ward(data)
    custom_time = time.time() - start_time

    start_time = time.time()
    scipy_linkage_matrix = linkage(data, method='ward')
    scipy_time = time.time() - start_time

    plot_dendrogram_custom(custom_linkage, labels)
    plot_dendrogram_scipy(scipy_linkage_matrix, labels)

    num_clusters = 3
    custom_clusters = fcluster(custom_linkage, num_clusters, criterion='maxclust')
    scipy_clusters = fcluster(scipy_linkage_matrix, num_clusters, criterion='maxclust')

    intra_custom, inter_custom = show_metrics(data, custom_clusters)

    clustering.set_data_custom([custom_time, intra_custom, inter_custom])

    intra_scipy, inter_scipy = show_metrics(data, scipy_clusters)

    clustering.set_data_etalon([scipy_time, intra_scipy, inter_scipy])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], c=custom_clusters, cmap='prism')
    plt.title('CUSTOM')

    plt.subplot(1, 2, 2)
    plt.scatter(data[:, 0], data[:, 1], c=scipy_clusters, cmap='prism')
    plt.title('ETALON')

    plt.tight_layout()
    plt.show()

    clustering.print_data()
