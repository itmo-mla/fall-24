import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from time import time

class HierarchicalClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.linkage_matrix = []

    def fit_predict(self, X):
        # Инициализация расстояний и кластеров
        distance_matrix = cdist(X, X)
        np.fill_diagonal(distance_matrix, np.inf)
        cluster_count = len(distance_matrix)
        cluster_sizes = list(np.ones(cluster_count, dtype=np.int64))
        cluster_ids = list(range(cluster_count))
        clusters = [[i] for i in range(cluster_count)]
        linkage_distances = {}
        cluster_powers = {c_id: size for (c_id, size) in zip(cluster_ids, cluster_sizes)}

        while cluster_count > self.n_clusters:
            # Поиск ближайших кластеров
            closest_pair_idx = np.argmin(distance_matrix)
            cluster_a, cluster_b = np.unravel_index(closest_pair_idx, distance_matrix.shape)

            size_a, size_b = cluster_sizes[cluster_a], cluster_sizes[cluster_b]
            dist = distance_matrix[cluster_a, cluster_b]
            linkage_distances[(cluster_ids[cluster_a], cluster_ids[cluster_b])] = dist

            # Объединение кластеров
            max_cluster_id = max(cluster_ids)
            merged_cluster = []
            for idx in sorted([cluster_a, cluster_b], reverse=True):
                del cluster_sizes[idx]
                del cluster_ids[idx]
                merged_cluster += clusters[idx]

            # Обновление расстояний (Ward linkage)
            alpha_a = (np.array(cluster_sizes) + size_a) / (np.array(cluster_sizes) + size_a + size_b)
            alpha_b = (np.array(cluster_sizes) + size_b) / (np.array(cluster_sizes) + size_a + size_b)
            beta = -np.array(cluster_sizes) / (np.array(cluster_sizes) + size_a + size_b)

            distance_matrix = np.delete(distance_matrix, [cluster_a, cluster_b], axis=0)
            updated_distances = np.sqrt(
                alpha_a * np.square(distance_matrix[:, cluster_a]) +
                alpha_b * np.square(distance_matrix[:, cluster_b]) + beta * (dist ** 2)
            )

            distance_matrix = np.delete(distance_matrix, [cluster_a, cluster_b], axis=1)
            extended_matrix = np.pad(distance_matrix, ((0, 1), (0, 1)), constant_values=np.inf)
            extended_matrix[-1, :-1] = updated_distances
            extended_matrix[:-1, -1] = updated_distances
            distance_matrix = extended_matrix

            # Обновление кластеров и их размеров
            cluster_sizes.append(size_a + size_b)
            cluster_ids.append(max_cluster_id + 1)
            cluster_powers[max_cluster_id + 1] = size_a + size_b
            clusters = [c for idx, c in enumerate(clusters) if idx not in [cluster_a, cluster_b]]
            clusters.append(merged_cluster)
            cluster_count -= 1

        # Завершение настройки
        distance_matrix = np.min(distance_matrix).reshape((1, 1))
        linkage_distances[(cluster_ids[0], cluster_ids[1])] = distance_matrix[0, 0]
        self.linkage_matrix = linkage_distances
        self.cluster_powers = cluster_powers
        self._build_linkage_matrix()

        # Присвоение меток кластеров
        self.labels_ = np.zeros(len(X), dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            for point in cluster:
                self.labels_[point] = cluster_idx

        return self.labels_

    def _build_linkage_matrix(self):
        linkage_matrix = np.array([]).reshape((0, 4))
        for (cluster_a, cluster_b), dist in self.linkage_matrix.items():
            row = np.array([cluster_a, cluster_b, dist, self.cluster_powers[cluster_a] + self.cluster_powers[cluster_b]])
            linkage_matrix = np.vstack([linkage_matrix, row])
        self.linkage = linkage_matrix


def compute_metrics(X, labels):
    """
    Вычисление внутрикластерных и межкластерных расстояний.
    """
    unique_clusters = np.unique(labels)
    intra_cluster_distances = []
    inter_cluster_distances = []

    for cluster in unique_clusters:
        cluster_points = X[labels == cluster]
        if len(cluster_points) > 1:
            intra_cluster_distances.append(np.mean(cdist(cluster_points, cluster_points)))

        for other_cluster in unique_clusters:
            if cluster != other_cluster:
                other_points = X[labels == other_cluster]
                inter_cluster_distances.append(np.mean(cdist(cluster_points, other_points)))

    return {
        "mean_intra_cluster_distance": np.mean(intra_cluster_distances) if intra_cluster_distances else 0,
        "mean_inter_cluster_distance": np.mean(inter_cluster_distances) if inter_cluster_distances else 0,
    }


def compare_hierarchical(X, n_clusters):
    """
    Сравнение кастомной и эталонной реализации иерархической кластеризации.
    """
    # Кастомная реализация
    start_time = time()
    custom_model = HierarchicalClustering(n_clusters=n_clusters)
    custom_labels = custom_model.fit_predict(X)
    custom_execution_time = time() - start_time
    custom_metrics = compute_metrics(X, custom_labels)

    # Эталонная реализация
    start_time = time()
    etalon_model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    etalon_labels = etalon_model.fit_predict(X)
    etalon_execution_time = time() - start_time
    etalon_metrics = compute_metrics(X, etalon_labels)

    # Визуализация результатов с TSNE
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=custom_labels, cmap='Paired', s=10)
    plt.title(f"Custom Hierarchical (Time: {custom_execution_time:.4f}s)")

    plt.subplot(1, 2, 2)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=etalon_labels, cmap='Paired', s=10)
    plt.title(f"Etalon Hierarchical (Time: {etalon_execution_time:.4f}s)")

    plt.show()

    # Построение дендрограмм
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if n_clusters > 2:
        custom_model = HierarchicalClustering(n_clusters=2)
        custom_labels = custom_model.fit_predict(X)
    dendrogram(custom_model.linkage, ax=axes[0])
    axes[0].set_title("Custom algorithm")

    dendrogram(linkage(X, "ward"), ax=axes[1])
    axes[1].set_title("Scipy Implementation")
    plt.show()

    print("Custom Metrics: ", custom_metrics)
    print("Custom Time: ", custom_execution_time)
    print("Etalon Metrics: ", etalon_metrics)
    print("Etalon Time: ", etalon_execution_time)


# Пример использования
if __name__ == "__main__":
    from load_data import load_iris, load_wine, encod

    # Загрузка и подготовка данных Iris
    iris = np.array(encod(load_iris()))
    compare_hierarchical(iris, n_clusters=3)

    # Загрузка и подготовка данных Wine
    wine = np.array(encod(load_wine()))
    compare_hierarchical(wine, n_clusters=3)
