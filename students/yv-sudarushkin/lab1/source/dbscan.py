import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from time import time


def dbscan_custom(X, epsilon, min_samples):
    """
    Реализация DBSCAN.

    Parameters:
        X (np.array): массив данных размером (n_samples, n_features).
        epsilon (float): радиус для определения соседей.
        min_samples (int): минимальное количество точек для образования кластера.

    Returns:
        roles (np.array): роли точек (1 - ядро, 0 - пограничная, -1 - шум).
        clusters (np.array): массив кластерных меток.
    """
    distances = cdist(X, X)  # Вычисление расстояний между всеми точками
    np.fill_diagonal(distances, np.inf)  # Исключаем расстояния до самой себя

    N = len(X)
    clusters = np.full(N, np.nan)  # Инициализация кластеров
    roles = np.full(N, np.nan)  # Роли точек (ядро, пограничная, шум)

    current_cluster_id = 0

    for i in range(N):
        if not np.isnan(clusters[i]):
            continue

        neighbors = np.where(distances[i] < epsilon)[0]
        if len(neighbors) < min_samples:
            roles[i] = -1  # Шум
            continue

        roles[i] = 1  # Ядро
        clusters[i] = current_cluster_id

        candidates = set(neighbors)
        while candidates:
            candidate = candidates.pop()
            if not np.isnan(clusters[candidate]):
                continue

            clusters[candidate] = current_cluster_id
            candidate_neighbors = np.where(distances[candidate] < epsilon)[0]

            if len(candidate_neighbors) >= min_samples:
                roles[candidate] = 1
                candidates.update(candidate_neighbors)
            else:
                roles[candidate] = 0  # Краевая точка

        current_cluster_id += 1

    clusters[np.isnan(clusters)] = -1  # Шумовые точки помечаем -1
    return roles, clusters


def compute_metrics(X, clusters):
    """
    Вычисление внутрикластерных и межкластерных расстояний.

    Parameters:
        X (np.array): массив данных.
        clusters (np.array): метки кластеров.

    Returns:
        dict: словарь с метриками (среднее внутрикластерное и межкластерное расстояния).
    """
    unique_clusters = np.unique(clusters[clusters != -1])
    intra_distances = []
    inter_distances = []

    for cluster in unique_clusters:
        cluster_points = X[clusters == cluster]
        if len(cluster_points) > 1:
            intra_distances.append(np.mean(cdist(cluster_points, cluster_points)))

        for other_cluster in unique_clusters:
            if cluster != other_cluster:
                other_points = X[clusters == other_cluster]
                inter_distances.append(np.mean(cdist(cluster_points, other_points)))

    metrics = {
        "mean_intra_cluster_distance": np.mean(intra_distances) if intra_distances else 0,
        "mean_inter_cluster_distance": np.mean(inter_distances) if inter_distances else 0,
    }
    return metrics


def compare_with_etalon(X, epsilon, min_samples):
    """
    Сравнение кастомной и эталонной реализации DBSCAN.

    Parameters:
        X (np.array): массив данных.
        epsilon (float): радиус для определения соседей.
        min_samples (int): минимальное количество точек для образования кластера.
    """
    # Кастомная реализация
    start = time()
    roles, clusters_custom = dbscan_custom(X, epsilon, min_samples)
    custom_time = time() - start
    custom_metrics = compute_metrics(X, clusters_custom)

    # Эталонная реализация
    start = time()
    db = DBSCAN(eps=epsilon, min_samples=min_samples)
    clusters_etalon = db.fit_predict(X)
    etalon_time = time() - start
    etalon_metrics = compute_metrics(X, clusters_etalon)

    # Визуализация результатов с TSNE
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters_custom, cmap='Paired', s=10)
    plt.title(f"Custom DBSCAN (Time: {custom_time:.4f}s)")

    plt.subplot(1, 2, 2)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters_etalon, cmap='Paired', s=10)
    plt.title(f"Etalon DBSCAN (Time: {etalon_time:.4f}s)")

    plt.show()

    print("Castom метрики: ", custom_metrics)
    print('Castim time: ', custom_time)
    print("Etalon метрики:", etalon_metrics)
    print("Etalon time:", etalon_time)


# Пример использования
if __name__ == "__main__":
    # Загрузка данных (например, Iris или Wine)
    from load_data import load_iris, load_wine, encod

    data = np.array(encod(load_iris()))
    X = data  # Для визуализации возьмем только первые две фичи

    epsilon = 1.5
    min_samples = 5

    compare_with_etalon(X, epsilon, min_samples)

    data2 = np.array(encod(load_wine()))
    X2 = data2
    epsilon = 2.3
    min_samples = 12

    compare_with_etalon(X2, epsilon, min_samples)
