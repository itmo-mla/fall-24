import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from time import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import multivariate_normal


class EMClusterer:
    def __init__(self, n_clusters=2, max_iter=50, tol=1e-4):
        """
        EM-кластеризатор для данных, основанный на гауссовой смеси.
        :param n_clusters: Количество кластеров.
        :param max_iter: Максимальное количество итераций.
        :param tol: Порог изменения центров для остановки алгоритма.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None  # Веса кластеров (априорные вероятности)
        self.means = None  # Средние значения кластеров
        self.covariances = None  # Матрицы ковариаций
        self.labels_ = None  # Метки кластеров
        self.g = None  # Матрица принадлежностей

    def _initialize_parameters(self, x):
        """
        Инициализация параметров.
        :param x: Данные для кластеризации.
        """
        n_samples, n_features = x.shape
        self.weights = np.ones(self.n_clusters) / self.n_clusters
        self.means = x[np.random.choice(n_samples, self.n_clusters, replace=False)]
        self.covariances = [np.eye(n_features) for _ in range(self.n_clusters)]

    def _e_step(self, x):
        n_samples = x.shape[0]
        self.g = np.zeros((n_samples, self.n_clusters))

        for k in range(self.n_clusters):
            prob = multivariate_normal.pdf(x, mean=self.means[k], cov=self.covariances[k])
            self.g[:, k] = self.weights[k] * prob

        self.g /= self.g.sum(axis=1, keepdims=True)

    def _m_step(self, x):
        n_samples, n_features = x.shape
        N_k = self.g.sum(axis=0)

        self.weights = N_k / n_samples

        self.means = np.dot(self.g.T, x) / N_k[:, np.newaxis]

        self.covariances = []
        for k in range(self.n_clusters):
            diff = x - self.means[k]
            cov = np.dot(self.g[:, k] * diff.T, diff) / N_k[k]
            # Добавляем небольшую поправку для стабильности (регуляризация)
            self.covariances.append(cov + 1e-6 * np.eye(n_features))

    def fit_predict(self, x):
        self._initialize_parameters(x)
        y_prev = None

        for iteration in range(self.max_iter):
            prev_means = self.means.copy()

            self._e_step(x)

            self._m_step(x)

            # Определение кластеров по максимальной вероятности
            y = np.argmax(self.g, axis=1)

            # Проверка на сходимость
            if y_prev is not None and np.array_equal(y, y_prev):
                print(f"Сошлось на итерации {iteration + 1}")
                break
            if np.linalg.norm(self.means - prev_means) < self.tol:
                print(f"Сошлось по изменениям центров на итерации {iteration + 1}")
                break

            y_prev = y

        self.labels_ = y
        return y


def compute_metrics(X, labels):
    """
    Вычисление средних внутрикластерных и межкластерных расстояний.
    """
    unique_clusters = np.unique(labels)
    intra_distances = []
    inter_distances = []

    for cluster in unique_clusters:
        cluster_points = X[labels == cluster]
        if len(cluster_points) > 1:
            intra_distances.append(np.mean(cdist(cluster_points, cluster_points)))

        for other_cluster in unique_clusters:
            if cluster != other_cluster:
                other_points = X[labels == other_cluster]
                inter_distances.append(np.mean(cdist(cluster_points, other_points)))

    metrics = {
        "mean_intra_cluster_distance": np.mean(intra_distances) if intra_distances else 0,
        "mean_inter_cluster_distance": np.mean(inter_distances) if inter_distances else 0,
    }
    return metrics


def compare_em(X, n_clusters, max_iter=100, tol=1e-4):
    """
    Сравнение кастомной и эталонной реализации EM.
    """
    # Кастомная реализация
    start = time()
    labels_custom = EMClusterer(n_clusters, max_iter, tol).fit_predict(X)
    # _, labels_custom = em_algo(X, n_clusters, max_iter)
    custom_time = time() - start
    custom_metrics = compute_metrics(X, labels_custom)

    # Эталонная реализация
    start = time()
    gmm = GaussianMixture(n_components=n_clusters, max_iter=max_iter, tol=tol, random_state=42)
    labels_etalon = gmm.fit_predict(X)
    etalon_time = time() - start
    etalon_metrics = compute_metrics(X, labels_etalon)

    # Визуализация результатов с TSNE
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels_custom, cmap='Paired', s=10)
    plt.title(f"Custom EM (Time: {custom_time:.4f}s)")

    plt.subplot(1, 2, 2)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels_etalon, cmap='Paired', s=10)
    plt.title(f"Etalon EM (Time: {etalon_time:.4f}s)")

    plt.show()

    print("Castom метрики: ", custom_metrics)
    print('Castim time: ', custom_time)
    print("Etalon метрики:", etalon_metrics)
    print("Etalon time:", etalon_time)

# Пример использования
if __name__ == "__main__":
    from load_data import load_iris, load_wine, encod

    data = np.array(encod(load_iris()))
    X = data

    compare_em(X, n_clusters=2)

    data2 = np.array(encod(load_wine()))
    X2 = data2

    compare_em(X2, n_clusters=3)
