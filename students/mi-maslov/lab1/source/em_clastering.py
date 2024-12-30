import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.mixture import GaussianMixture
from clustering import show_metrics, Clustering


def em_custom(points: np.array, num_clusters: int, random_state: int = 2):
    np.random.seed(random_state)
    num_features = points.shape[1]
    num_points = points.shape[0]

    # Инициализация весов, средних и ковариаций для каждого кластера
    weights = np.ones((num_clusters, 1)) / num_clusters
    means = np.random.rand(num_clusters, num_features) * np.max(points, axis=0)
    covariances = np.ones((num_clusters, num_features)) / 2

    points = points[np.newaxis, :, :]

    means = means[:, np.newaxis, :]
    covariances = covariances[:, np.newaxis, :]

    previous_means = means.copy() + 0.5  # Инициализация для проверки изменения средних значений
    iteration_data = []

    while np.sum((previous_means - means) ** 2) > 0.1:
        previous_means = means.copy()

        # E-фаза: вычисление вероятностей
        distances = np.sum((points - means) ** 2 / covariances, axis=2)
        likelihood = np.exp(-0.5 * distances) / (np.sqrt((2 * np.pi) ** num_features * np.prod(covariances, axis=2)))

        # Вычисление принадлежности точек к кластерам
        responsibilities = (weights * likelihood) / (np.sum(weights * likelihood, axis=0) + 1e-200)
        unclustered_points = np.sum(responsibilities, axis=0) == 0
        responsibilities[:, unclustered_points] = distances[:, unclustered_points] / np.sum(
            distances[:, unclustered_points], axis=0)

        iteration_data.append(
            {'responsibilities': responsibilities, 'weights': weights, 'means': means, 'covariances': covariances})

        # M-фаза: обновление параметров
        weights = np.sum(responsibilities, axis=1, keepdims=True) / num_points
        means = (responsibilities @ points / (weights * num_points)).reshape((num_clusters, 1, num_features))
        covariances = np.diagonal(
            np.sum(responsibilities[:, :, np.newaxis] * (points - means[:, np.newaxis, :]) ** 2, axis=2) / (
                        weights * num_points)).T[:, np.newaxis, :]
        covariances = np.where(covariances == 0, 1e-10, covariances)

    predicted_clusters = np.argmax(responsibilities, axis=0)
    return iteration_data, predicted_clusters


def em_etalon(data, num_clusters):
    start = time()
    model = GaussianMixture(n_components=num_clusters)
    predicted = model.fit_predict(data)
    end_time = time() - start

    inside, outside = show_metrics(data, predicted, iconic=True)
    plt.scatter(data[:, 0], data[:, 1], c=predicted, cmap='Paired')
    plt.title("EM - ETALON")
    plt.show()

    return end_time, inside, outside


if __name__ == "__main__":
    num_clusters = 3
    data = pd.read_csv("../datasets/iris_norm.csv").to_numpy()

    cl = Clustering(data=data, name="EM")

    start_time = time()
    iteration_data, predicted_clusters = em_custom(data, num_clusters)
    end_time = time() - start_time

    inside, outside = show_metrics(data, predicted_clusters)

    cl.set_data_custom([end_time, inside, outside])

    plt.scatter(data[:, 0], data[:, 1], c=predicted_clusters, cmap='Paired')
    plt.title("EM - CUSTOM")
    plt.show()
    end_time, inside, outside = em_etalon(data, num_clusters)
    cl.set_data_etalon([end_time, inside, outside])

    cl.print_data()
