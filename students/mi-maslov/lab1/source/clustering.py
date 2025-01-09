import pandas as pd
import numpy as np


class Clustering:
    def __init__(self, name="Clustering algorithm", data=None):
        if data is None:
            data = list([])
        self.name = name
        exec_data = {
            "TYPE": ["CUSTOM", "ETALON"],
            "EXEC TIME": [0.0, 0.0],
            "DISTANCE INSIDE CLUSTER": [0.0, 0.0],
            "DISTANCE BEETWEN CLUSTERS": [0.0, 0.0]
        }
        self.exec_data = pd.DataFrame(exec_data)
        self.exec_time = 0
        self.custom_func = 0
        self.etalon_func = 0
        self.data = data

    def print_data(self):
        print(f"Name: {self.name}")
        print(self.exec_data)

    def plot_cluster_custom(self):
        pass

    def plot_cluster_etalon(self):
        pass

    def set_data_custom(self, data):
        self.exec_data.loc[0] = ["CUSTOM", data[0], data[1], data[2]]

    def set_data_etalon(self, data):
        self.exec_data.loc[1] = ["ETALON", data[0], data[1], data[2]]

def euclidean_distance(a: np.array, b: np.array):
    """
    Вычисление Евклидова расстояния между двумя точками a и b.
    """
    return np.sqrt(np.sum((a - b) ** 2))


def mean_intracluster_distance(points: np.array, labels: np.array):
    """
    Вычисляет среднее внутрикластерное расстояние.
    """
    unique_labels = np.unique(labels)
    intra_distances = []

    for label in unique_labels:
        # Выбираем точки, принадлежащие текущему кластеру
        cluster_points = points[labels == label]

        # Считаем попарные расстояния внутри кластера
        if len(cluster_points) > 1:
            distances = []
            for i in range(len(cluster_points)):
                for j in range(i + 1, len(cluster_points)):
                    distances.append(euclidean_distance(cluster_points[i], cluster_points[j]))
            intra_distances.append(np.mean(distances))

    # Среднее внутрикластерное расстояние по всем кластерам
    return np.mean(intra_distances) if intra_distances else 0


def mean_intercluster_distance(points: np.array, labels: np.array):
    """
    Вычисляет среднее межкластерное расстояние.
    """
    unique_labels = np.unique(labels)
    cluster_centers = []

    # Находим центры каждого кластера
    for label in unique_labels:
        cluster_points = points[labels == label]
        cluster_center = np.mean(cluster_points, axis=0)
        cluster_centers.append(cluster_center)

    # Считаем попарные расстояния между центрами кластеров
    inter_distances = []
    for i in range(len(cluster_centers)):
        for j in range(i + 1, len(cluster_centers)):
            inter_distances.append(euclidean_distance(cluster_centers[i], cluster_centers[j]))

    # Среднее межкластерное расстояние
    return np.mean(inter_distances) if inter_distances else 0


def show_metrics(points: np.array, now_pred: np.array, iconic: bool = False):
    intra_dist = mean_intracluster_distance(points, now_pred)
    inter_dist = mean_intercluster_distance(points, now_pred)

    return intra_dist, inter_dist



