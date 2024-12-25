import numpy as np

def euclidean_distance(a:np.array, b:np.array):
    """
    Вычисление Евклидова расстояния между двумя точками a и b.
    """
    return np.sqrt(np.sum((a - b) ** 2))

def mean_intracluster_distance(points:np.array, labels:np.array):
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

def mean_intercluster_distance(points:np.array, labels:np.array):
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

def show_metrics(points:np.array, now_pred:np.array, iconic:bool=False):
    
    intra_dist = mean_intracluster_distance(points, now_pred)
    inter_dist = mean_intercluster_distance(points, now_pred)

    print(f"\tСреднее внутрикластерное расстояние для {'моей' if not iconic else 'эталонной'} реализации: {intra_dist}")
    print(f"\tСреднее межкластерное расстояние для {'моей' if not iconic else 'эталонной'} реализации: {inter_dist}")



if __name__ == "__main__":
    # Пример использования
    points = np.array([[1, 2], [2, 3], [5, 6], [8, 8], [9, 10]])
    labels = np.array([0, 0, 1, 1, 1])

    intra_dist = mean_intracluster_distance(points, labels)
    inter_dist = mean_intercluster_distance(points, labels)

    print(f"Среднее внутрикластерное расстояние: {intra_dist}")
    print(f"Среднее межкластерное расстояние: {inter_dist}")


