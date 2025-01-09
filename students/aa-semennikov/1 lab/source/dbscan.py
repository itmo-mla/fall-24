import numpy as np
import matplotlib.pyplot as plt
from wine import read_wine
from weapons import read_weapons
from sklearn.decomposition import PCA

def dbscan_algo(data, eps, min_samples):    
    data = data.to_numpy()
    n = data.shape[0]
    labels = -np.ones(n)
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)

    # Идем по всем точкам, пропуская ранее посещенные
    for i in range(n):
        if visited[i]:
            continue
        
        visited[i] = True
        # Получаем соседей точки в e-окрестности
        neighbors = np.where(np.linalg.norm(data - data[i], axis=1) <= eps)[0]

        # Если соседей меньше, чем нужно для получения точкой статуса корневой, помечаем её возможно шумовой
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            # Если соседей достаточно, то увеличиваем число кластеров на 1, точка становится корневой
            cluster_id += 1
            labels[i] = cluster_id
            
            # Идем по всем соседям корневой точки, чтобы расширить кластер
            next_door_neighbors = list(neighbors)
            while len(next_door_neighbors) > 0:
                # Заходим к соседу выбранной точки
                current_point = next_door_neighbors.pop()
                labels[current_point] = cluster_id
                # Если не заходили ранее, то помечаем, что зашли сейчас, узнаем список его соседей
                if not visited[current_point]:
                    visited[current_point] = True
                    neighbors = np.where(np.linalg.norm(data - data[current_point], axis=1) <= eps)[0]
                    # Если соседей current_point достаточно (корневая), то добавляем их в список соседей корневой точки i (не в e),
                    # для которых тоже проверяем, корневые ли они
                    if len(neighbors) >= min_samples:
                        next_door_neighbors.extend(neighbors)

    return labels

def plot(ax, data, labels):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            color = 'black'
        else:
            color = plt.cm.Spectral(label / len(unique_labels))
        ax.scatter(data_pca[labels == label, 0], data_pca[labels == label, 1], color=color, s=30)
        
if __name__ == "__main__":
    wine = read_wine()
    weapons = read_weapons()
    wine_res = dbscan_algo(wine, 2.5, 12)
    weapons_res = dbscan_algo(weapons, 1, 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plot(ax1, wine, wine_res)
    plot(ax2, weapons, weapons_res)
    ax1.set_title('Wine clustered')
    ax2.set_title('Weapons clustered')
    plt.tight_layout()
    plt.show()