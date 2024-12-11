import numpy as np
import matplotlib.pyplot as plt
from wine import read_wine
from weapons import read_weapons
from hierarchy_algo import hierarchy_algo
from em import em_algo
from dbscan import dbscan_algo
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from time import time
from sklearn.metrics import pairwise_distances

def intra_cluster_distance(data, labels):
    unique_labels = np.unique(labels)
    distances = []

    for label in unique_labels:
        if label == -1:  # Страховка от шума в случае DBSCAN
            continue
            
        cluster_points = data[labels == label]
        if len(cluster_points) > 1:
            # Вычисляем все попарные расстояния внутри кластера
            dists = pairwise_distances(cluster_points)
            # Берем среднее расстояние между всеми точками в кластере
            mean_distance = np.mean(dists)
            distances.append(mean_distance)

    return np.mean(distances) if distances else 0

def inter_cluster_distance(data, labels):
    unique_labels = np.unique(labels)
    cluster_centers = []

    for label in unique_labels:
        if label == -1:
            continue
            
        cluster_points = data[labels == label]
        center = np.mean(cluster_points, axis=0)
        cluster_centers.append(center)

    # Вычисляем все попарные расстояния между центрами кластеров
    if len(cluster_centers) > 1:
        dists = pairwise_distances(cluster_centers)
        return np.mean(dists)
    
    return 0

if __name__ == "__main__":
    wine = read_wine()
    weapons = read_weapons()
    times = []
    intra_dist = []
    inter_dist = []

    # Мои алгоритмы
    # Иерархический
    start_time = time()
    clusters = hierarchy_algo(wine, 2)[3]
    labels = []
    for i in clusters:
        labels.append(i[1])
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(intra_cluster_distance(wine, labels))
    inter_dist.append(inter_cluster_distance(wine, labels))

    start_time = time()
    clusters = hierarchy_algo(weapons, 2)[3]
    labels = []
    for i in clusters:
        labels.append(i[1])
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(intra_cluster_distance(weapons, labels))
    inter_dist.append(inter_cluster_distance(weapons, labels))

    # EM
    start_time = time()
    labels = em_algo(wine, 2)[1]
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(intra_cluster_distance(wine, labels))
    inter_dist.append(inter_cluster_distance(wine, labels))

    start_time = time()
    labels = em_algo(weapons, 2)[1]
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(intra_cluster_distance(weapons, labels))
    inter_dist.append(inter_cluster_distance(weapons, labels))

    # DBSCAN
    start_time = time()
    labels = dbscan_algo(wine, 2.5, 12)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(intra_cluster_distance(wine, labels))
    inter_dist.append(inter_cluster_distance(wine, labels))

    start_time = time()
    labels = dbscan_algo(weapons, 1, 4)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(intra_cluster_distance(weapons, labels))
    inter_dist.append(inter_cluster_distance(weapons, labels))


    # Алгоритмы из sklearn
    # Иерархический
    agg_clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
    start_time = time()
    labels = agg_clustering.fit_predict(wine)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(intra_cluster_distance(wine, labels))
    inter_dist.append(inter_cluster_distance(wine, labels))

    agg_clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
    start_time = time()
    labels = agg_clustering.fit_predict(weapons)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(intra_cluster_distance(weapons, labels))
    inter_dist.append(inter_cluster_distance(weapons, labels))

    # EM
    gmm = GaussianMixture(n_components=2)
    start_time = time()
    gmm.fit(wine)
    labels = gmm.predict(wine)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(intra_cluster_distance(wine, labels))
    inter_dist.append(inter_cluster_distance(wine, labels))

    gmm = GaussianMixture(n_components=2)
    start_time = time()
    gmm.fit(weapons)
    labels = gmm.predict(weapons)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(intra_cluster_distance(weapons, labels))
    inter_dist.append(inter_cluster_distance(weapons, labels))

    # DBSCAN
    db = DBSCAN(eps=2.5, min_samples=12)
    start_time = time()
    labels = db.fit_predict(wine)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(intra_cluster_distance(wine, labels))
    inter_dist.append(inter_cluster_distance(wine, labels))

    db = DBSCAN(eps=1, min_samples=4)
    start_time = time()
    labels = db.fit_predict(weapons)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(intra_cluster_distance(weapons, labels))
    inter_dist.append(inter_cluster_distance(weapons, labels))

    print("Моё VS sklearn (wine)")

    print("\nИерархический")
    print(f"Ср. S внутри:\t{np.round(intra_dist[0], 2)}\tVS\t{np.round(intra_dist[6], 2)}")
    print(f"Ср. S м-у кластерами:\t{np.round(inter_dist[0], 2)}\tVS\t{np.round(inter_dist[6], 2)}")
    print(f"Время:\t{np.round(times[0], 2)} s \tVS\t{times[6]} s")

    print("\nEM")
    print(f"Ср. S внутри:\t{np.round(intra_dist[2], 2)}\tVS\t{np.round(intra_dist[8], 2)}")
    print(f"Ср. S м-у кластерами:\t{np.round(inter_dist[2], 2)}\tVS\t{np.round(inter_dist[8], 2)}")
    print(f"Время:\t{times[2]} s \tVS\t{times[8]} s")

    print("\nDBSCAN")
    print(f"Ср. S внутри:\t{np.round(intra_dist[4], 2)}\tVS\t{np.round(intra_dist[10], 2)}")
    print(f"Ср. S м-у кластерами:\t{np.round(inter_dist[4], 2)}\tVS\t{np.round(inter_dist[10], 2)}")
    print(f"Время:\t{times[4]} s \tVS\t{times[10]} s")
    
    print("\n\nМоё VS sklearn (weapons)")

    print("\nИерархический")
    print(f"Ср. S внутри:\t{np.round(intra_dist[1], 2)}\tVS\t{np.round(intra_dist[7], 2)}")
    print(f"Ср. S м-у кластерами:\t{np.round(inter_dist[1], 2)}\tVS\t{np.round(inter_dist[7], 2)}")
    print(f"Время:\t{np.round(times[1], 2)} s \tVS\t{times[7]} s")

    print("\nEM")
    print(f"Ср. S внутри:\t{np.round(intra_dist[3], 2)}\tVS\t{np.round(intra_dist[9], 2)}")
    print(f"Ср. S м-у кластерами:\t{np.round(inter_dist[3], 2)}\tVS\t{np.round(inter_dist[9], 2)}")
    print(f"Время:\t{times[3]} s \tVS\t{times[9]} s")

    print("\nDBSCAN")
    print(f"Ср. S внутри:\t{np.round(intra_dist[5], 2)}\tVS\t{np.round(intra_dist[11], 2)}")
    print(f"Ср. S м-у кластерами:\t{np.round(inter_dist[5], 2)}\tVS\t{np.round(inter_dist[11], 2)}")
    print(f"Время:\t{times[5]} s \tVS\t{times[11]} s")

    # Графики
    wine_times = [times[i] for i in range(0, len(times), 2)]
    weapons_times = [times[i] for i in range(1, len(times), 2)]

    wine_intra_dist = [intra_dist[i] for i in range(0, len(intra_dist), 2)]
    weapons_intra_dist = [intra_dist[i] for i in range(1, len(intra_dist), 2)]

    wine_inter_dist = [inter_dist[i] for i in range(0, len(inter_dist), 2)]
    weapons_inter_dist = [inter_dist[i] for i in range(1, len(inter_dist), 2)]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # График для времени
    x = ['Hierarchical', 'EM', 'DBSCAN']
    width = 0.2
    index = np.arange(len(x))

    axs[0].bar(index - width*1.5, wine_times[:3], width, color='b', label='Wine (мое)')
    axs[0].bar(index - width/2, weapons_times[:3], width, color='c', label='Weapons (мое)')
    axs[0].bar(index + width/2, wine_times[3:], width, color='g', label='Wine (sklearn)')
    axs[0].bar(index + width*1.5, weapons_times[3:], width, color='m', label='Weapons (sklearn)')

    axs[0].set_xticks(index)
    axs[0].set_xticklabels(x)
    axs[0].set_ylabel('Время выполнения')
    axs[0].set_title('Время кластеризации')
    axs[0].legend()

    axs[0].set_yticks(np.arange(0, 1.1, 0.1))
    axs[0].set_ylim([0, 1.1])

    # График для внутрикластерного расстояния
    axs[1].bar(index - width*1.5, wine_intra_dist[:3], width, color='b', label='Wine (мое)')
    axs[1].bar(index - width/2, weapons_intra_dist[:3], width, color='c', label='Weapons (мое)')
    axs[1].bar(index + width/2, wine_intra_dist[3:], width, color='g', label='Wine (sklearn)')
    axs[1].bar(index + width*1.5, weapons_intra_dist[3:], width, color='m', label='Weapons (sklearn)')

    axs[1].set_xticks(index)
    axs[1].set_xticklabels(x)
    axs[1].set_ylabel('Расстояние')
    axs[1].set_title('Внутрикластерное расстояние')
    axs[1].legend()

    # График для межкластерного расстояния
    axs[2].bar(index - width*1.5, wine_inter_dist[:3], width, color='b', label='Wine (мое)')
    axs[2].bar(index - width/2, weapons_inter_dist[:3], width, color='c', label='Weapons (мое)')
    axs[2].bar(index + width/2, wine_inter_dist[3:], width, color='g', label='Wine (sklearn)')
    axs[2].bar(index + width*1.5, weapons_inter_dist[3:], width, color='m', label='Weapons (sklearn)')

    axs[2].set_xticks(index)
    axs[2].set_xticklabels(x)
    axs[2].set_ylabel('Расстояние')
    axs[2].set_title('Межкластерное расстояние')
    axs[2].legend()

    plt.suptitle('Производительность алгоритмов кластеризации', fontsize=16)
    plt.tight_layout()
    plt.show()