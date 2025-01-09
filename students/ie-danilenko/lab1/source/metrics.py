import numpy as np
from hierarhy import hierarhy_alg
from sklearn.cluster import AgglomerativeClustering
from em import em
from sklearn.mixture import GaussianMixture
from dbscan import dbscan
from sklearn.cluster import DBSCAN
import pandas as pd
from time import time
import matplotlib.pyplot as plt

def get_intra_dist(clusters):
    intracluster_dists = []
    for i in range(len(clusters)):
        cluster = np.asarray(clusters[i])
        cluster_dists = np.linalg.norm(cluster - cluster.mean(axis=0))
        intracluster_dists.append(np.mean(cluster_dists))

    return np.mean(intracluster_dists)

def get_inter_dist(clusters):
    interclusters_dists = []
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if i != j:
                for k in range(len(clusters[i])):
                    for k2 in range(len(clusters[j])):
                        interclusters_dists.append(np.linalg.norm(np.asarray(clusters[i][k]) - np.asarray(clusters[j][k2])))

    return np.mean(interclusters_dists)

def get_clusters(data, labels, dbscan=False):
    count = np.unique(labels).shape[0]
    if dbscan:
        if np.min(labels) == -1:
            count -= 1
    cluster = [[] for _ in range(count)]

    for i in range(labels.shape[0]):
        if labels[i] != -1:
            cluster[labels[i]].append(data.iloc[i])

    return cluster

if __name__ == "__main__":
    data_wine = pd.read_csv("datasets/wine-clustering.csv")
    data_crimes = pd.read_csv("datasets/crimes.csv", index_col='Unnamed: 0')
    del data_crimes['State']

    times = []
    intra_dist = []
    inter_dist = []

    # my
    start_time = time()
    clusters, _ = hierarhy_alg(data_wine, 3, 'ward')
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(get_intra_dist(clusters))
    inter_dist.append(get_inter_dist(clusters))

    start_time = time()
    clusters, _ = hierarhy_alg(data_crimes, 5, 'ward')
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(get_intra_dist(clusters))
    inter_dist.append(get_inter_dist(clusters))

    start_time = time()
    clusters = em(data_wine, 3)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(get_intra_dist(clusters))
    inter_dist.append(get_inter_dist(clusters))

    start_time = time()
    clusters = em(data_crimes, 5)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(get_intra_dist(clusters))
    inter_dist.append(get_inter_dist(clusters))

    start_time = time()
    clusters, _ = dbscan(data_wine, 40, 6)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(get_intra_dist(clusters))
    inter_dist.append(get_inter_dist(clusters))

    start_time = time()
    clusters, _ = dbscan(data_crimes, 200, 2)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(get_intra_dist(clusters))
    inter_dist.append(get_inter_dist(clusters))

    # etalon

    #hierarhy
    agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
    start_time = time()
    labels = agg_clustering.fit_predict(data_wine)
    clusters = get_clusters(data_wine, labels)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(get_intra_dist(clusters))
    inter_dist.append(get_inter_dist(clusters))

    agg_clustering = AgglomerativeClustering(n_clusters=5, linkage='ward')
    start_time = time()
    labels = agg_clustering.fit_predict(data_crimes)
    clusters = get_clusters(data_crimes, labels)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(get_intra_dist(clusters))
    inter_dist.append(get_inter_dist(clusters))

    # em
    gmm = GaussianMixture(n_components=3)
    start_time = time()
    gmm.fit(data_wine)
    labels = gmm.predict(data_wine)
    clusters = get_clusters(data_wine, labels)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(get_intra_dist(clusters))
    inter_dist.append(get_inter_dist(clusters))

    gmm = GaussianMixture(n_components=5)
    start_time = time()
    gmm.fit(data_crimes)
    labels = gmm.predict(data_crimes)
    clusters = get_clusters(data_crimes, labels)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(get_intra_dist(clusters))
    inter_dist.append(get_inter_dist(clusters))

    # dbscan
    db = DBSCAN(eps=40, min_samples=6)
    start_time = time()
    db.fit(data_wine)
    clusters = get_clusters(data_wine, db.labels_, True)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(get_intra_dist(clusters))
    inter_dist.append(get_inter_dist(clusters))

    db = DBSCAN(eps=200, min_samples=2)
    start_time = time()
    db.fit(data_crimes)
    clusters = get_clusters(data_crimes, db.labels_, True)
    end_time = time()
    times.append(end_time-start_time)
    intra_dist.append(get_intra_dist(clusters))
    inter_dist.append(get_inter_dist(clusters))

    print("task 4")
    print("HIERARHY ALG (MY)")

    print("\tWine dataset:")
    print(f"\t\tВремя: {times[0]} s")
    print(f"\t\tСреднее внутрикластерное расстояние: {intra_dist[0]}")
    print(f"\t\tСреднее межкластерное расстояние: {inter_dist[0]}")

    print("\tCrime dataset:")
    print(f"\t\tВремя: {times[1]} s")
    print(f"\t\tСреднее внутрикластерное расстояние: {intra_dist[1]}")
    print(f"\t\tСреднее межкластерное расстояние: {inter_dist[1]}")

    print("EM ALG (MY)")
    
    print("\tWine dataset:")
    print(f"\t\tВремя: {times[2]} s")
    print(f"\t\tСреднее внутрикластерное расстояние: {intra_dist[2]}")
    print(f"\t\tСреднее межкластерное расстояние: {inter_dist[2]}")

    print("\tCrime dataset:")
    print(f"\t\tВремя: {times[3]} s")
    print(f"\t\tСреднее внутрикластерное расстояние: {intra_dist[3]}")
    print(f"\t\tСреднее межкластерное расстояние: {inter_dist[3]}")

    print("DBSCAN ALG (MY)")

    print("\tWine dataset:")
    print(f"\t\tВремя: {times[4]} s")
    print(f"\t\tСреднее внутрикластерное расстояние: {intra_dist[4]}")
    print(f"\t\tСреднее межкластерное расстояние: {inter_dist[4]}")

    print("\tCrime dataset:")
    print(f"\t\tВремя: {times[5]} s")
    print(f"\t\tСреднее внутрикластерное расстояние: {intra_dist[5]}")
    print(f"\t\tСреднее межкластерное расстояние: {inter_dist[5]}")


    print("task 5")
    print("HIERARHY ALG (SKLEARN)")

    print("\tWine dataset:")
    print(f"\t\tВремя: {times[6]} s")
    print(f"\t\tСреднее внутрикластерное расстояние: {intra_dist[6]}")
    print(f"\t\tСреднее межкластерное расстояние: {inter_dist[6]}")

    print("\tCrime dataset:")
    print(f"\t\tВремя: {times[7]} s")
    print(f"\t\tСреднее внутрикластерное расстояние: {intra_dist[7]}")
    print(f"\t\tСреднее межкластерное расстояние: {inter_dist[7]}")

    print("EM ALG (SKLEARN)")
    
    print("\tWine dataset:")
    print(f"\t\tВремя: {times[8]} s")
    print(f"\t\tСреднее внутрикластерное расстояние: {intra_dist[8]}")
    print(f"\t\tСреднее межкластерное расстояние: {inter_dist[8]}")

    print("\tCrime dataset:")
    print(f"\t\tВремя: {times[9]} s")
    print(f"\t\tСреднее внутрикластерное расстояние: {intra_dist[9]}")
    print(f"\t\tСреднее межкластерное расстояние: {inter_dist[9]}")

    print("DBSCAN ALG (SKLEARN)")

    print("\tWine dataset:")
    print(f"\t\tВремя: {times[10]} s")
    print(f"\t\tСреднее внутрикластерное расстояние: {intra_dist[10]}")
    print(f"\t\tСреднее межкластерное расстояние: {inter_dist[10]}")

    print("\tCrime dataset:")
    print(f"\t\tВремя: {times[11]} s")
    print(f"\t\tСреднее внутрикластерное расстояние: {intra_dist[11]}")
    print(f"\t\tСреднее межкластерное расстояние: {inter_dist[11]}")
    

    # task 6
    alg = ['Hierarhy', 'EM', 'DBSCAN']
    x = np.arange(len(alg))
    width = 0.1

    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    bar_dict = {
        "My(Wine)" : (times[0], times[2], times[4]),
        "My(Crime)" : (times[1], times[3], times[5]),
        "Etalon(Wine)": (times[6], times[8], times[10]),
        "Etalon(Crime)": (times[7], times[9], times[11])
    }

    for attribute, measurement in bar_dict.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Время (s)')
    ax.set_title('Время работы алгоритма')
    ax.set_xticks(x + width, alg)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, 1.5)

    plt.show()

    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    bar_dict = {
        "My(Wine)" : (intra_dist[0], intra_dist[2], intra_dist[4]),
        "My(Crime)" : (intra_dist[1], intra_dist[3], intra_dist[5]),
        "Etalon(Wine)": (intra_dist[6], intra_dist[8], intra_dist[10]),
        "Etalon(Crime)": (intra_dist[7], intra_dist[9], intra_dist[11])
    }

    for attribute, measurement in bar_dict.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Расстояние')
    ax.set_title('Среднее внутрикластерное расстояние')
    ax.set_xticks(x + width, alg)
    ax.legend(loc='upper left', ncols=4)

    plt.show()

    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    bar_dict = {
        "My(Wine)" : (inter_dist[0], inter_dist[2], inter_dist[4]),
        "My(Crime)" : (inter_dist[1], inter_dist[3], inter_dist[5]),
        "Etalon(Wine)": (inter_dist[6], inter_dist[8], inter_dist[10]),
        "Etalon(Crime)": (inter_dist[7], inter_dist[9], inter_dist[11])
    }

    for attribute, measurement in bar_dict.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Расстояние')
    ax.set_title('Среднее межкластерное расстояние')
    ax.set_xticks(x + width, alg)
    ax.legend(loc='upper left', ncols=4)

    plt.show()