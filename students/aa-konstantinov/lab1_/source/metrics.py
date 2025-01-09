import  numpy as np
import time
from scipy.spatial.distance import pdist
from itertools import combinations


def average_in_cluster(data, labels):
    labels = np.array(labels)
    clusters = np.unique(labels)
    mas_of_distances = []
    
    for cluster in clusters:
        cluster_points = data[labels == cluster]
        if len(cluster_points) > 1:
            distances = pdist(cluster_points, metric='euclidean')
            mas_of_distances.append(np.mean(distances))
        else:
            mas_of_distances.append(0)

    return np.mean(mas_of_distances)


def average_in_clusters(data, labels):
    clusters = np.unique(labels)
    mas_of_distances = []
    if len(clusters) > 1:
        for cluster1, cluster2 in combinations(clusters, 2):
            points1 = data[labels == cluster1]
            points2 = data[labels == cluster2]
            if len(points1) == 0 or len(points2) == 0:
                continue
            distances = pdist(np.vstack([points1, points2]), metric='euclidean')
            len1 = len(points1)
            len2 = len(points2)
            inter_dist = distances[:len1 * len2].reshape(len1, len2)
            mas_of_distances.append(inter_dist.mean())
    else:
        return 0
    return np.mean(mas_of_distances)
