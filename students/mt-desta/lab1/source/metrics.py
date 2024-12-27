import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import time 
from my_hierarchy import my_HierarchicalClustering
from em import my_GaussianMixtureEM
from dbscan import my_dbscan
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, AgglomerativeClustering

class Metrics:
    def get_clusters(data, labels):
        count = np.unique(labels).shape[0]
        cluster = [[] for _ in range(count)]

        for i in range(labels.shape[0]):
            if labels[i] != -1:
                cluster[labels[i]].append(data.iloc[i])

        return cluster

    def mean_intracluster_distance_hier(lables):
        intracluster_dists = []
        for i in range(len(lables)):
            cluster = np.asarray(lables[i])
            cluster_dists = np.linalg.norm(cluster - cluster.mean(axis=0))
            intracluster_dists.append(np.mean(cluster_dists))
        
        return np.mean(intracluster_dists)

    def mean_intercluster_distance_hier(lables):
        interclusters_dists = []
        for i in range(len(lables)):
            for j in range(len(lables)):
                if i != j:
                    for k in range(len(lables[i])):
                        for k2 in range(len(lables[j])):
                            interclusters_dists.append(np.linalg.norm(np.asarray(lables[i][k]) - np.asarray(lables[j][k2])))

        return np.mean(interclusters_dists)

    def mean_intracluster_distance(self,x: pd.DataFrame, labels):
        X = x.to_numpy()
        d = self._labels_to_dict(X, labels)
        sum_dist = 0
        n_pairs = 0
        for points in d.values():
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    sum_dist += np.linalg.norm(points[i] - points[j])
                    n_pairs += 1

        return sum_dist / n_pairs

    def mean_intercluster_distance(self,x: pd.DataFrame, labels):
        X = x.to_numpy()
        d = self._labels_to_dict(X, labels)
        sum_dist = 0
        n_pairs = 0
        cluster_values = list(d.values())
        for i in range(len(cluster_values)):
            for j in range(i + 1, len(cluster_values)):
                for cluster_i_point in cluster_values[i]:
                    for cluster_j_point in cluster_values[j]:
                        sum_dist += np.linalg.norm(cluster_i_point - cluster_j_point)
                        n_pairs += 1

        return sum_dist / n_pairs

    def _labels_to_dict(X: np.ndarray, labels):
        d = {}
        for i, label in enumerate(labels):
            if label not in d:
                d[label] = []
            d[label].append(X[i])
        return d
    
    #clustering speed
    def measure_ref_em_clustering_speed(data,n_components):
        start_time = time.time()

        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit_predict(data)

        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        return elapsed_time

    def measure_ref_DBSCAN_clustering_speed(data,eps=0.5, min_samples = 5):
        # Start the timer
        start_time = time.perf_counter()
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(data)
        dbscan.fit_predict(data)
        
        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        
        return elapsed_time
    
    def measure_ref_Hierar_clustering_speed(data,n_clusters):
        # Start the timer
        start_time = time.perf_counter()
        
        agg = AgglomerativeClustering(n_clusters,linkage='ward')
        agg.fit_predict(data)
        
        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        
        return elapsed_time
    
    #clustering speed
    
    def measure_DBSCAN_clustering_speed(data,eps=0.5, min_samples = 5):
        # Start the timer
        start_time = time.perf_counter()
        
        # Perform DBSCAN clustering
        dbscan = my_dbscan(eps=eps, min_samples=min_samples)
        dbscan.fit(data)
        dbscan.fit_predict(data)
        
        # Stop the timer
        end_time = time.perf_counter()
        
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        
        return elapsed_time
    
    def measure_em_clustering_speed(data,n_components, max_iter):
        start_time = time.perf_counter()

        gmm = my_GaussianMixtureEM(n_components=n_components, max_iter=max_iter)
        gmm.fit(data)
        gmm.fit_predict(data)

        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        
        return elapsed_time
    
    def measure_hierar_clustering_speed(data,n_clusters):
        start_time = time.perf_counter()
        
        h = my_HierarchicalClustering(n_clusters,'ward')
        h.fit(data)

        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        
        return elapsed_time


