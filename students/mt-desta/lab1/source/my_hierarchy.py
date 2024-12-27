import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

class my_HierarchicalClustering:
    def __init__(self,n_clusters,method='min'):
        self.n_clusters = n_clusters
        self.method = method

    def lance_williams(self, r_us, r_vs, r_uv, u_len, v_len, s_len, method='min'):
        """Apply the Lance-Williams formula to compute the new distance between merged clusters."""
        au = av = b = g = 0

        if method == 'min':  # Single linkage (minimum distance)
            au = 0.5
            av = 0.5
            b = 0
            g = -0.5
        elif method == 'max':  # Complete linkage (maximum distance)
            au = 0.5
            av = 0.5
            b = 0
            g = 0.5
        elif method == 'mean':  # Average linkage (mean distance)
            au = u_len / (u_len + v_len)
            av = v_len / (u_len + v_len)
            b = g = 0
        elif method == 'center':  # Centroid linkage
            au = u_len / (u_len + v_len)
            av = v_len / (u_len + v_len)
            b = -au * av
            g = 0
        elif method == 'ward':  # Ward's method
            au = (s_len + u_len) / (s_len + u_len + v_len)
            av = (s_len + v_len) / (s_len + u_len + v_len)
            b = -s_len / (s_len + u_len + v_len)
            g = 0
        else:
            raise Exception("Invalid method. Choose one of: 'min', 'max', 'mean', 'center', 'ward'.")

        return au * r_us + av * r_vs + b * r_uv + g * np.abs(r_us - r_vs)

    def init_clusters(self, n):
        clusters = [[i] for i in range(n)]
        cluster_flags = [True for _ in range(n)]

        return clusters,cluster_flags
    
    def calculate_distance_matrix(self,data,n):
        distance_matrix = np.zeros((n,n))
        
        for i in range(n):
            for j in range(i+1,n):
                dist = np.linalg.norm(data.iloc[i] - data.iloc[j])
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist

        return distance_matrix

    def find_closest_clusters(self,clusters,cluster_flags,distance_matrix):
        min_dist = np.inf
        pair = (0,0)

        for i in range(len(clusters)):
            for j in range(i+1,len(clusters)):
                if cluster_flags[i] and cluster_flags[j]:
                    if distance_matrix[i][j] < min_dist:
                        min_dist = distance_matrix[i][j]
                        pair = (i,j)
        return pair[0], pair[1], min_dist
    
    def merge_clusters(self,clusters,c1,c2):
        return clusters[c1] + clusters[c2]
    
    def update_distance_matrix(self,distance_matrix,clusters,cluster_flags,c1,c2):
        distance_matrix = np.append(distance_matrix,np.zeros((len(clusters) - 1,1)), axis=1)
        distance_matrix = np.append(distance_matrix,np.zeros((1,len(clusters))), axis= 0)

        for i in range(len(clusters) - 1):
            if i != c1 and i != c2:
                if cluster_flags[i]:
                    r_us = distance_matrix[c1][i]
                    r_vs = distance_matrix[c2][i]
                    r_uv = distance_matrix[c1][c2]
                    u_len = len(clusters[c1])
                    v_len = len(clusters[c2])
                    s_len = len(clusters[i])

                    new_d = self.lance_williams(r_us,r_vs,r_uv,u_len,v_len,s_len,method=self.method)
                    distance_matrix[i][-1] = new_d
                    distance_matrix[-1][i] = new_d
    
        return distance_matrix
    
    def perform_clustering(self, distance_matrix, clusters,cluster_flags):
        linkage_matrix = []
        linkage_history = []


        while cluster_flags.count(True) > self.n_clusters:
            c1,c2, min_dist = self.find_closest_clusters(clusters,cluster_flags,distance_matrix)

            newc = self.merge_clusters(clusters,c1,c2)

            linkage_matrix.append([c1,c2,min_dist, len(newc)])

            cluster_flags[c1] = False
            cluster_flags[c2] = False
            cluster_flags.append(True)
            linkage_history.append(deepcopy(cluster_flags))
            clusters.append(newc)
            
            distance_matrix = self.update_distance_matrix(distance_matrix,clusters,cluster_flags,c1,c2)


        return linkage_history,linkage_matrix


    
    def fit(self,data):
        n = data.shape[0]

        clusters ,cluster_flags = self.init_clusters(n)

        distance_matrix = self.calculate_distance_matrix(data,n)

        linkage_history , linkage_matrix = self.perform_clustering(distance_matrix,clusters,cluster_flags)

        self.labels_ = []
        for i in range(len(clusters)):
            if cluster_flags[i]:
                c1 = []
                for j in range(len(clusters[i])):
                    c1.append(data.iloc[clusters[i][j]].to_numpy().tolist())
                self.labels_.append(c1)

        return linkage_history, linkage_matrix
        
    def plot_clusters(self):
        """Plot the clusters."""
        if self.labels_ is None:
            raise ValueError("You must fit the model before plotting.")
        
        plt.figure(figsize=(8, 6))
        
        for i in range(len(self.labels_)):
            plt.scatter(np.array(self.labels_[i])[:, 0], np.array(self.labels_[i])[:, 1])
        
        plt.title(f'Hierarchical Clustering (n_clusters={self.n_clusters}, linkage={self.method})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.show()


def plot_all_linkage_methods(X, n_clusters=2):
    """Plot hierarchical clustering with different linkage methods using unique colors for each cluster."""
    linkage_methods = ['min', 'max', 'mean', 'center', 'ward']
    fig, axs = plt.subplots(1, len(linkage_methods), figsize=(20, 5))
    
    for i, method in enumerate(linkage_methods):
        model = my_HierarchicalClustering(n_clusters=n_clusters, method=method)
        model.fit(X)
        
        for j in range(len(model.labels_)):
            axs[i].scatter(np.array(model.labels_[j])[:, 0], np.array(model.labels_[j])[:, 1])

        axs[i].set_title(f'Linkage: {method}')
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()
