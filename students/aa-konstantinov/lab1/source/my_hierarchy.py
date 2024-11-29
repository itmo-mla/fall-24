import numpy as np
import pandas as pd
import random

def euclidean(w, s):
    return np.sqrt(np.sum(np.square(w - s)))

def ward(w, s):
    W = len(w)
    S = len(s)
    return ((W * S) / (W + S)) * np.square(euclidean(np.mean(w, axis=0), np.mean(s, axis=0)))

class Hierarchy_:
    def __init__(self):
        pass

    def fclust(self, linkage, num_clusters: int):
        linkage = np.array(linkage)
    
        N = linkage.shape[0] + 1

        labels = np.zeros(N, dtype=int) 

        clusters = {}
        if num_clusters == 1:
            return labels       
        linkage_cl = linkage[:-(num_clusters - 1), :2]
        for index, clust in enumerate(linkage_cl):
            cluster_id = index + N
            new_cluster = []
            for c in clust:
                if c < N:
                    c = int(c)
                    new_cluster.append(c)
                else:
                    new_cluster.extend(clusters[c])
                    del clusters[c]
            clusters[cluster_id] = new_cluster

        current_label = 1
        for cluster_indices in clusters.values():
            labels[cluster_indices] = current_label
            current_label += 1

        return labels


 
    def linkage(self, data):
        data = np.array(data)
        clusters = {i: [i] for i in range(len(data))}  
        new_clusters = {}
        distances = {}
        linkage = []
        current_cluster_id = len(data)
    
        for i in clusters:
            for j in clusters:
                if i < j:
                    w =  ward(data[clusters[i]], data[clusters[j]])
                    distances[(i, j)] =  w
                       
        
        while len(clusters) > 1 :
            min_dist = min(zip(distances.values(), distances.keys()))
            i, j = min_dist[1]
            min_dist = min_dist[0]
            clusters[current_cluster_id] = clusters[i] + clusters[j] 
            linkage.append([i, j, min_dist, len(data[clusters[i]]) + len(data[clusters[j]]) ]) 
            new_clusters[current_cluster_id]  =  clusters[i] + clusters[j] 
            del clusters[j]    
            del clusters[i]   
            distances = {key: val for key, val in distances.items() if j not in key and i not in key}
            
            for k in clusters:
                if k != current_cluster_id:
                    w = ward(data[clusters[current_cluster_id]], data[clusters[k]])
                    if k < current_cluster_id:
                        distances[(k, current_cluster_id)] = w
                    else:
                        distances[(current_cluster_id, k)] = w

            current_cluster_id += 1

        return linkage


class DBSCAN_:
    def amount(self, index):
        near_index = [i  for i, value in enumerate(self.data)   if i != index and euclidean(value,self.data[index] ) < self.eps ]
        return near_index
    
    def __init__(self, data: pd.DataFrame, eps: float, m: int) -> None:
        self.data = np.array(data)
        self.eps = eps
        self.m = m
        self.U = list(range(len(data)))
        self.noise = []
        self.clusters = {}

    def fit(self):
        a = 0
        while self.U:
            random_dot = random.choice(self.U)
            clust = self.amount(random_dot)
            if len(clust) < self.m:
                self.noise.append(random_dot)
                self.U.remove(random_dot)
            else:
                a += 1
                self.clusters[a] = clust + [random_dot]
                for noise in self.clusters[a]:
                    if noise in self.U:
                        self.U.remove(noise)
                        noise_cl = self.amount(noise)
                        if len(noise_cl) >= self.m:
                            self.clusters[a] = self.clusters[a] + noise_cl
                        for near in noise_cl:
                            if near not in self.clusters[a]:
                                self.clusters[a].append(near)
                                if near in self.U:
                                        self.U.remove(near)
                                
    def predict(self):
        labels = np.full(len(self.data), -1)  
        for index, cluster in self.clusters.items():
            labels[cluster] = index 
        return labels



class EM_:
    def __init__(self, data: np.array, n: int, max_iters: int = 100):
        self.data = np.array(data)
        self.n = n
        self.max_iters = max_iters
        self.probabilities = np.ones(n) / n
        indices = np.random.choice(self.data.shape[0], size=n, replace=False)
        self.mu = self.data[indices]
        self.cov_mat = np.array([np.eye(self.data.shape[1]) for _ in range(n)])
        
    def fit(self):
        last_iter_class = None
        for _ in range(self.max_iters):
            self.expectation = self.E()
            self.maximization()
            if last_iter_class == [np.argmax(i) for i in self.expectation]:
                break
            last_iter_class = [np.argmax(i) for i in self.expectation]
        return last_iter_class


    def Gauss_density_ver(self, each_row, k):
        inv_cov = np.linalg.inv(self.cov_mat[k] + 1e-5 * np.eye(self.data.shape[1]))
        changes = each_row - self.mu[k]
        p2_y = np.dot(changes.T, np.dot(inv_cov, changes))
        det_cov = np.linalg.det(self.cov_mat[k]) #TODO 1e-5
        first_part = ((2 * np.pi) ** ( self.data.shape[1] / 2)) * np.sqrt(det_cov)
        density = np.exp(-0.5 * p2_y) / (first_part + 1e-5)
        return density

    def E(self):
        exp_res = np.zeros((self.data.shape[0], self.n))
        for cnum_cluster in range(self.n):
            densities = np.array([self.Gauss_density_ver(i, cnum_cluster) for i in self.data])
            exp_res[:, cnum_cluster] = self.probabilities[cnum_cluster] * densities
        sum_exp = exp_res.sum(axis=1, keepdims=True)
        sum_exp[sum_exp == 0] = 1e-5
        return exp_res / sum_exp

    def maximization(self):
        mu = np.zeros_like(self.mu)
        probabilities = np.zeros(self.n)
        cov_mat = np.zeros_like(self.cov_mat)

        for i in range(self.n):
            E_iter = self.expectation[:, i]
            E_sum = E_iter.sum()
            probabilities[i] = E_sum / len(self.data)
            mu[i] = np.dot(E_iter, self.data) / (E_sum*len(self.data))
            changes = self.data - mu[i]
            cov_mat[i] = np.dot(E_iter * changes.T, changes) / (E_sum*len(self.data))
            cov_mat[i] += 1e-5 * np.eye(self.data.shape[1])

      
        self.probabilities = probabilities
        self.mu = mu
        self.cov_mat = cov_mat

    




