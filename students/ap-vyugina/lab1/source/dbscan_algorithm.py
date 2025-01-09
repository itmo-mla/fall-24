import numpy as np
from scipy.spatial.distance import cdist
import random


def dbscan(X, epsilon, min_samples):
    distances = cdist(X, X)
    np.fill_diagonal(distances, np.inf)
    
    N = len(X)
    
    clusters = np.empty((N, ))
    clusters[:] = np.nan
    roles = clusters.copy()
    
    current_cluster_id = 0
    iter_counter = 0
    
    idxs_range = np.arange(N, dtype=np.uint32)
    
    while np.any(np.isnan(roles)):
        unassigned_idxs = np.argwhere(np.isnan(clusters)).reshape((-1, ))
        random_idx = random.choice(unassigned_idxs)

        d = distances[random_idx]
        neighbors = idxs_range[distances[random_idx] < epsilon]
        if len(neighbors) < min_samples:
            roles[random_idx] = -1
            K = [random_idx]
        else:
            roles[random_idx] = 1
            clusters[random_idx] = current_cluster_id
            
            K = np.append(neighbors, random_idx)
            while np.any(np.isnan(clusters[K])):
                idx2check = K[np.isnan(clusters[K])]
                idx = random.choice(idx2check)
                k_n = idxs_range[distances[idx] < epsilon]
                if len(k_n) >= min_samples:
                    roles[idx] = 1
                    K = np.union1d(K, k_n)
                else: 
                    roles[idx] = 0
                clusters[idx] = current_cluster_id
                # K = K.astype(np.uint32)
            current_cluster_id += 1

        distances[K] = np.inf
        
        iter_counter += 1
        if iter_counter == 500: break
    clusters[np.isnan(clusters)] = -1
    return roles, clusters