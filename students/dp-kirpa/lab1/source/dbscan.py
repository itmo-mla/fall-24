import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import ast
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import seaborn as sns
import plotly.express as px
from typing import Dict
from itertools import chain as chain_iters
from enum import Enum
import uuid
import numpy as np
from scipy.spatial.distance import euclidean
from time import perf_counter
from scipy.cluster.hierarchy import dendrogram


def dbscan(X, eps, min_samples):
    n_samples = X.shape[0]
    labels = -np.ones(n_samples)
    cluster_id = 0
    
    def find_neighbors(i):
        neighbors = []
        for j in range(n_samples):
            if np.linalg.norm(X[i] - X[j]) < eps:
                neighbors.append(j)
        return neighbors
    
    for i in range(n_samples):
        if labels[i] != -1:
            continue
            
        neighbors = find_neighbors(i)
        
        if len(neighbors) < min_samples:
            labels[i] = -1
            continue
        
        labels[i] = cluster_id
        
        seeds = set(neighbors)
        seeds.remove(i)
        
        while seeds:
            current_point = seeds.pop()
            
            if labels[current_point] != -1:
                continue
            
            labels[current_point] = cluster_id
            
            current_neighbors = find_neighbors(current_point)
            
            if len(current_neighbors) >= min_samples:
                seeds.update(current_neighbors)
        
        cluster_id += 1
    
    return labels
