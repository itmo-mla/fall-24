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


def multivariate_gaussian(x, mean, cov):
    k = len(mean)
    norm_factor = (2 * np.pi) ** (k / 2) * np.linalg.det(cov) ** 0.5
    return np.exp(-0.5 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean)) / (norm_factor + 1e-10)

def expectation(X, parameters):
    # пересчитываем принадлежность точек к кластерам
    weights, means, covs = parameters
    n_clusters = len(weights)
    n_samples = X.shape[0]
    responsibilities = np.zeros((n_samples, n_clusters))
    
    for k in range(n_clusters):
        for i in range(n_samples):
            responsibilities[i, k] = weights[k] * multivariate_gaussian(X[i], means[k], covs[k])
    
    responsibilities /= (responsibilities.sum(axis=1, keepdims=True) + 1e-10)
    return responsibilities

def maximization(X, responsibilities):
    # пересчитываем центры кластеров
    n_samples, n_features = X.shape
    n_clusters = responsibilities.shape[1]
    
    weights = np.zeros(n_clusters)
    means = np.zeros((n_clusters, n_features))
    covs = np.zeros((n_clusters, n_features, n_features))
    
    for k in range(n_clusters):
        weight = responsibilities[:, k].sum()
        mean = np.dot(responsibilities[:, k], X) / weight
        cov = (X - mean).T @ np.diag(responsibilities[:, k]) @ (X - mean) / weight
        
        weights[k] = weight / n_samples
        means[k] = mean
        covs[k] = cov + np.eye(n_features) * 1e-6
    
    return weights, means, covs

def em(X, n_clusters, n_iter=100):
    n_samples, n_features = X.shape
    
    initial_weights = np.ones(n_clusters) / n_clusters
    initial_means = X[np.random.choice(n_samples, n_clusters, replace=False)]
    initial_covs = np.array([np.cov(X, rowvar=False) for _ in range(n_clusters)])
    
    parameters = (initial_weights, initial_means, initial_covs)
    
    for _ in range(n_iter):
        responsibilities = expectation(X, parameters)
        parameters = maximization(X, responsibilities)
    
    return parameters + (responsibilities,)