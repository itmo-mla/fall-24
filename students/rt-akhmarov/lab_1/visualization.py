import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from itertools import combinations

def plot_least_correlated_pairs(X, responsibilities, n_components):
    n_features = X.shape[1]
    correlation_matrix = np.corrcoef(X, rowvar=False)
    pairs = list(combinations(range(n_features), 2))

    pair_correlation = [(i, j, abs(correlation_matrix[i, j])) for i, j in pairs]
    pair_correlation.sort(key=lambda x: x[2])  

    least_correlated_pairs = pair_correlation[:6]

    cluster_assignments = np.argmax(responsibilities, axis=1)
    colors = sns.color_palette("husl", n_components)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    for idx, (i, j, _) in enumerate(least_correlated_pairs):
        for cluster in range(n_components):
            cluster_points = X[cluster_assignments == cluster]
            axes[idx].scatter(cluster_points[:, i], cluster_points[:, j], label=f"Cluster {cluster}", c=colors[cluster])

        axes[idx].set_title(f"Feature {i} vs Feature {j}")
        axes[idx].set_xlabel(f"Feature {i}")
        axes[idx].set_ylabel(f"Feature {j}")
        axes[idx].legend()

    plt.tight_layout()
    plt.show()


def plot_pca_clustering(X, y, responsibilities, n_components=2):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    cluster_assignments = np.argmax(responsibilities, axis=1)
    colors = sns.color_palette("husl", n_components)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.7, cmap='jet')
    plt.title("PCA Before Clustering")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    print(n_components)
    plt.subplot(1, 2, 2)
    for cluster in range(n_components):
        cluster_points = X_pca[cluster_assignments == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}", c=colors[cluster])

    plt.title("PCA After Clustering")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_likelihood(log_likelihood):
    plt.figure(figsize=(12, 6))
    plt.plot(log_likelihood, c='red', alpha=0.7)
    plt.title("Log-Likelihood dymanic")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")