import numpy as np
import pandas as pd
import time
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import distance


def initialize_parameters(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    means = kmeans.fit(X).cluster_centers_
    covariances = [np.cov(X.T) + 1e-4 * np.eye(X.shape[1]) for _ in range(n_clusters)]
    weights = np.ones(n_clusters) / n_clusters
    return means, covariances, weights


def e_step(X, means, covariances, weights, n_clusters):
    n_samples = X.shape[0]
    probabilities = np.zeros((n_samples, n_clusters))

    for k in range(n_clusters):
        probabilities[:, k] = weights[k] * multivariate_normal.pdf(
            X, mean=means[k], cov=covariances[k], allow_singular=True
        )

    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def m_step(X, probabilities, n_clusters):
    N_k = probabilities.sum(axis=0)
    new_means = np.dot(probabilities.T, X) / N_k[:, np.newaxis]
    new_covariances = []

    for k in range(n_clusters):
        if N_k[k] == 0:
            new_covariances.append(np.eye(X.shape[1]))
        else:
            diff = X - new_means[k]
            cov_k = np.dot(probabilities[:, k] * diff.T, diff) / N_k[k]
            cov_k += 1e-4 * np.eye(X.shape[1])
            new_covariances.append(cov_k)

    new_weights = N_k / X.shape[0]
    return new_means, new_covariances, new_weights


def log_likelihood(X, means, covariances, weights, n_clusters):
    log_likelihood = 0
    for k in range(n_clusters):
        log_likelihood += weights[k] * multivariate_normal.pdf(
            X, mean=means[k], cov=covariances[k], allow_singular=True
        )
    return np.sum(np.log(log_likelihood))


def fit_em(X, n_clusters=2, max_iter=100, tol=1e-3):
    means, covariances, weights = initialize_parameters(X, n_clusters)
    prev_log_likelihood = None

    for _ in range(max_iter):
        probabilities = e_step(X, means, covariances, weights, n_clusters)
        new_means, new_covariances, new_weights = m_step(X, probabilities, n_clusters)

        current_log_likelihood = log_likelihood(X, new_means, new_covariances, new_weights, n_clusters)

        if prev_log_likelihood is not None and abs(current_log_likelihood - prev_log_likelihood) < tol:
            break

        means, covariances, weights = new_means, new_covariances, new_weights
        prev_log_likelihood = current_log_likelihood

    labels = np.argmax(probabilities, axis=1)
    return labels


# ========================
# Mall_Customers dataset
print("Mall_Customers dataset")
df1 = pd.read_csv("datasets/Mall_Customers.csv")
df1 = df1.drop("CustomerID", axis=1)
df1['Genre'] = df1['Genre'].replace({'Female': 0, 'Male': 1})

n_clusters = 5
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df1.values)
start_time = time.time()
labels_custom_em = fit_em(X_scaled, n_clusters=n_clusters, max_iter=1000)
end_time = time.time()
execution_time = end_time - start_time

intra_dist = distance.calculate_intra_cluster_distances_lib(df1, labels_custom_em)
inter_dist = distance.calculate_inter_cluster_distances_lib(df1, labels_custom_em)

print(f"Время кластеризации реализованным алгоритмом: {execution_time} секунд")
print(f"Среднее внутрикластерное расстояние: {intra_dist}")
print(f"Среднее межкластерное расстояние: {inter_dist}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_custom_em, s=40, cmap='viridis')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Custom EM Clustering (Mall_Customers)')
plt.savefig('img/em/1_manual_clustering.png')
plt.close()

sk_em = GaussianMixture(n_components=n_clusters, covariance_type='full')
start_time = time.time()
sk_em_labels = sk_em.fit_predict(X_scaled)
end_time = time.time()
execution_time = end_time - start_time

intra_dist = distance.calculate_intra_cluster_distances_lib(df1, sk_em_labels)
inter_dist = distance.calculate_inter_cluster_distances_lib(df1, sk_em_labels)

print(f"Время кластеризации библиотечным алгоритмом: {execution_time} секунд")
print(f"Среднее внутрикластерное расстояние: {intra_dist}")
print(f"Среднее межкластерное расстояние: {inter_dist}")

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=sk_em_labels, s=40, cmap='viridis')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Library EM Clustering (Mall_Customers)')
plt.savefig('img/em/1_lib_clustering.png')
plt.close()

# ============================
# wine-clustering dataset
print()
print("wine-clustering dataset")
df2 = pd.read_csv("datasets/wine-clustering.csv")
X_scaled = scaler.fit_transform(df2.values)

n_clusters = 2
start_time = time.time()
labels_custom_em = fit_em(X_scaled, n_clusters=n_clusters, max_iter=1000)
end_time = time.time()
execution_time = end_time - start_time

intra_dist = distance.calculate_intra_cluster_distances_lib(df2, labels_custom_em)
inter_dist = distance.calculate_inter_cluster_distances_lib(df2, labels_custom_em)

print(f"Время кластеризации реализованным алгоритмом: {execution_time} секунд")
print(f"Среднее внутрикластерное расстояние: {intra_dist}")
print(f"Среднее межкластерное расстояние: {inter_dist}")

X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_custom_em, s=40, cmap='plasma')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Custom EM Clustering (Wine)')
plt.savefig('img/em/2_manual_clustering.png')
plt.close()

sk_em = GaussianMixture(n_components=n_clusters, covariance_type='full')
start_time = time.time()
sk_em_labels = sk_em.fit_predict(X_scaled)
end_time = time.time()
execution_time = end_time - start_time

intra_dist = distance.calculate_intra_cluster_distances_lib(df2, sk_em_labels)
inter_dist = distance.calculate_inter_cluster_distances_lib(df2, sk_em_labels)

print(f"Время кластеризации библиотечным алгоритмом: {execution_time} секунд")
print(f"Среднее внутрикластерное расстояние: {intra_dist}")
print(f"Среднее межкластерное расстояние: {inter_dist}")

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=sk_em_labels, s=40, cmap='plasma')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Library EM Clustering (Wine)')
plt.savefig('img/em/2_lib_clustering.png')
plt.close()
