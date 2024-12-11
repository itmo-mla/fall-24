import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN as SklearnDBSCAN
import matplotlib.pyplot as plt
import distance

def dbscan_manual(X, eps, min_samples):
    n_samples = X.shape[0]
    labels = np.full(n_samples, -2) #Точка еще не посещена
    cluster_id = 0

    def region_query(point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= eps)[0]

    def expand_cluster(point_idx, neighbors):
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -2:  # Только если точка еще не посещена
                labels[neighbor_idx] = cluster_id
                new_neighbors = region_query(neighbor_idx)
                if len(new_neighbors) >= min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))
            elif labels[neighbor_idx] == -1:  # Если точка была шумом
                labels[neighbor_idx] = cluster_id
            i += 1

    for point_idx in range(n_samples):
        if labels[point_idx] != -2:
            continue
        neighbors = region_query(point_idx)
        if len(neighbors) < min_samples:
            labels[point_idx] = -1
        else:
            expand_cluster(point_idx, neighbors)
            cluster_id += 1

    return labels


# ========================
# Mall_Customers dataset
print("Mall_Customers dataset")
df1 = pd.read_csv("datasets/Mall_Customers.csv")
df1 = df1.drop("CustomerID", axis=1)
df1['Genre'] = df1['Genre'].replace({'Female': 0, 'Male': 1})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df1)

eps = 0.9
min_samples = 5

start_time = time.time()
labels_manual = dbscan_manual(X_scaled, eps=eps, min_samples=min_samples)
end_time = time.time()
execution_time = end_time - start_time

intra_dist = distance.calculate_intra_cluster_distances_lib(df1, labels_manual)
inter_dist = distance.calculate_inter_cluster_distances_lib(df1, labels_manual)

print(f"Время кластеризации реализованным DBSCAN: {execution_time} секунд")
print(f"Среднее внутрикластерное расстояние: {intra_dist}")
print(f"Среднее межкластерное расстояние: {inter_dist}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_manual, s=40, cmap='viridis')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Custom DBSCAN Clustering (Mall_Customers)')
plt.savefig('img/dbscan/1_manual_clustering.png')
plt.close()




sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_samples)
start_time = time.time()
labels_sklearn = sklearn_dbscan.fit_predict(X_scaled)
end_time = time.time()
execution_time = end_time - start_time

intra_dist = distance.calculate_intra_cluster_distances_lib(df1, labels_sklearn)
inter_dist = distance.calculate_inter_cluster_distances_lib(df1, labels_sklearn)

print(f"Время кластеризации библиотечным DBSCAN: {execution_time} секунд")
print(f"Среднее внутрикластерное расстояние: {intra_dist}")
print(f"Среднее межкластерное расстояние: {inter_dist}")


plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_sklearn, s=40, cmap='viridis')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Library DBSCAN Clustering (Mall_Customers)')
plt.savefig('img/dbscan/1_lib_clustering.png')
plt.close()


# ============================
# wine-clustering dataset
print()
print("wine-clustering dataset")
df2 = pd.read_csv("datasets/wine-clustering.csv")
X_scaled = scaler.fit_transform(df2)

start_time = time.time()
labels_manual = dbscan_manual(X_scaled, eps=eps, min_samples=min_samples)
end_time = time.time()
execution_time = end_time - start_time

intra_dist = distance.calculate_intra_cluster_distances_lib(df2, labels_manual)
inter_dist = distance.calculate_inter_cluster_distances_lib(df2, labels_manual)

print(f"Время кластеризации реализованным DBSCAN: {execution_time} секунд")
print(f"Среднее внутрикластерное расстояние: {intra_dist}")
print(f"Среднее межкластерное расстояние: {inter_dist}")

X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_manual, s=40, cmap='plasma')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Custom DBSCAN Clustering (Wine)')
plt.savefig('img/dbscan/2_manual_clustering.png')
plt.close()

start_time = time.time()
labels_sklearn = sklearn_dbscan.fit_predict(X_scaled)
end_time = time.time()
execution_time = end_time - start_time

intra_dist = distance.calculate_intra_cluster_distances_lib(df2, labels_sklearn)
inter_dist = distance.calculate_inter_cluster_distances_lib(df2, labels_sklearn)

print(f"Время кластеризации библиотечным DBSCAN: {execution_time} секунд")
print(f"Среднее внутрикластерное расстояние: {intra_dist}")
print(f"Среднее межкластерное расстояние: {inter_dist}")

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_sklearn, s=40, cmap='plasma')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Library DBSCAN Clustering (Wine)')
plt.savefig('img/dbscan/2_lib_clustering.png')
plt.close()
