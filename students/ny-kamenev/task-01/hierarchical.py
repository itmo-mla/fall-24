import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
import time
from sklearn.cluster import AgglomerativeClustering
import distance


def lance_williams_update(d_uv, d_us, d_vs, method="ward", nu=1, nv=1, ns=1):
    if method == "single":
        alpha_u = alpha_v = 0.5
        beta = 0
        gamma = -0.5
    elif method == "complete":
        alpha_u = alpha_v = 0.5
        beta = 0
        gamma = 0.5
    elif method == "average":
        alpha_u = nu / (nu + nv)
        alpha_v = nv / (nu + nv)
        beta = 0
        gamma = 0
    elif method == "ward":
        alpha_u = (nu + ns) / (nu + nv + ns)
        alpha_v = (nv + ns) / (nu + nv + ns)
        beta = -ns / (nu + nv + ns)
        gamma = 0
    else:
        raise ValueError("Неизвестный метод")

    return alpha_u * d_uv + alpha_v * d_us + beta * d_vs + gamma * abs(d_uv - d_us)


def hierarchical_clustering(data, n_clusters=1, method="ward"):
    clusters = {i: [i] for i in range(len(data))}
    cluster_sizes = {i: 1 for i in range(len(data))}

    distances = {
        (i, j): distance.euclidean_distance(data[i], data[j])
        for i in range(len(data)) for j in range(i + 1, len(data))
    }

    merge_matrix = []
    current_cluster = len(data)

    while len(clusters) > n_clusters:
        (cluster_a, cluster_b), min_distance = min(distances.items(), key=lambda x: x[1])

        size_a = cluster_sizes[cluster_a]
        size_b = cluster_sizes[cluster_b]

        clusters[current_cluster] = clusters.pop(cluster_a) + clusters.pop(cluster_b)
        cluster_sizes[current_cluster] = size_a + size_b

        new_distances = {}
        for cluster in clusters:
            if cluster != current_cluster:
                dist = lance_williams_update(
                    distances.get((min(cluster_a, cluster), max(cluster_a, cluster)), np.inf),
                    distances.get((min(cluster_b, cluster), max(cluster_b, cluster)), np.inf),
                    min_distance,
                    method=method,
                    nu=size_a,
                    nv=size_b,
                    ns=cluster_sizes[cluster]
                )
                new_distances[(min(current_cluster, cluster), max(current_cluster, cluster))] = dist

        distances = {k: v for k, v in distances.items() if cluster_a not in k and cluster_b not in k}

        distances.update(new_distances)

        merge_matrix.append([cluster_a, cluster_b, min_distance, cluster_sizes[current_cluster]])

        current_cluster += 1

    return np.array(merge_matrix), clusters


def visualize_clusters(data, clusters, plot_name):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    for i, (cluster_id, elements) in enumerate(clusters.items()):
        cluster_points = reduced_data[elements]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], cmap='rainbow')

    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    
    plt.savefig(plot_name)
    plt.close()





# ============================
# Mall_Customers dataset

print("Mall_Customers dataset")
df1 = pd.read_csv("datasets/Mall_Customers.csv")
df1 = df1.drop("CustomerID", axis=1)
df1['Genre'] = df1['Genre'].replace({'Female': 0, 'Male': 1})

print(df1.head())

pca = PCA(n_components=2)
data_2d = pca.fit_transform(df1)
plt.scatter(data_2d[:, 0], data_2d[:, 1])
plt.savefig("img/hierarchical/1_pca_scatter_plot.png")
plt.close()
# Выделяются 5 кластеров, связанных перемычками

plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(df1, method='ward'),
                            orientation='top',
                            distance_sort='ascending')
plt.title("Дендрограмма")
plt.xlabel("Индексы образцов")
plt.ylabel("Расстояние")

plt.savefig("img/hierarchical/1_lib_dendrogram.png")
plt.close()

start_time = time.time()
model = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
clusters = model.fit_predict(df1)
end_time = time.time()
execution_time = end_time - start_time

pca = PCA(n_components=2)
data_2d = pca.fit_transform(df1)
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters, cmap='rainbow')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.savefig("img/hierarchical/1_lib_clustering.png")
plt.close()

mean_intra_distance = distance.calculate_intra_cluster_distances_lib(df1.values, clusters)
mean_inter_distance = distance.calculate_inter_cluster_distances_lib(df1.values, clusters)
print(f"Время кластеризации библиотечными средствами: {execution_time} секунд")
print(f"Среднее внутрикластерное расстояние: {mean_intra_distance}")
print(f"Среднее межкластерное расстояние: {mean_inter_distance}")



data_array = df1.values

merge_matrix, clusters = hierarchical_clustering(data_array, 1, method="ward")

plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(merge_matrix,
                orientation='top',
               distance_sort='ascending')
plt.title("Дендрограмма")
plt.xlabel("Индексы образцов")
plt.ylabel("Расстояние")

plt.savefig("img/hierarchical/1_manual_dendrogram.png")
plt.close()

data_array = df1.values
n_clusters = 5

start_time = time.time()
merge_matrix, clusters = hierarchical_clustering(data_array, n_clusters=n_clusters, method="ward")
end_time = time.time()
execution_time = end_time - start_time

visualize_clusters(data_array, clusters, "img/hierarchical/1_manual_clustering.png")
mean_intra_distance = distance.calculate_intra_cluster_distances(data_array, clusters)
mean_inter_distance = distance.calculate_inter_cluster_distances(data_array, clusters)


print(f"Время кластеризации реализованным алгоритмом: {execution_time} секунд")
print(f"Среднее внутрикластерное расстояние: {mean_intra_distance}")
print(f"Среднее межкластерное расстояние: {mean_inter_distance}")


# ============================
# wine-clustering dataset

print()
print("wine-clustering dataset")
df2 = pd.read_csv("datasets/wine-clustering.csv")
print(df2.head())

pca = PCA(n_components=2)
data_2d = pca.fit_transform(df2)
plt.scatter(data_2d[:, 0], data_2d[:, 1])
# Выделяются 2 кластера, связанных перемычками

plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(df2, method='ward'),
                orientation='top',
               distance_sort='descending',
               show_leaf_counts=True)
plt.title("Дендрограмма")
plt.xlabel("Индексы образцов")
plt.ylabel("Расстояние")
plt.savefig("img/hierarchical/2_lib_dendrogram.png")
plt.close()

start_time = time.time()
model = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
clusters = model.fit_predict(df2)
end_time = time.time()
execution_time = end_time - start_time

pca = PCA(n_components=2)
data_2d = pca.fit_transform(df2)
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters, cmap='rainbow')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.savefig("img/hierarchical/2_lib_clustering.png")
plt.close()

mean_intra_distance = distance.calculate_intra_cluster_distances_lib(df2.values, clusters)
mean_inter_distance = distance.calculate_inter_cluster_distances_lib(df2.values, clusters)
print(f"Время кластеризации библиотечными средствами: {execution_time} секунд")
print(f"Среднее внутрикластерное расстояние: {mean_intra_distance}")
print(f"Среднее межкластерное расстояние: {mean_inter_distance}")

data_array = df2.values

merge_matrix, clusters = hierarchical_clustering(data_array, 1, method="ward")

plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(merge_matrix,
                orientation='top',
               distance_sort='ascending')
plt.title("Дендрограмма")
plt.xlabel("Индексы образцов")
plt.ylabel("Расстояние")
plt.savefig("img/hierarchical/2_manual_dendrogram.png")
plt.close()

data_array = df2.values
n_clusters = 2

start_time = time.time()
merge_matrix, clusters = hierarchical_clustering(data_array, n_clusters=n_clusters, method="ward")
end_time = time.time()
execution_time = end_time - start_time

visualize_clusters(data_array, clusters, "img/hierarchical/2_manual_clustering.png")
mean_intra_distance = distance.calculate_intra_cluster_distances(data_array, clusters)
mean_inter_distance = distance.calculate_inter_cluster_distances(data_array, clusters)


print(f"Время кластеризации реализованным алгоритмом: {execution_time} секунд")
print(f"Среднее внутрикластерное расстояние: {mean_intra_distance}")
print(f"Среднее межкластерное расстояние: {mean_inter_distance}")