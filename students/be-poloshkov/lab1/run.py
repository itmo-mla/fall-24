import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import sqeuclidean
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder

from dbscan import MyDBSCAN
from em import MyEM
from hierarchy import MyAgglomerative


def dist(x, y):
    return np.linalg.norm(x - y)

def mean_intracluster_distance(x: pd.DataFrame, labels):
    X = x.to_numpy()
    d = _labels_to_dict(X, labels)
    sum_dist = 0
    n_pairs = 0
    for points in d.values():
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                sum_dist += dist(points[i], points[j])
                n_pairs += 1

    return sum_dist / n_pairs

def mean_intercluster_distance(x: pd.DataFrame, labels):
    X = x.to_numpy()
    d = _labels_to_dict(X, labels)
    sum_dist = 0
    n_pairs = 0
    cluster_values = list(d.values())
    for i in range(len(cluster_values)):
        for j in range(i + 1, len(cluster_values)):
            for cluster_i_point in cluster_values[i]:
                for cluster_j_point in cluster_values[j]:
                    sum_dist += dist(cluster_i_point, cluster_j_point)
                    n_pairs += 1

    return sum_dist / n_pairs

def _labels_to_dict(X: np.ndarray, labels):
    d = {}
    for i, label in enumerate(labels):
        if label not in d:
            d[label] = []
        d[label].append(X[i])
    return d

def plot_dendrogram(linked):
    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True)
    plt.title('Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()

def make_linkage(model):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return linkage_matrix

def plot_countries(title, labels, df):
    plt.scatter(df['latitude'], df['longitude'], c=labels, cmap='rainbow')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.title(title)
    plt.show()

def plot_wine(title, labels, df):
    df['cluster'] = labels
    plt.scatter(df['x'], df['y'], c=df['cluster'], cmap='rainbow')
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    plt.title(title)
    plt.show()

def plot_elbow(wcss, title):
    plt.plot(range(len(wcss)), wcss)
    plt.title(f'Elbow method: {title}')
    plt.show()

def elbow_method(data):
    from sklearn.cluster import KMeans
    WCSS = [] # within cluster sum of squares
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, init='k-means++')
        kmeans.fit(data)
        WCSS.append(kmeans.inertia_)
    return WCSS

def main():
    # wine
    wine_data = pd.read_csv('wine-clustering.csv')
    print(wine_data)

    wine_data_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10).fit_transform(X=wine_data)

    plt.scatter(wine_data_tsne[:, 0], wine_data_tsne[:, 1], c='red')
    plt.show()

    wine_data_tsne = pd.DataFrame(data=wine_data_tsne, columns=['x', 'y'])

    wine_lnk = MyAgglomerative(n_clusters=1, metric='ward')
    wine_lnk.fit_predict(wine_data_tsne)
    plot_dendrogram(wine_lnk.lm)

    # Agglo Wine
    wine_my_hierarchy = MyAgglomerative(n_clusters=2, metric='ward')
    wine_my_agglo_start_time = time.time()
    wine_my_hierarchy_labels = wine_my_hierarchy.fit_predict(wine_data_tsne)
    wine_my_agglo_end_time = time.time()
    intra_dist = mean_intracluster_distance(wine_data_tsne, wine_my_hierarchy_labels)
    inter_dist = mean_intercluster_distance(wine_data_tsne, wine_my_hierarchy_labels)
    print(f'Wine My Agglo time: {wine_my_agglo_end_time - wine_my_agglo_start_time}, intra distance: {intra_dist}, inter distance: {inter_dist}')

    wine_sk_hierarchy = AgglomerativeClustering(n_clusters=2, metric='euclidean')
    wine_sk_agglo_start_time = time.time()
    wine_sk_hierarchy_labels = wine_sk_hierarchy.fit_predict(wine_data_tsne)
    wine_sk_agglo_end_time = time.time()
    intra_dist = mean_intracluster_distance(wine_data_tsne, wine_sk_hierarchy_labels)
    inter_dist = mean_intercluster_distance(wine_data_tsne, wine_sk_hierarchy_labels)
    print(f'Wine SK Agglo time: {wine_sk_agglo_end_time - wine_sk_agglo_start_time}, intra distance: {intra_dist}, inter distance: {inter_dist}')


    # countries
    countries_data = pd.read_csv('world_country_and_usa_states_latitude_and_longitude_values.csv')

    plt.scatter(countries_data['longitude'].to_numpy(), countries_data['latitude'].to_numpy())
    plt.show()

    countries_clustered_data = countries_data[['longitude', 'latitude']]
    countries_clustered_data = countries_clustered_data.dropna(axis=0, how='any')

    # optimal cluster number
    wcss = elbow_method(countries_clustered_data)
    plot_elbow(wcss, "countries")
    wcss = elbow_method(wine_data_tsne)
    plot_elbow(wcss, "wine")

    # Agglo
    countries_sk_aggl = AgglomerativeClustering(n_clusters=3)
    countries_sk_start_time = time.time()
    countries_sk_labels = countries_sk_aggl.fit_predict(countries_clustered_data)
    countries_sk_end_time = time.time()
    intra_dist = mean_intracluster_distance(countries_clustered_data, countries_sk_labels)
    inter_dist = mean_intercluster_distance(countries_clustered_data, countries_sk_labels)
    print(f'Time on SK Agglomerative is {countries_sk_end_time - countries_sk_start_time}, intra distance: {intra_dist}, inter distance: {inter_dist}')

    my_aggl_lnk = MyAgglomerative(n_clusters=1, metric='ward')
    my_aggl_lnk.fit_predict(countries_clustered_data)
    plot_dendrogram(my_aggl_lnk.lm)

    my_aggl = MyAgglomerative(n_clusters=3, metric='ward')
    my_aggl_start_time = time.time()
    my_aggl_labels = my_aggl.fit_predict(countries_clustered_data)
    my_aggl_end_time = time.time()
    intra_dist = mean_intracluster_distance(countries_clustered_data, my_aggl_labels)
    inter_dist = mean_intercluster_distance(countries_clustered_data, my_aggl_labels)
    print(f'Time on MyAgglomerative is {my_aggl_end_time - my_aggl_start_time}, intra distance: {intra_dist}, inter distance: {inter_dist}')

    # DBSCAN
    my_dbscan = MyDBSCAN(metric='euclidean', min_samples=5, eps=10)
    my_dbscan_start_time = time.time()
    my_dbscan_labels = my_dbscan.fit_predict(countries_clustered_data)
    my_dbscan_end_time = time.time()
    intra_dist = mean_intracluster_distance(countries_clustered_data, my_dbscan_labels)
    inter_dist = mean_intercluster_distance(countries_clustered_data, my_dbscan_labels)
    print(f'Time on MyDBSCAN Countries is {my_dbscan_end_time - my_dbscan_start_time}, , intra distance: {intra_dist}, inter distance: {inter_dist}')

    sk_dbscan = DBSCAN(eps=10, min_samples=5, metric='euclidean')
    sk_dbscan_start_time = time.time()
    sk_dbscan_labels = sk_dbscan.fit_predict(countries_clustered_data)
    sk_dbscan_end_time = time.time()
    intra_dist = mean_intracluster_distance(countries_clustered_data, sk_dbscan_labels)
    inter_dist = mean_intercluster_distance(countries_clustered_data, sk_dbscan_labels)
    print(f'Time on SK DBSCAN Countries is {sk_dbscan_end_time - sk_dbscan_start_time}, , intra distance: {intra_dist}, inter distance: {inter_dist}')

    # DBSCAN Wine
    my_dbscan_wine = MyDBSCAN(metric='euclidean', min_samples=5, eps=5)
    my_dbscan_start_time_wine = time.time()
    my_dbscan_labels_wine = my_dbscan_wine.fit_predict(wine_data_tsne)
    my_dbscan_end_time_wine = time.time()
    intra_dist = mean_intracluster_distance(wine_data_tsne, my_dbscan_labels_wine)
    inter_dist = mean_intercluster_distance(wine_data_tsne, my_dbscan_labels_wine)
    print(f'Time on MyDBSCAN Wine is {my_dbscan_end_time_wine - my_dbscan_start_time_wine}, , intra distance: {intra_dist}, inter distance: {inter_dist}')

    sk_dbscan_wine = DBSCAN(eps=5, min_samples=5, metric='euclidean')
    sk_dbscan_start_time_wine = time.time()
    sk_dbscan_labels_wine = sk_dbscan_wine.fit_predict(wine_data_tsne)
    sk_dbscan_end_time_wine = time.time()
    intra_dist = mean_intracluster_distance(wine_data_tsne, sk_dbscan_labels_wine)
    inter_dist = mean_intercluster_distance(wine_data_tsne, sk_dbscan_labels_wine)
    print(f'Time on SK DBSCAN Wine is {sk_dbscan_end_time_wine - sk_dbscan_start_time_wine}, , intra distance: {intra_dist}, inter distance: {inter_dist}')

    # EM
    my_em = MyEM(n_clusters=2, max_iter=10000)
    my_em_start_time = time.time()
    my_em_labels = my_em.fit_predict(countries_clustered_data)
    my_em_end_time = time.time()
    intra_dist = mean_intracluster_distance(countries_clustered_data, my_em_labels)
    inter_dist = mean_intercluster_distance(countries_clustered_data, my_em_labels)
    print(f'Time on MyEM Countries is {my_em_end_time - my_em_start_time}, , intra distance: {intra_dist}, inter distance: {inter_dist}')

    sk_em = GaussianMixture(n_components=2, covariance_type='full')
    sk_em_start_time = time.time()
    sk_em_labels = sk_em.fit_predict(countries_clustered_data)
    sk_em_end_time = time.time()
    intra_dist = mean_intracluster_distance(countries_clustered_data, sk_em_labels)
    inter_dist = mean_intercluster_distance(countries_clustered_data, sk_em_labels)
    print(f'Time on SK EM Countries is {sk_em_end_time - sk_em_start_time}, intra distance: {intra_dist}, inter distance: {inter_dist}')

    # EM Wine
    my_em_wine = MyEM(n_clusters=3, max_iter=1000)
    my_em_start_time_wine = time.time()
    my_em_labels_wine = my_em_wine.fit_predict(wine_data_tsne)
    my_em_end_time_wine = time.time()
    intra_dist = mean_intracluster_distance(wine_data_tsne, my_em_labels_wine)
    inter_dist = mean_intercluster_distance(wine_data_tsne, my_em_labels_wine)
    print(f'Time on MyEM Wine is {my_em_end_time_wine - my_em_start_time_wine}, intra distance: {intra_dist}, inter distance: {inter_dist}')

    sk_em_wine = GaussianMixture(n_components=3, covariance_type='full')
    sk_em_start_time_wine = time.time()
    sk_em_labels_wine = sk_em_wine.fit_predict(wine_data_tsne)
    sk_em_end_time_wine = time.time()
    intra_dist = mean_intracluster_distance(wine_data_tsne, sk_em_labels_wine)
    inter_dist = mean_intercluster_distance(wine_data_tsne, sk_em_labels_wine)
    print(f'Time on SK EM Wine is {sk_em_end_time_wine - sk_em_start_time_wine}, intra distance: {intra_dist}, inter distance: {inter_dist}')

    wine_data_tsne.copy()
    # Plotting
    plot_countries('Countries MyAgglomerative ', my_aggl_labels, countries_clustered_data)
    plot_countries('Countries SKAgglomerative ', countries_sk_labels, countries_clustered_data)

    plot_countries('Countries MyDBSCAN', my_dbscan_labels, countries_clustered_data)
    plot_countries('Countries SKDBSCAN', sk_dbscan_labels, countries_clustered_data)

    plot_countries('Countries MyEM ', my_em_labels, countries_clustered_data)
    plot_countries('Countries SKEM ', sk_em_labels, countries_clustered_data)

    plot_wine('Wine MyAgglomerative ', wine_my_hierarchy_labels, wine_data_tsne)
    plot_wine('Wine SkAgglomerative', wine_sk_hierarchy_labels, wine_data_tsne)

    plot_wine('Wine MyDBSCAN', my_dbscan_labels_wine, wine_data_tsne)
    plot_wine('Wine SKDBSCAN', sk_dbscan_labels_wine, wine_data_tsne)

    plot_wine('Wine MyEM ', my_em_labels_wine, wine_data_tsne)
    plot_wine('Wine SKEM ', sk_em_labels_wine, wine_data_tsne)


if __name__ == '__main__':
    main()