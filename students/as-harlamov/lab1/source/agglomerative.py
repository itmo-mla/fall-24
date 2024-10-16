from itertools import product

import pandas as pd

from clustering import Clustering


class AgglomerativeClustering(Clustering):
    def __init__(self, n_clusters: int = 3) -> None:
        self.n_clusters = n_clusters
        super().__init__()

    def fit_predict(self, X: pd.DataFrame):
        self.linkage = []

        n = len(X)
        cluster_points = {(i,) for i in range(n)}
        cluster_centroids = {(i,): X.iloc[i].values for i in range(n)}
        cluster_sizes = {(i,): 1 for i in range(n)}
        cluster_numbers = {(i,): i for i in range(n)}
        cluster_idx = n

        while len(cluster_points) > self.n_clusters:
            min_distance, cluster1, cluster2 = float('inf'), None, None

            for x, y in product(cluster_points, repeat=2):
                if x == y:
                    continue

                n1, n2 = cluster_sizes[x], cluster_sizes[y]
                centroid1, centroid2 = cluster_centroids[x], cluster_centroids[y]

                distance = (n1 * n2) / (n1 + n2) * self.euclidean_distance(centroid1, centroid2)
                if distance < min_distance:
                    min_distance = distance
                    cluster1, cluster2 = x, y

            if cluster1 is None or cluster2 is None:
                break

            self.linkage.append((cluster_numbers[cluster1], cluster_numbers[cluster2], min_distance, len(cluster1) + len(cluster2)))

            cluster_points -= {cluster1, cluster2}
            new_cluster = cluster1 + cluster2
            cluster_points.add(new_cluster)
            cluster_centroids[new_cluster] = self.centroid(X.iloc[list(new_cluster)].values)
            cluster_sizes[new_cluster] = len(new_cluster)
            cluster_numbers[new_cluster] = cluster_idx
            cluster_idx += 1

        self.labels_ = [0] * n
        for i, cluster in enumerate(cluster_points, 1):
            for point in cluster:
                self.labels_[point] = i

        return self.labels_


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.cluster import AgglomerativeClustering as SKAgglomerativeClustering

    X, _ = make_blobs(n_samples=100, centers=5, n_features=5, cluster_std=2.5, random_state=42)
    X = pd.DataFrame(X)
    X.columns = [f'col_{col}' for col in X.columns]

    my_aggl = SKAgglomerativeClustering(n_clusters=10)
    labels = my_aggl.fit_predict(X)
    print(Clustering.get_cluster_distances(X, labels))