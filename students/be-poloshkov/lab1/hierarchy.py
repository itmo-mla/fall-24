import numpy as np
import pandas as pd
from scipy.spatial.distance import sqeuclidean


class MyAgglomerative:
    def __init__(self, n_clusters=3, metric='euclidean'):
        self.n_clusters = n_clusters
        self.metric = metric
        self.lm = []

    @staticmethod
    def _ward_distance(c1, c2, l1, l2):
        c1_mean, c2_mean = [np.mean(c1, axis=0)], [np.mean(c2, axis=0)]
        sqeuclidean_dist = sqeuclidean(c1_mean, c2_mean)

        return (l1 * l2) / (l1 + l2) * sqeuclidean_dist

    def fit_predict(self, X: pd.DataFrame):
        observations, features = X.shape
        obs = X.to_numpy()

        d = {i:{i} for i in range(observations)}
        cluster_features = {i: obs[i] for i in range(observations)}

        it = 0
        while len(d) > self.n_clusters:
            imin, jmin, val = 0, 0, 10 ** 10
            for i, ci in d.items():
                for j, cj in d.items():
                    if i == j:
                        continue
                    dist = self._get_dist(cluster_features[i], cluster_features[j], len(ci), len(cj))
                    if dist < val:
                        imin, jmin, val = i, j, dist

            d[observations + it] = d[imin].union(d[jmin])
            cluster_features[observations + it] = np.sum([obs[i] for i in d[observations + it]], axis=0) / len(d[observations + it])
            del d[imin]
            del d[jmin]
            del cluster_features[imin]
            del cluster_features[jmin]

            self.lm.append([imin, jmin, val, len(d[observations + it])])
            it += 1

        res = [0] * observations
        for i, cluster in d.items():
            for val in cluster:
                res[val] = i
        return res

    def mean_intracluster_distance(self, x: pd.DataFrame, labels):
        X = x.to_numpy()
        d = self._labels_to_dict(X, labels)
        sum_dist = 0
        n_pairs = 0
        for points in d.values():
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    sum_dist += self._get_dist(points[i], points[j])
                    n_pairs += 1

        return sum_dist / n_pairs


    def _labels_to_dict(self, X: np.ndarray, labels):
        d = {}
        for i, label in enumerate(labels):
            if label not in d:
                d[label] = []
            d[label].append(X[i])
        return d

    def _get_dist(self, x, y, l1, l2):
        if self.metric == 'euclidean':
            return np.linalg.norm(x - y)
        if self.metric == 'chebyshev':
            return np.max(np.abs(x - y))
        if self.metric == 'manhattan':
            return np.sum(np.abs(x - y))
        if self.metric == 'cosine':
            return 1 - np.sum(np.multiply(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y))
        if self.metric == 'ward':
            return self._ward_distance(x, y, l1, l2)