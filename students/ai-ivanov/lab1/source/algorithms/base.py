from abc import ABC, abstractmethod
import pandas as pd
import heapq
from rich.progress import Progress
import gower
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np


class Cluster:
    def __init__(self, elems: set[int]):
        self.elems = elems


class ClusteringAlgorithm(ABC):
    def __init__(
        self,
        num_iters: int = 1000,
        recalc_step: int = 10,
        early_stop_threshold: float = None,
        initial_delta: float = 0.1,
        delta_increase_factor: float = 1.5,
    ):
        self.num_iters = num_iters
        self.recalc_step = recalc_step
        self.early_stop_threshold = early_stop_threshold
        self.initial_delta = initial_delta
        self.delta_increase_factor = delta_increase_factor
        self.clusters = []
        self.dist_matrix: list[list[float]] = []
        self.active_cluster_ids = set()
        self.merge_history = []
        self.close_pairs = []
        self.current_delta = initial_delta
        self.min_dists = []

    @abstractmethod
    def cluster(
        self, data: pd.DataFrame, progress: Progress | None = None
    ) -> tuple[list[Cluster], int]:
        pass

    def _initialize_clusters(self, data: pd.DataFrame):
        self.clusters = [Cluster({i}) for i in range(len(data))]
        self.active_cluster_ids = set(range(len(data)))

    def _initialize_dist_matrix(self, data: pd.DataFrame):
        gower_matrix = gower.gower_matrix(data)
        self.dist_matrix = gower_matrix.tolist()
        self._update_close_pairs()

    def _calc_avg_inner_dist(self) -> float:
        total_dist = 0
        count = 0
        for i in self.active_cluster_ids:
            cluster = self.clusters[i]
            elems = list(cluster.elems)
            for j in range(len(elems)):
                for k in range(j + 1, len(elems)):
                    total_dist += self.dist_matrix[elems[j]][elems[k]]
                    count += 1
        return total_dist / count if count > 0 else 0

    def _update_close_pairs(self):
        self.close_pairs = []
        active_list = list(self.active_cluster_ids)
        for i in range(len(active_list)):
            u = active_list[i]
            for j in range(i + 1, len(active_list)):
                v = active_list[j]
                dist = self.dist_matrix[u][v]
                if dist < self.current_delta:
                    heapq.heappush(self.close_pairs, (dist, u, v))

    def _find_min_dist_pair(self) -> tuple[int, int]:
        while not self.close_pairs:
            self.current_delta *= self.delta_increase_factor
            if self.current_delta > 1.0:
                return -1, -1

            self._update_close_pairs()

        while self.close_pairs:
            _, u, v = heapq.heappop(self.close_pairs)
            if u in self.active_cluster_ids and v in self.active_cluster_ids:
                return u, v

        return self._find_min_dist_pair()

    @abstractmethod
    def _merge_clusters(self, u: int, v: int):
        pass

    def _find_optimal_num_iters(self) -> int:
        if len(self.min_dists) < 2:
            return len(self.clusters)

        max_diff = 0.0
        optimal_num_iters = 0
        for i in range(len(self.min_dists) - 1):
            diff = self.min_dists[i + 1] - self.min_dists[i]
            if diff > max_diff:
                max_diff = diff
                optimal_num_iters = i

        return optimal_num_iters

    def plot_dendrogram(self):
        if not self.merge_history:
            print("No merge history available. Run clustering first.")
            return

        Z = np.array(self.merge_history, dtype=np.float64)

        plt.figure(figsize=(10, 7))
        dendrogram(Z, leaf_rotation=90.0, leaf_font_size=8.0)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.show()
