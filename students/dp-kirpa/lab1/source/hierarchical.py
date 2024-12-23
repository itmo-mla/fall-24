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


class Cluster:
    def __init__(self, elements: tuple):
        self._id = str(uuid.uuid4())
        self._elements = elements
    
    def __add__(self, other):
        if isinstance(other, Cluster):
            return Cluster(self._elements + other._elements)
        raise NotImplemented
    
    @property
    def elements(self) -> tuple:
        return self._elements
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def size(self) -> int:
        return len(self._elements)


class Lance_Williams_modes(Enum):
    closest_neighbour = 'closest_neighbour'
    farthest_neighbour = 'farthest_neighbour'
    group_mean = 'group_mean'
    centres = 'centres'
    Word = 'Word'

    
def Lance_Williams_distance(
    u: Cluster,
    v: Cluster,
    s: Cluster,
    Lance_Williams_map: Dict[str, Dict[str, float]],
    mode: Lance_Williams_modes=Lance_Williams_modes.Word
) -> float:
    '''Вход - кластера, словарь сохранённых расстояний и режим. Отдаёт новое значение расстония.'''
    allowed_modes = {
        Lance_Williams_modes.closest_neighbour,
        Lance_Williams_modes.farthest_neighbour,
        Lance_Williams_modes.group_mean,
        Lance_Williams_modes.centres,
        Lance_Williams_modes.Word
    }
    assert mode in allowed_modes, f'Режим работы может быть одним из следующих: {", ".join(allowed_mode.value for allowed_mode in allowed_modes)}'
    
    assert u.size and v.size and s.size, 'Мощность всех кластеров должна быть больше 0'
    
    w_size = u.size + v.size
    
    if mode == Lance_Williams_modes.closest_neighbour:
        a_u, a_v, b, g = 0.5, 0.5, 0, -0.5
    elif mode == Lance_Williams_modes.farthest_neighbour:
        a_u, a_v, b, g = 0.5, 0.5, 0, 0.5
    elif mode == Lance_Williams_modes.group_mean:
        a_u, a_v, b, g = u.size / w_size, v.size / w_size, 0, 0
    elif mode == Lance_Williams_modes.centres:
        a_u, a_v = u.size / w_size, v.size / w_size
        b, g = -a_u * a_v, 0
    else:
        a_u = (s.size + u.size) / (s.size + w_size)
        a_v = (s.size + v.size) / (s.size + w_size)
        b = -s.size / (s.size + w_size)
        g = 0
    
    rus = Lance_Williams_map[u.id][s.id]
    rvs = Lance_Williams_map[v.id][s.id]
    ruv = Lance_Williams_map[u.id][v.id]
    
    return a_u * rus + a_v * rvs + b * ruv + g * abs(rus - rvs)



def normalize_and_plain(vector):
    vector = vector._asdict()
    vector.pop('Index')
    vector = np.array(list(vector.values()))
    norm = np.linalg.norm(vector)
    if norm == 0: 
        return vector
    return vector / norm


class Clusterization:
    def __init__(self, data: pd.DataFrame):
        self._Lance_Williams_map = {}
        self._clusters = {}
        self._clusters_to_merge = None
        self._clusters_to_merge_distance = None
        self._data = data
        self._linkage_matrix = []
        self._max_cluster_number = -1
        self._clusters_ids_to_number = {}
        self._max_delta = -1
        self._prev_dist = None
        self._cluster_n_rec = 0
    
    def _init_clusterization(self) -> None:
        self._max_cluster_number = len(self._data)
        for cn, data_row in enumerate(self._data.itertuples(index=True)):
            cluster = Cluster((data_row,))
            self._clusters[cluster.id] = cluster
            self._clusters_ids_to_number[cluster.id] = cn
        for cluster_1_id, cluster_1 in self._clusters.items():
            self._Lance_Williams_map.setdefault(cluster_1_id, {})
            for cluster_2_id, cluster_2 in self._clusters.items():
                if cluster_1_id == cluster_2_id:
                    continue
                self._Lance_Williams_map.setdefault(cluster_2_id, {})
                distance = euclidean(
                    normalize_and_plain(cluster_1.elements[0]),
                    normalize_and_plain(cluster_2.elements[0])
                )
                self._Lance_Williams_map[cluster_1_id][cluster_2_id] = distance
                self._Lance_Williams_map[cluster_2_id][cluster_1_id] = distance
                if self._clusters_to_merge is None:
                    self._clusters_to_merge = (cluster_1_id, cluster_2_id)
                    self._clusters_to_merge_distance = distance
                else:
                    min_cluster_1_id, min_cluster_2_id = self._clusters_to_merge
                    if self._Lance_Williams_map[min_cluster_1_id][min_cluster_2_id] > distance:
                        self._clusters_to_merge = (cluster_1_id, cluster_2_id)
                        self._clusters_to_merge_distance = distance
    
    def _merge_closest_clusters(self, distance_mode: Lance_Williams_modes) -> None:
        min_cluster_1_id, min_cluster_2_id = self._clusters_to_merge
        
        if self._prev_dist is None:  # прогон впервые
            self._prev_dist = self._clusters_to_merge_distance
        
        if self._max_delta < (self._clusters_to_merge_distance - self._prev_dist):
            self._max_delta = self._clusters_to_merge_distance
            self._cluster_n_rec = len(self._clusters)
        self._prev_dist = self._clusters_to_merge_distance
        
        min_cluster_1 = self._clusters.pop(min_cluster_1_id)
        min_cluster_2 = self._clusters.pop(min_cluster_2_id)
        
        new_cluster = min_cluster_1 + min_cluster_2
        
        self._linkage_matrix.append([
            self._clusters_ids_to_number[min_cluster_1_id],
            self._clusters_ids_to_number[min_cluster_2_id],
            self._clusters_to_merge_distance,
            new_cluster.size
        ])
        
        self._Lance_Williams_map[new_cluster.id] = {}
                
        for neighbour_cluster_id, neighbour_cluster in self._clusters.items():
            neighbour_distance = Lance_Williams_distance(
                min_cluster_1, min_cluster_2, neighbour_cluster, self._Lance_Williams_map, distance_mode
            )
            self._Lance_Williams_map[new_cluster.id][neighbour_cluster_id] = neighbour_distance
            self._Lance_Williams_map[neighbour_cluster_id][new_cluster.id] = neighbour_distance
        
        self._clusters[new_cluster.id] = new_cluster
        self._clusters_ids_to_number[new_cluster.id] = self._max_cluster_number
        self._max_cluster_number += 1
        
        min_cluster_1_distances = self._Lance_Williams_map.pop(min_cluster_1_id)
        for cluster_1_neighbour_id in min_cluster_1_distances:
            self._Lance_Williams_map[cluster_1_neighbour_id].pop(min_cluster_1_id)
        
        min_cluster_2_distances = self._Lance_Williams_map.pop(min_cluster_2_id)
        for cluster_2_neighbour_id in min_cluster_2_distances:
            self._Lance_Williams_map[cluster_2_neighbour_id].pop(min_cluster_2_id)
        
        self._clusters_to_merge = None
        for cluster_1_id in self._clusters:
            for cluster_2_id in self._clusters:
                if cluster_1_id == cluster_2_id:
                    continue
                if self._clusters_to_merge is None:
                    self._clusters_to_merge = (cluster_1_id, cluster_2_id)
                    self._clusters_to_merge_distance = self._Lance_Williams_map[cluster_1_id][cluster_2_id]
                else:
                    min_cluster_1_id, min_cluster_2_id = self._clusters_to_merge
                    if self._Lance_Williams_map[min_cluster_1_id][min_cluster_2_id] > self._Lance_Williams_map[cluster_1_id][cluster_2_id]:
                        self._clusters_to_merge = (cluster_1_id, cluster_2_id)
                        self._clusters_to_merge_distance = self._Lance_Williams_map[cluster_1_id][cluster_2_id]
    
    def clusterize(self, num_clusters: int, distance_mode: Lance_Williams_modes = Lance_Williams_modes.Word) -> None:
        self._init_clusterization()
        while len(self._clusters) > num_clusters:
            print('Количество кластеров:', len(self._clusters))
            self._merge_closest_clusters(distance_mode)
        return self._clusters