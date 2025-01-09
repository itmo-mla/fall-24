import itertools
from enum import IntEnum
from typing import Any

import numpy as np

def make_condensed_distance_matrix_index(m: int):
    def resolve_index(i: int, j: int) -> int:
        if i == j:
            raise ValueError("`i` must be less `j`")
        i, j = sorted((i, j))
        return m * i + j - (i + 2) * (i + 1) // 2

    return resolve_index


def pairwise_distance(X: np.ndarray) -> np.ndarray:
    sample_size = X.shape[0]
    pairs = np.array(list(itertools.combinations(range(sample_size), 2)))
    differences = X[pairs[:, 0]] - X[pairs[:, 1]]
    distances = np.linalg.norm(differences, axis=1)
    return distances


class LinkageTableIndex(IntEnum):
    DISTANCE = 2
    CLUSTER_SIZE = 3


def link(
        distances_matrix: np.ndarray
) -> np.ndarray:
    samples_count = int((1 + 8 * len(distances_matrix)) ** 0.5 + 1) // 2
    cluster_index_map = [i if i < samples_count else None for i in range(samples_count * 2 - 1)]
    resolve = make_condensed_distance_matrix_index(samples_count)
    c = set(range(samples_count))
    sigma, sigma_step = 0.0, 0.1
    table: list[list] = []

    for new_cluster_number in np.arange(samples_count, samples_count * 2 - 1):
        while not (p := [
            (i, j) for i, j in itertools.combinations(c, 2)
            if
            cluster_index_map[i] is not None and
            cluster_index_map[j] is not None and
            distances_matrix[
                resolve(cluster_index_map[i], cluster_index_map[j])
            ] <= sigma
        ]) and len(
            c) > 1:
            sigma += sigma_step

        if not p:
            return np.array(table)

        new_cluster = min(
            p,
            key=lambda cluster_pair_indexes:
            distances_matrix[
                resolve(
                    cluster_index_map[cluster_pair_indexes[0]],
                    cluster_index_map[cluster_pair_indexes[1]]
                )
            ]
        )
        v, u = sorted(new_cluster)
        r_u_v = distances_matrix[resolve(cluster_index_map[v], cluster_index_map[u])]
        if v < samples_count:
            power_v = 1
        else:
            power_v = table[v - samples_count][LinkageTableIndex.CLUSTER_SIZE]
        if u < samples_count:
            power_u = 1
        else:
            power_u = table[u - samples_count][LinkageTableIndex.CLUSTER_SIZE]
        power_u_v = power_v + power_u
        table.append([v, u, r_u_v, power_u_v])
        dead_cluster = cluster_index_map[v]

        for i, s in enumerate(cluster_index_map):
            if s is None or i in [v, u]:
                continue
            power_s = 1 if i < samples_count else table[i - samples_count][3]
            r_u_s = distances_matrix[resolve(cluster_index_map[u], cluster_index_map[i])]
            r_v_s = distances_matrix[resolve(cluster_index_map[v], cluster_index_map[i])]
            alpha_u = (power_s + power_u) / (power_s + power_u_v)
            alpha_v = (power_s + power_v) / (power_s + power_u_v)
            beta = -power_s / (power_s + power_u_v)
            r_u_v_s = r_u_s ** 2 * alpha_u + r_v_s ** 2 * alpha_v + r_u_v ** 2 * beta
            distances_matrix[resolve(dead_cluster, cluster_index_map[i])] = r_u_v_s ** 0.5

        c -= {v, u}
        c.add(new_cluster_number)
        cluster_index_map[v] = cluster_index_map[u] = None
        cluster_index_map[new_cluster_number] = dead_cluster

    return np.array(table)


def find_threshold_cluster(link_table: np.ndarray) -> np.signedinteger[Any]:
    return np.argmax(np.diff(link_table[:, 2]))


def allocate_clusters(
        samples: np.ndarray[np.float32],
        link_table: np.ndarray,
        threshold_cluster: int,
) -> np.ndarray:
    clusters = {i: {i} for i in range(len(samples))}

    for cluster, (i, j, distance, size) in enumerate(link_table, start=len(samples)):
        clusters[cluster] = clusters.pop(i) | clusters.pop(j)
        if cluster == len(samples) + threshold_cluster:
            break

    cluster_labels = [(sample, cluster) for cluster, samples in clusters.items() for sample in samples]
    sorted_cluster_labels = np.array(sorted(cluster_labels, key=lambda record: record[0]))
    return np.unique(sorted_cluster_labels[:, 1], return_inverse=True)[1]
