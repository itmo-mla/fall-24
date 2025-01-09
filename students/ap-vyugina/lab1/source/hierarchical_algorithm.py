import numpy as np

def hierarchy(distances):
    N = len(distances)
    powers = list(np.ones(N, dtype=np.int64))

    cluster_ids = list(range(N))
    pairwise_distances = {}
    all_clusters_power = {c_id: p for (c_id, p) in zip(cluster_ids, powers)}
    
    while N > 2:
        idx_ = np.argmin(distances)
        idx_i, idx_j = np.unravel_index(idx_, distances.shape)

        i_p = powers[idx_i]
        j_p = powers[idx_j]

        d = distances[idx_i, idx_j] # R(u, v)
        pairwise_distances[(cluster_ids[idx_i], cluster_ids[idx_j])] = d

        max_cluster_id = max(cluster_ids)
        for idx in sorted([idx_i, idx_j], reverse=True):
            del powers[idx]
            del cluster_ids[idx]

        alpha_u = (np.array(powers) + i_p) / (np.array(powers) + i_p + j_p)
        alpha_v = (np.array(powers) + j_p) / (np.array(powers) + i_p + j_p)
        beta = -np.array(powers) / (np.array(powers) + i_p + j_p)

        distances = np.delete(distances, [idx_i, idx_j], axis=0) # удаляем строки. В колонках останется расстояние от U|V до любого кластера S
        distance_ = alpha_u * distances[:, idx_i] + alpha_v * distances[:, idx_j] + beta * d

        # scipy ward formula
        distance_ = np.sqrt(alpha_u * np.square(distances[:, idx_i]) + \
            alpha_v * np.square(distances[:, idx_j]) + beta * (d**2))


        distances = np.delete(distances, [idx_i, idx_j], axis=1)
        # удалены строки/столбцы, отвечающие за U|V

        distances_ = np.zeros((N-1, N-1))
        distances_[:-1, :-1] = distances
        distances_[-1, -1] = np.inf
        distances_[-1, :-1] = distance_.T
        distances_[:-1, -1] = distance_

        distances = distances_

        powers.append(i_p+j_p)
        cluster_ids.append(max_cluster_id+1)
        all_clusters_power[max_cluster_id+1] = i_p+j_p
        N -= 1


    distances = np.min(distances).reshape((1, 1))
    pairwise_distances[(cluster_ids[0], cluster_ids[1])] = distances[0, 0]
    powers = [sum(powers)]
    cluster_ids = [max(cluster_ids)+1]
    return pairwise_distances, all_clusters_power