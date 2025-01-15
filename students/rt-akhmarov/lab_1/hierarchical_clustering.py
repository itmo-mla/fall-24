import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances


def lw_update(d_uv, d_us, d_vs, method="ward", nu=1, nv=1, ns=1):
    """Обновление дистанции по формуле Лэнса-Уильямса."""
    params = {
        "single":   lambda nu, nv, ns: (0.5, 0.5, 0, -0.5),
        "complete": lambda nu, nv, ns: (0.5, 0.5, 0,  0.5),
        "average":  lambda nu, nv, ns: (nu/(nu+nv), nv/(nu+nv), 0, 0),
        "ward":     lambda nu, nv, ns: (
            (nu + ns)/(nu + nv + ns),
            (nv + ns)/(nu + nv + ns),
            -ns/(nu + nv + ns),
            0
        )
    }
    alpha_u, alpha_v, beta, gamma = params[method](nu, nv, ns)
    return alpha_u * d_uv + alpha_v * d_us + beta * d_vs + gamma * abs(d_uv - d_us)


def hierarchical_clustering(X, k=1, method="ward"):
    """Иерархическая кластеризация с ручным вычислением и матрицей слияний."""
    n = len(X)
    D = distance_matrix(X, X)
    np.fill_diagonal(D, np.inf)  # Прячем нули на диагонали
    ids = list(range(n))         # Номера объектов / кластеров
    sizes = [1] * n              # Размер каждого кластера
    merges = []                  # Матрица слияний
    cid = n                      # Идентификатор нового кластера

    while len(ids) > k:
        # Ищем минимальный элемент в матрице расстояний
        idx = np.argmin(D)
        i, j = divmod(idx, D.shape[1])
        if i > j:
            i, j = j, i
        dmin = D[i, j]

        merges.append([ids[i], ids[j], dmin, sizes[i] + sizes[j]])

        # Обновляем расстояния для объединённого кластера i
        for t in range(D.shape[0]):
            if t not in (i, j):
                D[i, t] = lw_update(D[i, t], D[j, t], dmin, method, sizes[i], sizes[j], sizes[t])
                D[t, i] = D[i, t]

        # Обновляем учёт кластеров и их размеры
        ids[i] = cid
        sizes[i] += sizes[j]
        ids.pop(j)
        sizes.pop(j)
        D = np.delete(D, j, axis=0)
        D = np.delete(D, j, axis=1)
        D[i, i] = np.inf
        cid += 1

    # Возвращаем матрицу слияний и текущие индексы
    return np.array(merges), {ids[a]: a for a in range(len(ids))}


def plot_dendrogram(merge_matrix, **kwargs):
    """Построение дендограммы на основе матрицы слияний."""
    n = merge_matrix.shape[0] + 1
    Z = np.zeros_like(merge_matrix)
    mapper = {}
    idx_new = 0

    def get_id(old):
        nonlocal idx_new
        if old not in mapper:
            mapper[old] = idx_new
            idx_new += 1
        return mapper[old]

    # Назначаем новые индексы одиночным точкам
    for i in range(n):
        get_id(i)

    # Формируем матрицу для дендограммы
    for i in range(merge_matrix.shape[0]):
        a, b, d, s = merge_matrix[i]
        a_new = get_id(a)
        b_new = get_id(b)
        Z[i] = [a_new, b_new, d, s]
        get_id(n + i)

    dendrogram(Z, **kwargs)


def get_clusters_by_cut(merge_matrix, n_samples, n_clusters):
    """Получение меток кластеров по итогам 'отрезания' на нужном уровне."""
    labs = np.arange(n_samples)
    cid = n_samples
    curr = n_samples

    for row in merge_matrix:
        if curr <= n_clusters:
            break
        a, b, _, _ = row
        la = (labs == a)
        lb = (labs == b)
        # Подстраховка для случаев, когда a/b уже переопределены
        if not la.any():
            la = (labs == np.unique(labs[labs == a])[0])
        if not lb.any():
            lb = (labs == np.unique(labs[labs == b])[0])

        labs[la] = cid
        labs[lb] = cid
        cid += 1
        curr -= 1

    # Переобзываем метки кластеров в 0..(n_clusters-1)
    unique_labs = np.unique(labs)
    map_l = {unique_labs[i]: i for i in range(len(unique_labs))}
    return np.array([map_l[x] for x in labs], dtype=int)


def find_optimal_clusters(X, merge_matrix, max_k=10):
    """Подбор оптимального числа кластеров по метрике силуэта."""
    n = len(X)
    best_k = 2
    best_score = -1
    limit = min(int(np.sqrt(n)), max_k + 1)

    for k in range(2, limit):
        labs = get_clusters_by_cut(merge_matrix, n, k)
        s = silhouette_score(X, labs)
        if s > best_score:
            best_score = s
            best_k = k

    return best_k, best_score


def compute_distances(X, labs):
    """Средние внутрикластерные и межкластерные расстояния."""
    clusts = np.unique(labs[labs != -1])
    intra = []
    inter = []

    for c in clusts:
        pts_c = X[labs == c]
        pts_oth = X[labs != c]

        if len(pts_c) > 1:
            intra.append(pairwise_distances(pts_c).mean())

        if len(pts_oth) > 0:
            inter.append(pairwise_distances(pts_c, pts_oth).mean())

    return np.mean(intra), np.mean(inter)


def linkage_matrix_to_labels(linkage_matrix, n_samples, n_clusters):
    """Преобразование матрицы слияний (linkage) в метки кластеров."""
    labs = np.arange(n_samples)
    curr_cid = n_samples

    # Объединяем кластеры в соответствии с матрицей до нужного количества
    for row in linkage_matrix[:len(linkage_matrix) - (n_clusters - 1)]:
        ca, cb = int(row[0]), int(row[1])
        labs[labs == ca] = curr_cid
        labs[labs == cb] = curr_cid
        curr_cid += 1

    # Сжимаем метки к 0..(n_clusters-1)
    uniq_labs = np.unique(labs)
    lab_map = {old: new for new, old in enumerate(uniq_labs)}
    return np.array([lab_map[l] for l in labs])
