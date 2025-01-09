import numpy as np


def extract_locality(X: np.ndarray, x: np.ndarray, e: float, ) -> np.ndarray:
    return np.nonzero(np.linalg.norm(X - x, axis=1) < e)[0]


def cluster_by_dbscan(X: np.ndarray, m: int, e: float) -> np.ndarray:
    l = len(X)

    labels = np.zeros(shape=(l,), dtype=np.int8)
    a = np.zeros(shape=(l,), dtype=np.int8)
    a_current = 0

    UNLABELED, NOISE, BORDER, CORE = 0, 1, 2, 3
    while np.any(unlabeled_points := np.nonzero(labels == 0)[0]):
        x_idx = np.random.choice(unlabeled_points)
        x = X[x_idx]
        locality_idx = extract_locality(X, x, e)
        if len(locality_idx) < m:
            labels[x_idx] = NOISE
        else:
            k = set(locality_idx)
            a_current = a_current + 1
            processed_points = {x_i for x_i in k if labels[x_i] in (UNLABELED, NOISE)}

            while processed_points:
                x_streak_idx = next(iter(processed_points))

                sub_locality_idx = extract_locality(X, X[x_streak_idx], e)
                if len(sub_locality_idx) >= m:
                    labels[x_streak_idx] = CORE
                    k |= set(sub_locality_idx)
                    processed_points |= {x_i for x_i in sub_locality_idx if labels[x_i] in (UNLABELED, NOISE)}
                else:
                    labels[x_streak_idx] = BORDER
                processed_points.remove(x_streak_idx)
            for x_i in k:
                a[x_i] = a_current

    return a - 1