import numpy as np
from sklearn.datasets import fetch_california_housing


def load_house() -> tuple[np.ndarray, np.ndarray]:
    """

    :return: X(20640, 8), y (20640,)
    """
    data = fetch_california_housing()
    X, y = data.data, data.target
    return np.array(X), y
