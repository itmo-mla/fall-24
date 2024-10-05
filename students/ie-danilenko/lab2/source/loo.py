from read_fish import *
import numpy as np
from knn import knn
import matplotlib.pylab as plt
from tqdm import tqdm

if __name__ == '__main__':
    X, y = read_fish("dataset/fish.csv")
    loo = np.array([])
    n = 2000
    X = X[:n]
    y = y[:n]
    for k in tqdm(range(1, n-2)):
        k_err = 0
        for i in range(X.shape[0]):
            x_del = np.delete(X, i, axis=0)
            y_del = np.delete(y, i)
            x = np.expand_dims(X[i], axis=0)
            y_pred = knn(x_del, y_del, x, k)

            k_err += int(y_pred[0] != y[i])
        loo = np.append(loo, k_err)

    print(f"Лучший k: {np.argmin(loo) + 1}")
    plt.plot(range(1, loo.shape[0] + 1), loo)
    plt.show()