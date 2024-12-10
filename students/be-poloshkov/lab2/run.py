from collections.abc import Callable
from typing import Tuple

import numpy as np
import pandas as pd
import time

from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from knn import MyKNNClf
def leave_one_out(X: np.ndarray, y: np.array):
    loo = []
    for idx in range(len(X)):
         loo.append((X[idx], y[idx], np.delete(X, idx, 0), np.delete(y, idx)))
    return loo

def run_leave_one_out(X: np.ndarray, y: np.array):
    errs = []
    for k in range(1, len(X) - 1):
        errs.append([k, 0])
        for xi, yi, x_wo_i, y_wo_i in leave_one_out(X, y):
            knn = MyKNNClf(k)
            knn.fit(x_wo_i, y_wo_i)
            ans = knn.predict([xi])
            errs[-1][1] += (ans != yi)
        errs[-1][1] = errs[-1][1] / X.shape[0]
        print(f'done step {k}')

    return errs

def run_empiric(X_train: np.ndarray, y_train: np.array, X_test: np.ndarray, y_test: np.array):
    errs = []
    for k in range(1, len(X_train) - 1):
        knn = MyKNNClf(k)
        knn.fit(X_train, y_train)
        y_pred= knn.predict(X_test)
        errs.append((k, np.mean(y_pred != y_test)))
        print(f'done step {k}')
    return errs

def compare_with_sklearn(X_train: np.ndarray, y_train: np.array, X_test: np.ndarray, y_test: np.array, ks: Tuple[int, ...], impls):
    for impl in impls:
        for k in ks:
            start = time.monotonic()
            neigh = impl(k)
            neigh.fit(X_train, y_train)
            y_pred = neigh.predict(X_test)
            speed = time.monotonic() - start
            print(f'{impl.__name__}(k={k})\n'
                  f'Time: {speed}s\n'
                  f'Accuracy: {np.mean(y_pred == y_test)}\n')

if __name__ == '__main__':
    data = pd.read_csv('updated_pollution_dataset.csv')

    X = data.drop('Air Quality', axis=1)
    y = data['Air Quality']

    X_numpy, y_numpy = X.to_numpy(), y.to_numpy()

    le = preprocessing.LabelEncoder()
    le.fit(y_numpy)
    y_numpy = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_numpy, y_numpy, train_size=0.05, test_size=0.05, random_state=42)

    # LOO
    errs_loo = run_leave_one_out(X_train, y_train)
    errs_loo_sorted = sorted(errs_loo, key=lambda x: x[1])
    print("LOO best ks")
    for i in range(10):
        print(f'k={errs_loo_sorted[i][0]}, error={errs_loo_sorted[i][1]}')

    # Holdout
    errs_hout = run_empiric(X_train, y_train, X_test, y_test)
    errs_hout_sorted = sorted(errs_hout, key=lambda x: x[1])
    print("Holdout best ks")
    for i in range(10):
        print(f'k={errs_hout_sorted[i][0]}, error={errs_hout_sorted[i][1]}')

    plt.title('Best K param')
    plt.plot(*zip(*errs_loo), color='r', label='LOO')
    plt.plot(*zip(*errs_hout), color='g', label='Holdout')
    plt.xlabel('k')
    plt.ylabel('Error rate')
    plt.legend()
    plt.show()


    compare_with_sklearn(X_train, y_train, X_test, y_test, ks=(2, 5, 10, 30), impls=(MyKNNClf, KNeighborsClassifier))
