from machine_failure import read_machine_failure
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from time import time
from test_fish import read_fish

def knn_algo(data, y, point, k):
    y_pred, classes = [], np.unique(y)
    # Cчитаем Евклидово расстояние до других точек
    point = point.reshape(1, -1)
    dist = np.linalg.norm(point - data, axis=1)
    nearest_indices = np.argsort(dist, axis=0)[:k]
    nearest_classes = y[nearest_indices]
    w = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (dist[nearest_indices] / dist[nearest_indices[-1]]) ** 2) # Гауссово ядро
    # Записываем в словарь сумму весов для каждого класса среди ближайших соседей
    class_weights = {}
    for classs in classes:
        class_weights[classs] = np.sum(w[nearest_classes.flatten() == classs])
    y_pred.append(max(class_weights, key = class_weights.get))
        
    return y_pred

def leave_one_out(k, X, y):
    y_pred = np.empty(y.shape, dtype=np.int32)
    for i in range(len(X)):
        y_pred[i] = knn_algo(np.delete(X, i, axis=0), np.delete(y, i, axis=0), X[i], k=k)
    return np.mean(y_pred != y, axis=0)


if __name__ == '__main__':
    data, y = read_machine_failure()
    #data, y = read_fish()
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)

    test = []
    for k in range(1, 50):
        test.append(leave_one_out(k, X_train, y_train))
    plt.plot(np.arange(1, 50), test, markersize=6)
    plt.xlabel('k')
    plt.ylabel('Частота ошибок')
    plt.show()

    y_pred = []
    t1 = []
    start1 = time()
    for i in range(len(y_test)):
        y_pred.append(knn_algo(X_test, y_test, X_test[i], 31))
    end1 = time()
    print(f'Report for my KNN:\n {classification_report(y_test, y_pred)}\nTime: {np.round(end1-start1, 4)}\n')

    start2 = time()
    KNN = KNeighborsClassifier(n_neighbors=31).fit(X_train, y_train.flatten())
    y_pred_sk = KNN.predict(X_test)
    end2 = time()
    print(f'Report for Sklearn KNN:\n {classification_report(y_test, y_pred_sk)}\nTime: {np.round(end2-start2, 4)}\n')