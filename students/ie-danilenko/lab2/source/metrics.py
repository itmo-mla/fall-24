from sklearn.neighbors import KNeighborsClassifier
from knn import knn
from read_fish import read_fish
from sklearn.model_selection import train_test_split
from time import time
import numpy as np
import matplotlib.pyplot as plt

def accuracy(y_pred, y):
    return np.sum(y_pred == y) / y.shape[0]

if __name__ == '__main__':
    X, y = read_fish("dataset/fish.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    accs = []
    times = []
    start_time = time()
    sk_knn = KNeighborsClassifier(1)
    sk_knn.fit(X_train, y_train)
    accs.append(accuracy(sk_knn.predict(X_test), y_test))
    end_time = time()
    times.append(end_time - start_time)

    start_time = time()
    accs.append(accuracy(knn(X_train, y_train, X_test, 1), y_test))
    end_time = time()
    times.append(end_time - start_time)

    alg = ['KNN']
    x = np.arange(len(alg))
    width = 0.1

    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    bar_dict = {
        "My" : (accs[1]),
        "sklearn" : (accs[0])
    }

    for attribute, measurement in bar_dict.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Accuracy')
    ax.set_title('Качество работы')
    ax.set_xticks(x + width, alg)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, 1.2)

    plt.show()

    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    bar_dict = {
        "My" : (times[1]),
        "sklearn" : (times[0])
    }

    for attribute, measurement in bar_dict.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Время (s)')
    ax.set_title('Время работы алгоритма')
    ax.set_xticks(x + width, alg)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, 0.6)

    plt.show()