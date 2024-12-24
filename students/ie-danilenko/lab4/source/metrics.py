import numpy as np
from core.optim import MomentumSGD
from core.loss import Lin, RMSE
from core.regular import L2
from core.weights import CorrelationWeightGenerator
from core.model import LinearClassificator
from read import read_data
from sklearn.model_selection import train_test_split
from time import time
from core.utils import accuracy
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

if __name__ == "__main__":
    times = []
    acc = []

    X, y = read_data('/Users/ilyadanilenko/Documents/GitHub/fall-24/students/ie-danilenko/lab4/source/dataset/Iris.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    optim = MomentumSGD(1e-5, 0.5)
    regu = L2(0.3)

    weight_gen = CorrelationWeightGenerator(X_train, y_train)
    model = LinearClassificator([Lin, RMSE], optim, regu, weight_gen)

    start_time = time()
    model.train(X_train, y_train, X_test, y_test, epochs=50)
    acc.append(accuracy(model.predict(X_test), y_test))
    times.append(time() - start_time)

    model_etalon = SGDClassifier(loss="log_loss", penalty="l2", max_iter=50, learning_rate='constant', eta0=1e-5)

    start_time = time()
    model_etalon.fit(X_train, y_train)
    acc.append(accuracy(model_etalon.predict(X_test), y_test.reshape(-1)))
    times.append(time() - start_time)    

    alg = ['Linear Classification']
    x = np.arange(len(alg))
    width = 0.1
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    bar_dict = {
        "My" : (acc[0]),
        "sklearn" : (acc[1])
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
    ax.set_ylim(0, 1.0)
    plt.savefig('../img/metrics_acc.png')

    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    bar_dict = {
        "My" : (times[0]),
        "sklearn" : (times[1])
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
    ax.set_ylim(0, 2)

    plt.savefig('../img/metrics_time.png')
    