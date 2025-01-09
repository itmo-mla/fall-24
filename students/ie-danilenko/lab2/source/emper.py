from read_fish import *
from sklearn.model_selection import train_test_split
from knn import knn
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm

def square(y_pred, y):
    return (y_pred - y)**2
    

if __name__ == '__main__':
    X, y = read_fish("dataset/fish.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    loss = np.array([])
    for k in tqdm(range(1, X_train.shape[0] - 1)):
        y_pred = knn(X_train, y_train, X_test, k)
        loss = np.append(loss, np.mean(square(y_pred, y_test)))

    plt.plot(range(1, loss.shape[0] + 1), loss)
    plt.xlabel('k')
    plt.ylabel('Эпирический риск')
    plt.show()
    