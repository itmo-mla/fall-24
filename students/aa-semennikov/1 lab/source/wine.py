import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA

# 1.1 - метки классов отсутствуют
# 1.3 - кластеры, соединённые перемычками
# 1.4 - предполагаемое количество кластеров = 3

def read_wine(plot = False):
    data = pd.read_csv("datasets/wine-clustering.csv").dropna()
    for c in data.columns:
        data[[c]] = preprocessing.StandardScaler().fit_transform(data[[c]])
    if plot == False:
        return data
    else:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
        plt.show()       

if __name__ == "__main__":
    read_wine(plot = True)