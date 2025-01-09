import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from hierarhy import hierarhy_alg
import pandas as pd


data_wine = pd.read_csv("datasets/wine-clustering.csv")
data_crimes = data = pd.read_csv("datasets/crimes.csv", index_col='Unnamed: 0')
del data_crimes['State']

def optim_count(history, linkage):
    index = np.argmax(np.asarray(linkage)[:, 2])
    m = np.max(np.asarray(linkage)[:, 2])
    return m, history[index - 1].count(True)

method = 'ward'
history, linked_wine = hierarhy_alg(data_wine, 1, method, dendr=True)
dist, count = optim_count(history, linked_wine)
print("Wine")
print(f"\tМаксимальное расстояние: {dist}")
print(f"\tОптимальное кол-во кластеров: {count}")

history, linked_crimes = hierarhy_alg(data_crimes, 1, method, dendr=True)
dist, count = optim_count(history, linked_crimes)
print("Crime")
print(f"\tМаксимальное расстояние: {dist}")
print(f"\tОптимальное кол-во кластеров: {count}")

fig, (ax1, ax2) = plt.subplots(1, 2)
dendrogram(linked_wine, orientation='top', distance_sort='descending', show_leaf_counts=True, ax=ax1)
dendrogram(linked_crimes, orientation='top', distance_sort='descending', show_leaf_counts=True, ax=ax2)
ax1.set_title(f'Wine ({method})')
ax1.set(xlabel='Идентификаторы объектов', ylabel='Расстояние')

ax2.set_title(f'Crimes ({method})')
ax2.set(xlabel='Идентификаторы объектов', ylabel='Расстояние')
plt.show()