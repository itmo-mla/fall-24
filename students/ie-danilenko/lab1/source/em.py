import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def em(data, count_clusters, max_iter=100):
    X = data.to_numpy()

    w = np.ones(count_clusters) / count_clusters
    mu = X[np.random.choice(X.shape[0], count_clusters, replace=False)]
    si = np.random.random((count_clusters, X.shape[1]))
    si = np.where(si[:, :] == 0.0, 1e-8, si[:, :])
    pred_label = np.array([])
    labels = np.array([])
    
    for _ in range(max_iter):
        # e-step
        g = np.zeros((X.shape[0], count_clusters))
        for k in range(count_clusters):
            p = np.prod(1 / (np.sqrt(2 * np.pi) * np.sqrt(si[k])) * np.exp(-((X - mu[k]) ** 2) / (2 * np.sqrt(si[k]) ** 2)), axis=1)
            g[:, k] = w[k] * p

        sum_g = g.sum(axis=1, keepdims=True)
        sum_g[sum_g == 0] = 1
        g /= sum_g
    
        # m-step
        w = g.mean(axis=0)
        mu = np.dot(g.T, X) / g.sum(axis=0).reshape(-1, 1)
        si = np.zeros((count_clusters, X.shape[1]))
        
        for k in range(count_clusters):
            diff = X - mu[k]
            si[k] = np.dot(g[:, k], diff**2) / g[:, k].sum()
        
        si = np.where(si[:, :] == 0.0, 1e-8, si[:, :])

        labels = np.argmax(g, axis=1)
        if np.array_equal(pred_label, labels):
            break
        else:
            pred_label = labels.copy()
        
    cluster = [[] for _ in range(count_clusters)]

    for i in range(X.shape[0]):
        cluster[labels[i]].append(data.iloc[i])

    return cluster

def plot(clusters, subplot):
    for i in range(len(clusters)):
        xs = np.array(clusters[i])[:, 0]
        ys = np.array(clusters[i])[:, 1]

        subplot.scatter(xs, ys)

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('EM-алгоритм')

    data_wine = pd.read_csv("datasets/wine-clustering.csv")[['Alcohol', 'Proline']]
    clusters_wine = em(data_wine, 3)

    data_crimes = pd.read_csv("datasets/crimes.csv", index_col='Unnamed: 0')[['K&A', 'WT']]
    clusters_crimes = em(data_crimes, 5)

    ax[0].set_title(f'Wine')
    plot(clusters_wine, ax[0])

    ax[1].set_title(f'Crimes')
    plot(clusters_crimes, ax[1])

    plt.show()