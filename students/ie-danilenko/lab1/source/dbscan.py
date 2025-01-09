import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def dbscan(data_pd, eps, m):
    data = data_pd.to_numpy()
    n = data.shape[0]
    labels = np.zeros(n, dtype=int)
    cluster_count = 0
    
    for i in range(len(data)):
        if labels[i] > 0:
            continue

        distances = np.linalg.norm(data[i] - data, axis=1)
        neighbors = np.where(distances <= eps)[0]
        
        if len(neighbors) < m:
            labels[i] = -1
        else:
            cluster_count += 1
            labels[i] = cluster_count
            
            k = 0
            while k < len(neighbors):
                current_index = neighbors[k]
                
                if labels[current_index] == -1:
                    labels[current_index] = cluster_count
                
                if labels[current_index] == 0:
                    labels[current_index] = cluster_count
                    
                    distance = np.linalg.norm(data[current_index] - data, axis=1)
                    neighbor_neighbors = np.where(distance <= eps)[0]
                    
                    if len(neighbor_neighbors) >= m:
                        neighbors = np.append(neighbors, neighbor_neighbors)
                
                k += 1

    cluster = [[] for _ in range(cluster_count)]
    noise = []
    for i in range(n):
        if labels[i] == -1:
            noise.append(data_pd.iloc[i])
        else:
            cluster[labels[i] - 1].append(data_pd.iloc[i])
    return cluster, noise

def plot(clusters, subplot):
    for i in range(len(clusters)):
        xs = np.array(clusters[i])[:, 0]
        ys = np.array(clusters[i])[:, 1]

        subplot.scatter(xs, ys)

if __name__ == "__main__":
    fig, ax = plt.subplots(2, 2)
    fig.suptitle('DBSCAN')

    data_wine = pd.read_csv("datasets/wine-clustering.csv")[['Alcohol', 'Proline']]
    clusters_wine, noise_wine = dbscan(data_wine, 40, 6)
    noise_wine = np.asarray(noise_wine)
    del data_wine
    
    ax[0][0].set_title(f'Wine (with noise)')
    plot(clusters_wine, ax[0][0])
    if noise_wine.shape[0] > 0:
        ax[0][0].scatter(noise_wine[0:-1, 0], noise_wine[0:-1, 1], c='black')

    ax[1][0].set_title(f'Wine (without noise)')
    plot(clusters_wine, ax[1][0])
    
    data_crimes = pd.read_csv("datasets/crimes.csv", index_col='Unnamed: 0')[['K&A', 'WT']]
    clusters_crimes, noise_crimes = dbscan(data_crimes, 200, 2)
    noise_crimes = np.asarray(noise_crimes)
    del data_crimes

    ax[0][1].set_title(f'Crimes (with noise)')
    plot(clusters_crimes, ax[0][1])
    if noise_crimes.shape[0] > 0:
        ax[0][1].scatter(noise_crimes[0:-1, 0], noise_crimes[0:-1, 1], c='black')

    ax[1][1].set_title(f'Crimes (without noise)')
    plot(clusters_crimes, ax[1][1])
    plt.show()