import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

def lance_williams(r_us, r_vs, r_uv, u_len, v_len, s_len, method = 'min'):
    au = av = b = g = 0
    if method == 'min':
        au = 0.5
        av = 0.5
        b = 0
        g = -0.5
    elif method == 'max':
        au = av = 0.5
        b = 0
        g = 0.5
    elif method == 'mean':
        au = u_len / (u_len + v_len)
        av =  v_len / (u_len + v_len)
        b = g = 0
    elif method == 'center':
        au = u_len / (u_len + v_len)
        av = v_len / (u_len + v_len)
        b = - au * av
        g = 0
    elif method == 'ward':
        au = (s_len + u_len) / (s_len + u_len + v_len)
        av = (s_len + v_len) / (s_len + u_len + v_len)
        b = - s_len / (s_len + u_len + v_len)
        g = 0
    else:
        raise Exception("The method argument must be one of the following values: 'min', 'max', 'mean', 'center', 'ward'.")

    return au * r_us + av * r_vs + b * r_uv + g * np.absolute(r_us - r_vs)


def hierarhy_alg(data, count_clusters, method='min', dendr = False):
    n = data.shape[0]
    clusters = [[i] for i in range(n)]
    clusters_flag = [True for _ in range(n)]
    flag_history = []
    distances = np.zeros((n, n))
    linkage_info = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(data.iloc[i] - data.iloc[j])
            distances[i][j] = dist
            distances[j][i] = dist
    
    while clusters_flag.count(True) > count_clusters:
        min_dist = np.inf
        pair = (0, 0)
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if clusters_flag[i] and clusters_flag[j]:
                    if distances[i][j] < min_dist:
                        min_dist = distances[i][j]
                        pair = (i, j)
        
        cluster_a, cluster_b = pair
        new_cluster = clusters[cluster_a] + clusters[cluster_b]
        linkage_info.append([cluster_a, cluster_b, min_dist, len(new_cluster)])
        
        clusters_flag[cluster_a] = False
        clusters_flag[cluster_b] = False
        clusters_flag.append(True)
        flag_history.append(deepcopy(clusters_flag))
        clusters.append(new_cluster)
        distances = np.append(distances, np.zeros((len(clusters) - 1, 1)), axis=1)
        distances = np.append(distances, np.zeros((1, len(clusters))), axis=0)

        for k in range(len(clusters) - 1):
            if k != cluster_a and k != cluster_b:
                if clusters_flag[k]:
                    r_us = distances[cluster_a][k]
                    r_vs = distances[cluster_b][k]
                    r_uv = distances[cluster_a][cluster_b]
                    u_len = len(clusters[cluster_a])
                    v_len = len(clusters[cluster_b])
                    s_len = len(clusters[k])

                    new_distance = lance_williams(r_us, r_vs, r_uv, u_len, v_len, s_len, method)
                    distances[k][-1] = new_distance
                    distances[-1][k] = new_distance
    
    clusters_data = []
    for i in range(len(clusters)):
        if clusters_flag[i]:
            cl = []
            for j in range(len(clusters[i])):
                cl.append(data.iloc[clusters[i][j]].to_numpy().tolist())
            clusters_data.append(cl)
    
    if not dendr:
        return clusters_data, linkage_info
    else:
        return flag_history, linkage_info

def plot(clusters, subplot):
    for i in range(len(clusters)):
        xs = np.array(clusters[i])[:, 0]
        ys = np.array(clusters[i])[:, 1]

        subplot.scatter(xs, ys)

if __name__ == "__main__":
    data_wine = pd.read_csv("datasets/wine-clustering.csv")[['Alcohol', 'Proline']]
    wine_cl = [
        hierarhy_alg(data_wine, 3, 'min')[0],
        hierarhy_alg(data_wine, 3, 'max')[0],
        hierarhy_alg(data_wine, 3, 'mean')[0],
        hierarhy_alg(data_wine, 3, 'center')[0],
        hierarhy_alg(data_wine, 3, 'ward')[0]
    ]
    del data_wine

    data_crimes = pd.read_csv("datasets/crimes.csv", index_col='Unnamed: 0')[['K&A', 'WT']]
    crime_cl = [
        hierarhy_alg(data_crimes, 5, 'min')[0],
        hierarhy_alg(data_crimes, 5, 'max')[0],
        hierarhy_alg(data_crimes, 5, 'mean')[0],
        hierarhy_alg(data_crimes, 5, 'center')[0],
        hierarhy_alg(data_crimes, 5, 'ward')[0]
    ]
    del data_crimes


    fig, ax = plt.subplots(3, 4)
    fig.suptitle('Иерархический алгоритм')

    ax[0][0].set_title(f'Wine (min)')
    plot(wine_cl[0], ax[0][0])

    ax[0][1].set_title(f'Wine (max)')
    plot(wine_cl[1], ax[0][1])

    ax[1][0].set_title(f'Wine (mean)')
    plot(wine_cl[2], ax[1][0])

    ax[1][1].set_title(f'Wine (center)')
    plot(wine_cl[3], ax[1][1])

    ax[2][0].set_title(f'Wine (ward)')
    plot(wine_cl[4], ax[2][0])

    ax[0][2].set_title(f'Crimes (min)')
    plot(crime_cl[0], ax[0][2])

    ax[0][3].set_title(f'Crimes (max)')
    plot(crime_cl[1], ax[0][3])

    ax[1][2].set_title(f'Crimes (mean)')
    plot(crime_cl[2], ax[1][2])

    ax[1][3].set_title(f'Crimes (center)')
    plot(crime_cl[3], ax[1][3])

    ax[2][2].set_title(f'Crimes (ward)')
    plot(crime_cl[4], ax[2][2])
    plt.show()