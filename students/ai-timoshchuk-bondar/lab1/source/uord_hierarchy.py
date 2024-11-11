import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def dist_matr(points:np.array, dist_type="d2"):
    if dist_type == "d2":
        # print(points)
        squared_norms = np.sum(points ** 2, axis=1).reshape(-1, 1)
        # print(squared_norms)
        # dist_matrix = np.sqrt((squared_norms + squared_norms.T) - 2 * points @ points.T)
        dist_matrix = (squared_norms + squared_norms.T) - 2 * points @ points.T
        # print(dist_matrix)
    elif dist_type == "d1":
        single_norm = np.sum(points, axis=1).reshape(-1, 1)
        dist_matrix = np.abs(single_norm - single_norm.T)
    else:
        raise Exception("Метод не добавлен")
    return dist_matrix



def calck_uord_dist(clasters:list, points:np.array):
    matrix = np.array([[0 for i in range(len(clasters))] for j in range(len(clasters))], dtype=np.float64)
    claster_lens = np.array([len(i) for i in clasters]).reshape(-1, 1)
    # print(points[clasters[0]])
    points = np.array([np.sum(points[claster1], axis=0)/len(claster1) for claster1 in clasters] )
    # print(points)
    # for i, claster1 in enumerate(clasters):
    #     for j,  claster2 in enumerate(clasters):
            
            # matrix[i,j] = (len(claster1)*len(claster2)/(len(claster1) + len(claster2))) * dist_matr(np.array([np.sum(points[claster1],axis=0)/len(claster1), np.sum(points[claster2], axis=0)/len(claster2)]).reshape(-1,2))[0,1]
            # print((len(claster1)*len(claster2)/(len(claster1) + len(claster2))))
            # print(matrix)
            # time.sleep(5)
    matrix = claster_lens * claster_lens.T / (claster_lens + claster_lens.T) * dist_matr(points=points)
#    print("uord", matrix)
    np.fill_diagonal(matrix, 0)
    return matrix

def get_delta(matrix:np.array, n1:int = 20, n2:int = 20):
    ind_dist = np.triu_indices(matrix.shape[0], k=1)
    if ind_dist[0].shape[0] > n1:
        return np.min(np.random.choice(matrix[np.triu_indices(matrix.shape[0], k=1)], size=n2, replace=False))
    
    return np.max(matrix[np.triu_indices(matrix.shape[0], k=1)])


def recalc_row_col(clasters:list, points:np.matrix, matrix:np.array, indexes:list, delta:np.float64):
    U = len(clasters[indexes[0]]) 
    V = len(clasters[indexes[1]])

    S = np.array([len(i) for i in clasters])
    a_u = (S + U)/(S + U + V)
    a_v = (S + V)/(S + U + V)
    b_b = - S / (S + U + V)

    matrix[indexes[0], :] = a_u * matrix[indexes[0], :] + a_v * matrix[indexes[1], :] + b_b * matrix[indexes[0], indexes[1]]
    matrix[:, indexes[0]] = matrix[indexes[0], :].T
    matrix[indexes[0], indexes[0]] = 0
    # К сожалению от этого никуда не деться из-за ошибки связанной с около-нулевыми значениями
    np.fill_diagonal(matrix, 0)
    return matrix
    
def recalс_clasters(clasters:list, i_min:int, i_max:int, clasters_for_dend:list):
    clasters[i_min] = clasters[i_min] + clasters[i_max]
    clasters_for_dend[i_min] = max(clasters_for_dend) + 1
    clasters_for_dend.pop(i_max)
    clasters.pop(i_max)
    

def recalс_matrix(clasters:list, points:np.matrix, matrix:np.array, indexes:np.array, delta:np.float64, clasters_for_dend:list, min_dists:np.array):
    i_min, i_max = np.sort(indexes)
    if i_min == i_max:
        print("clasters", clasters)
        print("claster", clasters[i_min])
        raise Exception(f"i_min==i_max\n{matrix}")
    recalc_row_col(clasters=clasters, points=points, matrix=matrix, indexes=[i_min, i_max], delta=delta)
    # Блок удаления строк
    matrix = np.delete(np.delete(matrix, i_max, axis=1), i_max, axis=0)
    recalс_clasters(clasters=clasters, i_min=i_min, i_max=i_max, clasters_for_dend=clasters_for_dend)
    upd_dist_indexes(min_dists=min_dists, max_i=i_max)
#    print("recalck", matrix)
    return matrix


def get_underdelta(dists:np.array, min_dists:np.array, delta:np.float64, n1: int, n2:int, min_i:int=None):
    '''
    dist, first_claster_ind, second_claster_ind
    [
    [0.0 0 0]
    [1.0 0 1]
    [1.0 1 0]
    [0.0 1 1]
    [4.0 1 2]
    [4.0 2 1]
    [0.0 2 2]
    [0.0 3 3]
    ]
    '''
    under_indexes = np.where((dists <= delta) * (dists > 0.00001))

    claster_inds = under_indexes

    if min_i is not None:
        claster_inds = [min_i for i in range(under_indexes[0].shape[0])], under_indexes[0]
        # print(under_indexes)

    if under_indexes[0].shape[0]:
        if min_dists.shape[0]:
            return np.append(min_dists, np.array(list(zip(dists[under_indexes], *claster_inds)), dtype=object), axis=0)

        #Если только создаём минимальные дистанции
        return np.array(list(zip(dists[under_indexes], *claster_inds)), dtype=object)
    
    return min_dists


def del_united_dists(min_dists:np.array, indexes:list):
    # НАДО ПЕРЕОПРЕДЕЛЯТЬ ИНДЕКСЫ КЛАСТЕРОВ КОТОРЫЕ СТОЯТ ПОСЛЕ ОБЪЕДИНЁННОГО КЛАСТЕРА
    return min_dists[~np.any(np.isin(min_dists[:, 1:], indexes), axis=1)]


def upd_dist_indexes(min_dists:np.array, max_i:int):
    index_to_upd_i,  index_to_upd_j= np.where(min_dists[:, 1:] >= max_i)
    index_to_upd_j += 1
    # Учитываем смещение клестера
    min_dists[index_to_upd_i, index_to_upd_j] -= 1
    # print(sorted(min_dists, key=lambda x: x[0]))
    # print(np.array(sorted(min_dists, key=lambda x: x[0])))


def get_nearest_clasters(min_dists:np.array, claster_combo_story:list, clasters:list, clasters_for_dend) -> list[int]:
    i = np.argmin(min_dists[:, 0])
    dist, iclast_1, iclast_2 = min_dists[i]
    # claster_combo_story.append([dist, iclast_1, iclast_2, len(clasters[iclast_1]) + len(clasters[iclast_2])])

    # print(iclast_1, iclast_2, len(clasters))                                                
                                                                                      #А ДОЛЖНА ПО-ДРУГОМУ СЧИТАТЬСЯ!!!! В алгосе ошибка
    M_c1, M_c2 = len(clasters[iclast_1]), len(clasters[iclast_2])                     #(M_c1 + M_c2) / (M_c1 * M_c2)
    claster_combo_story.append([clasters_for_dend[iclast_1], clasters_for_dend[iclast_2], (2 * dist)**0.5, M_c1 + M_c2])
    return np.sort((iclast_1, iclast_2))


def get_mindists_delta(uord_matrix:np.array, min_dists:np.array, delta:np.float64, min_i:int, n1:int, n2:int):
    min_dists = get_underdelta(dists = uord_matrix[min_i, :], min_dists=min_dists, delta=delta, min_i=min_i, n1=n1, n2=n2)
    if min_dists.shape[0] == 0:
        delta = get_delta(uord_matrix, n1=n1, n2=n2)
        print("new delta is", delta)
        min_dists = get_underdelta(dists = uord_matrix, min_dists=min_dists, delta=delta, min_i=None, n1=n1, n2=n2)

    return min_dists, delta 



# points = np.array([[2,2, 2], [3,3, 2], [5,5, 2], [8,8, 2], [9,9, 2]])
# print(points)

def uord_hierarchy(points:np.array, n1:int = 200, n2:int = 200):

    clasters = [[i] for i in range(points.shape[0])]
    clasters_for_dend = [i for i in range(points.shape[0])]
    indexes = []
    claster_combo_story = []

    min_dists = np.array([])

    uord_matrix = calck_uord_dist( points=points, clasters=clasters)
    delta = get_delta(matrix = uord_matrix, n1=n1, n2=n2)
    print("delta", delta)
    min_dists = get_underdelta(dists = uord_matrix, min_dists=min_dists, delta=delta, min_i=None, n1=n1, n2=n2)
    while uord_matrix.shape[0]>1:
    #    print(min_dists)
        indexes = get_nearest_clasters(min_dists, claster_combo_story, clasters, clasters_for_dend)
        min_dists = del_united_dists(min_dists=min_dists, indexes=indexes)
    #    print("min_dists del", min_dists)
        uord_matrix = recalс_matrix(clasters=clasters, points=points, matrix=uord_matrix, indexes=indexes, delta=delta, clasters_for_dend=clasters_for_dend, min_dists=min_dists)
        # Выбираем расстояния меньше дельты
        if uord_matrix.shape[0]>1:
            min_dists, delta = get_mindists_delta(uord_matrix = uord_matrix, min_dists=min_dists, delta=delta, min_i=indexes[0], n1=n1, n2=n2)
        # min_dists = get_underdelta(dists = uord_matrix[indexes[0], :], min_dists=min_dists, delta=delta, min_i=indexes[0], n1=n1, n2=n2)
        # if min_dists.shape[0] == 0:
        #     delta = get_delta(uord_matrix, n1=n1, n2=n2)
        #     print("new delta is", delta)
        #     min_dists = get_underdelta(dists = uord_matrix, min_dists=min_dists, delta=delta, min_i=None, n1=n1, n2=n2)
    
    return claster_combo_story


def plot_dendrogram(points:np.array, claster_combo_story:list):
    from scipy.cluster.hierarchy import dendrogram

    plt.figure(figsize=(10, 7))
    dendrogram(claster_combo_story, orientation='top', labels=[[i] for i in range(points.shape[0])], show_leaf_counts=True)

    # Настройка заголовка и меток
    plt.title('Дендрограмма')
    plt.xlabel('Объекты')
    plt.ylabel('Расстояние')

    # Отображение графика
    plt.show()

def calc_amm_clasters(claster_combo_story:list):
    # Т.к claster_combo_story на 1 меньше количества точек и np.diff на 2 меньще количества точек, то 1 к результату прибавлять не надО. так как мы уже эту единицу учитываем
    return len(claster_combo_story) - np.argmax(np.diff(np.array(claster_combo_story)[:, 2]))
    

 
# print(uord_matrix)
# print(claster_combo_story)

# check = np.array(claster_combo_story, dtype=np.int64)
# check = check[:, :2]
# indexes_clasters, count_clasters = np.unique(check, return_counts=True)
# strange = indexes_clasters[count_clasters>1]
# print(strange)
# print(check[np.any(np.isin(check, strange), axis=1)])





if __name__=="__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn import preprocessing
    points = np.array([[2,2], [3,3], [5,5], [8,8], [9,9]])

    # task 1
    label_encoder = preprocessing.LabelEncoder()
    points = pd.read_csv("./fall-24/students/AI_Timoshchuk-bondar/lab1/source/dirty/crimes.csv", index_col='Unnamed: 0')
    points['State'] = label_encoder.fit_transform(points['State'])
    # points = pd.read_csv("./fall-24/students/AI_Timoshchuk-bondar/lab1/source/dirty/wine-clustering.csv")[['Alcohol', "Proline"]]
    
    points = points.to_numpy(dtype=np.float32)

    n1=200
    n2=200
    start = time.time()
    claster_combo_story = uord_hierarchy(points=points, n1=n1, n2=n2)
    print(f"Время выполнения: {time.time() - start}")
    # print(claster_combo_story)
    plot_dendrogram(points=points, claster_combo_story=claster_combo_story)

    import scipy.cluster.hierarchy as sch

    start = time.time()
    dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))
    print(f"Время выполнения iconic: {time.time() - start}")

    plt.title('ETALON - Dendrogam', fontsize=20)
    plt.xlabel('Customers')
    plt.ylabel('Ecuclidean Distance')
    plt.show()





