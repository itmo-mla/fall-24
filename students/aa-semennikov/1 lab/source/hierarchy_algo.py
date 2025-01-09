import numpy as np
import matplotlib.pyplot as plt
from wine import read_wine
from weapons import read_weapons
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram as dnd, linkage

def hierarchy_algo(data, clusters_number):
    #Расчёт межкластерного расстояния по формуле Ланса-Уильямса (расстояние Уорда)
    def intercluster_distance(r_us, r_vs, r_uv, u_len, v_len, s_len):
        sw_len = (s_len + u_len + v_len)
        a_u = (s_len + u_len) / sw_len
        a_v = (s_len + v_len) / sw_len
        beta = - s_len / sw_len
        # gamma = 0
        return a_u * r_us + a_v * r_vs + beta * r_uv # + gamma * np.abs(r_us - r_vs)
    
    #Ini
    n = data.shape[0]
    clusters = [[i] for i in range(n)]
    active_clusters = [True for _ in range(n)]
    activity_history = []
    distance_matrix = np.zeros((n, n))
    combination_history = []

    #Считаем расстояния между точками (начальными кластерами)
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(data.iloc[i] - data.iloc[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
     
    #Пока желаемое количество кластеров меньше числа активных кластеров
    while clusters_number < active_clusters.count(True):
        min_distance = np.inf
        current_clusters_count = len(clusters)

        #Проходим по всем ещё не объединявшимся на этой итерации кластерам и ищем среди них наименее удаленную пару
        #cluster_1_index и cluster_2_index - индексы кластеров, выбранных для объединения
        for i in range(current_clusters_count):
            for j in range(i + 1, current_clusters_count):
                if active_clusters[i] is True and active_clusters[j] is True:
                    if distance_matrix[i][j] < min_distance:
                        min_distance = distance_matrix[i][j]
                        cluster_1_index, cluster_2_index = i, j
        
        #Объединяем кластеры, сохраняем информацию об объединении
        combined_cluster = clusters[cluster_1_index] + clusters[cluster_2_index]
        combination_history.append([cluster_1_index, cluster_2_index, min_distance, len(combined_cluster)])
        
        #Помечаем предыдущие кластеры неактивными и добавляем объединенный как активный
        active_clusters[cluster_1_index] = False
        active_clusters[cluster_2_index] = False
        active_clusters.append(True)
        #Делаем слепок текущих активных кластеров
        activity_history.append(list(active_clusters))
        #Добавляем объединенный кластер в общий список
        clusters.append(combined_cluster)

        #Расширяем матрицу расстояний
        new_distances = np.zeros((current_clusters_count + 1, current_clusters_count + 1))
        new_distances[:current_clusters_count, :current_clusters_count] = distance_matrix
        distance_matrix = new_distances

        #Проходим по всем старым активным кластерам (cluster_1_index и cluster_2_index туда не попадают)
        for i in range(current_clusters_count):
            if active_clusters[i] is True:
                #Считаем расстояние от объединенного кластера до старых
                r_us = distance_matrix[cluster_1_index][i]
                r_vs = distance_matrix[cluster_2_index][i]
                r_uv = distance_matrix[cluster_1_index][cluster_2_index]
                u_len = len(clusters[cluster_1_index])
                v_len = len(clusters[cluster_2_index])
                s_len = len(clusters[i])
                new_distance = intercluster_distance(r_us, r_vs, r_uv, u_len, v_len, s_len)
                # Обновляем расстояния до нового кластера в матрице
                distance_matrix[i][-1] = distance_matrix[-1][i] = new_distance

    # Идем по активным кластерам (по сути можно только по последним нескольким) и записываем 
    # точки, которые в них состоят (строки датафрейма), в списки, потом эти списки записываем 
    # в финальный список, содержащий финальные кластеры
    result_clusters = []
    labels = []
    for i in range(current_clusters_count - 3 , current_clusters_count + 1):
        if active_clusters[i] is True:
            cluster = []
            for j in range(len(clusters[i])):
                cluster.append(data.iloc[clusters[i][j]])
                labels.append((clusters[i][j], i))
            result_clusters.append(cluster)

    return result_clusters, activity_history, combination_history, labels

def plot(wine_clustered, weapons_clustered):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    i = 0

    for clusters in (wine_clustered, weapons_clustered):
        cluster_1 = np.array(clusters[0])
        cluster_2 = np.array(clusters[1])
        data = np.vstack((cluster_1, cluster_2))
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
        reduced_cluster_1 = reduced_data[:len(cluster_1)]
        reduced_cluster_2 = reduced_data[len(cluster_1):]
        axs[i].scatter(reduced_cluster_1[:, 0], reduced_cluster_1[:, 1], color='#9F8170')
        axs[i].scatter(reduced_cluster_2[:, 0], reduced_cluster_2[:, 1], color='yellow')
        i += 1

    axs[0].set_title(f'Wine clustered')
    axs[1].set_title(f'Weapons clustered')
    plt.tight_layout()
    plt.show()

def plot_dendro(wine_history, wine_combinations, weapons_history, weapons_combinations):
    # Определяем макс. расстояние м-у кластерами, сокращенное при их объединении, а также оптимальное число кластеров
    def opt_clusters_number(activity_history, combination_history):
        # Находим индекс макс. сокр. расстояния
        max_reduced_distance_index = np.argmax(np.asarray(combination_history)[:, 2]) 
        # Находим само макс. сокр. расстояние
        max_reduced_distance = np.max(np.asarray(combination_history)[:, 2])
        # Оптимальное число кластеров - это число активных кластеров до объединения, давшего макс. сокращение расстояния
        opt_clusters_number = activity_history[max_reduced_distance_index - 1].count(True)
        return max_reduced_distance, opt_clusters_number

    wine_max_dist_inc, wine_opt_cl_number = opt_clusters_number(wine_history, wine_combinations)
    weapons_max_dist_inc, weapons_opt_cl_number = opt_clusters_number(weapons_history, weapons_combinations)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))
    dnd(wine_combinations, orientation='top', distance_sort='descending',  ax=ax1)
    dnd(weapons_combinations, orientation='top', distance_sort='descending', ax=ax2)
    ax1.set(xlabel='Object_id', ylabel='Distance')
    ax2.set(xlabel='Object_id', ylabel='Distance')
    ax1.text(0.5, 1.1, f"Max_reduced_dist: {np.round(wine_max_dist_inc, 2)}\nOpt. clusters number: {wine_opt_cl_number}", 
        transform=ax1.transAxes, fontsize=12, ha='center', va='top')
    ax2.text(0.5, 1.1, f"Max_reduced_dist: {np.round(weapons_max_dist_inc, 2)}\nOpt. clusters number: {weapons_opt_cl_number}", 
        transform=ax2.transAxes, fontsize=12, ha='center', va='top')
    plt.show()

if __name__ == "__main__":
    wine = read_wine()
    weapons = read_weapons()

    dendro = False
    if dendro is False: # кластеризация и вывод диаграмм
        plot(hierarchy_algo(wine, 2)[0], hierarchy_algo(weapons, 2)[0])
    else: # вывод дендрограммы
        wine_clustered, weapons_clustered = hierarchy_algo(wine, 1), hierarchy_algo(weapons, 1)
        wine_history, wine_combinations = wine_clustered[1], wine_clustered[2]
        weapons_history, weapons_combinations = weapons_clustered[1], weapons_clustered[2]
        plot_dendro(wine_history, wine_combinations, weapons_history, weapons_combinations)

    # weapons_combinations = hierarchy_algo(weapons, 1)[2]
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))
    # dnd(weapons_combinations, orientation='top', distance_sort='descending', ax=ax1)
    # l = linkage(weapons, method='ward')
    # dnd(l, orientation='top', distance_sort='descending', ax=ax2) # библиотечная дендрограмма
    # ax1.set(xlabel='Object_id', ylabel='Distance')
    # ax2.set(xlabel='Object_id', ylabel='Distance')
    # plt.tight_layout()
    # plt.show()