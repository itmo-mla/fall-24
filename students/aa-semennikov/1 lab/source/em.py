import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from wine import read_wine
from weapons import read_weapons
from sklearn.decomposition import PCA

def em_algo(data, clusters_number, max_iter=50):
    # Инициализация
    data = data.to_numpy()
    n_samples, n_features = data.shape
    # Веса (априорная вероятность)
    weights = np.ones(clusters_number) / clusters_number
    # Выбираем случайные центры кластеров (мат. ожидание)
    means = data[np.random.choice(data.shape[0], clusters_number, replace=False)]
    # Инициализируем матрицы ковариации
    covariances = [np.eye(n_features) for _ in range(clusters_number)]

    def e_step(data, weights, means, covariances):
        g = np.zeros((n_samples, clusters_number))
        # Для каждого кластера считаем вероятность принадлежности ему каждой взятой точки
        for k in range(clusters_number):
            p = multivariate_normal.pdf(data, mean=means[k], cov=covariances[k])
            g[:, k] = weights[k] * p
        # Построчно нормируем
        g /= g.sum(axis=1, keepdims=True)
        return g

    def m_step(data, g):
        # Обновляем веса (g, отображающую вероятность принадлежности наблюдения к кластеру, суммируем для каждого
        # кластера и делим на кол-во наблюдений)
        N_k = g.sum(axis=0)
        weights = N_k / n_samples
        # Обновляем мат.ожидания
        means = np.dot(g.T, data) / N_k[:, np.newaxis]
        covariances = []
        for k in range(clusters_number):
            # Считаем отклонение между данными и новыми центрами и обновляем дисперсии
            diff = data - means[k]
            si = np.dot(g[:, k] * diff.T, diff) / N_k[k]
            covariances.append(si + 1e-6 * np.eye(n_features))

        return weights, means, covariances

    y_prev = None
    for _ in range(max_iter):
        g = e_step(data, weights, means, covariances)
        weights, means, covariances = m_step(data, g)
        y = np.argmax(g, axis=1)

        if y_prev is not None and np.array_equal(y, y_prev):
            break

        y_prev = y

    return means, y


def plot_em_results(ax, data, means, y):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    colors = ['yellow', '#9F8170', 'red']
    ax.scatter(data_pca[:, 0], data_pca[:, 1], c=[colors[y[i]] for i in range(len(data))])
    # Центры кластеров в пр-ве пониженной размерности
    means_pca = pca.transform(means)
    ax.scatter(means_pca[:, 0], means_pca[:, 1], c='green', marker='P', s=120)

if __name__ == "__main__":
    wine = read_wine()
    weapons = read_weapons()
    means_1, y_1 = em_algo(wine, 2)
    means_2, y_2 = em_algo(weapons, 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plot_em_results(ax1, wine, means_1, y_1)
    plot_em_results(ax2, weapons, means_2, y_2)
    ax1.set_title('Wine clustered')
    ax2.set_title('Weapons clustered')
    plt.tight_layout()
    plt.show()