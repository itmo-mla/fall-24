import numpy as np
import pandas as pd
from time import time
from vizualization import make_gif
from metrics import show_metrics
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def calck_EM_clastering(points:np.array, clasters_amm:int, random_state:int = 2):
    np.random.seed(random_state)
    N = points.shape[1]
    L = points.shape[0]
    print(N)
    w_y = np.ones((clasters_amm, 1))/clasters_amm
    # u_y = [[np.random.randint(min(points[:, i]), max(points[:, i])) for i in range(points.shape[1])] for j in range(clasters_amm)]
    mu_y = np.random.rand(clasters_amm, N) * np.max(points, axis=0)
    # mu_y = points[np.random.choice(points.shape[0], clasters_amm, replace=False)]
    print(mu_y)
    sigma_y = np.ones((clasters_amm, N)) / 2 
    # mu_y[0] = [2.5, 2.5]

    points = points[np.newaxis, :, :]

    mu_y = mu_y[:, np.newaxis, :]
    sigma_y = sigma_y[ :, np.newaxis, :]
    # dist = np.sum((points - mu_y[0])**2/sigma_y[0], axis=1)
    # dist2 = np.sum((points - mu_y[1])**2/sigma_y[1], axis=1)
    # dist3 = np.sum((points - mu_y[2])**2/sigma_y[2], axis=1)

    mu_y_prev = mu_y.copy() + 2
    iterations_data = []
    while np.sum((mu_y_prev - mu_y)**2)>0.1:

        # prev_pred = now_pred.copy()
        mu_y_prev = mu_y.copy()
        dist = np.sum((points - mu_y)**2 / sigma_y, axis=2)
        # print((points - mu_y)**2 / sigma_y)


        probability =  np.exp(-0.5 * dist) / (((2 * np.pi) ** (N / 2)) * np.prod(np.sqrt(sigma_y), axis=2)) #+ np.random.rand(clasters_amm, L) * 1e-200
        # probability = np.where(probability==0, np.random.rand(1) 1.e-13, probability)
        
        # print(f"{w_y=}")
        # print(f"{mu_y=}")
        # print(f"{sigma_y=}")
        # print(f"{dist=}")
        # # print(points - mu_y[:, np.newaxis, :])
        # # print(w_y * probability)
        # print(f"{probability=}")


        # EXPECTATION
        # s_w_y = np.sum(w_y * probability, axis=0)
        
        ##TODO ПРОВЕРИТЬ НАЧИНАЯ ВОТ ОТ СЮДА!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        g = (w_y * probability) / (np.sum(w_y * probability, axis=0) + 1e-200) #np.where(s_w_y==0, np.random.rand(1) *1.e-13, s_w_y) 
        unclastered_points = np.sum(g, axis=0) == 0
        g[:, unclastered_points] = dist[:,unclastered_points]/np.sum(dist[:, unclastered_points], axis=0)
        # print("gggg", g)
        # g = (w_y * probability) / np.sum(w_y * probability, axis=0, keepdims=True) 
        # print(f"{g=}")
        # print(np.sum(g))
        iterations_data.append(
                {
            'g': g,
            'w_y': w_y,
            'mu_y': mu_y,
            'sigma_y':sigma_y},
        )


        # MAXIMIZATION
        ## А НУЖНА ЛИ ТУТ 1/points.shape[1] если по сути у нас G уже по сумме равна единице?
        w_y = np.sum(g, axis=1, keepdims=True) /L 
        mu_y = (g @ points / (w_y * L)).reshape((clasters_amm, 1, N)) 
        sigma_y = np.diagonal(np.sum(g[:, :, np.newaxis] * (points - mu_y[:, np.newaxis, :])**2, axis=2)  / (w_y * L)).T[:, np.newaxis, :]  #+1e-8 #!!!* 1/L
        sigma_y = np.where(sigma_y==0, 1e-10, sigma_y)  
        now_pred = np.argmax(g, axis=0)
        # print(mu_y) 
        # print(sigma_y)
        # print(np.sum(w_y))
        # print(now_pred)
        # print(f"{mu_y=}")
        # print(f"{mu_y_prev=}")
    return iterations_data, now_pred


def EM_etalon(X, n_clusters):
    start = time()
    hier = GaussianMixture(n_components=n_clusters)
    y_pred = hier.fit_predict(X)
    print(f"Iconic сalc time is: {time()-start}")
    show_metrics(X, n_clusters, iconic=True)
    plt.scatter(X[:,0], X[:,1],c=y_pred, cmap='Paired')
    plt.title("ETALON - EM Clustering")
    plt.show()

    
if __name__ == "__main__":
    clasters_amm = 3
    points = np.array([[2,2], [3,4], [5,6], [8,9], [9,9]], dtype=np.float64)#, [10000, 10000], [20000, 20000]], dtype=np.float64)
    # points = pd.read_csv("./fall-24/students/AI_Timoshchuk-bondar/lab1/source/dirty/crimes.csv", index_col='Unnamed: 0')[['K&A', 'WT']]
    points = pd.read_csv("./fall-24/students/AI_Timoshchuk-bondar/lab1/source/dirty/wine-clustering.csv")#[['Alcohol', "Proline"]]
    points = points.to_numpy()
    # points /= np.max(points, axis=0)
    start = time()
    iterations_data, now_pred = calck_EM_clastering(points=points, clasters_amm=3)
    print(f"Calc time is: {time()-start}")

    # intra_dist = mean_intracluster_distance(points, now_pred)
    # inter_dist = mean_intercluster_distance(points, now_pred)

    # print(f"Среднее внутрикластерное расстояние для моей реализации: {intra_dist}")
    # print(f"Среднее межкластерное расстояние: {inter_dist}")
    show_metrics(points, now_pred)

    EM_etalon(points, n_clusters=clasters_amm)

    points1 = points[np.newaxis, :, :]
    make_gif(points=points1, iterations_data=iterations_data)
    

