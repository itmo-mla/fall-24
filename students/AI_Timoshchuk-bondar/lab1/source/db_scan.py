import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.cluster import DBSCAN

from metrics import show_metrics

def db_scan(points:np.array, num_neighb:int, max_dist:int):
    clasters = np.zeros(points.shape[0], dtype=np.int64)
    point_amm = points.shape[0]
    # print(point_amm)
    now_claster = 0

    for i in range(points.shape[0]):

        if clasters[i] > 0:
            continue

        neighb_mask =  np.sqrt(np.sum((points[i] - points)**2, axis=1)) < max_dist
        # print(neighb_mask)

        if neighb_mask.sum() < num_neighb:
            clasters[i] = -1
            continue
        
        now_claster +=1 
        # clasters[i] = now_claster
        all_neighb_mask = neighb_mask.copy()

        while neighb_mask.sum() > 0:
            clasters[neighb_mask] = now_claster
            new_neighb_mask = np.zeros(point_amm, dtype=bool)

            for neighb in points[neighb_mask]:
                tmp_neighb_mask =  np.sqrt(np.sum((neighb - points)**2, axis=1)) < max_dist
                # print(f"{tmp_neighb_mask=}")
                if np.sum(tmp_neighb_mask) < num_neighb:
                    continue
                
                # print(f"{all_neighb_mask=}")

                # Для исключения повторяющихся точек
                tmp_neighb_mask = tmp_neighb_mask > all_neighb_mask

                # print(f" After {tmp_neighb_mask=}")
                new_neighb_mask += tmp_neighb_mask

            neighb_mask = new_neighb_mask.copy()
            all_neighb_mask += new_neighb_mask
            # print(f" Final {all_neighb_mask=}")
        # print(now_claster)

    
    clasters_plot = [[point for point in points[clasters==(mask+1)]] for mask in range(now_claster)]
    red_flag = np.array([point for point in points[clasters==-1]])

    return clasters_plot, red_flag, clasters


def dbscan_etalon(X, eps, min_samples):
    #ss = StandardScaler()
    #X = ss.fit_transform(X)
    start = time()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(X)
    y_pred = db.fit_predict(X)
    print(f"Iconic сalc time is: {time()-start}")
    show_metrics(points=X, now_pred=y_pred, iconic=True)
    plt.scatter(X[:,0], X[:,1],c=y_pred, cmap='Paired')
    plt.title("ETALON - DBSCAN")
    plt.show()



if __name__ == "__main__":
    max_dist=300
    num_neighb=10
    points = pd.read_csv("./fall-24/students/AI_Timoshchuk-bondar/lab1/source/dirty/crimes.csv", index_col='Unnamed: 0')[['K&A', 'WT']]
    # points = pd.read_csv("./fall-24/students/AI_Timoshchuk-bondar/lab1/source/dirty/wine-clustering.csv")#[['Alcohol', "Proline"]]
    points = points.to_numpy()

    start = time()
    clasters_plot, red_flag, clasters = db_scan(points=points, num_neighb=num_neighb, max_dist=max_dist)
    print(f"Calc time is: {time()-start}")
    # print(clasters_plot)
    # print(red_flag)
    points1 = points[clasters>0]
    clasters1 = clasters[clasters>0]
    
    
    show_metrics(points=points1, now_pred=clasters1)

        
    for group in clasters_plot:
        # print(group[:,0])
        group = np.array(group)
        plt.scatter(group[:,0], group[:,1] )
    if red_flag.size>0:
        plt.scatter(red_flag[:,0], red_flag[:,1], c="black")
    plt.show()

    dbscan_etalon(points, max_dist, num_neighb)