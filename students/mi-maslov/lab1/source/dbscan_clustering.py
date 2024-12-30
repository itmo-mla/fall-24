import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.cluster import DBSCAN
from clustering import show_metrics
from clustering import Clustering

def dbscan_custom(points: np.array, min_points: int, epsilon: int):
    labels = np.zeros(points.shape[0], dtype=np.int64)
    num_points = points.shape[0]
    cluster_id = 0

    for idx in range(num_points):

        if labels[idx] > 0:
            continue

        neighbor_mask = np.sqrt(np.sum((points[idx] - points) ** 2, axis=1)) < epsilon

        if neighbor_mask.sum() < min_points:
            labels[idx] = -1
            continue

        cluster_id += 1
        combined_mask = neighbor_mask.copy()

        while neighbor_mask.sum() > 0:
            labels[neighbor_mask] = cluster_id
            new_neighbors = np.zeros(num_points, dtype=bool)

            for neighbor in points[neighbor_mask]:
                temp_neighbor_mask = np.sqrt(np.sum((neighbor - points) ** 2, axis=1)) < epsilon
                if temp_neighbor_mask.sum() < min_points:
                    continue
                temp_neighbor_mask = temp_neighbor_mask & (~combined_mask)
                new_neighbors |= temp_neighbor_mask

            neighbor_mask = new_neighbors.copy()
            combined_mask |= new_neighbors

    clusters = [[point for point in points[labels == (cluster + 1)]] for cluster in range(cluster_id)]
    noise_points = np.array([point for point in points[labels == -1]])

    return clusters, noise_points, labels


def dbscan_etalon(X, eps, min_samples):
    start = time()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(X)
    y_pred = db.fit_predict(X)
    end_time = time()-start
    inside, outside = show_metrics(points=X, now_pred=y_pred, iconic=True)
    plt.scatter(X[:,0], X[:,1],c=y_pred, cmap='Paired')
    plt.title("ETALON - DBSCAN")
    plt.show()

    return end_time, inside, outside


if __name__ == "__main__":
    max_dist = 3
    num_neighb = 5
    points = pd.read_csv("../datasets/iris_norm.csv")

    points = points.to_numpy()

    start = time()
    clasters_plot, red_flag, clasters = dbscan_custom(points=points, min_points=num_neighb, epsilon=max_dist)
    end_time = time() - start
    points_ = points[clasters>0]
    clasters_ = clasters[clasters>0]

    inside, outside = show_metrics(points=points_, now_pred=clasters_)

    cl = Clustering(data=points, name="DBSCAN")

    cl.set_data_custom([end_time, inside, outside])
        
    for group in clasters_plot:
        group = np.array(group)
        plt.scatter(group[:,0], group[:,1])
    if red_flag.size>0:
        plt.scatter(red_flag[:,0], red_flag[:,1], c="black")
    plt.show()

    end_time, inside, outside = dbscan_etalon(points, max_dist, num_neighb)

    cl.set_data_etalon([end_time, inside, outside])

    cl.print_data()