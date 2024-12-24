import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class my_dbscan:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()  # Convert DataFrame to NumPy array if needed
        
        n_points = len(X)
        self.labels_ = -1 * np.ones(n_points)  # Initialize all points as noise with label -1
        cluster_id = 0

        for point_idx in range(n_points):
            if self.labels_[point_idx] != -1:  # If point is already labeled, skip
                continue

            # Get neighbors for the current point
            neighbors = self.region_query(X, point_idx)

            # If the point is not a core point, label it as noise (-1)
            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = -1  # Mark as noise
            else:
                # Otherwise, expand cluster
                self.expand_cluster(X, point_idx, neighbors, cluster_id)
                cluster_id += 1

    def region_query(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        neighbors = np.where(distances <= self.eps)[0]  # Get all points within eps
        return neighbors.tolist()

    def expand_cluster(self, X, point_idx, neighbors, cluster_id):
        self.labels_[point_idx] = cluster_id  # Assign current point to the current cluster

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if self.labels_[neighbor_idx] == -1:  # If it was marked as noise, change to cluster
                self.labels_[neighbor_idx] = cluster_id

            if self.labels_[neighbor_idx] == -1:  # If not visited yet
                self.labels_[neighbor_idx] = cluster_id
                new_neighbors = self.region_query(X, neighbor_idx)

                # If new_neighbors has enough points to be considered core, add to neighbors list
                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)

            i += 1

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def plot_clusters(self, X, title):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()  # Convert DataFrame to NumPy array if needed

        unique_labels = np.unique(self.labels_)
        
        # Define distinct colors
        colors = cm.get_cmap('tab10', len(unique_labels))  # Use 'tab10' colormap for distinct colors

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot with noise
        axs[0].set_title(f"DBSCAN Clustering (With Noise) of {title}")
        for k, col in zip(unique_labels, colors(np.arange(len(unique_labels)))):
            class_member_mask = (self.labels_ == k)
            xy = X[class_member_mask]
            
            if k == -1:
                col = [0, 0, 0, 1]  # Noise points are black
                axs[0].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
            else:
                axs[0].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

        # Plot without noise (only clusters)
        axs[1].set_title(f"DBSCAN Clustering (Without Noise) of {title}")
        for k, col in zip(unique_labels, colors(np.arange(len(unique_labels)))):
            if k == -1:
                continue  # Skip noise points
            class_member_mask = (self.labels_ == k)
            xy = X[class_member_mask]
            axs[1].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

        plt.show()




