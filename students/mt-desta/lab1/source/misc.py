from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

def plot_elbow_method(X,title):
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title(f'Elbow Method for Optimal k: {title}')
    plt.show()

def plot_dendro(dataset1,dataset2):
    # Perform hierarchical clustering
    Z1 = linkage(dataset1, method='ward')
    Z2 = linkage(dataset2, method='ward')

    # Create a figure with two subplots for the dendrograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Dendrogram for the first dataset
    dendrogram(Z1, ax=axes[0])
    axes[0].set_title('Dendrogram - Mall Customers')

    # Dendrogram for the second dataset
    dendrogram(Z2, ax=axes[1])
    axes[1].set_title('Dendrogram - Iris')

    # Show the plots
    plt.tight_layout()
    plt.show()

def calculate_silhouette(X,title):
    sil = []
    K = range(2, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        labels = kmeans.labels_
        sil.append(silhouette_score(X, labels, metric='euclidean'))

    plt.figure(figsize=(6, 4))
    plt.plot(K, sil, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette Score for Optimal k: {title}')
    plt.show()

# df = pd.read_csv('data/Mall_Customers.csv')[['Annual Income (k$)', 'Spending Score (1-100)']]

# plot_elbow_method(df, 'Mall Customers')
# calculate_silhouette(df ,'Mall Customers')
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def hierarchical_clustering(df, method='single', num_clusters=1):
    """
    Performs hierarchical clustering using the Lance-Williams formula to compute the linkage matrix.
    
    Parameters:
    - df: A pandas DataFrame where each row is a data point.
    - method: The linkage method ('single', 'complete', 'average', 'centroid', 'ward').
    - num_clusters: The desired number of clusters to stop the merging process.
    
    Returns:
    - linkage_matrix: A numpy array containing the linkage matrix.
    - clusters: The final clusters as a list of lists.
    """
    
    # Convert the DataFrame to a numpy array
    data = df.values
    
    def compute_distance_matrix(data):
        """Computes the initial pairwise distance matrix between all points."""
        return pdist(data)
    
    def update_distance_matrix(dist_vector, a, b, clusters, method):
        """Updates the distance matrix using the Lance-Williams formula based on the method."""
        n = len(clusters)
        new_distances = []
        cluster_a_size = len(clusters[a][1])
        cluster_b_size = len(clusters[b][1])
        
        for i in range(n):
            if i != a and i != b:
                da = dist_vector[_condensed_index(n, a, i)]
                db = dist_vector[_condensed_index(n, b, i)]
                
                if method == 'single':
                    new_dist = min(da, db)
                elif method == 'complete':
                    new_dist = max(da, db)
                elif method == 'average':
                    new_dist = (cluster_a_size * da + cluster_b_size * db) / (cluster_a_size + cluster_b_size)
                elif method == 'centroid':
                    new_dist = np.sqrt((cluster_a_size * da**2 + cluster_b_size * db**2) / (cluster_a_size + cluster_b_size) - 
                                       (cluster_a_size * cluster_b_size * dist_vector[_condensed_index(n, a, b)]**2) / 
                                       ((cluster_a_size + cluster_b_size)**2))
                elif method == 'ward':
                    new_dist = np.sqrt(((cluster_a_size + len(clusters[i][1])) * da**2 +
                                        (cluster_b_size + len(clusters[i][1])) * db**2 -
                                        len(clusters[i][1]) * dist_vector[_condensed_index(n, a, b)]**2) /
                                       (cluster_a_size + cluster_b_size + len(clusters[i][1])))
                new_distances.append(new_dist)
        return new_distances
    
    def merge_clusters(clusters, a, b,index):
        """Merges clusters a and b into a new cluster."""
        new_cluster_id = index
        new_cluster = (new_cluster_id,clusters[a][1] + clusters[b][1])
        clusters = [c for i, c in enumerate(clusters) if i not in (a, b)]
        clusters.append(new_cluster)
        return clusters
    
    def _condensed_index(n, i, j):
        """
        Calculate the condensed index for two elements in a condensed distance matrix.
        """
        if i < j:
            return n*i - (i*(i+1)//2) + (j-i-1)
        elif i > j:
            return n*j - (j*(j+1)//2) + (i-j-1)
        else:
            return 0
    
    n = data.shape[0]
    clusters = [(i, [i]) for i in range(n)]  # Initialize each point as its own cluster
    dist_vector = compute_distance_matrix(data)
    linkage_matrix = []
    index = n
    
    while len(clusters) > num_clusters:
        # Find the pair of clusters with the smallest distance
        min_dist = np.inf
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = dist_vector[_condensed_index(n, clusters[i][1][0], clusters[j][1][0])]
                if dist < min_dist:
                    min_dist = dist
                    a, b = i, j
        
        linkage_matrix.append([clusters[a][0], clusters[b][0], min_dist, len(clusters[a][1]) + len(clusters[b][1])])
        
        # Update distance matrix using Lance-Williams formula
        new_distances = update_distance_matrix(dist_vector, a, b, clusters, method)
        
        # Merge the clusters
        clusters = merge_clusters(clusters, a, b,index)
        index += 1
        
        # Update the distance vector
        dist_vector = np.concatenate([dist_vector[:_condensed_index(n, a, b)],
                                      dist_vector[_condensed_index(n, a, b)+1:],
                                      new_distances])
    
    linkage_matrix = np.array(linkage_matrix)
    return linkage_matrix[linkage_matrix[:, 2].argsort()], clusters

def plot_dendrogram(linkage_matrix, labels=None):
    """
    Plots a dendrogram using the linkage matrix.
    
    Parameters:
    - linkage_matrix: A numpy array containing the linkage matrix (output from hierarchical clustering).
    - labels: The labels for each original data point (optional).
    """
    def get_leaves(linkage_matrix, num_leaves):
        """
        This function returns the order of leaves for plotting the dendrogram.
        It reconstructs the order of original data points based on cluster merges.
        """
        clusters = {i: [i] for i in range(num_leaves)}  # Initialize leaves as individual clusters
        
        for i, (a, b, dist, size) in enumerate(linkage_matrix):
            # Ensure the clusters exist (some might have been merged already)
            cluster_a = clusters.pop(int(a), [int(a)])  # Use pop with a default to avoid KeyError
            cluster_b = clusters.pop(int(b), [int(b)])  # Pop removes the cluster after it's merged
            new_cluster = cluster_a + cluster_b  # Combine the two clusters
            clusters[num_leaves + i] = new_cluster  # Assign the new cluster to a new index
        
        # Get the order of the final cluster (the whole dataset)
        final_order = list(clusters.values())  # Only one cluster should remain at the end
        return final_order

    def plot_lines(ax, node_positions, linkage_matrix):
        """
        Draws the connecting lines between clusters in the dendrogram.
        """
       
        for i, (a, b, dist, size) in enumerate(linkage_matrix):
            # Get the positions of the two clusters being merged
            for key_tuple,pos in node_positions.items():
                # print(key_tuple,pos)
                # Check if `a` is in any tuple key
                if a in key_tuple:
                    x1, y1 = node_positions[key_tuple]
                
                # Check if `b` is in any tuple key
                if b in key_tuple:
                    x2, y2 = node_positions[key_tuple]
            
            # Calculate the new cluster's x-position (average of the two clusters being merged)
            new_x = (x1 + x2) / 2
            new_y = dist  # The height of the new cluster is the distance between them
            
            # Draw vertical lines from each cluster to the new cluster
            ax.plot([x1, x1], [y1, new_y], c='k')
            ax.plot([x2, x2], [y2, new_y], c='k')
            
            # Draw horizontal line connecting the two clusters
            ax.plot([x1, x2], [new_y, new_y], c='k')
            
            # Store the position of the new cluster
            node_positions[(len(node_positions),)] = (new_x, new_y)
    
    num_leaves = len(linkage_matrix) + 1
    leaves_order = get_leaves(linkage_matrix, num_leaves)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate node positions for leaves
    node_positions = {tuple(leaves_order[i]): (i, 0) for i in range(len(leaves_order))}
    
    # Plot lines connecting clusters based on the linkage matrix
    plot_lines(ax, node_positions, linkage_matrix)
    
    # Label the leaves
    if labels is None:
        labels = [str(i) for i in range(num_leaves)]

    first_ten_keys = list(node_positions.keys())[:10]
    print(first_ten_keys)
    ax.set_xticks([node_positions[i][0] for i in first_ten_keys])
    # ax.set_xticklabels([labels[i] for i in leaves_order], rotation=90, ha='right')
    
    ax.set_ylim(0, np.max(linkage_matrix[:, 2]) * 1.1)  # Add some space above the highest linkage
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.show()

import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import sqeuclidean


class MyAgglomerative:
    def __init__(self, n_clusters=3, metric='euclidean'):
        self.n_clusters = n_clusters
        self.metric = metric
        self.lm = []

    @staticmethod
    def _ward_distance(c1, c2, l1, l2):
        c1_mean, c2_mean = [np.mean(c1, axis=0)], [np.mean(c2, axis=0)]
        sqeuclidean_dist = sqeuclidean(c1_mean, c2_mean)

        return (l1 * l2) / (l1 + l2) * sqeuclidean_dist

    def fit_predict(self, X: pd.DataFrame):
        observations, features = X.shape
        obs = X.to_numpy()

        d = {i:{i} for i in range(observations)}
        cluster_features = {i: obs[i] for i in range(observations)}

        it = 0
        while len(d) > self.n_clusters:
            imin, jmin, val = 0, 0, 10 ** 10
            for i, ci in d.items():
                for j, cj in d.items():
                    if i == j:
                        continue
                    dist = self._get_dist(cluster_features[i], cluster_features[j], len(ci), len(cj))
                    if dist < val:
                        imin, jmin, val = i, j, dist

            d[observations + it] = d[imin].union(d[jmin])
            cluster_features[observations + it] = np.sum([obs[i] for i in d[observations + it]], axis=0) / len(d[observations + it])
            del d[imin]
            del d[jmin]
            del cluster_features[imin]
            del cluster_features[jmin]

            self.lm.append([imin, jmin, val, len(d[observations + it])])
            it += 1

        res = [0] * observations
        for i, cluster in d.items():
            for val in cluster:
                res[val] = i
        return res

    def mean_intracluster_distance(self, x: pd.DataFrame, labels):
        X = x.to_numpy()
        d = self._labels_to_dict(X, labels)
        sum_dist = 0
        n_pairs = 0
        for points in d.values():
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    sum_dist += self._get_dist(points[i], points[j])
                    n_pairs += 1

        return sum_dist / n_pairs


    def _labels_to_dict(self, X: np.ndarray, labels):
        d = {}
        for i, label in enumerate(labels):
            if label not in d:
                d[label] = []
            d[label].append(X[i])
        return d

    def _get_dist(self, x, y, l1, l2):
        if self.metric == 'euclidean':
            return np.linalg.norm(x - y)
        if self.metric == 'chebyshev':
            return np.max(np.abs(x - y))
        if self.metric == 'manhattan':
            return np.sum(np.abs(x - y))
        if self.metric == 'cosine':
            return 1 - np.sum(np.multiply(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y))
        if self.metric == 'ward':
            return self._ward_distance(x, y, l1, l2)
        
