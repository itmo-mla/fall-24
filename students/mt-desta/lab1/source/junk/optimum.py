from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

df = pd.read_csv('data/Mall_Customers.csv')[['Annual Income (k$)', 'Spending Score (1-100)']]

plot_elbow_method(df, 'Mall Customers')
calculate_silhouette(df ,'Mall Customers')