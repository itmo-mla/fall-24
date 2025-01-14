import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_iris():
    df = pd.read_csv(f'csv/iris.csv')
    return df


def load_wine():
    df = pd.read_csv(f'csv/wine-clustering.csv')
    return df


def encod(df):
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return df


if __name__ == "__main__":
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    iris = load_iris()
    iris = encod(iris)
    pca = PCA(n_components=2).fit_transform(iris)
    tsne = TSNE(n_components=2).fit_transform(iris)


    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].scatter(tsne[:, 0], tsne[:, 1], c='red', edgecolors='k')
    axes[0].set_title("TSNE")

    axes[1].scatter(pca[:, 0], pca[:, 1], c='blue', edgecolors='k')
    axes[1].set_title("PCA")
    plt.suptitle('Iris')
    plt.show()

    wine = load_wine()
    wine = encod(wine)
    pca2 = PCA(n_components=2).fit_transform(wine)
    tsne2 = TSNE(n_components=2).fit_transform(wine)

    fig2, axes2 = plt.subplots(1, 2, figsize=(8, 3))
    axes2[0].scatter(tsne2[:, 0], tsne2[:, 1], c='red', edgecolors='k')
    axes2[0].set_title("TSNE")

    axes2[1].scatter(pca2[:, 0], pca2[:, 1], c='blue', edgecolors='k')
    axes2[1].set_title("PCA")
    plt.suptitle('Vine')
    plt.show()
