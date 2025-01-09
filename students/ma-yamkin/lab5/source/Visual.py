import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Visual:
    def __init__(self, X, y, svm):
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)

        plt.figure(figsize=(10, 6))
        plt.scatter(X_reduced[y == 1, 0], X_reduced[y == 1, 1], color='blue', label='1', alpha=0.7)
        plt.scatter(X_reduced[y == -1, 0], X_reduced[y == -1, 1], color='red', label='-1', alpha=0.7)

        X_support_reduced = pca.transform(svm.support_vectors)
        plt.scatter(X_support_reduced[:, 0], X_support_reduced[:, 1], facecolors='none', edgecolors='k', s=100, label='Опорные вектора')

        point = pca.transform(svm.weights[None, :])
        x = float(point[0][0])
        y = float(point[0][1])
        b = svm.b
        k = (y - b) / x
        x = [-4, 2]
        y = [_ * k for _ in x]
        plt.plot(x, y)

        plt.title('SVM')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(loc='best')
        plt.grid()
        plt.show()
