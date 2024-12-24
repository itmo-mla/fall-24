import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dual import SVM
from read import read_student
from sklearn.model_selection import train_test_split

def plot_svm_pca(svm, X, y, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
    y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    mesh_pca = np.c_[xx.ravel(), yy.ravel()]
    mesh_orig = pca.inverse_transform(mesh_pca)
    
    Z = svm.predict(mesh_orig)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='blue', label='Class 1', alpha=0.7)
    plt.scatter(X_pca[y == -1, 0], X_pca[y == -1, 1], c='red', label='Class -1', alpha=0.7)
    
    support_vectors_pca = pca.transform(svm.support_vectors)
    plt.scatter(support_vectors_pca[:, 0], support_vectors_pca[:, 1], 
               s=100, linewidth=1, facecolors='none', edgecolors='k',
               label='Support Vectors')
    
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.title(f'SVM Decision Boundary with PCA ({title})')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    X, y = read_student('dataset/student.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train[0:100]
    y_train = y_train[0:100]
    X_test = X_test[0:100]
    y_test = y_test[0:100]
    
    svm_linear = SVM('linear', C=1.0)
    svm_linear.solve(X_train, y_train)
    plot_svm_pca(svm_linear, X_train, y_train, 'Linear Kernel')
    
    svm_rbf = SVM('rbf', C=1.0, gamma=0.5)
    svm_rbf.solve(X_train, y_train)
    plot_svm_pca(svm_rbf, X_train, y_train, 'RBF Kernel')
    
    svm_poly = SVM('polynomial', C=1.0, gamma=0.5, r=1.0, d=2.0)
    svm_poly.solve(X_train, y_train)
    plot_svm_pca(svm_poly, X_train, y_train, 'Polynomial Kernel')

    for svm_model, name in [(svm_linear, 'Linear'), (svm_rbf, 'RBF'), (svm_poly, 'Polynomial')]:
        predictions = svm_model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        print(f"{name} SVM Accuracy: {accuracy}")