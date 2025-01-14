from svm import SVM
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


df = pd.read_csv('./datasets/SVMtrain.csv')
df = df.drop(columns=['PassengerId', 'Embarked'], axis=1)
df['Sex'] = df['Sex'].map({'female': 0, 'Male': 1})
df['Survived'] = df['Survived'].map({0: -1, 1: 1})
print(df.head(5))

X = df.drop(['Survived'], axis=1)
y = df['Survived']

scaler = preprocessing.MinMaxScaler()
X = np.array(scaler.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

start_time = time.time()
svm = SVM(C=2, kernel_type='linear')
svm.fit(X_train_pca, y_train)
end_time = time.time()
y_pred = svm.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Linear kernel training time: {end_time - start_time:.4f} seconds")
print(f"Linear kernel accuracy: {accuracy:.4f}")
svm.plot_predictions(X_train_pca, y_train, "linear")

start_time = time.time()
svm = SVM(C=2, kernel_type='rbf', gamma=3)
svm.fit(X_train_pca, y_train)
end_time = time.time()
y_pred = svm.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"RBF kernel training time: {end_time - start_time:.4f} seconds")
print(f"RBF kernel accuracy: {accuracy:.4f}")
svm.plot_predictions(X_test_pca, y_test, "rbf")

start_time = time.time()
svm = SVM(C=2, kernel_type='polynom', degree=2)
svm.fit(X_train_pca, y_train)
end_time = time.time()
y_pred = svm.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Polynomial kernel 2 training time: {end_time - start_time:.4f} seconds")
print(f"Polynomial kernel 2 accuracy: {accuracy:.4f}")
svm.plot_predictions(X_train_pca, y_train, "polynom2")

start_time = time.time()
svm = SVM(C=2, kernel_type='polynom', degree=3)
svm.fit(X_train_pca, y_train)
end_time = time.time()
y_pred = svm.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Polynomial kernel 3 training time: {end_time - start_time:.4f} seconds")
print(f"Polynomial kernel 3 accuracy: {accuracy:.4f}")
svm.plot_predictions(X_train_pca, y_train, "polynom3")



start_time = time.time()
sklearn_svc = SVC(C=2, kernel='linear')
sklearn_svc.fit(X_train_pca, y_train)
end_time = time.time()
print(f"Sklearn Linear kernel training time: {end_time - start_time:.4f} seconds")
y_pred = sklearn_svc.predict(X_test_pca)
print(f"Sklearn Linear kernel accuracy: {accuracy_score(y_test, y_pred):.4f}")

start_time = time.time()
sklearn_svc = SVC(C=2, kernel='rbf', gamma=3)
sklearn_svc.fit(X_train_pca, y_train)
end_time = time.time()
print(f"Sklearn RBF kernel training time: {end_time - start_time:.4f} seconds")
y_pred = sklearn_svc.predict(X_test_pca)
print(f"Sklearn RBF kernel accuracy: {accuracy_score(y_test, y_pred):.4f}")

start_time = time.time()
sklearn_svc = SVC(C=2, kernel='poly', degree=2)
sklearn_svc.fit(X_train_pca, y_train)
end_time = time.time()
print(f"Sklearn Polynomial kernel 2 training time: {end_time - start_time:.4f} seconds")
y_pred = sklearn_svc.predict(X_test_pca)
print(f"Sklearn Polynomial kernel 3 accuracy: {accuracy_score(y_test, y_pred):.4f}")

start_time = time.time()
sklearn_svc = SVC(C=2, kernel='poly', degree=3)
sklearn_svc.fit(X_train_pca, y_train)
end_time = time.time()
print(f"Sklearn Polynomial kernel 3 training time: {end_time - start_time:.4f} seconds")
y_pred = sklearn_svc.predict(X_test_pca)
print(f"Sklearn Polynomial kernel 3 accuracy: {accuracy_score(y_test, y_pred):.4f}")