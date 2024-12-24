import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dual import SVM
from read import read_student
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X, y = read_student('dataset/student.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train[0:100]
    y_train = y_train[0:100]
    X_test = X_test[0:100]
    y_test = y_test[0:100]
    
    results = []

    start_time = time.time()
    my_svm = SVM('linear', C=1.0)
    my_svm.solve(X_train, y_train)
    my_pred = my_svm.predict(X_test)
    my_time = time.time() - start_time
    my_acc = accuracy_score(y_test, my_pred)
        
    start_time = time.time()
    sklearn_svm = SVC(kernel='linear', C=1.0)
    sklearn_svm.fit(X_train, y_train)
    sklearn_pred = sklearn_svm.predict(X_test)
    sklearn_time = time.time() - start_time
    sklearn_acc = accuracy_score(y_test, sklearn_pred)
        
    results.append({
        'kernel': 'linear',
        'my_acc': my_acc,
        'sklearn_acc': sklearn_acc,
        'my_time': my_time,
        'sklearn_time': sklearn_time
    })

    start_time = time.time()
    my_svm = SVM('rbf', C=1.0, gamma=0.5)
    my_svm.solve(X_train, y_train)
    my_pred = my_svm.predict(X_test)
    my_time = time.time() - start_time
    my_acc = accuracy_score(y_test, my_pred)
        
    start_time = time.time()
    sklearn_svm = SVC(kernel='rbf', C=1.0, gamma=0.5)
    sklearn_svm.fit(X_train, y_train)
    sklearn_pred = sklearn_svm.predict(X_test)
    sklearn_time = time.time() - start_time
    sklearn_acc = accuracy_score(y_test, sklearn_pred)
        
    results.append({
        'kernel': 'rbf',
        'my_acc': my_acc,
        'sklearn_acc': sklearn_acc,
        'my_time': my_time,
        'sklearn_time': sklearn_time
    })

    start_time = time.time()
    my_svm = SVM('polynomial', C=1.0, gamma=0.5, r=1.0, d=2.0)
    my_svm.solve(X_train, y_train)
    my_pred = my_svm.predict(X_test)
    my_time = time.time() - start_time
    my_acc = accuracy_score(y_test, my_pred)
        
    start_time = time.time()
    sklearn_svm = SVC(kernel='poly', C=1.0, gamma=0.5, coef0=1.0, degree=2)
    sklearn_svm.fit(X_train, y_train)
    sklearn_pred = sklearn_svm.predict(X_test)
    sklearn_time = time.time() - start_time
    sklearn_acc = accuracy_score(y_test, sklearn_pred)
        
    results.append({
        'kernel': 'polynomial',
        'my_acc': my_acc,
        'sklearn_acc': sklearn_acc,
        'my_time': my_time,
        'sklearn_time': sklearn_time
    })
    
    kernels = [r['kernel'] for r in results]
    x = np.arange(len(kernels))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    my_acc = [r['my_acc'] for r in results]
    sklearn_acc = [r['sklearn_acc'] for r in results]
    
    ax1.bar(x - width/2, my_acc, width, label='My')
    ax1.bar(x + width/2, sklearn_acc, width, label='Scikit-learn')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(kernels)
    ax1.legend()
    
    my_time = [r['my_time'] for r in results]
    sklearn_time = [r['sklearn_time'] for r in results]
    
    ax2.bar(x - width/2, my_time, width, label='My')
    ax2.bar(x + width/2, sklearn_time, width, label='Scikit-learn')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Time')
    ax2.set_xticks(x)
    ax2.set_xticklabels(kernels)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()