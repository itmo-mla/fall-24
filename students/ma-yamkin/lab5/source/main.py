from SVM import SVM
from Loader import Loader
from Visual import Visual

import numpy as np
from sklearn.metrics import accuracy_score


loader = Loader('iris')
X_train, X_test, y_train, y_test = loader.X_train, loader.X_test, loader.y_train, loader.y_test

svm = SVM(kernel='linear')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
y_test = np.where(np.array(y_test) == 0, -1, 1)

Visual(X_test, y_test, svm)

accuracy_score(y_test, y_pred)
