import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier

class CustomLinearClassifier:
    def __init__(self, method='sgd_momentum', lr=0.01, gamma=0.9, reg_alpha=0.0, epochs=50, selection=None, init_type='zeros'):
        self.method = method
        self.lr = lr
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.epochs = epochs
        self.selection = selection
        self.init_type = init_type
        self.w_ = None
        self.history_ = []

    def fit(self, X, y):
        if self.method == 'sgd_momentum':
            self.w_ = self._sgd_momentum(X, y)
        elif self.method == 'steepest_gd':
            self.w_ = self._steepest_gradient_descent(X, y)

    def predict(self, X):
        return np.sign(X.dot(self.w_))

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def _margin(self, w, x, y):
        return y * np.dot(w, x)

    def _correlation_based_initialization(self, X, y):
        w_init = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            w_init.append(corr)
        return np.array(w_init)

    def _initialize_weights(self, X, y):
        if self.init_type == 'zeros':
            return np.zeros(X.shape[1])
        elif self.init_type == 'random':
            return np.random.randn(X.shape[1]) * 0.01
        elif self.init_type == 'correlation':
            return self._correlation_based_initialization(X, y)
        return np.zeros(X.shape[1])

    def _clip_gradients(self, grad, max_val=500):
        return np.clip(grad, -max_val, max_val)

    def _compute_loss_gradient(self, w, x, y):
        grad = 2 * (np.dot(w, x) - y) * x
        grad += self.reg_alpha * w
        return self._clip_gradients(grad)

    @staticmethod
    def margin_based_presentation(X, y, w):
        margins = np.array([y[i] * np.dot(w, X[i]) for i in range(len(X))])
        idx_sorted = np.argsort(np.abs(margins))
        return X[idx_sorted], y[idx_sorted]

    def _compute_epoch_loss(self, X, y, w):
        return np.mean((y - X.dot(w))**2) + self.reg_alpha * np.sum(w**2)

    def _sgd_momentum(self, X, y):
        w = self._initialize_weights(X, y)
        v = np.zeros_like(w)
        for _ in range(self.epochs):
            if self.selection is not None:
                X, y = self.selection(X, y, w)
            idx = np.random.permutation(len(y))
            for i in idx:
                grad = self._compute_loss_gradient(w, X[i], y[i])
                v = self.gamma * v + self.lr * grad
                w -= v
            self.history_.append(self._compute_epoch_loss(X, y, w))
        return w

    def _steepest_gradient_descent(self, X, y):
        w = self._initialize_weights(X, y)
        for _ in range(self.epochs):
            if self.selection is not None:
                X, y = self.selection(X, y, w)
            diff = (y - X.dot(w))
            grad = -2 * X.T.dot(diff) + self.reg_alpha * w
            numerator = grad.dot(grad)
            X_grad = X.dot(grad)
            denominator = X_grad.dot(X_grad)
            step_length = numerator / denominator if denominator != 0 else 1.0
            w -= step_length * grad
            self.history_.append(self._compute_epoch_loss(X, y, w))
        return w

iris = load_iris()
X_full = iris.data
y_full = iris.target
mask = (y_full != 2)
X_2c = X_full[mask]
y_2c = y_full[mask]
y_2c = np.where(y_2c == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X_2c[:, :2], y_2c, test_size=0.3, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape )

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

def margin_selection(X, y, w):
    return CustomLinearClassifier.margin_based_presentation(X, y, w)

clf_sgd = CustomLinearClassifier(method='sgd_momentum', lr=0.01, gamma=0.9, reg_alpha=0.01, epochs=10000, selection=margin_selection, init_type='correlation')
clf_sgd.fit(X_train_s, y_train)
acc_sgd = clf_sgd.score(X_test_s, y_test)

clf_steepest = CustomLinearClassifier(method='steepest_gd', reg_alpha=0.01, epochs=10000, init_type='correlation')
clf_steepest.fit(X_train_s, y_train)
acc_steepest = clf_steepest.score(X_test_s, y_test)

sgd_lib = SGDClassifier(loss="squared_error", penalty="l2", max_iter=8000, random_state=42, alpha=0.01)
sgd_lib.fit(X_train_s, y_train)
acc_lib = accuracy_score(y_test, sgd_lib.predict(X_test_s))

print(acc_sgd, acc_steepest, acc_lib)


fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

classifiers = [
    (clf_sgd, "SGD+Momentum"),
    (clf_steepest, "SteepestGD"),
    (sgd_lib, "SGDClassifier")
]

xx, yy = np.meshgrid(
    np.linspace(X_train_s[:,0].min()-1, X_train_s[:,0].max()+1, 200),
    np.linspace(X_train_s[:,1].min()-1, X_train_s[:,1].max()+1, 200)
)

for i, (clf, title) in enumerate(classifiers):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[i].contourf(xx, yy, Z, alpha=0.3)
    axes[i].scatter(X_train_s[:,0], X_train_s[:,1], c=y_train, edgecolors='k')
    axes[i].set_title(title)

plt.show()