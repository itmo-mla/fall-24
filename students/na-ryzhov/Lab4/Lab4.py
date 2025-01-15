import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier

class LinearClassifier:
    def __init__(self, algorithm='momentum_sgd', learning_rate=0.01, momentum=0.9, regularization=0.0, max_epochs=50, sampling_fn=None, weight_init='zeros'):
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.sampling_fn = sampling_fn
        self.weight_init = weight_init
        self.weights = None
        self.loss_history = []

    def train(self, X, y):
        if self.algorithm == 'momentum_sgd':
            self.weights = self._momentum_sgd(X, y)
        elif self.algorithm == 'steepest_descent':
            self.weights = self._steepest_descent(X, y)

    def predict(self, X):
        return np.sign(X.dot(self.weights))

    def evaluate(self, X, y):
        return accuracy_score(y, self.predict(X))

    def _initialize_weights(self, X, y):
        if self.weight_init == 'zeros':
            return np.zeros(X.shape[1])
        elif self.weight_init == 'random':
            return np.random.normal(0, 0.01, X.shape[1])
        elif self.weight_init == 'correlation':
            return np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        return np.zeros(X.shape[1])

    def _gradient_clipping(self, grad, threshold=500):
        return np.clip(grad, -threshold, threshold)

    def _gradient(self, w, x, y):
        loss_grad = 2 * (np.dot(w, x) - y) * x + self.regularization * w
        return self._gradient_clipping(loss_grad)

    def _epoch_loss(self, X, y, w):
        return np.mean((y - X.dot(w))**2) + self.regularization * np.sum(w**2)

    def _momentum_sgd(self, X, y):
        w = self._initialize_weights(X, y)
        velocity = np.zeros_like(w)
        for _ in range(self.max_epochs):
            if self.sampling_fn is not None:
                X, y = self.sampling_fn(X, y, w)
            for i in np.random.permutation(len(y)):
                grad = self._gradient(w, X[i], y[i])
                velocity = self.momentum * velocity + self.learning_rate * grad
                w -= velocity
            self.loss_history.append(self._epoch_loss(X, y, w))
        return w

    def _steepest_descent(self, X, y):
        w = self._initialize_weights(X, y)
        for _ in range(self.max_epochs):
            if self.sampling_fn is not None:
                X, y = self.sampling_fn(X, y, w)
            diff = y - X.dot(w)
            grad = -2 * X.T.dot(diff) + self.regularization * w
            step_size = grad.dot(grad) / max(X.dot(grad).dot(X.dot(grad)), 1e-10)
            w -= step_size * grad
            self.loss_history.append(self._epoch_loss(X, y, w))
        return w

# Загружаем данные и готовим их для классификации
iris_data = load_iris()
X_data = iris_data.data
y_data = iris_data.target
binary_mask = y_data != 2
X_binary = X_data[binary_mask]
y_binary = np.where(y_data[binary_mask] == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X_binary[:, :2], y_binary, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def margin_based_sampling(X, y, w):
    margins = y * np.dot(X, w)
    indices = np.argsort(np.abs(margins))
    return X[indices], y[indices]

# Инициализация и обучение моделей
classifier_sgd = LinearClassifier(algorithm='momentum_sgd', learning_rate=0.01, momentum=0.9, regularization=0.01, max_epochs=10000, sampling_fn=margin_based_sampling, weight_init='correlation')
classifier_sgd.train(X_train_scaled, y_train)
sgd_accuracy = classifier_sgd.evaluate(X_test_scaled, y_test)

classifier_sd = LinearClassifier(algorithm='steepest_descent', regularization=0.01, max_epochs=10000, weight_init='correlation')
classifier_sd.train(X_train_scaled, y_train)
sd_accuracy = classifier_sd.evaluate(X_test_scaled, y_test)

sgd_builtin = SGDClassifier(loss="squared_error", penalty="l2", max_iter=8000, random_state=42, alpha=0.01)
sgd_builtin.fit(X_train_scaled, y_train)
builtin_accuracy = accuracy_score(y_test, sgd_builtin.predict(X_test_scaled))

# Вывод результатов
print(sgd_accuracy, sd_accuracy, builtin_accuracy)

# Визуализация решений
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
models = [(classifier_sgd, "SGD+Momentum"), (classifier_sd, "Steepest Descent"), (sgd_builtin, "SGDClassifier")]
xx, yy = np.meshgrid(
    np.linspace(X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1, 200),
    np.linspace(X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1, 200)
)

for i, (model, title) in enumerate(models):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[i].contourf(xx, yy, Z, alpha=0.3)
    axes[i].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, edgecolors='k')
    axes[i].set_title(title)

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier

class LinearClassifier:
    def __init__(self, algorithm='momentum_sgd', learning_rate=0.01, momentum=0.9, regularization=0.0, max_epochs=50, sampling_fn=None, weight_init='zeros'):
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.sampling_fn = sampling_fn
        self.weight_init = weight_init
        self.weights = None
        self.loss_history = []

    def train(self, X, y):
        if self.algorithm == 'momentum_sgd':
            self.weights = self._momentum_sgd(X, y)
        elif self.algorithm == 'steepest_descent':
            self.weights = self._steepest_descent(X, y)

    def predict(self, X):
        return np.sign(X.dot(self.weights))

    def evaluate(self, X, y):
        return accuracy_score(y, self.predict(X))

    def _initialize_weights(self, X, y):
        if self.weight_init == 'zeros':
            return np.zeros(X.shape[1])
        elif self.weight_init == 'random':
            return np.random.normal(0, 0.01, X.shape[1])
        elif self.weight_init == 'correlation':
            return np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        return np.zeros(X.shape[1])

    def _gradient_clipping(self, grad, threshold=500):
        return np.clip(grad, -threshold, threshold)

    def _gradient(self, w, x, y):
        loss_grad = 2 * (np.dot(w, x) - y) * x + self.regularization * w
        return self._gradient_clipping(loss_grad)

    def _epoch_loss(self, X, y, w):
        return np.mean((y - X.dot(w))**2) + self.regularization * np.sum(w**2)

    def _momentum_sgd(self, X, y):
        w = self._initialize_weights(X, y)
        velocity = np.zeros_like(w)
        for _ in range(self.max_epochs):
            if self.sampling_fn is not None:
                X, y = self.sampling_fn(X, y, w)
            for i in np.random.permutation(len(y)):
                grad = self._gradient(w, X[i], y[i])
                velocity = self.momentum * velocity + self.learning_rate * grad
                w -= velocity
            self.loss_history.append(self._epoch_loss(X, y, w))
        return w

    def _steepest_descent(self, X, y):
        w = self._initialize_weights(X, y)
        for _ in range(self.max_epochs):
            if self.sampling_fn is not None:
                X, y = self.sampling_fn(X, y, w)
            diff = y - X.dot(w)
            grad = -2 * X.T.dot(diff) + self.regularization * w
            step_size = grad.dot(grad) / max(X.dot(grad).dot(X.dot(grad)), 1e-10)
            w -= step_size * grad
            self.loss_history.append(self._epoch_loss(X, y, w))
        return w

# Загружаем данные и готовим их для классификации
iris_data = load_iris()
X_data = iris_data.data
y_data = iris_data.target
binary_mask = y_data != 2
X_binary = X_data[binary_mask]
y_binary = np.where(y_data[binary_mask] == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X_binary[:, :2], y_binary, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def margin_based_sampling(X, y, w):
    margins = y * np.dot(X, w)
    indices = np.argsort(np.abs(margins))
    return X[indices], y[indices]

# Инициализация и обучение моделей
classifier_sgd = LinearClassifier(algorithm='momentum_sgd', learning_rate=0.01, momentum=0.9, regularization=0.01, max_epochs=10000, sampling_fn=margin_based_sampling, weight_init='correlation')
classifier_sgd.train(X_train_scaled, y_train)
sgd_accuracy = classifier_sgd.evaluate(X_test_scaled, y_test)

classifier_sd = LinearClassifier(algorithm='steepest_descent', regularization=0.01, max_epochs=10000, weight_init='correlation')
classifier_sd.train(X_train_scaled, y_train)
sd_accuracy = classifier_sd.evaluate(X_test_scaled, y_test)

sgd_builtin = SGDClassifier(loss="squared_error", penalty="l2", max_iter=8000, random_state=42, alpha=0.01)
sgd_builtin.fit(X_train_scaled, y_train)
builtin_accuracy = accuracy_score(y_test, sgd_builtin.predict(X_test_scaled))

# Вывод результатов
print(f"sgd_accuracy: {sgd_accuracy:.4f}")
print(f"sd_accuracy: {sd_accuracy:.4f}")
print(f"builtin_accuracy: {builtin_accuracy:.4f}")

# Визуализация решений
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
models = [(classifier_sgd, "SGD+Momentum"), (classifier_sd, "Steepest Descent"), (sgd_builtin, "SGDClassifier")]
xx, yy = np.meshgrid(
    np.linspace(X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1, 200),
    np.linspace(X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1, 200)
)

for i, (model, title) in enumerate(models):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[i].contourf(xx, yy, Z, alpha=0.3)
    axes[i].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, edgecolors='k')
    axes[i].set_title(title)

plt.show()
