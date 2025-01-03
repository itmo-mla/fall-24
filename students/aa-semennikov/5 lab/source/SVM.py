import numpy as np
from scipy.optimize import minimize

class SVM:
    def __init__(self, kernel, C=1, gamma=0.25, d=3):
        # Тип ядра
        self.kernel = kernel
        # Параметр регуляризации
        self.C = C
        # Коэф-ты Лагранжа
        self.lambdas = None
        # Смещения
        self.b = None
        # Параметр RBF ядра
        self.gamma = gamma
        # Степень полинома (для соотв-го ядра)
        self.d = d
        # Опорные вектора
        self.sup_vec = None
        # Метки классов опорных векторов
        self.sup_vec_labels = None
        self.weights = None

    def fit(self, X, y):
        # Определяем ф. Лагранжа в зависимости от типа ядра
        def lagrangian(lambdas):
            if self.kernel == 'RBF':
                # Считаем матрицу ядра
                K = np.array([[self.rbf_kernel(X[i], X[j]) for j in range(X.shape[0])] for i in range(X.shape[0])])
            elif self.kernel == 'polynomial':
                # -||-
                K = np.array([[self.polynomial_kernel(X[i], X[j]) for j in range(X.shape[0])] for i in range(X.shape[0])])
            else: K = X @ X.T # -||-
            # Первый член (до знака '-') отвечает за максимизацию ширины отступа, а второй - за регуляризацию
            return 0.5 * np.sum(lambdas[:, None] * lambdas[None, :] * y[:, None] * y[None, :] * K) - np.sum(lambdas)

        # Ограничение множества значений лямбд
        lambdas_E = [(0, self.C) for _ in range(X.shape[0])]

        # Решение двойственной задачи по лямбда
        # Задаем ограничение для оптимизации: сумма ск. произведений коэф. Лагранжа на метки классов должна строго '= 0'
        constraints = {'type': 'eq', 'fun': lambda lambdas: lambdas @ y}
        # Используем scipy.optimize.minimize для минимизации лагранжиана с нужными ограничениями
        self.lambdas = minimize(fun=lagrangian, x0=np.zeros(X.shape[0]), bounds=lambdas_E, constraints=constraints).x

        # Индексы опорных векторов
        idx = self.lambdas > 1e-4
        # Опорные вектора
        self.sup_vec = X[idx]
        # Метки классов опорных векторов
        self.sup_vec_labels = y[idx]
        # Сохраняем в лямбды коэф-ты, соответствующие опорным объектам (остальные не нужны, они не влияют на положение гиперпл.)
        self.lambdas = self.lambdas[idx]

        # Считаем смещение: из y*(w*x + b) = 1 (условия на отступа для опорных объектов) следует, что b = y - w*X
        self.b = np.mean(self.sup_vec_labels - X[idx] @ self.get_weights())

    def rbf_kernel(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def polynomial_kernel(self, x1, x2):
        return (x1 @ x2 + 1) ** self.d

    # Считаем вектор весов как взвешенную сумму lambda^i * y^i * x^i
    def get_weights(self):
        return np.sum(self.lambdas[:, None] * self.sup_vec_labels[:, None] * self.sup_vec, axis=0)

    def predict(self, X):
        self.weights = self.get_weights()
        return np.sign(X @ self.weights + self.b)