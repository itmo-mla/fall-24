import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from time import time


class RidgeRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y, alpha=1.0):
        # Добавляем столбец единиц для интерсепта
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Выполняем SVD разложение
        U, s, Vt = np.linalg.svd(X_b, full_matrices=False)

        # Вычисляем коэффициенты с регуляризацией
        s_alpha = s / (s**2 + alpha)
        S_alpha = np.diag(s_alpha)
        self.coefficients = Vt.T @ S_alpha @ U.T @ y

        # Разделяем интерсепт и коэффициенты
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

        return self

    def predict(self, X):
        return X @ self.coefficients + self.intercept

    def score(self, X, y):
        return r2_score(y, self.predict(X))


def find_optimal_alpha(X_train, X_val, y_train, y_val, alphas=None):
    if alphas is None:
        alphas = np.logspace(-4, 4, 100)

    best_alpha = None
    best_score = -np.inf

    for alpha in alphas:
        model = RidgeRegression()
        model.fit(X_train, y_train, alpha=alpha)
        score = model.score(X_val, y_val)

        if score > best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha, best_score


# Загрузка данных
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Разделение данных
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Поиск оптимального alpha
best_alpha, best_score = find_optimal_alpha(
    X_train_scaled, X_val_scaled, y_train, y_val
)
print(f"Оптимальное значение alpha: {best_alpha:.4f}")
print(f"Лучший R² score на валидационной выборке: {best_score:.4f}")

# Обучение моделей с оптимальным alpha
start = time()
our_model = RidgeRegression()
our_model.fit(X_train_scaled, y_train, alpha=best_alpha)
our_score = our_model.score(X_test_scaled, y_test)
print(f"\nВремя работы моей модели: {time()-start}")
# Сравнение с эталонной реализацией
start = time()
sklearn_model = Ridge(alpha=best_alpha)
sklearn_model.fit(X_train_scaled, y_train)
sklearn_score = sklearn_model.score(X_test_scaled, y_test)
print(f"Время работы Sklearn модели: {time()-start}")

print("\nСравнение на тестовой выборке:")
print(f"Моя модель R²: {our_score:.4f}")
print(f"Sklearn модель R²: {sklearn_score:.4f}")
