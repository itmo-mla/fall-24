import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import time

class CustomRidge:
    def __init__(self):
        self.weights = None
        self.bias = None

    def train(self, features, target, regularization=1.0):
        extended_features = np.hstack([np.ones((features.shape[0], 1)), features])

        U, singular_values, V_transpose = np.linalg.svd(extended_features, full_matrices=False)

        adjusted_singular = singular_values / (singular_values**2 + regularization)
        diagonal_matrix = np.diag(adjusted_singular)

        coefficients = V_transpose.T @ diagonal_matrix @ U.T @ target

        self.bias = coefficients[0]
        self.weights = coefficients[1:]
        return self

    def predict(self, features):
        return np.dot(features, self.weights) + self.bias

    def evaluate(self, features, target):
        predictions = self.predict(features)
        return r2_score(target, predictions)

def optimal_regularization(train_features, val_features, train_target, val_target, alpha_values=None):
    if alpha_values is None:
        alpha_values = np.logspace(-4, 4, 100)

    best_alpha = None
    highest_r2 = float('-inf')

    for alpha in alpha_values:
        model = CustomRidge()
        model.train(train_features, train_target, regularization=alpha)
        r2 = model.evaluate(val_features, val_target)

        if r2 > highest_r2:
            highest_r2 = r2
            best_alpha = alpha

    return best_alpha, highest_r2

housing_data = fetch_california_housing()
data, targets = housing_data.data, housing_data.target

train_data, temp_data, train_targets, temp_targets = train_test_split(data, targets, test_size=0.3, random_state=42)
val_data, test_data, val_targets, test_targets = train_test_split(temp_data, temp_targets, test_size=0.5, random_state=42)

scaler = StandardScaler()
train_data_normalized = scaler.fit_transform(train_data)
val_data_normalized = scaler.transform(val_data)
test_data_normalized = scaler.transform(test_data)

best_alpha, best_r2 = optimal_regularization(train_data_normalized, val_data_normalized, train_targets, val_targets)
print(f"Наиболее подходящее значение регуляризации: {best_alpha:.4f}")
print(f"Максимальное значение R² на валидации: {best_r2:.4f}")

start_time = time.time()
custom_model = CustomRidge()
custom_model.train(train_data_normalized, train_targets, regularization=best_alpha)
custom_r2 = custom_model.evaluate(test_data_normalized, test_targets)
print(f"Время обучения и предсказания (собственная модель): {time.time() - start_time:.4f} секунд")

start_time = time.time()
sklearn_ridge = Ridge(alpha=best_alpha)
sklearn_ridge.fit(train_data_normalized, train_targets)
sklearn_r2 = sklearn_ridge.score(test_data_normalized, test_targets)
print(f"Время обучения и предсказания (scikit-learn): {time.time() - start_time:.4f} секунд")

print("\nСравнение на тестовых данных:")
print(f"R² собственной модели: {custom_r2:.4f}")
print(f"R² модели scikit-learn: {sklearn_r2:.4f}")