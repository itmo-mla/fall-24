from Regressor import RidgeRegression
from Loader import Loader
from metrics import mse, rmse, r2

import pandas as pd
from sklearn.linear_model import Ridge


loader = Loader()
X_train, X_test, y_train, y_test = loader.X_train, loader.X_test, loader.y_train, loader.y_test

alphas = [0.1, 1, 10, 100]
optimal_alpha = RidgeRegression.select_optimal_alpha(X_train, y_train, alphas)

model = RidgeRegression(alpha=optimal_alpha)
model.train(X_train, y_train)
pred = pd.Series(model.predict(X_test))
print(f'r2 ручной алгоритм: {r2(pred, y_test)}')
print(f'mse ручной алгоритм: {mse(pred, y_test)}')
print(f'rmse ручной алгоритм: {rmse(pred, y_test)}')


ridge_model = Ridge(alpha=optimal_alpha)
ridge_model.fit(X_train, y_train)
pred = ridge_model.predict(X_test)
print(f'r2 библиотечный алгоритм: {r2(pred, y_test)}')
print(f'mse библиотечный алгоритм: {mse(pred, y_test)}')
print(f'rmse библиотечный алгоритм: {rmse(pred, y_test)}')

