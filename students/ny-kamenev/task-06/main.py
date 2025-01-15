from regression import Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import time


df = pd.read_csv('./datasets/boston.csv')
print(df.head(5))
X = df.drop(['MEDV'], axis=1)
y = df['MEDV']

X_train, X_tail, y_train, y_tail = train_test_split(X, y, test_size=0.4, random_state=3)

X_val, X_test, y_val, y_test = train_test_split(X_tail, y_tail, test_size=0.5, random_state=3)

start_time = time.time()
model = Regression()
optimal_param = model.find_optimal(X_train, X_val, y_train, y_val, min=0, max=10, step=0.01, criterion='mse')
end_time = time.time()
print(f'Optimal parameter: {optimal_param}')
print(f"Time spent on finding the optimal parameter: {end_time - start_time:.4f} seconds")


print("\n\n\nManual")
start_time = time.time()
model = Regression(param=optimal_param)
model.fit(X_train, y_train)
pred = pd.Series(model.predict(X_test))
end_time = time.time()
print(f'r2: {r2_score(pred, y_test)}')
print(f'mse: {mean_squared_error(pred, y_test)}')
print(f'mae: {mean_absolute_error(pred, y_test)}')
print(f"Time taken manual: {end_time - start_time:.4f} seconds")

plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label='Actual values')
plt.plot(range(len(pred)), pred, label='Predicted values')
plt.title('Manual')
plt.legend()
plt.grid(True)
plt.savefig('./img/manual.png')
plt.close()

print("\n\n\nSklearn")
start_time = time.time()
model = Ridge(alpha=optimal_param)
model.fit(X_train, y_train)
pred_sklearn = model.predict(X_test)
end_time = time.time()
print(f'r2: {r2_score(pred_sklearn, y_test)}')
print(f'mse: {mean_squared_error(pred_sklearn, y_test)}')
print(f'mae: {mean_absolute_error(pred_sklearn, y_test)}')
print(f"Time taken manual: {end_time - start_time:.4f} seconds")

plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label='Actual values')
plt.plot(range(len(pred_sklearn)), pred_sklearn, label='Predicted values')
plt.title('Sklearn')
plt.legend()
plt.grid(True)
plt.savefig('./img/sklearn.png')
plt.close()


plt.figure(figsize=(10, 6))
plt.plot(range(len(pred)), pred, label='Manual')
plt.plot(range(len(pred_sklearn)), pred_sklearn, label='Sklearn')
plt.title('Compare')
plt.legend()
plt.grid(True)
plt.savefig('./img/compare.png')
plt.close()
