import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

data = pd.read_csv('insurance.csv')

data = pd.get_dummies(data, drop_first=True)

X = data.drop('charges', axis=1).values
y = data['charges'].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.35, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(X_train)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

X_val = scaler_X.transform(X_val)
y_val = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

X_test = scaler_X.transform(X_test)
y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

def ridge_regression_svd(X, y, tau):
    # SVD разложение
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # Обратная к D^2 + tau
    D_inv = np.diag(S / (S**2 + tau))
    # Вычисляем веса
    w_tau = Vt.T @ D_inv @ U.T @ y
    return w_tau

def find_best_tau(X_train, y_train, X_val, y_val, taus):
    best_tau = None
    best_mse = float('inf')
    for tau in taus:
        w_tau = ridge_regression_svd(X_train, y_train, tau)
        y_pred = X_val @ w_tau
        mse = mean_squared_error(y_val, y_pred)
        if mse < best_mse:
            best_mse = mse
            best_tau = tau
    return best_tau, best_mse


# taus = np.logspace(-2, 6, 100)
taus = np.linspace(0, 1, 10000)
best_tau, best_mse = find_best_tau(X_train, y_train, X_val, y_val, taus)

w_best = ridge_regression_svd(X_train, y_train, best_tau)

y_test_pred = X_test @ w_best
test_mse = mean_squared_error(y_test, y_test_pred)

ridge_model = Ridge(alpha=best_tau)
ridge_model.fit(X_train, y_train)
sklearn_test_mse = mean_squared_error(y_test, ridge_model.predict(X_test))

print(f"Best tau: {best_tau}")
print(f"Test MSE (custom implementation): {round(test_mse, 4)}")
print(f"Test MSE (scikit-learn): {round(sklearn_test_mse, 4)}")

mse_values = []
for tau in taus:
    w_tau = ridge_regression_svd(X_train, y_train, tau)
    y_pred = X_val @ w_tau
    mse_values.append(mean_squared_error(y_val, y_pred))

plt.figure(figsize=(8, 6))
plt.plot(taus, mse_values, label='Validation MSE', marker='o')
plt.axvline(best_tau, color='r', linestyle='--', label=f'Best tau: {best_tau:.2f}')
plt.xscale('log')
plt.xlabel('Tau (Regularization Parameter)')
plt.ylabel('Mean Squared Error')
plt.title('Validation MSE vs. Tau')
plt.legend()
plt.grid(True)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(20, 6))

axes[0].scatter(y_test, y_test_pred, alpha=0.5, label='Custom Implementation', color='blue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
axes[0].legend()

axes[1].scatter(y_test, ridge_model.predict(X_test), alpha=0.5, label='scikit-learn', color='orange')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
axes[1].legend()

plt.tight_layout()
plt.show()
