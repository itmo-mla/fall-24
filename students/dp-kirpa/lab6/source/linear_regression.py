import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


data = pd.read_csv("Food_Delivery_Times.csv")

X, y = data["Distance_km"].to_numpy(), data["Delivery_Time_min"].to_numpy()

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train_centered = X_train - np.mean(X_train)
y_train_centered = y_train - np.mean(y_train)

T = 0

U, S, Vt = np.linalg.svd(X_train_centered.reshape(-1, 1), full_matrices=False)

betas = Vt.T @ np.divide(S, S * S + T, where=(S != 0)) @ U.T @ y_train

b = np.mean(y_train) - betas * np.mean(X_train)

best_t = 0
best_quality = 10**9
qualities = []

for current_t in range(251):
  checked_betas = Vt.T @ np.divide(S, S * S + current_t, where=(S != 0)) @ U.T @ y_train
  if (current_quality := np.linalg.norm(checked_betas * X_val + b - y_val)) < best_quality:
    best_quality = current_quality
    best_t = current_t
  qualities.append(current_quality)

y_pred = betas * X_test + b

quality = np.linalg.norm(y_pred - y_test)
