from read import read
from sklearn.model_selection import train_test_split
from regression import RidgeRegression
from sklearn.linear_model import Ridge
import numpy as np
from time import time
import matplotlib.pyplot as plt

def mse(y_pred, y):
    return np.mean(np.square(y - y_pred))

X, y = read('dataset/Food_Delivery_Times.csv')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = RidgeRegression(alpha=1.0)
start_time_custom = time()
model.fit(X_train, y_train)
custom_pred = model.predict(X_val)
custom_time = time() - start_time_custom
custom_mse = mse(custom_pred, y_val)

sk_model = Ridge(alpha=1.0)
start_time_sklearn = time()
sk_model.fit(X_train, y_train)
sklearn_pred = sk_model.predict(X_val)
sklearn_time = time() - start_time_sklearn
sklearn_mse = mse(sklearn_pred, y_val)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar(['My Ridge', 'Sklearn Ridge'], [custom_mse, sklearn_mse], label=['My', 'Sklearn'], color=['blue', 'orange'])
ax1.set_title('MSE')
ax1.set_ylabel('MSE')
ax1.legend()

ax2.bar(['My Ridge', 'Sklearn Ridge'], [custom_time, sklearn_time], label=['My', 'Sklearn'], color=['blue', 'orange'])
ax2.set_title('Time')
ax2.set_ylabel('Time (seconds)')
ax2.legend()

plt.tight_layout()
plt.show()