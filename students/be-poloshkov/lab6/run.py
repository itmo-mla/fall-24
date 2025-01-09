import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from linear_reg import MyRidgeRegression

def main():
    df = pd.read_csv('./Student_Performance.csv')

    le = LabelEncoder()
    df['Extracurricular Activities'] = le.fit_transform(df['Extracurricular Activities'])

    X, y = df.drop(columns=['Performance Index']), df['Performance Index']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    btau, bq = 0, np.inf
    mse, r2, qual = [], [], []
    tau_range = range(1001)

    for tau in tau_range:
        my_reg = MyRidgeRegression(tau=tau)
        my_reg.fit(X_train, y_train)
        q = my_reg.quality(X_test, y_test)
        if q < bq:
            bq = q
            btau = tau

        qual.append(q)
        y_pred = my_reg.predict(X_test)
        r2.append(r2_score(y_test, y_pred))
        mse.append(mean_squared_error(y_test, y_pred))

    graph_history('mse', tau_range, mse)
    graph_history('r2', tau_range, r2)
    graph_history('quality', tau_range, qual)

    models = [('MyRidgeRegression', MyRidgeRegression(tau=589)), ('SKLearn Ridge Regression', Ridge())]
    compare_models(models, X_train, X_test, y_train, y_test)

    print(f'best tau: {btau}, best quality: {bq}')

def graph_history(name, taus, arr):
    plt.title(name)
    plt.plot(taus, arr)
    plt.show()

def compare_models(models, X_train, X_test, y_train, y_test):
    x = list(range(len(X_test)))

    for label, model in models:
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end = time.time()

        print(label)
        print(f'Time elapsed: {end - start}')
        print(f'R^2: {r2_score(y_test, y_pred)}')
        print(f'MSE: {mean_squared_error(y_test, y_pred)}')

        plt.title(label)
        plt.plot(x, y_pred, label='Predicted')
        plt.plot(x, y_test, label='True')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()