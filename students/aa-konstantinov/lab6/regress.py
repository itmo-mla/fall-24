import numpy as np 


class Log_reg:
    def __init__(self, reg = 0.001 ):
        self.reg = reg
    def fit(self, X, y):
        V, D, U = np.linalg.svd(X, full_matrices=False)
        w = D / (self.reg + D ** 2)
        self.coef_ = U.T @ (w * (V.T @ y))
    def predict(self, X):
        return X @ self.coef_
    def calc_reg(self, X_train, X_test, y_train, y_test):
        '''
        Функция для подбора оптимального параметра регуляризации (см. слайд 13).
        Модель обучается с различными значениями параметра регуляризации и выбирает то, при котором среднеквадратичная ошибка (MSE) на тестовой выборке минимально.
        '''
        reg = np.arange(0, 100, 0.001)
        mse = []
        for i in reg:
            self.reg = i
            self.fit(X_train,y_train )
            mse.append(np.mean((X_test @ self.coef_ - y_test)**2))
        self.reg = reg[np.argmin(np.array(mse))]


        




    