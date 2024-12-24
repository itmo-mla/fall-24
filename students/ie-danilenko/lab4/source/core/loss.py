import numpy as np

class MSE:
    @staticmethod
    def get(y_pred, y):
        return np.mean(np.square(y - y_pred))

    @staticmethod
    def dget(y_pred, y):
        n = y.shape[0]
        return (2/n)*(y_pred - y)

class RMSE:
    @staticmethod
    def get(y_pred, y):
        return np.sqrt(MSE.get(y_pred, y))
    
    @staticmethod
    def dget(y_pred, y):
       return  -(y - y_pred) / RMSE.get(y_pred, y)

class Lin:
    @staticmethod
    def get(X, y, weight):
        mul = np.matmul(X, weight.T)
        return mul * y
    
    @staticmethod
    def dget(X, y, weight):
        dm = 2 * (1 - Lin.get(X, y, weight))
        mul = X * y
        return -np.matmul(dm.T, mul)
