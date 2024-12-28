import numpy as np
from scipy.optimize import minimize

def optimizer(y_train, C, kernel):
    def _optim_func(lam):
        N= lam.shape[0]
        N1 = y_train.shape[0]
        lam1 = lam.reshape(N , 1)
        lam2 = lam.reshape(1,N)
        y_train1 = lam.reshape(N1 , 1)
        y_train2 = lam.reshape( 1,N1)

        return -np.sum(lam) + 0.5 * np.sum(
            lam1 * lam2 * y_train1 * y_train2 * kernel
        )
    min_it = minimize(
        _optim_func, np.zeros(len(y_train)),
        method = 'SLSQP',
        constraints=[
            {'type': 'ineq', 'fun': lambda x: x},  
            {'type': 'ineq', 'fun': lambda x: C - x},  
            {'type': 'eq', 'fun': lambda x: np.dot(x, y_train)}  
        ],
        bounds=[(0, C) for _ in range( len(y_train) )]
    )
    return min_it.x

class SVC:
    def __init__(self, C):
        self.w0 = 0
        self.C = C

    def kernel_calc(self, X, X_):
        if self.kernel_type == 'linear':
            return np.dot(X, X_.T)
        if self.kernel_type == 'rbf':
            X1 = X ** 2
            X2 = X_ ** 2
            X1 = np.sum(X1 , axis = 1).reshape(X1.shape[0], 1)
            X2 = np.sum(X2 , axis = 1).reshape( 1, X2.shape[0])
            return np.exp(-self.gamma * (X1 - 2 * np.dot(X , X_.T) + X2))
        if self.kernel_type == 'polynom':
            return (np.dot(X, X_.T) + 1 ) ** self.d


    def fit(self, X_train, y_train, kernel_type, gamma=None, d=None):
        self.kernel_type = kernel_type
        self.X = X_train
        self.gamma = gamma
        self.d = d
        kernel = self.kernel_calc(X_train, X_train)
        self.lam = optimizer(y_train, self.C, kernel)
        
        indices = self.lam <= self.C
        self.X_ind = X_train[indices]
        self.Y_ind = y_train[indices]
        self.lam = self.lam[indices]
        kernel_id = kernel[indices][:, indices]
        if kernel_type == 'linear':
            self.w = np.sum(self.lam[:, None] * self.Y_ind[:, None] * self.X_ind, axis=0)
            self.w0 = np.mean(self.Y_ind - np.dot(self.X_ind, self.w))
        else:
            self.w0 = np.mean(
                self.Y_ind - np.sum(self.lam[:, None] * self.Y_ind[:, None] * kernel_id, axis=0)
            )
       


        
    def predict(self, X):
        kernel = self.kernel_calc(X, self.X_ind)
        if self.kernel_type == 'linear':
            decision = np.dot(X, self.w) + self.w0
        else:
            decision = np.sum(self.lam[:, None] * self.Y_ind[:, None] * kernel.T, axis=0) + self.w0
        return np.sign(decision)



