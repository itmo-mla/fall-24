import random

import numpy as np


def quadratic_margin_loss(w, X, y):
    M = (w @ X) * y
    return (1-M)**2

def quadratic_margin_dloss(w, X, y):
    M = (w @ X) * y
    return -2*(1-M)*(y @ X.T)


class LinearClassifier:
    def __init__(self, n_features):
        self.n_features = n_features
        self.Q = None

        self.v = 0

        self.loss_history = []
        self.Q_history = []
    
    def init_weights(self, w=None):
        if w is None:
            self.w = np.random.uniform(low=-1/(2*self.n_features), high=1/(2*self.n_features), size=(1, self.n_features))
        else: self.w = w
        
    def fit(self, X, Y, n_iter, lr, lambda_, reg, momentum, gamma, optimize_lr, use_margins):
        if self.Q is None:
            random_sample = np.random.choice(range(len(X)), size=(30))
            random_X_sample = X[random_sample]
            random_y_sample = Y[random_sample]

            self.Q = np.mean([quadratic_margin_loss(self.w, x, y) for (x, y) in zip(random_X_sample, random_y_sample)])
        
        for _ in range(n_iter):
            if use_margins:
                margins = ((X @ self.w.T) * Y).flatten()
                abs_inv_margins = max(abs(margins)) - abs(margins)
                abs_inv_margins = abs_inv_margins / sum(abs_inv_margins)
                idx = np.random.choice(np.arange(len(X)), p=abs_inv_margins)
                x, y = X[idx], Y[idx]
            else:
                x, y = random.choice(list(zip(X, Y)))

            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            
            l = quadratic_margin_loss(self.w, x, y)

            if optimize_lr:
                lr = 1 / sum(x**2)
            if momentum:
                self.v = gamma * self.v + (1 - gamma) * quadratic_margin_dloss(self.w - lr*gamma*self.v, x, y)
                self.w = self.w * (1-lr*reg) - lr*self.v
            else:
                self.w = self.w * (1-lr*reg) - lr*quadratic_margin_dloss(self.w, x, y)
            
            self.Q = lambda_*l + (1-lambda_)*self.Q

            self.loss_history += [l[0, 0]]
            self.Q_history += [self.Q[0, 0]]

    def predict(self, X):
        return np.sign(X @ self.w.T)