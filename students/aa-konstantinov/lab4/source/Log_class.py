import numpy as np


class Long_cl:
    def __init__(self, reg, weights_type, lam, gam, epochs=100):
        self.weights_type = weights_type
        self.reg = reg
        self.lam = lam
        self.gam = gam
        self.epochs = epochs
        self.w = None
        self.v = None  

    def init_weights(self, weights_type, X, y, multi = 20):
        if weights_type == "corr":
            w = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                w[i] = np.dot(y,X[:, i]) /     \
                    (np.dot(X[:, i], X[:, i]))
            #Check for corr
            correlation_matrix = np.corrcoef(X, rowvar=False)
            off_diagonal = correlation_matrix - np.diag(np.diag(correlation_matrix))
            if np.any(off_diagonal > 0.7) :
                print('Корреляци присутствует')
        
            return w   
        elif weights_type == "random":
            otr = 1 / (2 * X.shape[1])
            return np.random.uniform(-otr, otr, X.shape[1])
        elif weights_type == "multi":
            l = 1
            w_ = None
            for _ in range(multi):
                self.w = self.init_weights('random', X, y)
                losses = self.loss(X, y)
                if losses < l:
                    w_ = self.w 
                    l = losses
            return w_

    def indent(self, X, y):
        z = np.dot(X, self.w) * y
        return z

    def loss(self, X, y):
        loss = (y - np.dot(X, self.w)) ** 2 + self.reg * np.sum(self.w ** 2) / 2 
        return np.mean(loss)

    def gloss(self, X, y, W):
        grad = -2 * np.dot(X.T, (y - np.dot(X, W))) / len(y)
        grad += self.reg * self.w  # регул
        return grad
    def indexes(self, X_train, y_train):
        N, _ = X_train.shape
        indents = np.abs(self.indent(X_train, y_train))
        proba = 1 / (1 + indents)/ np.sum(1 / (1 + indents)) 
        index = np.random.choice(N, 10, False, proba)
        return index

    def fit(self, X_train, y_train):
        self.w = self.init_weights(self.weights_type, X_train, y_train)
        index = self.indexes(X_train, y_train)
        self.v = np.zeros_like(self.w)
        q = np.mean(self.loss(X_train[index], y_train[index]))  # Q
        for _ in range(self.epochs):
            index = self.indexes(X_train, y_train)
            h = 1 / np.linalg.norm(X_train[index])**2  # Нормированный шаг
            g_l = self.gloss(X_train[index], y_train[index], W=self.w - h * self.gam * self.v ) 
            self.v = self.gam * self.v + (1 - self.gam) * g_l  
            self.w = self.w*(1-h*self.reg) - h * self.v  # Учет регул
            ei = self.loss(X_train[index], y_train[index]) 
            q = self.lam * ei + (1 - self.lam) * q

    def predict(self, X):
        return np.sign(np.dot(X, self.w))

    
