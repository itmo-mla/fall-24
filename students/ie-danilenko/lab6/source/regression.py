import numpy as np
from read import read
from sklearn.model_selection import train_test_split

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef = None
        
    def fit(self, X, y):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        s_diagonal = np.diag(s / (s**2 + self.alpha))
        self.coef = Vt.T @ s_diagonal @ U.T @ y
    
    def predict(self, X):
        return np.dot(X, self.coef)
    
def get_optim_alpha(X, y, alphas):
    best_alpha = None
    best_error = float('inf')

    for alpha in alphas:
        model = RidgeRegression(alpha)
        model.fit(X, y)
        pred = model.predict(X)
        error = np.mean((y - pred) ** 2)

        if error < best_error:
            best_error = error
            best_alpha = alpha

    return best_alpha
    
if __name__ == '__main__':
    X, y = read('dataset/Food_Delivery_Times.csv')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RidgeRegression()
    model.fit(X_train, y_train)
    pred_y = model.predict(X_val)
    print('Predict value: ', pred_y)
    print("True  value:", y_val)

    optim_alpha = get_optim_alpha(X_train, y_train, np.linspace(1 / 1e6, 10., int(1e6)))
    print('Optimal alpha:', optim_alpha)
    optim_model = RidgeRegression(optim_alpha)
    optim_model.fit(X_train, y_train)
    pred_y = optim_model.predict(X_val)
    print('Predict value: ', pred_y)
    print("True  value:", y_val)