import numpy as np
import matplotlib.pyplot as plt


class LinRegress:
    def __init__(self, tau=0):
        self.W = None
        self.tau = tau

    def fit(self, x, y, tau=None):
        if tau is not None:
            self.tau = tau
        U, D, V_T = np.linalg.svd(x, full_matrices=False)
        self.W = V_T.T @ np.linalg.inv(np.diag(D**2) + self.tau * np.eye(D.shape[0])) @ np.diag(D) @ U.T @ y

    def fit_tau(self, x_train, x_val, y_train, y_val, tau_range: np.ndarray):
        q_history = []
        best_weights, best_tau, best_q = None, 0, float('inf')
        U, D, V_T = np.linalg.svd(x_train, full_matrices=False)
        for tau in tau_range:
            self.tau = tau
            D_tau = np.diag(D) / (D**2 + tau)
            q = np.linalg.norm(x_val @ V_T.T @ D_tau @ U.T @ y_train - y_val) ** 2
            if q < best_q:
                best_tau = tau
                best_q = q

            q_history.append(q)
        self.tau = best_tau
        self.W = V_T.T @ np.linalg.inv(np.diag(D**2) + self.tau * np.eye(D.shape[0])) @ np.diag(D) @ U.T @ y_train
        return q_history

    def predict(self, x):
        if self.W is None:
            raise NotImplementedError('Сначала необходимо выполнить fit')
        else:
            return x @ self.W


def plot_result(y_test, y_pred, idxs):
    x_range = np.arange(idxs[0], idxs[1])
    plt.plot(x_range, y_test[idxs[0]:idxs[1]], label='True')
    plt.plot(x_range, y_pred[idxs[0]:idxs[1]], label='Pred')
    plt.legend(loc='upper left')
    plt.show()
