import numpy as np

class MomentumSGD:
    def __init__(self, lr=1e-2, momentum=0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def init_weight(self, weights, biases):
        if self.velocity_w is None:
            self.velocity_w = np.zeros_like(weights)
        if self.velocity_b is None:
            self.velocity_b = np.zeros_like(biases)

    def calc_step(self, dW, db):
        self.velocity_w = self.momentum * self.velocity_w + (1 - self.momentum) * dW
        self.velocity_b = self.momentum * self.velocity_b + (1 - self.momentum) * db

        return self.lr * self.velocity_w, self.lr * self.velocity_b
    
class FastGD:
    def __init__(self, lr):
        np.seterr(all='warn')
        self.lr = lr

    def init_weight(self, weights, biases):
        pass

    def calc_step(self, dW, db, X):
        h = 1. / (np.sum(X**2, axis=1) + self.lr)
        return h * dW, h * db
