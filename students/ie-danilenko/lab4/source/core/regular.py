import numpy as np

class L2:
    def __init__(self, lam = 0.01):
        self.lam = lam

    def get(self, weight):
        return self.lam / 2 * np.sum(weight ** 2)
    
    def dget(self, weight):
        return self.lam * weight