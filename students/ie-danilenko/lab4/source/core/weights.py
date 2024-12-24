import numpy as np

class StaticWeightGenerator:
    def __init__(self, value):
        self.value = value

    def generate(self, shape):
        return np.ones(shape) * self.value
    

class RandomWeightGenerator:
    def generate(self, shape):
        value = shape[0] * shape[1]
        min_value = - 1  / (value * 2)
        max_value = 1  / (value * 2)
        return np.random.uniform(low=min_value, high=max_value, size=shape)
    
class CorrelationWeightGenerator:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def generate(self, shape):
        mean_X = np.mean(self.X, axis=0)
        mean_Y = np.mean(self.y, axis=0)

        X_centered = self.X - mean_X
        Y_centered = self.y - mean_Y

        weights = np.zeros(shape)
        n_outputs, n_features = shape

        for j in range(n_outputs):
            for i in range(n_features):
                covariance = np.sum(X_centered[:, i] * Y_centered[:, j]) / (self.X.shape[0] - 1)
                
                variance = np.sum(X_centered[:, i] ** 2) / (self.X.shape[0] - 1)
                if variance != 0:
                    weights[j, i] = covariance / variance

        return weights