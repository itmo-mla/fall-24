import numpy as np
from copy import deepcopy

class Linear:
    def __init__(self, input_size, count, activate, weight_gen):
        self.count = count
        self.weights = weight_gen.generate((count, input_size))
        self.biases = weight_gen.generate((1, count))
        self.activate = activate

        self.input = None
        self.output = None

        self.dW = None
        self.db = None

        self.start_weight = deepcopy(self.weights)
        self.start_biases = deepcopy(self.biases)

    def forward(self, input):
        self.input = input
        self.output = np.matmul(input, self.weights.T) + self.biases
        self.output = self.activate.get(self.output)
        return self.output
    
    def backward(self, pred_delta, pred_weights=None):
        new_delta = None
        if pred_weights is None:
            new_delta = pred_delta
        else:
            new_delta = self.activate.dget(self.output) * np.matmul(pred_delta, pred_weights)

        self.dW = new_delta * self.input
        if len(new_delta.shape) == 1:
            self.db = new_delta
        else:
            self.db = np.sum(new_delta)
        return new_delta

    def update_weights(self, dW, db):
        self.weights -= dW
        self.biases -= db

    def get_weights(self):
        return [self.weights]
    
    def get_biases(self):
        return [self.biases]
    
    def get_dW(self):
        return [self.dW]
    
    def get_db(self):
        return [self.db]