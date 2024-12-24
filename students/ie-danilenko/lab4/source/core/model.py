from core.activation import Sign
from core.layer import Linear
from core.optim import MomentumSGD, FastGD
from core.weights import RandomWeightGenerator
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from core.utils import get_batches, get_batches_margins
from time import time

class LinearClassificator:
    def __init__(self, loss, method, regularization, weight_gen):
        self.loss = loss
        self.method = method
        self.regularization = regularization

        self.layers = [
            Linear(4, 1, Sign, weight_gen)
        ]

    def _feedforward(self, X):
        inputs = X
        for i in range(len(self.layers)):
            inputs = self.layers[i].forward(inputs)
        return inputs
    
    def _backprop(self, y):
        reg = self.regularization.dget(self.layers[-1].weights) if self.regularization is not None else np.zeros_like(self.layers[-1].input)
        delta = self.loss[0].dget(self.layers[-1].input, y, self.layers[-1].weights)
        delta += reg
        for i in range(len(self.layers)-1, -1, -1):
            if i != len(self.layers) - 1:
                delta = self.layers[i].backward(delta, self.layers[i+1].weights)
            else:
                delta = self.layers[i].backward(delta)

    def _update_params(self, X):
        for i in range(len(self.layers)):
            grad_dw = self.layers[i].dW
            grad_db = self.layers[i].db
            if type(self.method) == FastGD:
                dW, db = self.method.calc_step(grad_dw, grad_db, X)
            else:
                dW, db = self.method.calc_step(grad_dw, grad_db)
            
            self.layers[i].update_weights(dW, db)

    def _compute_loss(self, X, y):
        y_pred = self.predict(X)
        return self.loss[1].get(y_pred, y)
    
    def predict(self, X, ):
        self._feedforward(X)
        pred = self.layers[-1].output
        return pred
    
    def train(self, X, y, X_test, y_test, epochs=1):
        self.method.init_weight(self.layers[-1].weights, self.layers[-1].biases)

        epoch_losses = np.array([])
        dataset = list(zip(X, y))

        plt.ion()
        for i in tqdm(range(epochs)):
            for (X_batch, y_batch) in get_batches(dataset, 1):
                self._feedforward(X_batch)
                self._backprop(y_batch)
                self._update_params(X_batch)
            epoch_losses = np.append(epoch_losses, self._compute_loss(X_test, y_test))
            plt.plot(epoch_losses, c='red')
            plt.draw()
            plt.gcf().canvas.flush_events()
            plt.pause(0.01)
            plt.ioff()
        return epoch_losses
    

class LinearClassificatorModule(LinearClassificator): 
    def __init__(self, loss, method, regularization, weight_gen):
        super().__init__(loss, method, regularization, weight_gen)

    def train(self, X, y, X_test, y_test, epochs=1):
        self.method.init_weight(self.layers[-1].weights, self.layers[-1].biases)

        epoch_losses = np.array([])

        plt.ion()
        for i in tqdm(range(epochs)):
            for (X_batch, y_batch) in get_batches_margins(X, y, self.layers[0].weights):
                self._feedforward(X_batch)
                self._backprop(y_batch)
                self._update_params(X_batch)
            epoch_losses = np.append(epoch_losses, self._compute_loss(X_test, y_test))
            plt.plot(epoch_losses, c='red')
            plt.draw()
            plt.gcf().canvas.flush_events()
            plt.pause(0.01)
            plt.ioff()
        return epoch_losses
    
class Multistart:
    def __init__(self, loss, method, regularization, count):
        self.count = count
        self.model = None
        self.loss = loss
        self.method = method
        self.regularization = regularization

    def get_min_loss_index(self, array):
        min_row = None
        min_index = -1

        for index, row in enumerate(array):
            if min_row is None or (row[0] < min_row[0]) or (row[0] == min_row[0] and row[1] < min_row[1]):
                min_row = row
                min_index = index

        return min_index
    
    def train(self, X, y, X_test, y_test, epochs=1):
        model_stat = []
        models = []
        for i in range(self.count):
            np.random.seed(int(time()))
            models.append(LinearClassificator(
                self.loss,
                self.method,
                self.regularization,
                RandomWeightGenerator()
            ))
            losses = models[-1].train(X, y, X_test, y_test, epochs)
            model_stat.append([losses.min(),losses.argmin()])

        model_stat = np.asarray(model_stat)
        min_index = self.get_min_loss_index(model_stat)
        print(model_stat, min_index)
        self.model = models[min_index]

        del models

    def predict(self, X):
        return self.model.predict(X)
    
    def get_best_w_b(self):
        return self.model.layers[0].start_weight, self.model.layers[0].start_biases