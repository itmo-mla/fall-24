from enum import StrEnum
from tqdm import tqdm
import numpy as np
import abc
from typing import Callable


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x + 1e-6))


def sigmoid_derivative(x: float) -> float:
    return x * (1 - x)


def cross_entropy(y: float, y_pred: float) -> float:
    return - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred + 1e-6))


def cross_entropy_derivative(y: float, y_pred: float) -> float:
    return y_pred - y


class Loss(abc.ABC):
    function: Callable[[float, float], float]
    derivative: Callable[[float, float], float]


class Activation(abc.ABC):
    function: Callable[[float], float]
    derivative: Callable[[float], float]


class Sigmoid(Activation):
    def __init__(self):
        self.function = sigmoid
        self.derivative = sigmoid_derivative


class CrossEntropy(Loss):
    def __init__(self):
        self.function = cross_entropy
        self.derivative = cross_entropy_derivative


class WeightInitialization(StrEnum):
    ZERO = "ZERO"
    RANDOM = "RANDOM"
    CORRELATION = "CORRELATION"
    MULTI_START = "MULTI_START"


class LinearClassifier:
    a: np.ndarray
    a_inertia: float
    b: float
    b_inertia: float
    x: np.ndarray
    y_pred: float
    activation: Activation
    loss: Loss
    learning_rate: float
    forgetting_rate: float
    regularizator: float

    def __init__(self, size: int, loss: Loss, activation: Activation, forgetting_rate: float,
                 regularizator: float = 0.0):
        self.a = np.random.rand(size)
        self.a_biased = np.zeros(size)
        self.a_inertia = 0.0
        self.b = 0.0
        self.b_biased = 0.0
        self.b_inertia = 0.0
        self.loss = loss
        self.activation = activation
        self.forgetting_rate = forgetting_rate
        self.regularizator = regularizator

    def _fit_epoch(self, X: np.ndarray, Y: np.ndarray) -> float:
        preliminary_predictions = self.predict_bulk(X)
        less_sure_predictions = 1 / (np.abs(preliminary_predictions) + 1e-6)
        shuffling_probability_coefficients = less_sure_predictions / np.sum(less_sure_predictions)
        sample_indices = np.random.choice(a=len(X), size=X.shape[0], replace=False,
                                          p=shuffling_probability_coefficients)
        loss = 0.0
        for idx, (x, y) in enumerate(zip(X[sample_indices], Y[sample_indices])):
            loss += self._fit_iteration(x, y)
        return loss

    def fit(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            epoch_count: int = 5,
            weight_init: WeightInitialization = WeightInitialization.RANDOM
    ) -> np.ndarray:
        self._weight_init(weight_init, X, Y)
        losses = np.zeros(epoch_count)
        for epoch in tqdm(range(epoch_count)):
            losses[epoch] = self._fit_epoch(X, Y)
        return np.array(losses) / len(X)

    def _weight_init(self, method: WeightInitialization, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
        self.a_inertia = 0.0
        self.a_biased = 0.0
        self.b_inertia = 0.0
        self.b_biased = 0.0

        match method:
            case WeightInitialization.ZERO:
                self.a = np.zeros(self.a.shape[0])
                self.b = 0.0
            # обучить со случайным предъявлением и с п.8;
            case WeightInitialization.RANDOM:
                self.a = (np.random.rand(self.a.shape[0]) - 0.5) / len(X)
                self.b = (np.random.rand() - 0.5 )/ len(X)
            # обучить с инициализацией весов через корреляцию (п.3, слайд 14);
            case WeightInitialization.CORRELATION:
                for j in range(self.a.shape[0]):
                    self.a[j] = np.sum(Y * X.T[j]) / np.sum(X.T[j] ** 2)
                self.b = np.mean(y) - np.dot(np.mean(X, axis=0), self.a)
            # обучить со случайной инициализацией весов через мультистарт (п.5, слайд 14);
            case WeightInitialization.MULTI_START:
                variance_count = 10

                A, B = np.zeros((variance_count, self.a.shape[0])), np.zeros(variance_count)
                losses = np.zeros(variance_count)
                for variance in range(variance_count):
                    self._weight_init(WeightInitialization.RANDOM, X, Y)
                    A[variance], B[variance], losses[variance] = self.a, self.b, self._fit_epoch(X, Y)
                best_variance = np.argmin(losses)
                self.a, self.b = A[best_variance], B[best_variance]
        return self.a, self.b

    def predict(self, x: np.ndarray) -> float:
        return self._forward(x)

    def predict_bulk(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict(x) for x in X])

    def _fit_iteration(self, x: np.ndarray, y_true: float):
        y_pred = self._forward(x)

        loss_derivative = self.loss.derivative(y_true, y_pred)

        self._backward(loss_derivative)
        return self.loss.function(y_true, y_pred)

    def _forward(self, x: np.ndarray) -> float:
        self.x = x.copy()
        self.learning_rate = np.power(np.linalg.norm(self.x) + 1e-7, np.float32(-2.0))

        self.a_biased = self.a - self.learning_rate * self.forgetting_rate * self.a_inertia
        self.b_biased = self.b - self.learning_rate * self.forgetting_rate * self.b_inertia

        self.y_pred = self.activation.function(np.dot(self.x, self.a_biased) + self.b_biased)
        return self.y_pred

    def _backward(self, error_signal: float) -> None:
        gradient = error_signal * self.activation.derivative(self.y_pred)

        self.a_inertia = self.a_inertia * self.forgetting_rate + (1 - self.forgetting_rate) * gradient * self.x
        self.b_inertia = self.b_inertia * self.forgetting_rate + (1 - self.forgetting_rate) * gradient
        self.a[...] = self.a * (1 - self.learning_rate * self.regularizator) - self.learning_rate * self.a_inertia
        self.b += - self.learning_rate * self.b_inertia