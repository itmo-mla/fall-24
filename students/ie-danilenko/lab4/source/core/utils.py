import numpy as np
from core.loss import Lin

def calculate_probabilities(elements):
    elements = np.array(elements, dtype=float)

    min_value = np.min(elements)
    if min_value <= 0:
        elements -= min_value + 1

    inverse_values = 1 / elements

    probabilities = inverse_values / np.sum(inverse_values)
    return probabilities
def accuracy(y_pred, y):
    return np.sum(y_pred == y) / y.shape[0]

def get_batches(data, batch_size):
  n = len(data)
  get_X = lambda z: z[0]
  get_y = lambda z: z[1]
  for i in range(0, n, batch_size):
    batch = data[i:i+batch_size]
    yield np.array([get_X(b) for b in batch]), np.array([get_y(b) for b in batch])

def get_batches_margins(X, y, weight):
    margins = np.abs(Lin.get(X, y, weight))
    proba = calculate_probabilities(margins.flatten())
    data = list(zip(X, y))
    for i in range(len(data)):
      index = np.random.choice(len(data), size=1, p=proba)[0]
      batch = data[index]
      yield np.array([batch[0]]), np.array([batch[1]])