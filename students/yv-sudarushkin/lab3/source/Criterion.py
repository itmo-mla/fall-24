import numpy as np
import enum

def entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities, where=probabilities > 0))


def multiClassEntropyCriterion(X, y):
    unique_thresholds = np.unique(X)
    best_gain = -1
    best_threshold = None

    for threshold in unique_thresholds[:-1]:
        left_mask = X <= threshold
        right_mask = ~left_mask

        y_left, y_right = y[left_mask], y[right_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            continue

        left_entropy = entropy(y_left)
        right_entropy = entropy(y_right)

        total_entropy = (len(y_left) * left_entropy + len(y_right) * right_entropy) / len(y)
        information_gain = entropy(y) - total_entropy

        if information_gain > best_gain:
            best_gain = information_gain
            best_threshold = threshold

    return best_gain, best_threshold

def DonskoyCriterion(X, y):
    best_gain = -1
    beta = None

    values = np.unique(X)

    for curr in values[:-1]:
        left_group = X <= curr
        right_group = X > curr

        information_gain = np.sum(np.not_equal.outer(y[left_group], y[right_group]))

        if information_gain > best_gain:
            best_gain = information_gain
            beta = curr

    return best_gain, beta


def uncertainty_measure(Y):
    y = 1/len(Y) * np.sum(Y)
    return 1/len(Y) * np.sum((Y-y)**2)


def mseCriterion(X, y):
    best_gain = -1
    beta = None

    l = len(y)

    for curr in sorted(np.unique(X))[:-1]:
        idxs = X > curr
        y_0 = y[idxs]
        y_1 = y[~idxs]

        information_gain = uncertainty_measure(y) - len(y_0) / l * uncertainty_measure(y_0) - len(y_1) / l * uncertainty_measure(y_1)

        if information_gain > best_gain:
            best_gain = information_gain
            beta = curr

    return best_gain, beta

class Criterions(enum.Enum):
    multiClassEntropy = multiClassEntropyCriterion
    donskoy = DonskoyCriterion
    mse = mseCriterion