import numpy as np


### Classification

h = lambda z: -z * np.log2(z, out=np.zeros_like(z, dtype=np.float64), where=(z!=0))

def entropy(y):
    P_c = np.array([sum(y == y_lbl) for y_lbl in np.unique(y)])
    return sum(h(P_c/sum(P_c)))


# X - набор какого-то конкретного признака
def multiClassEntropyCriterion(X, y):

    max_information_gain = -1
    best_weight = None

    l = len(X)

    Y = np.unique(y)
    Y.sort()

    P_c = np.array([sum(y == y_lbl) for y_lbl in Y])

    for predicat_weight in sorted(np.unique(X))[:-1]:
        p = sum(X > predicat_weight)
        p_c = np.array([sum((y==y_lbl) * (X > predicat_weight)) for y_lbl in Y])

        I = sum(h(P_c/l)) - p/l * sum(h(p_c/p)) - (l-p)/l * sum(h((P_c - p_c)/(l-p)))

        if I > max_information_gain:
            max_information_gain = round(I, 5)
            best_weight = predicat_weight

    return max_information_gain, best_weight


def DonskoyCriterion(X, y):
    max_information_gain = -1
    best_weight = None

    for predicat_weight in sorted(np.unique(X))[:-1]:
        p = X > predicat_weight
        I = np.sum((p[:, None] != p) & (y[:, None] != y))

        if I > max_information_gain:
            max_information_gain = round(I, 5)
            best_weight = predicat_weight
    return max_information_gain, best_weight


### Regression

def uncertainty_measure(Y):
    y = 1/len(Y) * np.sum(Y)
    return 1/len(Y) * np.sum((Y-y)**2)


def MSECriterion(X, y):

    max_information_gain = -1
    best_weight = None

    l = len(y)

    for predicat_weight in sorted(np.unique(X))[:-1]:
        idxs = X > predicat_weight
        y_0 = y[idxs]
        y_1 = y[~idxs]

        I = uncertainty_measure(y) - len(y_0)/l * uncertainty_measure(y_0) - len(y_1)/l * uncertainty_measure(y_1)

        if I > max_information_gain:
            max_information_gain = round(I, 5)
            best_weight = predicat_weight

    return max_information_gain, best_weight