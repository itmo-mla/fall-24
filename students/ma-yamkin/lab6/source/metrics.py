import numpy as np


# mse
def mse(y_pred, y_targ):
    y_pred = np.array(y_pred)
    y_targ = np.array(y_targ)
    return np.mean(np.square(y_targ - y_pred))


# rmse
def rmse(y_pred, y_targ):
    y_pred = np.array(y_pred)
    y_targ = np.array(y_targ)
    return (np.sum((y_targ - y_pred) ** 2) / len(y_pred)) ** 0.5


# r2
def r2(y_pred, y_targ):
    y_pred = np.array(y_pred)
    y_targ = np.array(y_targ)

    sse = np.sum((y_pred - y_targ) ** 2)
    sst = np.sum((y_pred - np.mean(y_pred)) ** 2)
    return abs(1 - sse / sst)
