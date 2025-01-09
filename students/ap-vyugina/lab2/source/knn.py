import numpy as np
from scipy.spatial.distance import cdist

'''
X: датасет, (N_samples, N_features)
Y: метки, (N_samples, )
u: вектор для предсказания, (1, N_features)
k: число ближайших соседей
'''
def knn(u, X, y, k, mode='simple', h=2):
    d = cdist(X, u.reshape((1, -1)))
    
    topk_idxs = np.argsort(d, axis=0)[:k]
    
    topk = d[topk_idxs].flatten()
    lbls = y[topk_idxs].flatten()
    
    v = np.zeros((k, 5))
    if mode == "simple":
        topk = np.ones((k, ))
    elif mode == "parzen_fixed":
        topk = 1/np.sqrt(2*np.pi) * np.exp(-1/2 * (topk / h)**2) 
    elif mode == "parzen_nonfixed":
        h = np.sort(d, axis=0)[k]
        topk = 1/np.sqrt(2*np.pi) * np.exp(-1/2 * (topk / h)**2) 
        
    v[np.arange(k), lbls] = topk
    return np.argmax(np.sum(v, axis=0)).astype(np.int32)


def leave_one_out(k, X, y):
    y_pred = np.empty(y.shape, dtype=np.int32)
    for i in range(len(X)):
        y_pred[i] = knn(X[i], np.delete(X, i, axis=0), np.delete(y, i), k=k, mode='parzen_nonfixed')
    return np.sum(y_pred != y) / len(y)