import random
random.seed(0)

import numpy as np


def em(X, N_features, K, max_step=30):
    N = len(X)
    
    centers = [random.choice(X) for _ in range(K)]
    centers = np.array(centers) # (K, N_features)
    
    sigmas = np.empty((K, N_features))
    for k in range(K):
        diff = X - centers[k]
        sigmas[k] = np.diag(np.dot(diff.T, diff)) / N ## sigma**2
    weights = 1/K * np.ones((K, 1))
    
    y_prev = None
    for _ in range(max_step):
        probs = []
        for i in range(K):
            ro_sq = np.sum((X[:, :] - centers[i])**2 / sigmas[i] , axis=1)
            probs += [np.power((2*np.pi), -N_features/2) / np.prod(np.sqrt(sigmas[i])) * np.exp(-1/2*ro_sq)]
        probs = np.array(probs, dtype=np.float32).T # (N, K)

        ## e-step
        g = weights.T * probs
        g /= g.sum(axis=1)[:, np.newaxis]

        ## m-step
        weights = 1/N * np.sum(g, axis=0)
        centers = g.T @ X / weights[:, np.newaxis] / N
        sigmas = np.empty((K, N_features))
        
        for k in range(K):
            sigmas[k] = g[:, k] @ (X - centers[k])**2
        sigmas /= weights[:, np.newaxis] 
        sigmas /= N

        y = np.argmax(g, axis=1)
        if y_prev is not None:
            if np.array_equal(y, y_prev): return y
        y_prev = y
    return y
        
    
    