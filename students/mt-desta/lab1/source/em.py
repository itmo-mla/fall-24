import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal

class my_GaussianMixtureEM:
    def __init__(self, n_components, tol=1e-6, max_iter=100, reg_covar=1e-6):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.reg_covar = reg_covar  # Regularization term for covariance matrices

    def fit(self, X):
        X = X.values

        # Means, covariances, and weights initialization
        np.random.seed(42)
        self.n_samples, self.n_features = X.shape
        self.means = X[np.random.choice(self.n_samples, self.n_components, replace=False)]
        self.covariances = [np.cov(X.T) for _ in range(self.n_components)]
        self.weights = np.ones(self.n_components) / self.n_components

        prev_log_likelihood = -np.inf

        # EM Iterations
        for iteration in range(self.max_iter):
            # E-step remains the same
            responsibilities = np.zeros((self.n_samples, self.n_components))
            for k in range(self.n_components):
                responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(X, mean=self.means[k], cov=self.covariances[k])
            
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)

            # M-step: Update the parameters (weights, means, and covariances)
            N_k = responsibilities.sum(axis=0)
            self.weights = N_k / self.n_samples
            self.means = (responsibilities.T @ X) / N_k[:, np.newaxis]

            # Regularize the covariance matrices
            self.covariances = [
                (responsibilities[:, k] * (X - self.means[k]).T) @ (X - self.means[k]) / N_k[k] + self.reg_covar * np.eye(self.n_features)
                for k in range(self.n_components)
            ]

            # Log-likelihood calculation and convergence check
            log_likelihood = np.sum(np.log(np.sum([self.weights[k] * multivariate_normal.pdf(X, mean=self.means[k], cov=self.covariances[k]) for k in range(self.n_components)], axis=0)))
            
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood

    def fit_predict(self,X):
        n_samples,  _ = X.shape
        likelihoods = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            diff = X - self.means[k].reshape(1,-1)
            maha_dist = np.sum(np.dot(diff, np.linalg.inv(self.covariances[k])) * diff, axis=1)
            likelihoods[:, k] = self.weights[k] * np.exp(-0.5 * maha_dist) / np.sqrt((2 * np.pi) ** X.shape[1] * np.linalg.det(self.covariances[k]))
        
        # Assign each sample to the cluster with the highest likelihood
        predictions = np.argmax(likelihoods, axis=1)
        
        return predictions
    

