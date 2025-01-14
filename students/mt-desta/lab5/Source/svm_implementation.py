import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.svm import SVC

class SVM:
    def __init__(self, kernel_type='linear', C=1.0, gamma='scale', degree=3, coef0=1.0):
        self.kernel_type = kernel_type
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.gamma_value = None  # Will be computed during fit
        
        if kernel_type == "linear":
            self.kernel = lambda x, y: x @ y.T
        elif kernel_type == "rbf":
            def rbf_kernel(x, y):
                x2 = np.sum(x**2, axis=1)[:, np.newaxis]
                y2 = np.sum(y**2, axis=1)[np.newaxis, :]
                xy = x @ y.T
                distances = x2 + y2 - 2 * xy
                return np.exp(-self.gamma_value * distances)
            self.kernel = rbf_kernel
        elif kernel_type == "poly":
            self.kernel = lambda x, y: (self.gamma_value * (x @ y.T) + self.coef0) ** self.degree
        else:
            raise ValueError(f"Invalid kernel type: {kernel_type}")

    def _compute_gamma(self, X):
        if self.gamma == 'scale':
            return 1.0 / (X.shape[1] * X.var())
        elif self.gamma == 'auto':
            return 1.0 / X.shape[1]
        else:
            return self.gamma

    def fit(self, X, y, eps=1e-4):
        n_samples = X.shape[0]
        
        # Compute gamma value
        self.gamma_value = self._compute_gamma(X)
        
        # Convert labels to -1, 1
        y = np.where(y <= 0, -1, 1)

        # Compute kernel matrix
        K = self.kernel(X, X)
        
        # Compute P matrix
        P = (y[:, np.newaxis] * y[np.newaxis, :]) * K
        
        # Define objective function
        def objective(alpha):
            return 0.5 * alpha @ P @ alpha - alpha.sum()
        
        # Define gradient
        def gradient(alpha):
            return P @ alpha - 1
        
        # Set up constraints
        constraints = [
            {'type': 'eq', 'fun': lambda alpha: alpha @ y},  # sum(alpha_i * y_i) = 0
        ]
        
        # Set up bounds
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        # Solve optimization problem
        result = minimize(
            fun=objective,
            x0=np.zeros(n_samples),
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )

        if not result.success:
            print(f"Optimization failed: {result.message}")
            return False

        # Get alphas and identify support vectors
        self.alphas = result.x
        sv_mask = self.alphas > eps
        
        if not np.any(sv_mask):
            print("No support vectors found")
            return False

        # Store support vectors and related data
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]
        self.support_vector_alphas = self.alphas[sv_mask]
        self.X = X
        self.y = y

        # Compute bias (w0)
        margin_vectors = (self.alphas > eps) & (self.alphas < self.C - eps)
        if np.any(margin_vectors):
            K_margin = self.kernel(X[margin_vectors], self.support_vectors)
            self.w0 = np.mean(
                y[margin_vectors] - 
                np.sum(self.support_vector_alphas * self.support_vector_labels * K_margin, axis=1)
            )
        else:
            K_sv = self.kernel(self.support_vectors, self.support_vectors)
            predictions = np.sum(
                self.support_vector_alphas * self.support_vector_labels * K_sv, 
                axis=1
            )
            pos_sv = self.support_vector_labels == 1
            neg_sv = self.support_vector_labels == -1
            if np.any(pos_sv) and np.any(neg_sv):
                self.w0 = -0.5 * (np.max(predictions[neg_sv]) + np.min(predictions[pos_sv]))
            else:
                self.w0 = 0

        print(f"Number of support vectors: {len(self.support_vectors)}")
        print(f"Support vectors per class: {np.bincount(y[sv_mask] == 1)}")
        print(f"Objective value: {result.fun:.6f}")
        
        return True

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        K = self.kernel(X, self.support_vectors)
        decision_values = np.sum(
            self.support_vector_alphas * self.support_vector_labels * K, 
            axis=1
        ) - self.w0
        
        return np.where(decision_values <= 0, 0, 1)

    def visualize(self, X, y, feature_names=None):
        """
        Visualize SVM decision boundaries for specific feature pairs
        """
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
            
        # Define specific feature pairs
        feature_pairs = [
            ('radius_mean', 'texture_mean'),
            ('perimeter_mean', 'area_mean'),
            ('smoothness_mean', 'compactness_mean'),
            ('radius_mean', 'area_mean'),
            ('texture_mean', 'smoothness_mean'),
            ('compactness_mean', 'concavity_mean')
        ]
        
        # Create figure with 2x3 subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Plot each feature pair
        for idx, (feat1, feat2) in enumerate(feature_pairs):
            ax = axes[idx]
            
            # Get feature indices
            i = feature_names.index(feat1)
            j = feature_names.index(feat2)
            
            # Get current feature pair
            X_pair = X[:, [i, j]]
            
            # Create mesh grid
            x_min, x_max = X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1
            y_min, y_max = X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            
            # Create full-dimensional data for prediction
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            full_mesh_points = np.zeros((mesh_points.shape[0], self.X.shape[1]))
            full_mesh_points[:, [i, j]] = mesh_points
            
            # Predict and reshape
            Z = self.predict(full_mesh_points).reshape(xx.shape)
            
            # Plot decision boundary and regions with distinct colors
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
            ax.contour(xx, yy, Z, levels=[0.5], alpha=0.8, colors='k', linewidths=2)
            
            # Plot data points with distinct colors and markers
            scatter = ax.scatter(X_pair[:, 0], X_pair[:, 1], c=y, 
                               cmap='RdYlBu', edgecolors='k', alpha=0.7)
            
            # Plot support vectors
            if hasattr(self, 'support_vectors'):
                ax.scatter(self.support_vectors[:, i], self.support_vectors[:, j],
                         s=100, linewidth=1, facecolors='none', edgecolors='k',
                         label='Support Vectors')
            
            ax.set_xlabel(feat1)
            ax.set_ylabel(feat2)
            ax.legend()
        
        plt.suptitle(f'SVM Decision Boundaries ({self.kernel_type} kernel)', y=1.02)
        plt.tight_layout()
        plt.show()

    def compare_with_sklearn(self, X_train, y_train, X_test, y_test, feature_names=None):
        """
        Compare with sklearn's SVM implementation
        """
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(X_train.shape[1])]
            
        # Train sklearn SVM
        clf = SVC(kernel=self.kernel_type, C=self.C, gamma=self.gamma, degree=self.degree)
        clf.fit(X_train, y_train)
        
        # Calculate and print accuracy
        accuracy = clf.score(X_test, y_test)
        print(f'Sklearn SVM accuracy: {accuracy * 100:.2f}%')
        
        # Define specific feature pairs
        feature_pairs = [
            ('radius_mean', 'texture_mean'),
            ('perimeter_mean', 'area_mean'),
            ('smoothness_mean', 'compactness_mean'),
            ('radius_mean', 'area_mean'),
            ('texture_mean', 'smoothness_mean'),
            ('compactness_mean', 'concavity_mean')
        ]
        
        # Create figure with 2x3 subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Plot each feature pair
        for idx, (feat1, feat2) in enumerate(feature_pairs):
            ax = axes[idx]
            
            # Get feature indices
            i = feature_names.index(feat1)
            j = feature_names.index(feat2)
            
            # Get current feature pair
            X_train_pair = X_train[:, [i, j]]
            
            # Create mesh grid
            x_min, x_max = X_train_pair[:, 0].min() - 1, X_train_pair[:, 0].max() + 1
            y_min, y_max = X_train_pair[:, 1].min() - 1, X_train_pair[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            
            # Create feature pair data for prediction
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            full_mesh_points = np.zeros((mesh_points.shape[0], X_train.shape[1]))
            full_mesh_points[:, [i, j]] = mesh_points
            
            # Plot decision boundary
            Z = clf.decision_function(full_mesh_points)
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
            ax.contour(xx, yy, Z, levels=[0], alpha=0.8, colors='k', linewidths=2)
            
            scatter = ax.scatter(X_train_pair[:, 0], X_train_pair[:, 1], 
                               c=y_train, cmap='RdYlBu', edgecolors='k', alpha=0.7)
            
            if hasattr(clf, 'support_vectors_'):
                ax.scatter(clf.support_vectors_[:, i], clf.support_vectors_[:, j],
                         s=100, linewidth=1, facecolors='none', edgecolors='k',
                         label='Support Vectors')
            
            ax.set_xlabel(feat1)
            ax.set_ylabel(feat2)
            ax.legend()
        
        plt.suptitle(f'Sklearn SVM Decision Boundaries ({self.kernel_type} kernel)', y=1.02)
        plt.tight_layout()
        plt.show()