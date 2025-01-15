import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class LinearClassifier:
    def __init__(self, n_features=None):
        self.n_features = n_features
        self.w = None
        self.v = 0  # Velocity for momentum
        self.Q = None  # Moving average quality
        self.history = {
            'loss': [],
            'Q': [],  # Quality history
            'margins': [],  # Track margins for analysis
            'sample_weights': [],  # Track sample selection weights
            'correlations': None  # Store feature correlations
        }
    
    def init_weights(self, w=None, method='random', X=None, y=None):
        """
        Initialize weights using different methods
        
        Parameters:
        - w: pre-defined weights (if None, will be initialized)
        - method: initialization method ('random', 'correlation', 'zero')
        - X: features (needed for correlation initialization)
        - y: targets (needed for correlation initialization)
        """
        if w is not None:
            self.w = w
        else:
            if method == 'random':
                # Xavier initialization
                limit = 1/(4*np.sqrt(self.n_features))
                self.w = np.random.uniform(-limit, limit, (1, self.n_features))
            
            elif method == 'correlation':
                if X is None or y is None:
                    raise ValueError("X and y must be provided for correlation initialization")
                
                # Compute correlations between features and target
                correlations = []
                for i in range(self.n_features):
                    # Handle constant features
                    if np.std(X[:, i]) > 1e-10:
                        corr = np.corrcoef(X[:, i], y)[0, 1]
                        correlations.append(0 if np.isnan(corr) else corr)
                    else:
                        correlations.append(0)
                
                correlations = np.array(correlations)
                
                # Store correlations for analysis
                self.history['correlations'] = correlations
                
                # Scale correlations to have similar magnitude as random init
                if np.std(correlations) > 0:
                    scale = 1/(4*np.sqrt(self.n_features))
                    correlations = correlations * scale / np.std(correlations)
                else:
                    # Fallback to random if all correlations are zero
                    limit = 1/(4*np.sqrt(self.n_features))
                    correlations = np.random.uniform(-limit, limit, self.n_features)
                
                self.w = correlations.reshape(1, -1)
            
            elif method == 'zero':
                self.w = np.zeros((1, self.n_features))
            
            else:
                raise ValueError(f"Unknown initialization method: {method}")
        
        # Reset momentum
        self.v = 0
        
        return self.w
    
    def plot_feature_correlations(self):
        """Plot feature correlations with target"""
        if self.history['correlations'] is None:
            print("No correlation data available. Use correlation initialization first.")
            return
        
        correlations = self.history['correlations']
        
        plt.figure(figsize=(12, 4))
        
        # Plot correlation distribution
        plt.subplot(121)
        plt.hist(correlations, bins=50)
        plt.title('Distribution of Feature-Target Correlations')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Count')
        
        # Plot correlation vs feature index
        plt.subplot(122)
        plt.plot(correlations, 'b.', alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Feature-Target Correlations')
        plt.xlabel('Feature Index')
        plt.ylabel('Correlation Coefficient')
        
        plt.tight_layout()
        plt.show()
        
        # Print correlation statistics
        print("\nCorrelation Statistics:")
        print(f"Mean absolute correlation: {np.mean(np.abs(correlations)):.4f}")
        print(f"Max absolute correlation: {np.max(np.abs(correlations)):.4f}")
        print(f"Number of features with |correlation| > 0.1: {np.sum(np.abs(correlations) > 0.1)}")
    
    def _margin_loss(self, x, y):
        """Compute quadratic margin loss for a single sample"""
        M = (self.w @ x) * y
        return (1 - M) ** 2
    
    def _margin_dloss(self, x, y):
        """Compute gradient of quadratic margin loss for a single sample"""
        M = (self.w @ x) * y
        return -2 * (1 - M) * (y @ x.T)
    
    def _compute_margins(self, X, y):
        """Compute margins for all samples"""
        return ((X @ self.w.T) * y.reshape(-1, 1)).flatten()
    
    def analyze_margins(self, X, y):
        """
        Analyze the margins (indentation) of data points
        Returns dictionary with margin statistics
        """
        margins = self._compute_margins(X, y)
        
        stats = {
            'mean': np.mean(margins),
            'std': np.std(margins),
            'min': np.min(margins),
            'max': np.max(margins),
            'median': np.median(margins),
            'positive_rate': np.mean(margins > 0),
            'margin_distribution': margins
        }
        
        return stats
    
    def plot_margin_distribution(self, X, y, title="Margin Distribution"):
        """
        Plot the distribution of margins (object indentation)
        """
        margins = self._compute_margins(X, y)
        
        plt.figure(figsize=(10, 6))
        
        # Plot margin distribution for each class
        for label, label_name in [(-1, 'Class -1'), (1, 'Class 1')]:
            class_margins = margins[y == label]
            sns.kdeplot(class_margins, label=label_name)
        
        plt.axvline(x=0, color='r', linestyle='--', label='Decision Boundary')
        plt.xlabel('Margin (Distance from Decision Boundary)')
        plt.ylabel('Density')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Print margin statistics
        stats = self.analyze_margins(X, y)
        print("\nMargin Statistics:")
        print(f"Mean margin: {stats['mean']:.4f}")
        print(f"Std margin: {stats['std']:.4f}")
        print(f"Min margin: {stats['min']:.4f}")
        print(f"Max margin: {stats['max']:.4f}")
        print(f"Median margin: {stats['median']:.4f}")
        print(f"Correctly classified rate: {stats['positive_rate']:.4f}")
    
    def plot_margin_evolution(self):
        """
        Plot how margins evolve during training
        """
        if len(self.history['margins']) == 0:
            print("No margin history available. Set track_margins=True during training.")
            return
        
        plt.figure(figsize=(10, 6))
        margins_array = np.array(self.history['margins'])
        
        plt.plot(np.mean(margins_array, axis=1), label='Mean Margin')
        plt.fill_between(range(len(margins_array)), 
                        np.percentile(margins_array, 25, axis=1),
                        np.percentile(margins_array, 75, axis=1),
                        alpha=0.3, label='25-75 Percentile')
        
        plt.xlabel('Iteration')
        plt.ylabel('Margin')
        plt.title('Evolution of Margins During Training')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def fit(self, X, y, n_iter=1000, lr=0.01, lambda_=0.9, reg=0.01, 
            momentum=True, gamma=0.9, optimize_lr=False, use_margins=True,
            track_margins=False):
        """
        Train the classifier using stochastic gradient descent
        
        Additional Parameters:
        - track_margins: whether to track margin evolution during training
        """
        if self.n_features is None:
            self.n_features = X.shape[1]
            self.init_weights()
        
        # Initialize Q with random sample average loss
        if self.Q is None:
            random_sample = np.random.choice(range(len(X)), size=30)
            random_X = X[random_sample]
            random_y = y[random_sample]
            
            losses = []
            for x_i, y_i in zip(random_X, random_y):
                x_i = x_i.reshape(-1, 1)
                y_i = np.array([[y_i]])
                losses.append(self._margin_loss(x_i, y_i))
            self.Q = np.mean(losses)
        
        best_w = None
        best_accuracy = 0
        no_improve_count = 0
        
        # Store parameters
        self.params = {
            'n_iter': n_iter,
            'lr': lr,
            'lambda_': lambda_,
            'reg': reg,
            'momentum': momentum,
            'gamma': gamma,
            'optimize_lr': optimize_lr,
            'use_margins': use_margins,
            'track_margins': track_margins
        }
        
        for iter_num in range(n_iter):
            if use_margins:
                # Margin-based sample selection with temperature annealing
                margins = self._compute_margins(X, y)
                if track_margins and iter_num % 10 == 0:  # Save margins every 10 iterations
                    self.history['margins'].append(margins)
                    
                temperature = max(0.1, 1.0 - iter_num/n_iter)  # Annealing temperature
                abs_inv_margins = np.max(np.abs(margins)) - np.abs(margins)
                probs = np.exp(abs_inv_margins / temperature)
                probs = probs / np.sum(probs)
                idx = np.random.choice(np.arange(len(X)), p=probs)
                x, y_i = X[idx], y[idx]
            else:
                # Random sample selection
                idx = np.random.randint(len(X))
                x, y_i = X[idx], y[idx]
                
                if track_margins and iter_num % 10 == 0:
                    margins = self._compute_margins(X, y)
                    self.history['margins'].append(margins)
            
            # Reshape for matrix operations
            x = x.reshape(-1, 1)
            y_i = np.array([[y_i]])
            
            # Compute current loss
            loss = self._margin_loss(x, y_i)
            
            # Adaptive learning rate if requested
            if optimize_lr:
                x_norm = np.sum(x**2)
                current_lr = min(lr, 1.0 / (x_norm + 1e-8))  # Bounded adaptive learning rate
            else:
                current_lr = lr
            
            # Update weights using momentum if enabled
            if momentum:
                self.v = gamma * self.v + (1 - gamma) * self._margin_dloss(x, y_i)
                grad_update = current_lr * self.v
            else:
                grad_update = current_lr * self._margin_dloss(x, y_i)
            
            # Update with gradient clipping
            grad_norm = np.linalg.norm(grad_update)
            if grad_norm > 1.0:
                grad_update = grad_update / grad_norm
            
            # Weight update with L2 regularization
            self.w = self.w * (1 - current_lr * reg) - grad_update
            
            # Update quality estimate
            self.Q = lambda_ * loss + (1 - lambda_) * self.Q
            
            # Record history
            self.history['loss'].append(float(loss))
            self.history['Q'].append(float(self.Q))
            
            # Track best weights
            if iter_num % 100 == 0:  # Check periodically
                current_accuracy = self.score(X, y)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_w = self.w.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Early stopping with restoration
                if no_improve_count >= 10:  # No improvement for 1000 iterations
                    break
        
        # Restore best weights
        if best_w is not None:
            self.w = best_w
    
    def predict(self, X):
        """Predict class labels"""
        return np.sign(X @ self.w.T)
    
    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions.flatten() == y)
    
    def compute_metrics(self, X, y):
        """
        Compute comprehensive classification metrics
        
        Returns dictionary with metrics:
        - accuracy: overall accuracy
        - precision: precision for each class
        - recall: recall for each class
        - f1: F1 score for each class
        - confusion_matrix: confusion matrix
        - support: number of samples in each class
        """
        predictions = self.predict(X).flatten()
        
        # Compute confusion matrix
        classes = np.unique(y)
        n_classes = len(classes)
        conf_matrix = np.zeros((n_classes, n_classes))
        
        for i in range(len(y)):
            true_idx = np.where(classes == y[i])[0][0]
            pred_idx = np.where(classes == predictions[i])[0][0]
            conf_matrix[true_idx, pred_idx] += 1
        
        # Compute metrics for each class
        metrics = {
            'accuracy': np.mean(predictions == y),
            'precision': {},
            'recall': {},
            'f1': {},
            'support': {},
            'confusion_matrix': conf_matrix,
            'classes': classes
        }
        
        for i, cls in enumerate(classes):
            # True positives, false positives, false negatives
            tp = conf_matrix[i, i]
            fp = np.sum(conf_matrix[:, i]) - tp
            fn = np.sum(conf_matrix[i, :]) - tp
            
            # Compute precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics['precision'][cls] = precision
            metrics['recall'][cls] = recall
            metrics['f1'][cls] = f1
            metrics['support'][cls] = np.sum(conf_matrix[i, :])
        
        return metrics
    
    def plot_confusion_matrix(self, X, y, title="Confusion Matrix"):
        """Plot confusion matrix with seaborn"""
        metrics = self.compute_metrics(X, y)
        conf_matrix = metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
                   xticklabels=metrics['classes'],
                   yticklabels=metrics['classes'])
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    
    def print_classification_report(self, X, y):
        """Print detailed classification report"""
        metrics = self.compute_metrics(X, y)
        
        print("\nClassification Report:")
        print("-" * 60)
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print("\nPer-class metrics:")
        print("-" * 60)
        print(f"{'Class':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 60)
        
        for cls in metrics['classes']:
            print(f"{cls:>10.0f} {metrics['precision'][cls]:>10.4f} "
                  f"{metrics['recall'][cls]:>10.4f} {metrics['f1'][cls]:>10.4f} "
                  f"{metrics['support'][cls]:>10.0f}")
        
        print("-" * 60) 