# Lab 4 Linear Classification

## Task 1: Dataset

For this lab we will be using the [mashroom dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification). 

## Task 2: Implementing object margin
The margin is calculated using the following function:

```python
def _compute_margins(self, X, y):
        return ((X @ self.w.T) * y.reshape(-1, 1)).flatten()
```
The visualization of the margin distribution is done using the following function:

```python
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

```

## Task 3: Implementing Loss Function

The loss function and its gradient are calculated using the following function: quadratic margin loss was used.

```python
def _margin_loss(self, x, y):
        """Compute quadratic margin loss for a single sample"""
        M = (self.w @ x) * y
        return (1 - M) ** 2
    
    def _margin_dloss(self, x, y):
        """Compute gradient of quadratic margin loss for a single sample"""
        M = (self.w @ x) * y
        return -2 * (1 - M) * (y @ x.T)
```
## Task 4: Recursive quality evaluation function

The quality (Q) is calculated using the following function:

```python
self.Q = lambda_ * loss + (1 - lambda_) * self.Q
```

This is a recursive exponential moving average (EMA) formula that's commonly used to track the quality or performance of a model over time. 
1. **lambda_** is a smoothing factor between 0 and 1 that determines how much weight to give to the new loss value
2. **loss** is the current loss value
3. **self.Q** is the historical quality metric. 

The formula combines:
* The new loss value **(lambda_ * loss)**
* The previous quality value **((1 - lambda_) * self.Q)**


## Task 5: Stochastic Gradient Descent with momentum

The stochastic gradient descent with Nesterov momentum and regularization is implemented using the following code:

```python
self.v = gamma * self.v + (1 - gamma) * self._margin_dloss(x, y_i)
grad_update = current_lr * self.v

```

## Task 6: L2 Regularization

L2 regularization adds a penalty term to the loss function proportional to the squared magnitude of weights. This helps prevent overfitting by keeping the weights small and the model simpler.

L2 regularization is implemented using the following code:

```python
self.w = self.w * (1 - current_lr * reg) - grad_update
```

## Task 7: Fast Gradient Descent

Fast Gradient Descent (FGD) is a method used to optimize the weights of a linear classifier by iteratively updating the weights in the direction of the negative gradient of the loss function. This approach is often used in conjunction with regularization techniques to prevent overfitting.

## Task 8: Presentation of objects to the indent module

```python 

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
```

The code implements margin-based sample selection with temperature annealing for training:

1. Computes margins between samples and decision boundary using _compute_margins()

2. Uses temperature annealing schedule that decreases from 1.0 to 0.1 as training progresses:
   temperature = max(0.1, 1.0 - iter_num/n_iter)

3. Converts margins to selection probabilities:
   - Inverts absolute margins so smaller margins have higher probability
   - Applies softmax with temperature to get normalized probabilities
   - abs_inv_margins = max|margins| - |margins|
   - probs = exp(abs_inv_margins/temperature) / sum(exp(...))

4. Randomly samples training examples weighted by these probabilities
   - Focuses training on examples closer to decision boundary
   - Temperature controls exploration vs exploitation tradeoff

5. Optionally tracks margin history during training for analysis

## Task 9: Training

Several implementations of linear classifiers were evaluated on the mushroom dataset, with the following results:

### Best Performing Models:

1. **Margin-based Sampling**
   - Training Accuracy: 100.00%
   - Test Accuracy: 100.00%
   - Perfect classification performance with optimal margin distribution

2. **Reference SGD Implementation**
   - Training Accuracy: 100.00%
   - Test Accuracy: 100.00%
   - Perfect precision, recall and F1-scores for both classes

3. **Correlation Initialization**
   - Training Accuracy: 99.49%
   - Test Accuracy: 99.29%
   - Strong performance with feature correlation-based initialization

### Good Performing Models:

4. **Random Sampling**
   - Training Accuracy: 98.19%
   - Test Accuracy: 98.04%
   - Effective simple sampling strategy

5. **Classic with L2 + Nesterov**
   - Training Accuracy: 96.23%
   - Test Accuracy: 96.02%
   - Momentum-based optimization with regularization

6. **Classic with L2**
   - Training Accuracy: 95.86%
   - Test Accuracy: 94.89%
   - Basic implementation with regularization

### Lower Performing Model:

7. **Classic with L2 + Nesterov + Optimal LR**
   - Training Accuracy: 87.85%
   - Test Accuracy: 86.64%
   - Learning rate optimization didn't improve performance

### Implementation Details:

1. **Classic with L2**
   - Base implementation with L2 regularization
   - Uses stochastic gradient descent
   - Regularization helps prevent overfitting

2. **Classic with L2 + Nesterov**
   - Adds Nesterov momentum to basic implementation
   - Improved convergence speed
   - Better generalization than basic L2

3. **Classic with L2 + Nesterov + Optimal LR**
   - Attempts to optimize learning rate
   - Unexpectedly lower performance
   - May indicate sensitivity to learning rate selection

4. **Correlation Initialization**
   - Initializes weights based on feature correlations
   - Strong performance without complex optimization
   - Demonstrates importance of weight initialization

5. **Random Sampling**
   - Simple random selection of training examples
   - Surprisingly good performance
   - Efficient training approach

6. **Margin-based Sampling**
   - Focuses on examples near decision boundary
   - Achieves perfect classification
   - Most effective sampling strategy

7. **Reference SGD Implementation**
   - sklearn's SGDClassifier as benchmark
   - Perfect classification performance
   - Validates effectiveness of custom implementations

### Key Findings:

1. **Sampling Strategies**: Margin-based sampling proved highly effective, achieving perfect classification along with the reference implementation.

2. **Initialization Impact**: Correlation-based initialization showed excellent performance without requiring complex optimization techniques.

3. **Regularization Effects**: L2 regularization consistently helped prevent overfitting across implementations.

4. **Momentum Benefits**: Nesterov momentum generally improved performance when combined with L2 regularization.

5. **Learning Rate Sensitivity**: The attempt to optimize learning rate showed that model performance can be sensitive to learning rate selection.

### Visualizations:
![Training History](Images/output.png)

### Conclusions:

1. All implementations except one achieved >94% test accuracy, demonstrating the effectiveness of linear classification on this dataset.

2. Margin-based sampling and the reference implementation achieved perfect classification, showing the potential of sophisticated sampling strategies.

3. Correlation-based initialization proved highly effective, suggesting the importance of intelligent weight initialization.

4. The addition of Nesterov momentum to L2 regularization improved performance slightly.

5. Learning rate optimization requires careful tuning, as shown by the lower performance of the optimal learning rate variant.

6. The high performance across multiple implementations validates the robustness of the linear classification approach for this dataset.



