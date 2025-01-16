# Lab work #2. Metric classification

# Data

The [KNN Algorithm Dataset](https://www.kaggle.com/datasets/gkalpolukcu/knn-algorithm-dataset) from **_kaggle.com_** was selected for this lab work. Based on this dataset, the type of tumours can be classified in to Benign (B) or Mallignant (M). First, the dataset is checked for missing values and found that the column named 'Unnamed: 32' had 569 missing values like shown bellow:

```plaintext
id                           0
diagnosis                    0
radius_mean                  0
texture_mean                 0
perimeter_mean               0
area_mean                    0
smoothness_mean              0
compactness_mean             0
concavity_mean               0
concave points_mean          0
symmetry_mean                0
fractal_dimension_mean       0
radius_se                    0
texture_se                   0
perimeter_se                 0
area_se                      0
smoothness_se                0
compactness_se               0
concavity_se                 0
concave points_se            0
symmetry_se                  0
fractal_dimension_se         0
radius_worst                 0
texture_worst                0
perimeter_worst              0
area_worst                   0
smoothness_worst             0
compactness_worst            0
concavity_worst              0
concave points_worst         0
symmetry_worst               0
fractal_dimension_worst      0
Unnamed: 32                569
dtype: int64
```

So the column `id` is removed besause it doesn't affect the classification and the column `Unnamed: 32` is removed because it has too many missing values.

Then the data set was preprocessed by encoding categorical variables and then spliting the data into training data and testing data.

# Implementing KNN

## Reference KNN

The model `KNeighborsClassifier` from the library `sklearn.neighbors` is used to implement the _Reference KNN_

## Custom KNN

The custom implementation of the algorithm is shown below:

```py
class CustomKNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict_single(self, x):
        # Calculate distances
        distances = np.linalg.norm(self.X_train - x.reshape((1,-1)), axis=1)

        # Sort indices by distance
        sorted_indices = np.argsort(distances)

        # Select k nearest neighbors
        k_nearest_indices = sorted_indices[:self.k]
        k_nearest_distances = distances[k_nearest_indices]
        k_nearest_labels = self.y_train[k_nearest_indices]

        # Apply Gaussian kernel for weights
        weights = np.zeros((self.k, len(np.unique(self.y_train))))
        bandwidth = np.sort(distances, axis=0)[self.k]
        weights[np.arange(self.k), k_nearest_labels] = 1 / np.sqrt(2 * np.pi) * gaussian_kernel(k_nearest_distances, bandwidth)

        # Aggregate weights per class
        class_weights = {}
        for label, weight in zip(k_nearest_labels, weights):
            class_weights[label] = class_weights.get(label, 0) + weight

        # Predict class with max weight
        predicted_class = np.argmax(np.sum(weights, axis=0)).astype(np.int32)

        return predicted_class

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            predictions.append(self.predict_single(x))

        return np.array(predictions)
```

It takes the number `k` the number of neighnours as arguments during initialization. It applies the **Parzen Window** with **Variable Width**. It uses Gaussian Kernel as its kernel as defined in a the function:

```py
def gaussian_kernel(distance, bandwidth):
    return np.exp(-0.5 * (distance / bandwidth) ** 2)
```

# Parameter Selection

The value for k was selected using sliding control (LOO) method and plotting its empirical risk graph for both implementations.

## Reference KNN

The empirical risk graph

<img src='assets\risk_graph_ref.png'>

As we can see from the graph, the optimal values for k are shown to be 5 or 11. For this task, **k = 5**.

## Custom KNN

The empirical risk graph

<img src='assets\risk_graph_imp.png'>

As we can see from the graph, the optimal values for k are shown to be 5, 10 or 11. For this task, **k = 5**.

# Comparison of Metrics

These are the results optained by running the two libraries on the whole test set.

## Reference KNN

\*Reference KNN Execution Time: **0.0047 seconds** or **4.7 ms\***

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.96      | 0.96   | 0.96     | 71      |
| 1            | 0.93      | 0.93   | 0.93     | 43      |
| Accuracy     |           |        | 0.95     | 114     |
| Macro Avg    | 0.94      | 0.94   | 0.94     | 114     |
| Weighted Avg | 0.95      | 0.95   | 0.95     | 114     |

**_Confusion Matrix:_**

<img src='assets\con_mat_ref.png'>

## Custom KNN

\*Custom KNN Execution Time: **0.0283 seconds** or **28.3 ms\***

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.96      | 0.96   | 0.96     | 71      |
| 1            | 0.93      | 0.93   | 0.93     | 43      |
| Accuracy     |           |        | 0.95     | 114     |
| Macro Avg    | 0.94      | 0.94   | 0.94     | 114     |
| Weighted Avg | 0.95      | 0.95   | 0.95     | 114     |

**_Confusion Matrix:_**

<img src='assets\con_mat_imp.png'>

Based on the Metrics both perform exactly the same on accuracy. But the reference library was faster.

The performance was also compared by selecting a random test value.

```plaintext
Got random test value at index = 86
Implemented KNN prediction: 1 | Reference KNN prediction: 0
Elapsed Time: for Implemented KNN: 0.5533695220947266 ms | for Reference KNN: 1.508951187133789 ms
```

Here also the implemented algorithm performs better solely and highly on time elapsed.

# Conclusion

1. The implemented KNN outperforms the reference KNN on speed on a random chosen test value.
2. The reference KNN outperforms the implemented KNN on speed on the complete test.
3. They both have the same accuracy score across all metrics (Precision, Recall, F1-Score and Accuracy). But, for a large dataset the reference KNN will be much better as it will be able to handle the large volume.
