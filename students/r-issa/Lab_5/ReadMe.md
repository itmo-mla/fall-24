# Lab Work №4

## The used dataset:

In this lab The [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) was used. This is a dataset that contains data aimed at analyzing, understanding, and predicting depression among students (27901 students). The feature **Depression** was the target feature, where it was a boolean feature:

1. Depression = 0 -> **Not Depressed**
2. Depression = 1 -> **Depressed**

The dataset was preprocessed in the following steps:

1. Removing the row with `NaN` (missing) values.
2. Removing the `id` feature because it doesn't affect the classification.
3. Removing features with almost fixed value (more than 90% of its values are the same), and these features were [`Profession`, `Work Pressure`, `Job Satisfaction`].
4. Replace the feature `Depression` value `0` with `-1` for **Not Depressed**.
5. Using `LabelEncoder` to encode the features with `object` values.
6. Using the `MinMaxScaler` to scale the feature values.
7. Split the dataset into **_Features_** and **_Target_**.
8. Split the data into **_Training Data_** (80%) and **_Testing Data_** (20%).

## Reference Linear Classifier

For implementing a linear classifier with a library we used `SGDClassifier` from the library `sklearn.linear_model`.

### The Confusion Matrix:

<img src="assets\conf_mat_ref.png">

### The Results:

_Execution Time_: 2473 mcs

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| -1           | 0.87      | 0.72   | 0.79     | 2348    |
| 1            | 0.82      | 0.92   | 0.87     | 3232    |
| Accuracy     |           |        | 0.84     | 5580    |
| Macro Avg    | 0.84      | 0.82   | 0.83     | 5580    |
| Weighted Avg | 0.84      | 0.84   | 0.83     | 5580    |

## Custom Linear Classifier

In this task we compare Tree _Custom Linear Classifiers_:

- \*Custom Linear Classifier with **Random Weights Initialization\***: using the implemented class `CustomLinearClassifier` with `initialization="random"`.
- \*Custom Linear Classifier with **MultiStart Weights Initialization\***: using the implemented class `CustomLinearClassifier` with `initialization="random"` for a number of times (`n_start = 25`) then choosing the weights that resulted in the highest Accuracy.
- \*Custom Linear Classifier with **Correlation Weights Initialization\***: using the implemented class `CustomLinearClassifier` with `initialization="correlation"`.

### Custom Linear Classifier with **Random Weights Initialization**

#### The Loss and Quality Functional:

<img src="assets\loss_Q_cus_rand.png">

#### The Margins:

<img src="assets\M_cus_rand.png">

#### The Confusion Matrix:

<img src="assets\conf_mat_cus_rand.png">

#### The Results:

_Execution Time_: 712 mcs

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| -1           | 0.65      | 0.74   | 0.69     | 2348    |
| 1            | 0.79      | 0.71   | 0.75     | 3232    |
| Accuracy     |           |        | 0.72     | 5580    |
| Macro Avg    | 0.72      | 0.72   | 0.72     | 5580    |
| Weighted Avg | 0.73      | 0.72   | 0.72     | 5580    |

### Custom Linear Classifier with **MultiStart Weights Initialization**

The MultiStart was implemented on `n_start = 25` with these results:

```plaintext
[0]: Accuracy: 0.74
[1]: Accuracy: 0.59
[2]: Accuracy: 0.56
[3]: Accuracy: 0.71
[4]: Accuracy: 0.77
[5]: Accuracy: 0.75
[6]: Accuracy: 0.72
[7]: Accuracy: 0.66
[8]: Accuracy: 0.74
[9]: Accuracy: 0.44
[10]: Accuracy: 0.69
[11]: Accuracy: 0.49
[12]: Accuracy: 0.73
[13]: Accuracy: 0.64
[14]: Accuracy: 0.58
[15]: Accuracy: 0.60
[16]: Accuracy: 0.65
[17]: Accuracy: 0.56
[18]: Accuracy: 0.59
[19]: Accuracy: 0.76
[20]: Accuracy: 0.65
[21]: Accuracy: 0.63
[22]: Accuracy: 0.70
[23]: Accuracy: 0.53
[24]: Accuracy: 0.63
```

And the best weights:

```plaintext
Best Weights: [[-0.41615016 -0.16996526 -0.01537359  0.32515974 -0.15839347 -0.27275657
  -0.16120517 -0.10111889 -0.02488956  0.71295622  0.25876114  0.34328966
  -0.1029759 ]]
```

#### The Loss and Quality Functional:

<img src="assets\loss_Q_cus_multistart.png">

#### The Margins:

<img src="assets\M_cus_multistart.png">

#### The Confusion Matrix:

<img src="assets\conf_mat_cus_multistart.png">

#### The Results:

_Execution Time_: 1919 mcs

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| -1           | 0.82      | 0.58   | 0.68     | 2348    |
| 1            | 0.75      | 0.91   | 0.82     | 3232    |
| Accuracy     |           |        | 0.77     | 5580    |
| Macro Avg    | 0.78      | 0.74   | 0.75     | 5580    |
| Weighted Avg | 0.78      | 0.77   | 0.76     | 5580    |

### Custom Linear Classifier with **Correlation Weights Initialization**

#### The Loss and Quality Functional:

<img src="assets\loss_Q_cus_corr.png">

#### The Margins:

<img src="assets\M_cus_corr.png">

#### The Confusion Matrix:

<img src="assets\conf_mat_cus_corr.png">

#### The Results:

_Execution Time_: 8475 mcs

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| -1           | 0.74      | 0.51   | 0.61     | 2348    |
| 1            | 0.71      | 0.87   | 0.78     | 3232    |
| Accuracy     |           |        | 0.72     | 5580    |
| Macro Avg    | 0.72      | 0.69   | 0.69     | 5580    |
| Weighted Avg | 0.72      | 0.72   | 0.71     | 5580    |

# Conclusions

1. The Reference Linear Classifier was better in classifying the data than the Custom Linear Classifier.
2. The Custom Linear Classifier performs better with the **Random Weights Initialization** where it reached the **_Accuracy = 0.77_**.
3. The Custom Linear Classifier with the **Correlation Weights Initialization** had the smoothest _Quality Functional_ curve.
