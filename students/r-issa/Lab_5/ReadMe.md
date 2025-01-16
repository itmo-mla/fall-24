# Lab Work №5

## The used dataset:

In this lab The [Breast Cancer Dataset](https://www.kaggle.com/datasets/rahmasleam/breast-cancer) was used. This is a dataset that contains data aimed at predicting the diagnosis of tumors. The feature **diagnosis** was the target feature, where it was a boolean feature:

1. diagnosis = M -> **Malignant (Cancerous)**
2. diagnosis = B -> **Benign (non-Cancerous)**

The dataset was preprocessed in the following steps:

1. Removing the `id` feature because it doesn't affect the classification.
2. Replace the feature `diagnosis` value `B` with `-1` and the value `M` with `1`.
3. Using `LabelEncoder` to encode the features with `object` values.
4. Using the `MinMaxScaler` to scale the feature values.
5. Split the dataset into **_Features_** and **_Target_**.
6. Split the data into **_Training Data_** (80%) and **_Testing Data_** (20%).

## Reference SVM

For implementing SVM with a library we used `SVC` from the library `sklearn.svm`.

### SVM with Linear Kernel

#### The Decision Boundary

<img src="assets\db_svm_linear_ref.png">

#### The Confusion Matrix

<img src="assets\cm_svm_linear_ref.png">

#### The Results:

_Execution Time_: 1293 mcs

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| -1           | 0.97      | 0.99   | 0.98     | 71      |
| 1            | 0.98      | 0.95   | 0.96     | 43      |
| Accuracy     |           |        | 0.97     | 114     |
| Macro Avg    | 0.97      | 0.97   | 0.97     | 114     |
| Weighted Avg | 0.97      | 0.97   | 0.97     | 114     |

### SVM with RBF Kernel

#### The Decision Boundary

<img src="assets\db_svm_rbf_ref.png">

#### The Confusion Matrix

<img src="assets\cm_svm_rbf_ref.png">

#### The Results:

_Execution Time_: 1751 mcs

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| -1           | 0.96      | 0.99   | 0.97     | 71      |
| 1            | 0.98      | 0.93   | 0.95     | 43      |
| Accuracy     |           |        | 0.96     | 114     |
| Macro Avg    | 0.97      | 0.96   | 0.96     | 114     |
| Weighted Avg | 0.97      | 0.96   | 0.96     | 114     |

### SVM with Polynomial Kernel

#### The Decision Boundary

<img src="assets\db_svm_poly_ref.png">

#### The Confusion Matrix

<img src="assets\cm_svm_poly_ref.png">

#### The Results:

_Execution Time_: 1579 mcs

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| -1           | 0.92      | 1.00   | 0.96     | 71      |
| 1            | 1.00      | 0.86   | 0.93     | 43      |
| Accuracy     |           |        | 0.95     | 114     |
| Macro Avg    | 0.96      | 0.93   | 0.94     | 114     |
| Weighted Avg | 0.95      | 0.95   | 0.95     | 114     |

## Custom SVM

For the Custom SVM with we implemented the class `CustomSVM`.

### SVM with Linear Kernel

#### The Decision Boundary

<img src="assets\db_svm_linear_cus.png">

#### The Confusion Matrix

<img src="assets\cm_svm_linear_cus.png">

#### The Results:

_Execution Time_: 261 mcs

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| -1           | 1.00      | 0.94   | 0.97     | 71      |
| 1            | 0.91      | 1.00   | 0.96     | 43      |
| Accuracy     |           |        | 0.96     | 114     |
| Macro Avg    | 0.96      | 0.97   | 0.96     | 114     |
| Weighted Avg | 0.97      | 0.96   | 0.97     | 114     |

### SVM with RBF Kernel

#### The Decision Boundary

<img src="assets\db_svm_rbf_cus.png">

#### The Confusion Matrix

<img src="assets\cm_svm_rbf_cus.png">

#### The Results:

_Execution Time_: 673 mcs

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| -1           | 0.95      | 0.99   | 0.97     | 71      |
| 1            | 0.97      | 0.91   | 0.94     | 43      |
| Accuracy     |           |        | 0.96     | 114     |
| Macro Avg    | 0.96      | 0.95   | 0.95     | 114     |
| Weighted Avg | 0.96      | 0.96   | 0.96     | 114     |

### SVM with Polynomial Kernel

#### The Decision Boundary

<img src="assets\db_svm_poly_cus.png">

#### The Confusion Matrix

<img src="assets\cm_svm_poly_cus.png">

#### The Results:

_Execution Time_: 738 mcs

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| -1           | 0.99      | 0.99   | 0.99     | 71      |
| 1            | 0.98      | 0.98   | 0.98     | 43      |
| Accuracy     |           |        | 0.98     | 114     |
| Macro Avg    | 0.98      | 0.98   | 0.98     | 114     |
| Weighted Avg | 0.98      | 0.98   | 0.98     | 114     |
