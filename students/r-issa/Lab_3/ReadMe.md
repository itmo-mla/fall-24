# Lab Work №3

## The used dataset:

In this lab The [Star dataset to predict star types](https://www.kaggle.com/datasets/deepu1109/star-dataset) was used. This is a dataset consisting of several features of stars (240 stars of 6 classes) the feature **Star Type** was the target feature:

1. Star Type = 0 -> **Brown Dwarf**
2. Star Type = 1 -> **Red Dwarf**
3. Star Type = 2 -> **White Dwarf**
4. Star Type = 3 -> **Main Sequence**
5. Star Type = 4 -> **Supergiant**
6. Star Type = 5 -> **Hypergiant**

The data contains categorical and numerical features, and the gaps (missing values are added manually)

## Classification Task

In this task we compare Five _Decision Tree Classifiers_ with the `MAX_DEPTH = 6`:

- _Reference Decision Tree Classifier_: using the `DecisionTreeClassifier` from the library `sklearn.tree`.
- _Custom Decision Tree Classifier with criterion **Entropy** before pruning_: using the implemented class `CustomDecisionTreeClassifier` before using the method `prune`.
- _Custom Decision Tree Classifier with criterion **Entropy** after pruning_: using the implemented class `CustomDecisionTreeClassifier` after using the method `prune`.
- _Custom Decision Tree Classifier with criterion **Donskoy** before pruning_: using the implemented class `CustomDecisionTreeClassifier` before using the method `prune`.
- _Custom Decision Tree Classifier with criterion **Donskoy** after pruning_: using the implemented class `CustomDecisionTreeClassifier` after using the method `prune`.

### **Reference Decision Tree Classifier:**

**The Tree:**  
<img src='assets\classification_reference.png'>

**The Confusion Matrix:**  
<img src='assets\confusion_matrix_classification_reference.png'>

**The Results:**  
_Execution Time_: 1047 mcs  
| | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| 0 | 0.80 | 1.00 | 0.89 | 8 |
| 1 | 1.00 | 0.71 | 0.83 | 7 |
| 2 | 1.00 | 1.00 | 1.00 | 6 |
| 3 | 1.00 | 1.00 | 1.00 | 8 |
| 4 | 1.00 | 1.00 | 1.00 | 8 |
| 5 | 1.00 | 1.00 | 1.00 | 11 |
| Accuracy | | | 0.96 | 48 |
| Macro Avg | 0.97 | 0.95 | 0.95 | 48 |
| Weighted Avg | 0.97 | 0.96 | 0.96 | 48 |

### **Custom Decision Tree Classifier with criterion _Entropy_ before pruning:**

**The Tree:**  
<img src='assets\classification_entropy_custom.png'>

**The Confusion Matrix:**  
<img src='assets\confusion_matrix_classification_entropy_custom.png'>

**The Results:**  
_Execution Time_: 806 mcs  
| | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| 0 | 1.00 | 0.75 | 0.86 | 8 |
| 1 | 0.75 | 0.86 | 0.80 | 7 |
| 2 | 1.00 | 1.00 | 1.00 | 6 |
| 3 | 0.88 | 0.88 | 0.88 | 8 |
| 4 | 0.89 | 1.00 | 0.94 | 8 |
| 5 | 1.00 | 1.00 | 1.00 | 11 |
| Accuracy | | | 0.92 | 48 |
| Macro Avg | 0.92 | 0.91 | 0.91 | 48 |
| Weighted Avg | 0.92 | 0.92 | 0.92 | 48 |

### **Custom Decision Tree Classifier with criterion _Entropy_ after pruning:**

**The Tree:**  
<img src='assets\classification_entropy_custom_prun.png'>

**The Confusion Matrix:**  
<img src='assets\confusion_matrix_classification_entropy_custom_prun.png'>

**The Results:**  
_Execution Time_: 793 mcs  
| | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| 0 | 1.00 | 1.00 | 1.00 | 8 |
| 1 | 1.00 | 1.00 | 1.00 | 7 |
| 2 | 1.00 | 1.00 | 1.00 | 6 |
| 3 | 1.00 | 0.88 | 0.93 | 8 |
| 4 | 0.89 | 1.00 | 0.94 | 8 |
| 5 | 1.00 | 1.00 | 1.00 | 11 |
| Accuracy | | | 0.98 | 48 |
| Macro Avg | 0.98 | 0.98 | 0.98 | 48 |
| Weighted Avg | 0.98 | 0.98 | 0.98 | 48 |

### **Custom Decision Tree Classifier with criterion _Donskoy_ before pruning:**

**The Tree:**  
<img src='assets\classification_donskoy_custom.png'>

**The Confusion Matrix:**  
<img src='assets\confusion_matrix_classification_donskoy_custom.png'>

**The Results:**  
_Execution Time_: 524 mcs  
| | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| 0 | 1.00 | 1.00 | 1.00 | 8 |
| 1 | 1.00 | 0.86 | 0.92 | 7 |
| 2 | 0.86 | 1.00 | 0.92 | 6 |
| 3 | 0.88 | 0.88 | 0.88 | 8 |
| 4 | 1.00 | 1.00 | 1.00 | 8 |
| 5 | 1.00 | 1.00 | 1.00 | 11 |
| Accuracy | | | 0.96 | 48 |
| Macro Avg | 0.96 | 0.96 | 0.95 | 48 |
| Weighted Avg | 0.96 | 0.96 | 0.96 | 48 |

### **Custom Decision Tree Classifier with criterion _Donskoy_ after pruning:**

**The Tree:**  
<img src='assets\classification_donskoy_custom_prun.png'>

**The Confusion Matrix:**  
<img src='assets\confusion_matrix_classification_donskoy_custom_prun.png'>

**The Results:**  
_Execution Time_: 772 mcs  
| | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| 0 | 1.00 | 1.00 | 1.00 | 8 |
| 1 | 0.88 | 1.00 | 0.93 | 7 |
| 2 | 1.00 | 1.00 | 1.00 | 6 |
| 3 | 1.00 | 0.88 | 0.93 | 8 |
| 4 | 1.00 | 1.00 | 1.00 | 8 |
| 5 | 1.00 | 1.00 | 1.00 | 11 |
| Accuracy | | | 0.98 | 48 |
| Macro Avg | 0.98 | 0.98 | 0.98 | 48 |
| Weighted Avg | 0.98 | 0.98 | 0.98 | 48 |

## Regression Task

In this task we compare Three _Decision Tree Regressors_ with the `MAX_DEPTH = 6`:

- _Reference Decision Tree Regressor_: using the `DecisionTreeRegressor` from the library `sklearn.tree`.
- _Custom Decision Tree Regressor before pruning_: using the implemented class `CustomDecisionTreeRegressor` before using the method `prune`.
- _Custom Decision Tree Regressor after pruning_: using the implemented class `CustomDecisionTreeRegressor` after using the method `prune`.

### **The Trees:**

**Reference Decision Tree Regressor:**  
<img src='assets\regression_reference.png'>

**Custom Decision Tree Regressor before pruning:**  
<img src='assets\regression_custom.png'>

**Custom Decision Tree Regressor after pruning:**  
<img src='assets\regression_custom_prun.png'>

### **The R<sup>2</sup> Scores:**

|                     | Reference | Custom before pruning | Custom after pruning |
| ------------------- | --------- | --------------------- | -------------------- |
| Execution Time      | 932 mcs   | 767 mcs               | 1358 mcs             |
| R<sup>2</sup> score | 0.9870    | 0.9513                | 0.9610               |

# Conclusions

1. The Custom Decision Tree Classifier performs relatively close to the Reference Decision Tree Classifier.
2. The Custom Decision Tree Classifier performs better when pruned.
3. The Custom Decision Tree Regressor performs relatively close to the Reference Decision Tree Regressor.
4. The Custom Decision Tree Regressor performs better when pruned.

**Note:** The result may change when running the entire code again because the manually added gaps differ every run, but these conclusions stay true.
