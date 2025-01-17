# Lab Work №6

## The used dataset:

In this lab The [Laptop Price Dataset](https://www.kaggle.com/datasets/ironwolf437/laptop-price-dataset) was used. This is a dataset that contains data aimed at predicting the price of laptops based on their specifications. The feature **Price (Euro)** was the target feature.

The dataset was preprocessed in the following steps:

1. Using `LabelEncoder` to encode the features with `object` values.
2. Using the `MinMaxScaler` to scale the feature values.
3. Split the dataset into **_Features_** and **_Target_**.
4. Split the data into **_Training Data_** (80%) and **_Testing Data_** (20%).

## Reference Linear Regressor

For implementing Linear Regressor with a library we used `Ridge` from the library `sklearn.linear_model`.

### Alpha Optimization

<img src="assets\alpha_optimization.png">

### The Results:

<img src="assets\ref_results.png">

### The Metrics:

_Execution Time_: 678 mcs

_R2 Score_: 0.6950

## Custom Linear Regressor

For the Custom SVM with we implemented the class `CustomRidge`.

### Tau Optimization

<img src="assets\tau_optimization.png">

### The Results:

<img src="assets\cus_results.png">

### The Metrics:

_Execution Time_: 148 mcs

_R2 Score_: 0.6885

## Conclusions

|           | Execution Time | R2 Score |
| --------- | -------------- | -------- |
| Reference | 678 mcs        | 0.6950   |
| Custom    | 148 mcs        | 0.6885   |

1. We found that the best alpha for the reference implementation was `0.0001` and the **_R2 Score_** decreased as the alpha grew.
2. We found that the best tau for the custom implementation was `0.0001` and the **_R2 Score_** decreased as the tau grew.
3. the reference implementation was better than the custom implementation according to the **_R2 Score_**
4. the custom implementation was faster than the reference implementation according to the **_Execution Time_**
