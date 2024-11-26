from pydantic import BaseModel
from desicion_tree_classifier_id3 import DecisionTreeClassifierID3
from desicion_tree_regressor_id3 import DecisionTreeRegressorID3
from typing import Literal
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class ClassificationScores(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float


class RegressionScores(BaseModel):
    mse: float
    mae: float
    r2: float


def evaluate_classifier(
    clf: DecisionTreeClassifierID3 | DecisionTreeClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_types: list[Literal["categorical", "numeric"]] | None = None,
) -> ClassificationScores:
    """
    Evaluate a classifier using accuracy, precision, recall and F1 scores.

    Parameters
    ----------
    clf : DecisionTreeID3 or sklearn classifier
        The classifier to evaluate
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    feature_types : list[str] | None
        List of feature types ("categorical" or "numeric") for DecisionTreeClassifierID3.
        Not used for sklearn classifiers.

    Returns
    -------
    EvaluationMetrics
        Object containing accuracy, precision, recall and f1 scores
    """
    # Fit the classifier
    match clf:
        case DecisionTreeClassifierID3():
            if feature_types is None:
                raise ValueError(
                    "feature_types must be provided for DecisionTreeClassifierID3"
                )
            clf.fit(X_train, y_train, feature_types)
            y_pred = clf.predict(X_test)
        case DecisionTreeClassifier():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        case _:
            raise ValueError("Unsupported classifier")

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    return ClassificationScores(
        accuracy=accuracy, precision=precision, recall=recall, f1=f1
    )


def evaluate_regressor(
    clf: DecisionTreeRegressorID3 | DecisionTreeRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_types: list[Literal["categorical", "numeric"]] | None = None,
) -> RegressionScores:
    """
    Evaluate a regressor using MSE, MAE and R2 metrics.

    Parameters
    ----------
    clf : DecisionTreeRegressorID3 or sklearn regressor
        The regressor to evaluate
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    feature_types : list[str] | None
        List of feature types ("categorical" or "numeric") for DecisionTreeRegressorID3.
        Not used for sklearn regressors.

    Returns
    -------
    RegressionScores
        Object containing mse, mae and r2 scores
    """
    match clf:
        case DecisionTreeRegressorID3():
            if feature_types is None:
                raise ValueError(
                    "feature_types must be provided for DecisionTreeRegressorID3"
                )
            clf.fit(X_train, y_train, feature_types)
            y_pred = clf.predict(X_test)
        case DecisionTreeRegressor():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

    # Calculate metrics with NaN handling
    valid_indices = ~np.isnan(y_test) & ~np.isnan(y_pred)
    y_test_clean = y_test[valid_indices]
    y_pred_clean = y_pred[valid_indices]

    if len(y_test_clean) == 0:
        raise ValueError("No valid predictions after removing NaN values")

    mse = mean_squared_error(y_test_clean, y_pred_clean)
    mae = mean_absolute_error(y_test_clean, y_pred_clean)
    r2 = r2_score(y_test_clean, y_pred_clean)

    return RegressionScores(mse=mse, mae=mae, r2=r2)
