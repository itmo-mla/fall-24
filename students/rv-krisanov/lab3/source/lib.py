from sklearn.model_selection import train_test_split
from numpy import signedinteger, ndarray
from dataclasses import dataclass
from typing import Self, Callable, Literal
from enum import IntEnum
import numpy.typing as npt
import numpy as np
import pandas as pd

class FeatureType(IntEnum):
    CATEGORY = 0
    CONTINUOUS = 1


@dataclass(slots=True, frozen=True)
class Predicate:
    name: str
    value: int | float
    feature_type: FeatureType


@dataclass
class LeafNode:
    parent: "InnerNode"
    value: signedinteger | np.float_


@dataclass
class InnerNode:
    parent: Self
    predicate: Predicate
    children: list[LeafNode | Self]

    @property
    def right_child(self):
        return self.children[-1]

    @property
    def left_child(self):
        return self.children[0]


def apply_predicate_plural(x: pd.DataFrame, predicate: Predicate) -> npt.NDArray[np.bool_]:
    if predicate.feature_type == FeatureType.CATEGORY:
        return np.where(x[predicate.name].values == predicate.value)[0]
    elif predicate.feature_type == FeatureType.CONTINUOUS:
        return np.where(x[predicate.name].values >= predicate.value)[0]


def apply_predicate_singular(x: pd.Series, predicate: Predicate) -> bool:
    if predicate.feature_type == FeatureType.CATEGORY:
        return x[predicate.name] == predicate.value
    elif predicate.feature_type == FeatureType.CONTINUOUS:
        return x[predicate.name] >= predicate.value


def _uncertainty_measure_criterion(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    class_probabilities = counts / len(y)
    return class_probabilities


CriterionT = Callable[[np.ndarray], float]


def entropy(y: np.ndarray) -> float:
    class_probabilities = _uncertainty_measure_criterion(y)
    return - np.sum(class_probabilities * np.log2(class_probabilities + 1e-6))


def gini(y: np.ndarray) -> float:
    class_probabilities = _uncertainty_measure_criterion(y)
    return np.sum(class_probabilities * (1 - class_probabilities))


def donskoy(y: np.ndarray) -> float:
    return - gini(y)

def mse(y: np.ndarray) -> float:
    return np.mean(y ** 2)

def information_gain(
        x: pd.DataFrame,
        y: np.ndarray,
        predicate: Predicate,
        criterion: CriterionT,
) -> float:
    uncertainty_measure = criterion(y)

    right_leaf_indices = apply_predicate_plural(x, predicate)
    left_leaf_indices = np.setdiff1d(np.arange(len(y)), right_leaf_indices)

    left_leaf, right_leaf = y[left_leaf_indices], y[right_leaf_indices]
    left_uncertainty, right_uncertainty = criterion(left_leaf), criterion(right_leaf)

    return uncertainty_measure - (
            + left_uncertainty * len(left_leaf) / len(y)
            + right_uncertainty * len(right_leaf) / len(y)
    )


def information_gain2(x: pd.DataFrame, y: np.ndarray, predicate: Predicate) -> float:
    uncertainty_measure = entropy(y)
    right_leaf_indices = apply_predicate_plural(x, predicate)
    left_leaf_indices = np.setdiff1d(np.arange(len(y)), right_leaf_indices)
    left_leaf, right_leaf = y[left_leaf_indices], y[right_leaf_indices]
    return uncertainty_measure - (
            + entropy(left_leaf) * len(left_leaf) / len(y)
            + entropy(right_leaf) * len(right_leaf) / len(y)
    )


def predict(id3_root: LeafNode | InnerNode, x: pd.Series) -> signedinteger | np.float_:
    node = id3_root
    while isinstance(node, InnerNode):
        if apply_predicate_singular(x, node.predicate):
            node = node.right_child
        else:
            node = node.left_child
    return node.klass


def predict_bulk(id3_root: LeafNode | InnerNode, x: pd.DataFrame) -> ndarray:
    return np.array([predict(id3_root, sample) for _, sample in x.iterrows()])


class I3:
    root: LeafNode | InnerNode | None
    criterion: CriterionT
    feature_type: FeatureType

    def __init__(
            self,
            criterion: CriterionT,
            feature_type: FeatureType,
            feature_type_map: dict[Literal['Survived'], FeatureType]
    ):
        self.criterion = criterion
        self.root = None
        self.feature_type = feature_type
        self.feature_type_map = feature_type_map

    def _build_i3_recursive(
            self,
            parent: InnerNode | None,
            x: pd.DataFrame,
            y: np.ndarray,
    ) -> LeafNode | InnerNode:
        if (
                np.unique(y).size > 1
                and
                (relevant_predicates := [
                    Predicate(name=column, value=unique_value, feature_type=self.feature_type_map[column])
                    for column in x.columns if len(unique_values := x[column].dropna().unique()) > 1
                    for unique_value in unique_values
                ])
        ):
            best_predicate = relevant_predicates[
                np.argmax(
                    [
                        information_gain(x, y, predicate, self.criterion)
                        for predicate in relevant_predicates
                    ]
                )
            ]
            right_leaf_indices = apply_predicate_plural(x, best_predicate)
            left_leaf_indices = np.setdiff1d(np.arange(len(y)), right_leaf_indices)
            if left_leaf_indices.size and right_leaf_indices.size:
                new_inner_node = InnerNode(
                    parent=parent,
                    predicate=best_predicate,
                    children=[],
                )
                perspective_left_leaf, perspective_right_leaf = y[left_leaf_indices], y[right_leaf_indices]

                new_inner_node.children = [
                    self._build_i3_recursive(new_inner_node, x.iloc[left_leaf_indices], perspective_left_leaf),
                    self._build_i3_recursive(new_inner_node, x.iloc[right_leaf_indices], perspective_right_leaf),
                ]
                return new_inner_node

        return LeafNode(
            parent=parent,
            value=self._node_value(y),
        )

    def fit(self, x: pd.DataFrame, y: np.ndarray):
        self.root = self._build_i3_recursive(None, x, y)

    def _predict(self, node: LeafNode | InnerNode, x: pd.Series) -> signedinteger | np.float_:
        while isinstance(node, InnerNode):
            if apply_predicate_singular(x, node.predicate):
                node = node.right_child
            else:
                node = node.left_child
        return node.value

    def predict(self, x: pd.Series) -> ndarray:
        return self._predict(self.root, x)

    def predict_bulk(self, x: pd.DataFrame) -> ndarray:
        return self._predict_bulk(self.root, x)

    def _predict_bulk(self, node: LeafNode | InnerNode, x: pd.DataFrame) -> ndarray:
        return np.array([self._predict(node, sample) for _, sample in x.iterrows()])

    def _node_value(self, y: np.ndarray) -> signedinteger | np.float_:
        if self.feature_type is FeatureType.CATEGORY:
            return np.argmax(np.bincount(y))
        else:
            if len(y) == 0:
                raise Exception("Empty leaf")
            return np.mean(y)

    def _post_pruning_recursive(self, node: LeafNode | InnerNode, X_ctrl: pd.DataFrame, y_ctrl: np.ndarray) -> None:
        if isinstance(node, LeafNode):
            return
        right_leaf_indices = apply_predicate_plural(X_ctrl, node.predicate)
        left_leaf_indices = np.setdiff1d(np.arange(len(y_ctrl)), right_leaf_indices)
        if not left_leaf_indices.size:
            node.children[0] = LeafNode(
                parent=node,
                value=self._node_value(y_ctrl),
            )

        if not right_leaf_indices.size:
            node.children[1] = LeafNode(
                parent=node,
                value=self._node_value(y_ctrl),
            )
        self._post_pruning_recursive(node.left_child, X_ctrl.iloc[left_leaf_indices], y_ctrl[left_leaf_indices])
        self._post_pruning_recursive(node.right_child, X_ctrl.iloc[right_leaf_indices], y_ctrl[right_leaf_indices])
        if (parent := node.parent) is None:
            return

        error_count_parent = np.sum(y_ctrl != self._predict_bulk(node, X_ctrl))
        error_count_left = np.sum(y_ctrl != self._predict_bulk(node.left_child, X_ctrl))
        error_count_right = np.sum(y_ctrl != self._predict_bulk(node.right_child, X_ctrl))

        values = np.unique(y_ctrl)
        if self.feature_type is FeatureType.CATEGORY:
            error_counts_just_value = [np.sum(y_ctrl != value) for value in values]
        else:
            error_counts_just_value = [mse(value - y_ctrl) for value in values]

        NODE_ERROR_IDX, LEFT_CHILD_ERROR_IDX, RIGHT_CHILD_ERROR_IDX, *UNIVERSAL_VALUE_ERROR_IDS = \
            list(range(3 + len(np.unique(y_ctrl))))

        errors = [error_count_parent, error_count_left, error_count_right, *error_counts_just_value]
        # because when errors count are equals, we should prefer to reduce tree complexity

        argmin_index = len(errors) - 1 - np.argmin(errors[::-1])

        if argmin_index != NODE_ERROR_IDX:
            child_idx = [child is node for child in parent.children].index(True)


        if argmin_index == LEFT_CHILD_ERROR_IDX:
            left_child = node.left_child
            parent.children[child_idx] = left_child
            left_child.parent = parent
        elif argmin_index == RIGHT_CHILD_ERROR_IDX:
            right_child = node.right_child
            parent.children[child_idx] = right_child
            right_child.parent = parent
        elif argmin_index in UNIVERSAL_VALUE_ERROR_IDS:
            parent.children[child_idx] = LeafNode(
                parent=parent,
                value=values[argmin_index - len([NODE_ERROR_IDX, LEFT_CHILD_ERROR_IDX, RIGHT_CHILD_ERROR_IDX])],
            )

    def post_pruning(
            self,
            X_ctrl: pd.DataFrame,
            y_ctrl: np.ndarray
    ) -> LeafNode | InnerNode:
        self._post_pruning_recursive(self.root, X_ctrl, y_ctrl)
        return self.root

