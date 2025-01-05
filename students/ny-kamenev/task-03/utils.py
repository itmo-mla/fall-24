from graphviz import Digraph
import os
import random
import numpy as np

os.environ['PATH'] = os.environ["PATH"] + ';C:/Program Files/Graphviz/bin'


def visualize_tree(tree, dir="classification", name="tree"):
    dot = Digraph(name=name, format="png")
    node_counter = [0]

    def add_node(node, parent=None, edge_label=""):
        if node.is_leaf():
            node_id = str(node_counter[0])
            dot.node(node_id, label=f"Prediction: {node.prediction}", shape="box", style="filled", color="lightgrey")
            if parent is not None:
                dot.edge(str(parent), node_id, label=edge_label)
            node_counter[0] += 1
            return

        node_id = str(node_counter[0])
        dot.node(node_id, label=f"{node.feature} <= {node.threshold:.2f}")
        if parent is not None:
            dot.edge(str(parent), node_id, label=edge_label)
        node_counter[0] += 1

        add_node(node.left, parent=node_id, edge_label="Yes")
        add_node(node.right, parent=node_id, edge_label="No")

    add_node(tree)

    dot.render(name, dir, cleanup=True)


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None, left_proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction
        self.left_proba = left_proba
        self.num_elems = 0

    def is_leaf(self):
        return self.left is None and self.right is None


class SklearnTreeNode:
    def __init__(self, tree, node_id):
        self.tree = tree
        self.node_id = node_id

    def is_leaf(self):
        return (self.tree.children_left[self.node_id] == -1 and
                self.tree.children_right[self.node_id] == -1)

    @property
    def feature(self):
        return self.tree.feature[self.node_id]

    @property
    def threshold(self):
        return self.tree.threshold[self.node_id]

    @property
    def left(self):
        left_child = self.tree.children_left[self.node_id]
        if left_child == -1:
            return None
        return SklearnTreeNode(self.tree, left_child)

    @property
    def right(self):
        right_child = self.tree.children_right[self.node_id]
        if right_child == -1:
            return None
        return SklearnTreeNode(self.tree, right_child)

    @property
    def prediction(self):
        return self.tree.value[self.node_id].argmax()

    @property
    def left_proba(self):
        proba = self.tree.value[self.node_id]
        return proba[0, 0] / proba.sum()


def visualize_tree_sklearn(tree, dir="classification", name="tree", feature_names=None, is_regression=False):
    dot = Digraph(name=name, format="png")
    node_counter = [0]

    def add_node(node, parent=None, edge_label=""):
        if node.is_leaf():
            node_id = str(node_counter[0])
            if (is_regression):
                prediction = node.tree.value[node.node_id].flatten().mean()
                dot.node(
                    node_id,
                    label=f"Prediction: {prediction:.2f}",
                    shape="box",
                    style="filled",
                    color="lightgrey"
                )
            else:
                dot.node(
                    node_id,
                    label=f"Prediction: {node.prediction}\nProba: {node.left_proba:.2f}",
                    shape="box",
                    style="filled",
                    color="lightgrey"
                )
            if parent is not None:
                dot.edge(str(parent), node_id, label=edge_label)
            node_counter[0] += 1
            return

        feature_label = (
            feature_names[node.feature]
            if feature_names and node.feature != -2
            else f"Feature {node.feature}"
        )
        node_id = str(node_counter[0])
        dot.node(node_id, label=f"{feature_label} ≤ {node.threshold:.2f}")
        if parent is not None:
            dot.edge(str(parent), node_id, label=edge_label)
        node_counter[0] += 1

        if node.left:
            add_node(node.left, parent=node_id, edge_label="Yes")
        if node.right:
            add_node(node.right, parent=node_id, edge_label="No")

    add_node(SklearnTreeNode(tree.tree_, 0))
    dot.render(name, dir, cleanup=True)


def inject_nan_values(input, missing_ratio=0.02, skip_column=''):
    df = input.copy()

    rows_count, cols_count = df.shape
    total_cells = rows_count * cols_count
    num_nan = max(1, int(total_cells * missing_ratio))

    if skip_column and skip_column in df.columns:
        skip_index = df.columns.get_loc(skip_column)
    else:
        skip_index = -1

    for _ in range(num_nan):
        row_idx = random.randint(0, rows_count - 1)
        col_idx = random.randint(0, cols_count - 1)

        while col_idx == skip_index:
            col_idx = random.randint(0, cols_count - 1)

        df.iat[row_idx, col_idx] = np.nan

    return df
