import os

import numpy as np
import pydotplus

from classification_tree import ClassificationTreeNode
from regression_tree import RegressionTreeNode

os.environ['PATH'] = os.environ["PATH"] + ";D:\\PROGRAM FILES\\Graphviz\\bin"


def convert_classification_treenode2dict(node: ClassificationTreeNode):
    node_dict = {}
    node_dict["beta"] = node.beta
    node_dict["feature_idx"] = node.feature_idx
    node_dict["entropy"] = node.entropy
    node_dict["prob"] = node.prob

    if node.left is not None:
        node_dict["left"] = convert_classification_treenode2dict(node.left)
    if node.right is not None:
        node_dict["right"] = convert_classification_treenode2dict(node.right)
    
    return node_dict


def convert_regression_treenode2dict(node: RegressionTreeNode):
    node_dict = {}
    node_dict["beta"] = node.beta
    node_dict["feature_idx"] = node.feature_idx
    node_dict["uncertainty"] = node.uncertainty
    node_dict["value"] = node.value

    if node.left is not None:
        node_dict["left"] = convert_regression_treenode2dict(node.left)
    if node.right is not None:
        node_dict["right"] = convert_regression_treenode2dict(node.right)
    
    return node_dict


def convert_sklearn_treenode2dict(tree, tree_type, node_id=0):
    node_dict = {}
    node_dict["beta"] = tree.threshold[node_id]   
    node_dict["feature_idx"] = tree.feature[node_id]
    
    if tree_type == "classification":
        node_dict["entropy"] = tree.impurity[node_id]
    elif tree_type == "regression":
        node_dict["uncertainty"] = tree.impurity[node_id]
        node_dict["value"] = tree.value[node_id][0, 0]

    is_split_node = tree.children_left[node_id] != tree.children_right[node_id]
    if is_split_node:
        node_dict["left"] = convert_sklearn_treenode2dict(tree, tree_type, tree.children_left[node_id])
        node_dict["right"] = convert_sklearn_treenode2dict(tree, tree_type, tree.children_right[node_id])
    else:
        if tree_type == "classification":
            node_dict["lbl"] = np.argmax(tree.value[node_id]) 
            node_dict["prob"] = np.max(tree.value[node_id])
    
    return node_dict


def visualize_tree(tree, path, tree_type, tree_source, max_depth=4):
    if tree_source == "custom":
        if tree_type == "classification":
            tree_dict = convert_classification_treenode2dict(tree)
        elif tree_type == "regression":
            tree_dict = convert_regression_treenode2dict(tree)
        else:
            raise ValueError(f"Wrong tree type: {tree_type}") 
    elif tree_source == "sklearn":
        tree_dict = convert_sklearn_treenode2dict(tree, tree_type)
    else:
        raise ValueError(f"Wrong tree source: {tree_source}") 
    
    dot_data = pydotplus.Dot()
    current_depth = 0

    def add_node(parent_name, node, current_depth):
        if tree_source == "custom":
            if tree_type == "classification":
                label = f'x[{node["feature_idx"]}]<={node["beta"]}' +\
                    f'\nentropy={node["entropy"]:.4f}' if node["beta"] is not None \
                    else f'lbl={np.argmax(node["prob"])}\nprob={np.max(node["prob"])}'
                
            elif tree_type == "regression":

                label = f'x[{node["feature_idx"]}]<={node["beta"]}\n' if node["beta"] is not None else ''
                label += f'uncertainty={node["uncertainty"]:.4f}' + \
                    f'\nvalue={node["value"]:.2f}'
            else: raise ValueError(f"Wrong tree type: {tree_type}")

        elif tree_source == "sklearn":
            if tree_type == "classification":
                if node["feature_idx"] == -2: # это значит, что пришли к листу
                    label = f'lbl={node["lbl"]}\nprob={np.max(node["prob"])}'
                else:
                    label = f'x[{node["feature_idx"]}]<={node["beta"]:.2f}'+\
                    f'\nentropy={node["entropy"]:.4f}'
            elif tree_type == "regression":
                if node["feature_idx"] == -2: # это значит, что пришли к листу
                    label = f'uncertainty={node["uncertainty"]:.4f}' + \
                        f'\nvalue={node["value"]:.2f}'
                else:
                    label = f'x[{node["feature_idx"]}]<={node["beta"]}'+\
                        f'\nuncertainty={node["uncertainty"]:.4f}' + \
                        f'\nvalue={node["value"]:.2f}'
            else: raise ValueError(f"Wrong tree type: {tree_type}")
        
        else:
            raise ValueError(f"Wrong tree source: {tree_source}") 

        dot_data.add_node(pydotplus.Node(f'{parent_name}', shape='box', label=label))
        current_depth += 1
        if current_depth > max_depth: return
        if "left" in node.keys():
            dot_data.add_edge(pydotplus.Edge(parent_name, f'{parent_name}_{id(node["left"])}', label="yes"))
            add_node(f'{parent_name}_{id(node["left"])}', node["left"], current_depth=current_depth)
        if "right" in node.keys():
            dot_data.add_edge(pydotplus.Edge(parent_name, f'{parent_name}_{id(node["right"])}', label="no"))
            add_node(f'{parent_name}_{id(node["right"])}', node["right"], current_depth=current_depth)
    
    add_node("Tree", tree_dict, current_depth)
    dot_data.write(path, format="png")
