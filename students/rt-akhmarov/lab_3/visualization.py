import pydotplus
import numpy as np

def convert_classification_treenode2dict(node):
    """
    Конвертирует узел дерева классификации в словарь.
    """
    node_dict = {}
    node_dict["predicat"] = node.predicat
    node_dict["feature_idx"] = node.feature_idx
    node_dict["info_gain"] = node.info_gain
    node_dict["prob"] = node.prob

    if node.left is not None:
        node_dict["left"] = convert_classification_treenode2dict(node.left)
    if node.right is not None:
        node_dict["right"] = convert_classification_treenode2dict(node.right)

    return node_dict

def visualize_tree(tree, path, tree_type, tree_source, max_depth=4):
    """
    Визуализация дерева решений в формате PNG.

    Args:
        tree: Объект дерева решений (Node).
        path: Путь для сохранения изображения.
        tree_type: Тип дерева ("classification" или "regression").
        tree_source: Источник дерева ("custom" или "sklearn").
        max_depth: Максимальная глубина визуализации.
    """
    if tree_source == "custom":
        if tree_type == "classification":
            tree_dict = convert_classification_treenode2dict(tree)
        else:
            raise ValueError(f"Unsupported tree type for custom source: {tree_type}")
    else:
        raise ValueError(f"Unsupported tree source: {tree_source}")

    dot_data = pydotplus.Dot()
    current_depth = 0

    def add_node(parent_name, node, current_depth):
        """
        Рекурсивно добавляет узлы дерева в граф.
        """
        if tree_source == "custom":
            if tree_type == "classification":
                label = (
                    f'x[{node["feature_idx"]}] <= {node["predicat"]}\n'
                    f'info_gain={node["info_gain"]:.4f}'
                    if node["predicat"] is not None
                    else f'class={np.argmax(node["prob"])}\nprob={np.max(node["prob"]):.2f}'
                )
            else:
                raise ValueError(f"Unsupported tree type: {tree_type}")
        else:
            raise ValueError(f"Unsupported tree source: {tree_source}")

        dot_data.add_node(pydotplus.Node(parent_name, shape='box', label=label))
        current_depth += 1

        if current_depth > max_depth:
            return

        if "left" in node.keys():
            dot_data.add_edge(pydotplus.Edge(parent_name, f'{parent_name}_{id(node["left"])}', label="yes"))
            add_node(f'{parent_name}_{id(node["left"])}', node["left"], current_depth=current_depth)

        if "right" in node.keys():
            dot_data.add_edge(pydotplus.Edge(parent_name, f'{parent_name}_{id(node["right"])}', label="no"))
            add_node(f'{parent_name}_{id(node["right"])}', node["right"], current_depth=current_depth)

    add_node("Tree", tree_dict, current_depth)
    dot_data.write_png(path)
