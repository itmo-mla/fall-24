import numpy as np
import pandas as pd
import time
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

class ID3Tree:
    def __init__(self,
                 criterion="entropy",  # 'entropy' или 'donskoy' (только для классификации)
                 mode="classification", # 'classification' или 'regression'
                 max_depth=None,
                 min_samples_split=2):

        self.criterion = criterion
        self.mode = mode
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.classes_ = None

    # === Метрики ===
    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))

    def mse(self, y):
        mean_val = np.mean(y)
        return np.mean((y - mean_val) ** 2)
    
    def donskoy_criterion(self, X, y, feature_idx, threshold=None):
        mask_not_nan = ~pd.isnull(X[:, feature_idx])
        X_not_nan = X[mask_not_nan]
        y_not_nan = y[mask_not_nan]
        if len(X_not_nan) == 0:
            return 0
        if threshold is not None:
            left_mask = (X_not_nan[:, feature_idx] <= threshold)
            right_mask = ~left_mask
        else:
            left_mask = np.zeros(len(X_not_nan), dtype=bool)
            right_mask = np.zeros(len(X_not_nan), dtype=bool)

        return np.sum(y_not_nan[left_mask][:, None] != y_not_nan[right_mask])

    def split_dataset(self, X, y, feature_idx,
                      threshold=None, value=None, is_categorical=False):
        mask_not_nan = ~pd.isnull(X[:, feature_idx])
        X_not_nan = X[mask_not_nan]
        y_not_nan = y[mask_not_nan]

        if len(X_not_nan) == 0:
            return (np.empty((0, X.shape[1])), np.array([], dtype=y.dtype)), \
                   (np.empty((0, X.shape[1])), np.array([], dtype=y.dtype)), 0.0

        if is_categorical and value is not None:
            left_mask = (X_not_nan[:, feature_idx] == value)
            right_mask = ~left_mask
        else:
            left_mask = (X_not_nan[:, feature_idx] <= threshold)
            right_mask = ~left_mask

        X_left, y_left = X_not_nan[left_mask], y_not_nan[left_mask]
        X_right, y_right = X_not_nan[right_mask], y_not_nan[right_mask]

        q_v = 0.0
        if len(y_not_nan) > 0:
            q_v = len(y_left) / len(y_not_nan)

        return (X_left, y_left), (X_right, y_right), q_v

    def find_best_split(self, X, y, depth):
        n = len(y)
        if n == 0:
            return None

        if self.mode == "classification":
            if self.criterion == "entropy":
                current_score = self.entropy(y)
            elif self.criterion == "donskoy":
                current_score = 0
            else:
                current_score = 0
        else:
            current_score = self.mse(y)

        n_features = X.shape[1]
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_value = None
        best_is_categorical = False
        best_q = 0.0

        for feature_idx in range(n_features):
            mask_not_nan = ~pd.isnull(X[:, feature_idx])
            feature_vals = X[mask_not_nan, feature_idx]
            if len(feature_vals) <= 1:
                continue

            is_cat = feature_vals.dtype.kind in ['O', 'S', 'U']
            unique_vals = np.unique(feature_vals)

            if is_cat:
                for val in unique_vals:
                    (left_X, left_y), (right_X, right_y), q_v = \
                        self.split_dataset(X, y, feature_idx,
                                           value=val, is_categorical=True)
                    total = len(left_y) + len(right_y)
                    if total == 0:
                        continue

                    if self.mode == "classification":
                        if self.criterion == "entropy":
                            p_left = len(left_y) / total
                            p_right = len(right_y) / total
                            left_ent = self.entropy(left_y) if len(left_y) else 0.0
                            right_ent = self.entropy(right_y) if len(right_y) else 0.0
                            gain = current_score - (p_left * left_ent + p_right * right_ent)
                        else:  # donskoy
                            score = self.donskoy_criterion(X, y, feature_idx, None)
                            gain = -score
                    else:
                        p_left = len(left_y) / total
                        p_right = len(right_y) / total
                        left_mse = self.mse(left_y) if len(left_y) else 0.0
                        right_mse = self.mse(right_y) if len(right_y) else 0.0
                        gain = current_score - (p_left * left_mse + p_right * right_mse)

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = None
                        best_value = val
                        best_is_categorical = True
                        best_q = q_v

            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
                for thr in thresholds:
                    (left_X, left_y), (right_X, right_y), q_v = \
                        self.split_dataset(X, y, feature_idx,
                                           threshold=thr, is_categorical=False)

                    total = len(left_y) + len(right_y)
                    if total < self.min_samples_split:
                        continue
                    if total == 0:
                        continue

                    if self.mode == "classification":
                        if self.criterion == "entropy":
                            p_left = len(left_y) / total
                            p_right = len(right_y) / total
                            left_ent = self.entropy(left_y) if len(left_y) else 0.0
                            right_ent = self.entropy(right_y) if len(right_y) else 0.0
                            gain = current_score - (p_left * left_ent + p_right * right_ent)
                        else:  # donskoy
                            score = self.donskoy_criterion(X, y, feature_idx, threshold=thr)
                            gain = -score
                    else:
                        p_left = len(left_y) / total
                        p_right = len(right_y) / total
                        left_mse = self.mse(left_y) if len(left_y) else 0.0
                        right_mse = self.mse(right_y) if len(right_y) else 0.0
                        gain = current_score - (p_left * left_mse + p_right * right_mse)

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = thr
                        best_value = None
                        best_is_categorical = False
                        best_q = q_v

        if best_feature is None or best_gain <= 0:
            return None

        return (best_feature, best_threshold, best_value,
                best_is_categorical, best_gain, best_q)

    def build_tree(self, X, y, depth=0):
        n = len(y)
        if n == 0:
            if self.mode == "classification":
                return {"label": None}
            else:
                return {"value": 0.0}

        if self.mode == "classification":
            unique_classes = np.unique(y)
            if len(unique_classes) == 1:
                return {"label": unique_classes[0]}
        else:
            if np.allclose(y, y[0]):
                return {"value": y[0]}

        if self.max_depth is not None and depth >= self.max_depth:
            if self.mode == "classification":
                most_common = np.argmax(np.bincount(y))
                return {"label": most_common}
            else:
                return {"value": np.mean(y)}

        if len(y) < self.min_samples_split:
            if self.mode == "classification":
                most_common = np.argmax(np.bincount(y))
                return {"label": most_common}
            else:
                return {"value": np.mean(y)}

        split_res = self.find_best_split(X, y, depth)
        if split_res is None:
            if self.mode == "classification":
                most_common = np.argmax(np.bincount(y))
                return {"label": most_common}
            else:
                return {"value": np.mean(y)}

        (best_feature, best_threshold, best_value,
         best_is_categorical, best_gain, best_q) = split_res

        if best_is_categorical:
            (left_X, left_y), (right_X, right_y), q_v = \
                self.split_dataset(X, y, best_feature,
                                   value=best_value,
                                   is_categorical=True)
            node = {
                "feature": best_feature,
                "value": best_value,
                "is_categorical": True,
                "q": best_q,
                "left": self.build_tree(left_X, left_y, depth+1),
                "right": self.build_tree(right_X, right_y, depth+1)
            }
        else:
            (left_X, left_y), (right_X, right_y), q_v = \
                self.split_dataset(X, y, best_feature,
                                   threshold=best_threshold,
                                   is_categorical=False)
            node = {
                "feature": best_feature,
                "threshold": best_threshold,
                "is_categorical": False,
                "q": best_q,
                "left": self.build_tree(left_X, left_y, depth+1),
                "right": self.build_tree(right_X, right_y, depth+1)
            }
        return node

    def fit(self, X, y):

        X = np.array(X, dtype=object)
        y = np.array(y)

        if self.mode == "classification":
            self.classes_ = np.unique(y)

        self.tree = self.build_tree(X, y)

    def predict_single_classification(self, x, node):
        # Если лист
        if "label" in node:
            probs = np.zeros(len(self.classes_), dtype=float)
            label = node["label"]
            if label in self.classes_:
                idx = np.where(self.classes_ == label)[0][0]
            else:
                idx = int(label)
            probs[idx] = 1.0
            return probs

        feature = node["feature"]
        q_v = node["q"]

        if node["is_categorical"]:
            # Если пропуск
            if pd.isnull(x[feature]):
                p_left = self.predict_single_classification(x, node["left"])
                p_right = self.predict_single_classification(x, node["right"])
                return q_v * p_left + (1.0 - q_v) * p_right
            else:
                if x[feature] == node["value"]:
                    return self.predict_single_classification(x, node["left"])
                else:
                    return self.predict_single_classification(x, node["right"])
        else:
            # Числовой признак
            if pd.isnull(x[feature]):
                p_left = self.predict_single_classification(x, node["left"])
                p_right = self.predict_single_classification(x, node["right"])
                return q_v * p_left + (1.0 - q_v) * p_right
            else:
                if x[feature] <= node["threshold"]:
                    return self.predict_single_classification(x, node["left"])
                else:
                    return self.predict_single_classification(x, node["right"])

    def predict_single_regression(self, x, node):
        if "value" in node:
            return node["value"]

        feature = node["feature"]
        q_v = node["q"]

        if node["is_categorical"]:
            if pd.isnull(x[feature]):
                val_left = self.predict_single_regression(x, node["left"])
                val_right = self.predict_single_regression(x, node["right"])
                return q_v * val_left + (1.0 - q_v) * val_right
            else:
                if x[feature] == node["value"]:
                    return self.predict_single_regression(x, node["left"])
                else:
                    return self.predict_single_regression(x, node["right"])
        else:
            if pd.isnull(x[feature]):
                val_left = self.predict_single_regression(x, node["left"])
                val_right = self.predict_single_regression(x, node["right"])
                return q_v * val_left + (1.0 - q_v) * val_right
            else:
                if x[feature] <= node["threshold"]:
                    return self.predict_single_regression(x, node["left"])
                else:
                    return self.predict_single_regression(x, node["right"])

    def predict(self, X):
        X = np.array(X, dtype=object)

        preds = []
        if self.mode == "classification":
            for x in X:
                prob_vec = self.predict_single_classification(x, self.tree)
                cls = self.classes_[np.argmax(prob_vec)]
                preds.append(cls)
            return np.array(preds)
        else:
            for x in X:
                val = self.predict_single_regression(x, self.tree)
                preds.append(val)
            return np.array(preds)

    def predict_proba(self, X):
        if self.mode != "classification":
            raise ValueError("predict_proba доступен только для 'classification'.")

        X = np.array(X, dtype=object)
        all_probs = []
        for x in X:
            prob_vec = self.predict_single_classification(x, self.tree)
            all_probs.append(prob_vec)
        return np.vstack(all_probs)

    def prune(self, X_val, y_val):
        def prune_node(node):
            if "label" in node or "value" in node:
                return node

            original_node = node.copy()

            if self.mode == "classification":
                # До
                y_pred_before = self.predict(X_val)
                score_before = accuracy_score(y_val, y_pred_before)

                leaf = {"label": np.argmax(np.bincount(self.predict(X_val).astype(int)))}
                node.clear()
                node.update(leaf)

                y_pred_after = self.predict(X_val)
                score_after = accuracy_score(y_val, y_pred_after)

                if score_after < score_before:
                    node.clear()
                    node.update(original_node)
                else:
                    return node
            else:
                # mode='regression'
                y_pred_before = self.predict(X_val)
                mse_before = mean_squared_error(y_val, y_pred_before)

                leaf = {"value": np.mean(y_val)}  
                node.clear()
                node.update(leaf)

                y_pred_after = self.predict(X_val)
                mse_after = mean_squared_error(y_val, y_pred_after)

                if mse_after > mse_before:
                    node.clear()
                    node.update(original_node)
                else:
                    return node
            if "left" in node:
                node["left"] = prune_node(node["left"])
            if "right" in node:
                node["right"] = prune_node(node["right"])
            return node

        self.tree = prune_node(self.tree)


df = pd.read_csv("./mushrooms.csv")

df['class'] = df['class'].map({'e': 0, 'p': 1})

np.random.seed(42)
missing_mask = np.random.rand(len(df)) < 0.05
df.loc[missing_mask, 'cap-color'] = np.nan

unique_colors = df['cap-color'].unique()
color_map = {c: i for i, c in enumerate(unique_colors)}
df['cap-color-num'] = df['cap-color'].map(color_map)

y = df['class'].values
X = df.drop('class', axis=1)

X_encoded = pd.get_dummies(X, columns=[col for col in X.columns if col != 'cap-color-num'],
                           dummy_na=False)
X_encoded = X_encoded.values

X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Shapes (Грибы):")
print(X_train.shape, X_temp.shape, y_train.shape, y_temp.shape)

model_entropy = ID3Tree(criterion="entropy", mode="classification", max_depth=2, min_samples_split=5)
model_entropy.fit(X_train, y_train)

model_donskoy = ID3Tree(criterion="donskoy", mode="classification", max_depth=10, min_samples_split=5)
model_donskoy.fit(X_train, y_train)

y_pred_entropy = model_entropy.predict(X_test)
acc_entropy = accuracy_score(y_test, y_pred_entropy)

y_pred_donskoy = model_donskoy.predict(X_test)
acc_donskoy = accuracy_score(y_test, y_pred_donskoy)

df_b = pd.read_csv("./Boston.csv")

y_b = df_b["medv"].values
X_b = df_b.drop("medv", axis=1).values

scaler = StandardScaler()
X_b_scaled = scaler.fit_transform(X_b)

Xb_train, Xb_temp, yb_train, yb_temp = train_test_split(
    X_b_scaled, y_b, test_size=0.8, random_state=42
)
Xb_val, Xb_test, yb_val, yb_test = train_test_split(
    Xb_temp, yb_temp, test_size=0.5, random_state=42
)

model_reg = ID3Tree(mode="regression", max_depth=2, min_samples_split=5)
model_reg.fit(Xb_train, yb_train)


def custom_mse(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return 0.0
    sq_sum = 0.0
    for i in range(n):
        diff = y_true[i] - y_pred[i]
        sq_sum += diff * diff
    return sq_sum / n

yb_pred = model_reg.predict(Xb_test)
mse_before_prune = custom_mse(yb_test, yb_pred)

model_entropy.prune(X_val, y_val)
model_reg.prune(Xb_val, yb_val)

y_pred_entropy_pruned = model_entropy.predict(X_test)
acc_entropy_pruned = accuracy_score(y_test, y_pred_entropy_pruned)

yb_pred_pruned = model_reg.predict(Xb_test)
mse_after_prune = custom_mse(yb_test, yb_pred_pruned)

start_time = time.time()
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, y_train)
y_pred_sklearn = clf.predict(X_test)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
time_sklearn_class = time.time() - start_time

start_time = time.time()
y_pred_our = model_entropy.predict(X_test)
time_our_class = time.time() - start_time


start_time = time.time()
rgr = DecisionTreeRegressor(max_depth=2, random_state=42)
rgr.fit(Xb_train, yb_train)
yb_pred_sklearn = rgr.predict(Xb_test)
mse_sklearn = custom_mse(yb_test, yb_pred_sklearn)
time_sklearn_reg = time.time() - start_time

start_time = time.time()
yb_pred_our = model_reg.predict(Xb_test)
time_our_reg = time.time() - start_time

print("=== Классификация (Грибочки) ===")
print(f"Accuracy (ID3 Entropy): до редукции={acc_entropy:.4f}, после редукции={acc_entropy_pruned:.4f}")
print(f"Accuracy (ID3 Donskoy): {acc_donskoy:.4f}")
print(f"Accuracy (Sklearn DecisionTreeClassifier): {acc_sklearn:.4f}")
print(f"Время работы ID3 (предсказание): {time_our_class:.6f} с")
print(f"Время работы sklearn (предсказание): {time_sklearn_class:.6f} с")

print("\n=== Регрессия (Boston) ===")
print(f"MSE (ID3 до редукции)={mse_before_prune:.4f}, после редукции={mse_after_prune:.4f}")
print(f"MSE (Sklearn DecisionTreeRegressor)={mse_sklearn:.4f}")
print(f"Время работы ID3 (предсказание): {time_our_reg:.6f} с")
print(f"Время работы sklearn (предсказание): {time_sklearn_reg:.6f} с")
