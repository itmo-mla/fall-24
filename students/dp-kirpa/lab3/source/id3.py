from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.metrics import mean_squared_error


class Predicate:
  def __init__(self, func: Callable, possible_values = {0, 1}):
    self.func = func
    self.possible_values = possible_values

  def __call__(self, x) -> Any:
    if pd.isnull(np.array(x)).any():
      raise ValueError("Nan значение передано в предикат")
    return self.func(x)
  
  def is_categorical(data, feature_index_or_name) -> bool:
    if isinstance(data, pd.DataFrame):
        return data.dtypes[feature_index_or_name].name == 'object' or data.dtypes[feature_index_or_name].name == 'category'
    else:
        # Для массивов предполагаем, что все строковые данные категориальные
        return isinstance(data[0][feature_index_or_name], str)

def make_predicate_function(feature, bin_range, eq_value=None) -> Callable:
    if eq_value is not None:
        return lambda x: x[feature] == eq_value
    else:
        return lambda x: bin_range[0] <= x[feature] < bin_range[1]

def generate_predicates(data, n_bins=3) -> Iterable:
    if isinstance(data, pd.DataFrame):
        columns = data.columns
    else:
        columns = range(len(data[0]))

    for feature_index_or_name in columns:
        feature_data = data[feature_index_or_name] if isinstance(data, pd.DataFrame) else list(zip(*data))[feature_index_or_name]
        if is_categorical(data, feature_index_or_name):
            unique_values = np.unique(feature_data, return_counts=False)
            for value in unique_values:
                yield make_predicate_function(feature_index_or_name, None, eq_value=value)
        else:
            min_val, max_val = min(feature_data), max(feature_data)
            bin_edges = np.linspace(min_val, max_val, num=n_bins+1)
            for i in range(len(bin_edges) - 1):
                yield make_predicate_function(feature_index_or_name, (bin_edges[i], bin_edges[i+1]))

class ID3ClassifierNode:
  def __init__(self):
    self.predicate = None
    self.children = {}
    self.children_probas = {}
    self.child_sub_samples = {}
    # лист от внутренней вершины отличается тем, что у листа class_label != None
    self.class_label = None
    self.major_class = None
    # после стрижки здесь может быть дочерняя вершина, которая заменяет данную
    self.prunned_by = None

  def predict_probas(self, x) -> dict[int, Any]:
    # если листовая вершина
    if self.class_label:
      # для листовой children probas - это вероятности классов
      return self.children_probas

    if self.predicate:
      try:
        predicate_value = self.predicate(x)
      except Exception:
        # βv (x) не определено =⇒ пропорциональное распределение
        probas = {}
        for child_predicate_value, child in self.children.items():
          child_probas = child.predict_probas(x)
          for class_label, class_label_proba in child_probas.items():
            probas.setdefault(class_label, 0)
            probas[class_label] += (
                class_label_proba *
                self.children_probas.get(child_predicate_value, 0)
                )
        return probas
      else:
        # значение из параллельной ветки не учитывается
        return self.children[predicate_value].predict_probas(x)
    else:
      if self.prunned_by:
        return self.prunned_by.predict_probas(x)
      raise ValueError(
          "Не задан ни класс для листовой, ни предикат для внутренней вершины"
          )

  def forward(self, x) -> Any:
    probas = self.predict_probas(x)
    return max(probas, key=probas.get)

  def accuracy(self, X, y) -> float:
    hits_n = 0
    for i in range(len(X)):
      hits_n += int(self.forward(X.to_numpy()[i]) == y.iloc[i])
    return hits_n / len(y)

  def prune(self, X, y) -> None:
    # X := подмножество объектов Xk, дошедших до текущей вершины

    # для всех v ∈ Vвнутр
    if self.class_label or not self.predicate:
      return

    # если Sv = ∅ то
    if not len(X):
      self.class_label = self.major_class
      self.children_probas = {self.class_label: 1}

    # Оцениваем ошибку текущего узла, если бы он был листом
    class_labels, counts = np.unique(y, return_counts=True)
    major_class = class_labels[np.argmax(counts)]
    error_leaf = np.sum(y != major_class)  # ошибка, если бы узел был листом

    # Рассчитываем ошибку для текущего дерева и каждого из детей
    v_error = 1 - self.accuracy(X, y)
    v_children = [1 - child.accuracy(X, y) for child in self.children.values()]

    # Сравниваем ошибки и принимаем решение о прунинге
    if (min_error := min(error_leaf, v_error, *v_children)) == error_leaf:
      # Превратить узел в лист
      self.class_label = major_class
      self.children_probas = {self.class_label: 1}
      return

    if min_error == v_error:
      for child in self.children.values():
        for child_predicate_value, child in self.children.items():
          mask = (
              np.apply_along_axis(self.predicate, 1, X) == child_predicate_value
          )
          sub_X = X[mask]
          sub_y = y[mask]
          child.prune(sub_X, sub_y)
      return

    for v_child, child in zip(v_children, self.children.values()):
      if min_error == v_child:
        self.predicate = None
        self.prunned_by = child
        return

  @staticmethod
  def major_method(x: np.array) -> Any:
    return stats.mode(x).mode

  def backward(
      self, X: np.array, y: np.array, betas: Iterable[Predicate], criterium
      ) -> None:
    betas = list(betas)
    # если все объекты из U лежат в одном классе c ∈ Y,
    # то вернуть новый лист v, cv := c
    if len(np.unique(y)) == 1:
      self.class_label = y[0]
      self.children_probas = {self.class_label: 1}
      self.major_class = y[0]
      return

    # если β(xi) не определено,
    # то при вычислении I(β,U) объект xi исключается из выборки U
    nan_mask = pd.isnull(y)
    X = X[~nan_mask]
    y = y[~nan_mask]
    self.major_class = self.major_method(y)

    calc_mask = pd.isnull(X).any(axis=1)
    X_calc = X[~calc_mask]
    y_calc = y[~calc_mask]

    # найти предикат с максимальной информативностью
    predicate = Predicate(
        max(betas, key=lambda beta: criterium(beta, X_calc, y_calc))
        )
    predicate_values = predicate.possible_values

    # разбить выборку на две части U = U0 ∪ U1 по предикату β
    for el, el_class in zip(X, y):
      # если предикат не вычислился,
      # то разбить на данном уровне по данному признаку невозможно
      try:
        predicate_el = predicate(el)
        self.child_sub_samples.setdefault(predicate_el, [])
        self.child_sub_samples[predicate_el].append((el, el_class))
      except Exception:
        ...

    total_length = sum(
        len(list(used_lines)) for used_lines in self.child_sub_samples.values()
        )

    # если U0 = ∅ или U1 = ∅,
    # то вернуть новый лист v, cv := Мажоритарный класс(U);
    for predicate_value in predicate_values:
      child_sub_sample = self.child_sub_samples.get(predicate_value)
      if not child_sub_sample:
        self.class_label = self.major_class
        self.children_probas = {self.class_label: 1}
        return

    # создать новую внутреннюю вершину v: βv := β
    self.predicate = predicate

    # построить левое поддерево: Lv := LearnID3(U0);
    # построить правое поддерево: Rv := LearnID3(U1)
    for predicate_value, child_sub_sample in self.child_sub_samples.items():
      child_node = self.__class__()
      child_X, child_y = zip(*child_sub_sample)
      child_node.backward(
          np.array(child_X),
          np.array(child_y),
          betas,
          criterium
      )
      self.children[predicate_value] = child_node
      self.children_probas[predicate_value] = len(child_X) / total_length

def Donskoj_criterium(beta: Callable, X: np.array, y: np.array) -> float:
  # I(β, X) = #{(xi, xj) : β(xi) = β(xj) и yi = yj}
  count = 0
  n = len(X)
  for i in range(n):
    for j in range(i + 1, n):
      x_i, y_i = X[i], y[i]
      x_j, y_j = X[j], y[j]
      if beta(x_i) != beta(x_j) and y_i != y_j:
        count += 1
  return count


def entropy(p: float) -> float:
  # -p log2(p)
  return -p * np.log2(p)

def multiclass_ent_criterium(beta: Callable, X: np.array, y: np.array) -> float:
  # I(β, Xl), Pc = #{xi: yi = c}, p = #{xi: β(xi) = 1}, h(z) = −z log2z
  unique_classes = np.unique(y)
  P_c = {c: np.sum(y == c) for c in unique_classes}
  p_c = {
      c: np.sum((y == c) & (np.array(beta(x) for x in X) == 1))
      for c in unique_classes
      }
  p = np.sum(np.array(beta(x) for x in X) == 1)
  l = len(X)

  res = 0
  for c in unique_classes:
    res += (
        entropy(P_c[c] / l)
        - (p / l) * entropy(P_c[c] / (p + 0.001))
        - ((l - p) / l) * entropy((P_c[c] - p_c[c]) / (l - p + 0.001))
        )
  return res


def uncertainty(y) -> float:
  return np.mean((np.mean(y) - y) ** 2)

def regression_criterium(beta: Callable, X: np.array, y: np.array) -> float:
  beta_X_values = np.apply_along_axis(beta, 1, X)
  f_u_b = 0
  for predicate_value in {0, 1}:
    if len(y_part := y[beta_X_values == predicate_value]):
      f_u_b += uncertainty(y) * (len(y_part) / len(y))
  return uncertainty(y) - f_u_b


class ID3RegressorNode(ID3ClassifierNode):
  def mse(self, X, y) -> float:
    return mean_squared_error(np.apply_along_axis(self.forward, 1, X), y)

  @staticmethod
  def major_method(x: np.array) -> float:
    return np.mean(x)

  def prune(self, X, y) -> None:
    # X := подмножество объектов Xk, дошедших до текущей вершины

    # для всех v ∈ Vвнутр
    if self.class_label or not self.predicate:
      return

    # если Sv = ∅ то
    if not len(X):
      self.class_label = self.major_class
      self.children_probas = {self.class_label: 1}

    # Оцениваем ошибку текущего узла, если бы он был листом
    error_leaf = mean_squared_error(np.array([self.major_class for _ in y]), y)

    # Рассчитываем ошибку для текущего дерева и каждого из детей
    v_error = self.mse(X, y)
    v_children = [child.mse(X, y) for child in self.children.values()]

    # Сравниваем ошибки и принимаем решение о прунинге
    if (min_error := min(error_leaf, v_error, *v_children)) == error_leaf:
      # Превратить узел в лист
      self.class_label = self.major_class
      self.children_probas = {self.class_label: 1}
      return

    if min_error == v_error:
      for child in self.children.values():
        for child_predicate_value, child in self.children.items():
          mask = (
              np.apply_along_axis(self.predicate, 1, X) == child_predicate_value
          )
          sub_X = X[mask]
          sub_y = y[mask]
          child.prune(sub_X, sub_y)
      return

    for v_child, child in zip(v_children, self.children.values()):
      if min_error == v_child:
        self.predicate = None
        self.prunned_by = child
        return
