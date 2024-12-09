from abc import ABC, abstractmethod

class TreeNode(ABC):
    
    @abstractmethod
    def __init__(self):
        self.n_features = 0
        self.available_feature_idxs = list(range(self.n_features))

        self.feature_idx = None
        self.beta = None
        self.information_gain = 0
        self.entropy = -1


        self.left_prob = 1
        self.left = None

        self.right = None

    @abstractmethod
    def set_value(self, y):
        pass

    @abstractmethod
    def fit(self, X, y, criterion):
        pass

    @abstractmethod
    def predict(self, X_sample):
        pass
