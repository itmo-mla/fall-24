import numpy as np
from pydantic import BaseModel
from scipy.spatial.distance import cdist


class Simple(BaseModel):
    def __str__(self) -> str:
        return "Simple"


class ParzenFixed(BaseModel):
    h: float

    def __str__(self) -> str:
        return f"ParzenFixed(h={self.h})"


class ParzenAdaptive(BaseModel):
    def __str__(self) -> str:
        return "ParzenAdaptive"


class KNN:
    def __init__(
        self,
        k: int,
        num_classes: int,
        mode: Simple | ParzenFixed | ParzenAdaptive,
    ):
        """
        Initialize KNN classifier

        Args:
            k: number of nearest neighbors
            num_classes: number of classes
            mode: mode of prediction
        """
        self.k: int = k
        self.num_classes: int = num_classes
        self.mode: Simple | ParzenFixed | ParzenAdaptive = mode
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the classifier with training data

        Args:
            X: training samples, shape (N_samples, N_features)
            y: training labels, shape (N_samples,)
        """
        self.X = X
        self.y = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for input samples

        Args:
            X: input samples to predict, shape (N_samples, N_features)
        Returns:
            Predicted labels, shape (N_samples,)
        """
        if self.X is None or self.y is None:
            raise ValueError("Classifier must be fitted before making predictions")

        # Initialize array to store predictions for each input sample
        predictions = np.empty(len(X), dtype=np.int32)

        # Iterate through each input sample
        for i, x in enumerate(X):
            # Calculate distances between the input sample and all training samples
            d = cdist(self.X, x.reshape((1, -1)))

            # Get indices of k nearest neighbors
            topk_idxs = np.argsort(d, axis=0)[:self.k]

            # Get distances and labels of k nearest neighbors
            topk = d[topk_idxs].flatten()
            lbls = self.y[topk_idxs].flatten()

            # Initialize vote matrix (k neighbors x num_classes)
            v = np.zeros((self.k, self.num_classes))

            # Apply weighting based on the selected mode
            match self.mode:
                case Simple():
                    # Simple mode: equal weights for all neighbors
                    topk = np.ones((self.k,))
                case ParzenFixed(h=h):
                    # Fixed Parzen window: Gaussian kernel with fixed bandwidth
                    topk = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (topk / h) ** 2)
                case ParzenAdaptive():
                    # Adaptive Parzen window: Gaussian kernel with adaptive bandwidth
                    h = np.sort(d, axis=0)[self.k]  # Use distance to k-th neighbor as bandwidth
                    topk = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (topk / h) ** 2)

            # Assign weights to corresponding labels in the vote matrix
            v[np.arange(self.k), lbls] = topk

            # Predict the class with the highest total vote
            predictions[i] = np.argmax(np.sum(v, axis=0))

        return predictions

    def leave_one_out(self) -> float:
        """
        Calculate leave-one-out error
        """
        y_pred = np.empty(self.y.shape, dtype=np.int32)
        for i in range(len(self.X)):
            y_pred[i] = (
                KNN(k=self.k, mode=self.mode, num_classes=self.num_classes)
                .fit(np.delete(self.X, i, axis=0), np.delete(self.y, i))
                .predict(self.X[i].reshape((1, -1)))
            )
        return np.sum(y_pred != self.y) / len(self.y)
