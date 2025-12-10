"""
PLS-DA (Partial Least Squares Discriminant Analysis) implementation.
"""

from typing import Any, Dict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression


class PLSDA(BaseEstimator, ClassifierMixin):
    """
    PLS-DA classifier wrapper around sklearn's PLSRegression.

    Adapts PLSRegression for binary classification by treating
    the target as continuous and thresholding predictions.
    """

    def __init__(self, n_components: int = 2, threshold: float = 0.5):
        """
        Initialize PLS-DA classifier.

        Args:
            n_components: Number of PLS components
            threshold: Decision threshold for classification
        """
        self.n_components = n_components
        self.threshold = threshold
        self.pls_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSDA":
        """
        Fit the PLS-DA model.

        Args:
            X: Training features
            y: Training labels (binary: 0/1)

        Returns:
            Self
        """
        self.pls_ = PLSRegression(n_components=self.n_components)
        self.pls_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features to predict

        Returns:
            Predicted class labels (0 or 1)
        """
        y_prob = self.predict_proba(X)[:, 1]
        return (y_prob > self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features to predict

        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        if self.pls_ is None:
            raise RuntimeError("Model has not been fitted yet")

        y_raw = self.pls_.predict(X)[:, 0]

        # Clip to [0, 1] range
        y_prob = np.clip(y_raw, 0, 1)

        # Return probabilities for both classes
        return np.column_stack([1 - y_prob, y_prob])

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "n_components": self.n_components,
            "threshold": self.threshold
        }

    def set_params(self, **params) -> "PLSDA":
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
