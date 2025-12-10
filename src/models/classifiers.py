"""
Classification models and evaluation utilities.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB

from .plsda import PLSDA

logger = logging.getLogger(__name__)


class ClassifierFactory:
    """
    Factory for creating classification models.

    Provides consistent interface for multiple classifier types
    with configurable hyperparameters.
    """

    AVAILABLE_MODELS = ["logistic_regression", "naive_bayes", "pls_da"]

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize factory with configuration.

        Args:
            config: Configuration object with model parameters
        """
        self.config = config
        self._default_params = {
            "random_state": 42,
            "logistic_regression": {
                "max_iter": 1000,
                "solver": "lbfgs"
            },
            "pls_da": {
                "n_components": 2
            }
        }

        if config is not None and hasattr(config, "model_params"):
            self._default_params.update(config.model_params)

    def create(self, model_type: str, **kwargs) -> Any:
        """
        Create a classifier instance.

        Args:
            model_type: Type of model to create
            **kwargs: Additional parameters to override defaults

        Returns:
            Configured classifier instance
        """
        random_state = kwargs.pop("random_state", self._default_params["random_state"])

        if model_type == "logistic_regression":
            params = self._default_params.get("logistic_regression", {}).copy()
            params.update(kwargs)
            return LogisticRegression(random_state=random_state, **params)

        elif model_type == "naive_bayes":
            return GaussianNB(**kwargs)

        elif model_type == "pls_da":
            params = self._default_params.get("pls_da", {}).copy()
            params.update(kwargs)
            return PLSDA(**params)

        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {self.AVAILABLE_MODELS}"
            )

    def create_all(self) -> Dict[str, Any]:
        """
        Create all available classifiers.

        Returns:
            Dictionary mapping model names to classifier instances
        """
        return {
            "Logistic Regression": self.create("logistic_regression"),
            "Naive Bayes": self.create("naive_bayes"),
            "PLS-DA": self.create("pls_da"),
        }


class ClassifierEvaluator:
    """
    Evaluate classification models with various metrics.
    """

    def __init__(
        self,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize evaluator.

        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state

    def evaluate(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate a single model.

        Args:
            model: Classifier to evaluate
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred.astype(float)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = np.nan
            logger.warning("Could not compute AUC (possibly single class in test)")

        # Cross-validation accuracy
        cv_accuracy = self._cross_validate(model, X_train, y_train)

        return {
            "model": model,
            "accuracy": accuracy,
            "cv_accuracy": cv_accuracy,
            "auc": auc,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    def _cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Perform cross-validation.

        Args:
            model: Classifier to evaluate
            X: Features
            y: Labels

        Returns:
            Mean CV accuracy
        """
        # PLS-DA doesn't support sklearn's cross_val_score directly
        if isinstance(model, PLSDA):
            return self._manual_cv(model, X, y)

        kf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

        try:
            scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
            return scores.mean()
        except Exception as e:
            logger.warning(f"CV failed: {e}")
            return np.nan

    def _manual_cv(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Manual cross-validation for models that don't support sklearn CV.

        Args:
            model: Classifier to evaluate
            X: Features
            y: Labels

        Returns:
            Mean CV accuracy
        """
        kf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

        scores = []
        for train_idx, val_idx in kf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Clone model for each fold
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_tr, y_tr)
            y_pred = model_clone.predict(X_val)
            scores.append(accuracy_score(y_val, y_pred))

        return np.mean(scores)

    def evaluate_all(
        self,
        models: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple models.

        Args:
            models: Dictionary of model name -> classifier
            X_train, y_train, X_test, y_test: Train/test data

        Returns:
            Dictionary of model name -> evaluation results
        """
        results = {}
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            results[name] = self.evaluate(model, X_train, y_train, X_test, y_test)
            logger.info(
                f"  Accuracy: {results[name]['accuracy']:.4f}, "
                f"AUC: {results[name]['auc']:.4f}"
            )
        return results

    def get_roc_data(
        self,
        results: Dict[str, Dict[str, Any]],
        y_test: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
        """
        Extract ROC curve data from evaluation results.

        Args:
            results: Evaluation results dictionary
            y_test: True test labels

        Returns:
            Dictionary of model name -> (fpr, tpr, auc)
        """
        roc_data = {}
        for name, metrics in results.items():
            if not np.isnan(metrics["auc"]):
                fpr, tpr, _ = roc_curve(y_test, metrics["y_prob"])
                roc_data[name] = (fpr, tpr, metrics["auc"])
        return roc_data

    def to_dataframe(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert results to DataFrame for export.

        Args:
            results: Evaluation results dictionary

        Returns:
            DataFrame with performance metrics
        """
        rows = []
        for name, metrics in results.items():
            rows.append({
                "Model": name,
                "Test_Accuracy": metrics["accuracy"],
                "CV_Accuracy": metrics["cv_accuracy"],
                "AUC": metrics["auc"],
            })
        return pd.DataFrame(rows)
