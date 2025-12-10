"""
Data transformation utilities for methylation analysis.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataTransformer:
    """
    Transform methylation data for model training.

    Handles scaling, train/test splitting by tissue, and data preparation.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize transformer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler_: Optional[StandardScaler] = None
        self.imputer_means_: Optional[pd.Series] = None

    def split_by_tissue(
        self,
        df: pd.DataFrame,
        train_tissue: str = "Blood",
        test_tissue: str = "Sperm",
        tissue_col: str = "tissue",
        target_col: str = "target",
        meta_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Split data by tissue type for internal validation.

        Args:
            df: Merged DataFrame with methylation data and metadata
            train_tissue: Tissue type for training
            test_tissue: Tissue type for testing
            tissue_col: Column name for tissue type
            target_col: Column name for target variable
            meta_cols: Metadata columns to exclude from features

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        if meta_cols is None:
            meta_cols = [
                "Individual_ID", "geo_accession", "sample_name",
                "tissue", "status", "state", "target"
            ]

        train_df = df[df[tissue_col] == train_tissue].copy()
        test_df = df[df[tissue_col] == test_tissue].copy()

        # Separate features and target
        feature_cols = [c for c in df.columns if c not in meta_cols]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        logger.info(
            f"Split data: {len(X_train)} {train_tissue} (train), "
            f"{len(X_test)} {test_tissue} (test)"
        )

        return X_train, y_train, X_test, y_test

    def fit_transform_scale(
        self,
        X_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit scaler on training data and transform both sets.

        Args:
            X_train: Training features
            X_test: Optional test features

        Returns:
            Tuple of (X_train_scaled, X_test_scaled) as numpy arrays
        """
        self.scaler_ = StandardScaler()
        X_train_scaled = self.scaler_.fit_transform(X_train)

        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler_.transform(X_test)

        return X_train_scaled, X_test_scaled

    def transform_scale(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted scaler.

        Args:
            X: Features to scale

        Returns:
            Scaled features as numpy array
        """
        if self.scaler_ is None:
            raise RuntimeError("Scaler has not been fitted yet")
        return self.scaler_.transform(X)

    def fit_impute(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit imputer on training data and transform.

        Args:
            X_train: Training features with potential missing values

        Returns:
            Imputed training data
        """
        self.imputer_means_ = X_train.mean()
        return X_train.fillna(self.imputer_means_)

    def transform_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using fitted means.

        Args:
            X: Features with potential missing values

        Returns:
            Imputed data
        """
        if self.imputer_means_ is None:
            raise RuntimeError("Imputer has not been fitted yet")
        return X.fillna(self.imputer_means_)

    def prepare_for_training(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        train_tissue: str = "Blood",
        test_tissue: str = "Sperm",
        tissue_col: str = "tissue",
        target_col: str = "target"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete preparation pipeline for model training.

        Performs: tissue split -> imputation -> scaling

        Args:
            df: Merged DataFrame
            feature_cols: List of feature column names
            train_tissue: Tissue for training
            test_tissue: Tissue for testing
            tissue_col: Tissue column name
            target_col: Target column name

        Returns:
            Tuple of (X_train_scaled, y_train, X_test_scaled, y_test)
        """
        # Split by tissue
        train_df = df[df[tissue_col] == train_tissue].copy()
        test_df = df[df[tissue_col] == test_tissue].copy()

        X_train = train_df[feature_cols]
        y_train = train_df[target_col].values

        X_test = test_df[feature_cols]
        y_test = test_df[target_col].values

        # Impute missing values
        X_train = self.fit_impute(X_train)
        X_test = self.transform_impute(X_test)

        # Scale
        X_train_scaled, X_test_scaled = self.fit_transform_scale(X_train, X_test)

        logger.info(
            f"Prepared data: {X_train_scaled.shape[0]} train, "
            f"{X_test_scaled.shape[0]} test samples "
            f"with {X_train_scaled.shape[1]} features"
        )

        return X_train_scaled, y_train, X_test_scaled, y_test

    def get_scaler(self) -> StandardScaler:
        """Get the fitted scaler."""
        if self.scaler_ is None:
            raise RuntimeError("Scaler has not been fitted yet")
        return self.scaler_

    def get_imputer_means(self) -> pd.Series:
        """Get the imputation means."""
        if self.imputer_means_ is None:
            raise RuntimeError("Imputer has not been fitted yet")
        return self.imputer_means_.copy()
