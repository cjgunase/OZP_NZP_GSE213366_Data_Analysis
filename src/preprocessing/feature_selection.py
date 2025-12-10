"""
Feature selection methods for methylation data.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


class FeatureSelector(ABC):
    """Abstract base class for feature selection methods."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureSelector":
        """Fit the selector to the data."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to selected features."""
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    @abstractmethod
    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names."""
        pass


class CorrelationSelector(FeatureSelector):
    """
    Select features based on correlation between paired samples.

    Used for selecting probes with high blood-sperm correlation,
    ensuring biomarkers are transferable across tissues.
    """

    def __init__(
        self,
        threshold: float = 0.75,
        correlation_file: Optional[str] = None,
        min_samples: int = 3
    ):
        """
        Initialize correlation-based feature selector.

        Args:
            threshold: Minimum correlation coefficient for selection
            correlation_file: Path to pre-computed correlations (optional)
            min_samples: Minimum samples required for correlation calculation
        """
        self.threshold = threshold
        self.correlation_file = correlation_file
        self.min_samples = min_samples
        self.correlations_: Optional[pd.DataFrame] = None
        self.selected_features_: Optional[List[str]] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        paired_X: Optional[pd.DataFrame] = None
    ) -> "CorrelationSelector":
        """
        Fit selector by computing or loading correlations.

        Args:
            X: First tissue data (e.g., blood)
            y: Not used, present for API compatibility
            paired_X: Paired tissue data (e.g., sperm) - same individuals, same probes

        Returns:
            Self
        """
        if self.correlation_file is not None:
            self._load_correlations()
        elif paired_X is not None:
            self._compute_correlations(X, paired_X)
        else:
            raise ValueError(
                "Either correlation_file or paired_X must be provided"
            )

        # Select features above threshold
        self.selected_features_ = self.correlations_[
            self.correlations_["correlation"] > self.threshold
        ]["probe_id"].tolist()

        logger.info(
            f"Selected {len(self.selected_features_)} features "
            f"with correlation > {self.threshold}"
        )

        return self

    def fit_from_file(self, correlation_file: str, probe_col: str = "ID_REF", corr_col: str = "Pearson_Correlation") -> "CorrelationSelector":
        """
        Fit selector from pre-computed correlation file.

        Args:
            correlation_file: Path to CSV with correlation data
            probe_col: Column name for probe IDs
            corr_col: Column name for correlation values

        Returns:
            Self
        """
        logger.info(f"Loading correlations from {correlation_file}")
        corr_df = pd.read_csv(correlation_file)

        self.correlations_ = pd.DataFrame({
            "probe_id": corr_df[probe_col],
            "correlation": corr_df[corr_col]
        })

        self.selected_features_ = self.correlations_[
            self.correlations_["correlation"] > self.threshold
        ]["probe_id"].tolist()

        logger.info(
            f"Selected {len(self.selected_features_)} features "
            f"with correlation > {self.threshold}"
        )

        return self

    def _load_correlations(self):
        """Load pre-computed correlations from file."""
        logger.info(f"Loading correlations from {self.correlation_file}")
        self.correlations_ = pd.read_csv(self.correlation_file)

    def _compute_correlations(
        self,
        X1: pd.DataFrame,
        X2: pd.DataFrame
    ) -> None:
        """
        Compute probe-wise correlations between paired samples.

        Args:
            X1: First tissue data (samples x probes)
            X2: Second tissue data (samples x probes), must have same index as X1
        """
        logger.info("Computing probe-wise correlations...")

        # Ensure same probes in both datasets
        common_probes = list(set(X1.columns) & set(X2.columns))
        logger.info(f"Computing correlations for {len(common_probes)} common probes")

        results = []
        for probe in common_probes:
            vals1 = X1[probe].values
            vals2 = X2[probe].values

            # Remove NaN pairs
            valid = ~(np.isnan(vals1) | np.isnan(vals2))
            v1_valid = vals1[valid]
            v2_valid = vals2[valid]

            if len(v1_valid) >= self.min_samples:
                corr, pval = pearsonr(v1_valid, v2_valid)
            else:
                corr, pval = np.nan, np.nan

            results.append({
                "probe_id": probe,
                "correlation": corr,
                "p_value": pval,
                "n_pairs": len(v1_valid)
            })

        self.correlations_ = pd.DataFrame(results)
        logger.info(f"Computed correlations for {len(results)} probes")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select features from input data.

        Args:
            X: Data with probes as columns

        Returns:
            Data with only selected probe columns
        """
        if self.selected_features_ is None:
            raise RuntimeError("Selector has not been fitted yet")

        available = [f for f in self.selected_features_ if f in X.columns]
        if len(available) < len(self.selected_features_):
            logger.warning(
                f"Only {len(available)}/{len(self.selected_features_)} "
                "selected features available in input data"
            )

        return X[available]

    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names."""
        if self.selected_features_ is None:
            raise RuntimeError("Selector has not been fitted yet")
        return self.selected_features_.copy()

    def get_correlation_stats(self) -> pd.DataFrame:
        """Get correlation statistics for all probes."""
        if self.correlations_ is None:
            raise RuntimeError("Correlations have not been computed yet")
        return self.correlations_.copy()

    def save_correlations(self, output_path: str) -> None:
        """Save correlation results to CSV."""
        if self.correlations_ is None:
            raise RuntimeError("Correlations have not been computed yet")
        self.correlations_.to_csv(output_path, index=False)
        logger.info(f"Saved correlations to {output_path}")


class VarianceSelector(FeatureSelector):
    """Select features based on variance threshold."""

    def __init__(self, threshold: float = 0.01):
        """
        Initialize variance-based selector.

        Args:
            threshold: Minimum variance for feature selection
        """
        self.threshold = threshold
        self.variances_: Optional[pd.Series] = None
        self.selected_features_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "VarianceSelector":
        """Compute variances and select features."""
        self.variances_ = X.var()
        self.selected_features_ = self.variances_[
            self.variances_ > self.threshold
        ].index.tolist()

        logger.info(
            f"Selected {len(self.selected_features_)} features "
            f"with variance > {self.threshold}"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features from input data."""
        if self.selected_features_ is None:
            raise RuntimeError("Selector has not been fitted yet")

        available = [f for f in self.selected_features_ if f in X.columns]
        return X[available]

    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names."""
        if self.selected_features_ is None:
            raise RuntimeError("Selector has not been fitted yet")
        return self.selected_features_.copy()
