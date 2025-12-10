"""
Methylation data loader for various GEO datasets and probe sets.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from .base import DataLoader

logger = logging.getLogger(__name__)


class MethylationDataLoader(DataLoader):
    """
    Load and preprocess methylation data from various sources.

    Handles:
    - Different file formats (probes as rows vs columns)
    - Sample name normalization (X prefix handling)
    - Probe filtering by probe set
    - Missing value handling
    """

    def load(
        self,
        file_path: Union[str, Path],
        transpose: bool = True,
        sample_filter: Optional[List[str]] = None,
        probe_filter: Optional[List[str]] = None,
        handle_x_prefix: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load methylation data from CSV file.

        Args:
            file_path: Path to methylation matrix CSV
            transpose: If True, transpose so samples are rows (default behavior)
            sample_filter: Optional list of sample IDs to keep
            probe_filter: Optional list of probe IDs to keep
            handle_x_prefix: Handle X prefix in column names

        Returns:
            DataFrame with samples as rows and probes as columns
        """
        path = self._resolve_path(file_path)
        self._validate_file(path)

        cache_key = str(path)
        if cache_key in self._cache:
            logger.debug(f"Loading from cache: {path.name}")
            df = self._cache[cache_key].copy()
        else:
            logger.info(f"Loading methylation data from {path.name}...")
            df = pd.read_csv(path, index_col=0)
            self._cache[cache_key] = df.copy()

        # Handle X prefix in column names (R adds X to numeric column names)
        if handle_x_prefix:
            df = self._normalize_column_names(df)

        # Filter probes (rows before transpose)
        if probe_filter is not None:
            available_probes = [p for p in probe_filter if p in df.index]
            df = df.loc[available_probes]
            logger.info(f"Filtered to {len(available_probes)} probes")

        # Transpose if needed (standard format: samples as rows)
        if transpose:
            df = df.T
            df.index.name = "sample_id"

        # Filter samples
        if sample_filter is not None:
            available_samples = [s for s in sample_filter if s in df.index]
            df = df.loc[available_samples]
            logger.info(f"Filtered to {len(available_samples)} samples")

        logger.info(f"Loaded {df.shape[0]} samples x {df.shape[1]} probes")
        return df

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove X prefix from column names if present."""
        new_cols = []
        for col in df.columns:
            if col.startswith('X') and col[1:].replace('_', '').replace('R', '').replace('C', '').isdigit():
                new_cols.append(col[1:])
            else:
                new_cols.append(col)
        df.columns = new_cols
        return df

    def load_with_metadata(
        self,
        meth_path: Union[str, Path],
        meta_path: Union[str, Path],
        sample_col: str = "sample_name",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load methylation data and merge with metadata.

        Args:
            meth_path: Path to methylation matrix
            meta_path: Path to metadata CSV
            sample_col: Column name in metadata for sample IDs

        Returns:
            Merged DataFrame with metadata columns and probe values
        """
        # Load methylation data
        meth_df = self.load(meth_path, **kwargs)

        # Load metadata
        meta_df = pd.read_csv(self._resolve_path(meta_path))

        # Merge
        merged = meta_df.merge(
            meth_df,
            left_on=sample_col,
            right_index=True,
            how="inner"
        )

        logger.info(f"Merged data: {merged.shape[0]} samples with metadata")
        return merged

    def get_probe_values(
        self,
        df: pd.DataFrame,
        probe_ids: List[str],
        meta_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract values for specific probes.

        Args:
            df: DataFrame with merged methylation and metadata
            probe_ids: List of probe IDs to extract
            meta_cols: Metadata columns to exclude from probe search

        Returns:
            DataFrame with only the specified probe columns
        """
        if meta_cols is None:
            meta_cols = ["Individual_ID", "geo_accession", "sample_name",
                        "tissue", "status", "state", "target"]

        probe_cols = [p for p in probe_ids if p in df.columns]
        non_probe_cols = [c for c in df.columns if c in meta_cols]

        return df[non_probe_cols + probe_cols]

    def compute_m_values(self, beta_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert beta values to M-values (log2 ratio).

        M = log2(beta / (1 - beta))

        Args:
            beta_df: DataFrame with beta values (0-1)

        Returns:
            DataFrame with M-values
        """
        # Clip to avoid log(0) or division by zero
        beta_clipped = np.clip(beta_df.values, 0.001, 0.999)
        m_values = np.log2(beta_clipped / (1 - beta_clipped))
        return pd.DataFrame(m_values, index=beta_df.index, columns=beta_df.columns)

    def impute_missing(
        self,
        df: pd.DataFrame,
        method: str = "mean",
        reference_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Impute missing values in methylation data.

        Args:
            df: DataFrame with potential missing values
            method: Imputation method ("mean", "median", "zero")
            reference_df: Optional reference DataFrame for computing statistics
                         (use training data means for test data imputation)

        Returns:
            DataFrame with missing values imputed
        """
        if reference_df is None:
            reference_df = df

        if method == "mean":
            fill_values = reference_df.mean()
        elif method == "median":
            fill_values = reference_df.median()
        elif method == "zero":
            fill_values = 0
        else:
            raise ValueError(f"Unknown imputation method: {method}")

        return df.fillna(fill_values)
