"""
Metadata loader for sample annotations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .base import DataLoader

logger = logging.getLogger(__name__)


class MetadataLoader(DataLoader):
    """
    Load and process sample metadata from various formats.

    Handles:
    - Different column naming conventions
    - Tissue/status filtering
    - Matched pair identification
    """

    def load(
        self,
        file_path: Union[str, Path],
        column_mapping: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load metadata from CSV file.

        Args:
            file_path: Path to metadata CSV
            column_mapping: Optional mapping to standardize column names
                           e.g., {"sample_id": "sample_name", "tissue_type": "tissue"}

        Returns:
            DataFrame with standardized metadata
        """
        path = self._resolve_path(file_path)
        self._validate_file(path)

        logger.info(f"Loading metadata from {path.name}...")
        df = pd.read_csv(path)

        # Apply column mapping if provided
        if column_mapping:
            reverse_mapping = {v: k for k, v in column_mapping.items()}
            df = df.rename(columns=reverse_mapping)

        logger.info(f"Loaded metadata for {len(df)} samples")
        return df

    def filter_by_tissue(
        self,
        df: pd.DataFrame,
        tissue: Union[str, List[str]],
        tissue_col: str = "tissue"
    ) -> pd.DataFrame:
        """
        Filter metadata by tissue type.

        Args:
            df: Metadata DataFrame
            tissue: Tissue type(s) to keep
            tissue_col: Name of tissue column

        Returns:
            Filtered DataFrame
        """
        if isinstance(tissue, str):
            tissue = [tissue]

        # Handle case-insensitive matching
        df_tissue_lower = df[tissue_col].str.lower().str.strip()
        tissue_lower = [t.lower().strip() for t in tissue]

        mask = df_tissue_lower.isin(tissue_lower)
        filtered = df[mask].copy()

        logger.info(f"Filtered to {len(filtered)} samples for tissue(s): {tissue}")
        return filtered

    def filter_by_status(
        self,
        df: pd.DataFrame,
        status: Union[str, List[str]],
        status_col: str = "status"
    ) -> pd.DataFrame:
        """
        Filter metadata by fertility status.

        Args:
            df: Metadata DataFrame
            status: Status value(s) to keep
            status_col: Name of status column

        Returns:
            Filtered DataFrame
        """
        if isinstance(status, str):
            status = [status]

        mask = df[status_col].isin(status)
        filtered = df[mask].copy()

        logger.info(f"Filtered to {len(filtered)} samples for status: {status}")
        return filtered

    def get_matched_pairs(
        self,
        df: pd.DataFrame,
        individual_col: str = "Individual_ID",
        tissue_col: str = "tissue",
        tissues: tuple = ("Blood", "Sperm")
    ) -> pd.DataFrame:
        """
        Identify matched sample pairs (e.g., blood-sperm from same individual).

        Args:
            df: Metadata DataFrame
            individual_col: Column identifying individuals
            tissue_col: Column identifying tissue type
            tissues: Tuple of (tissue1, tissue2) to match

        Returns:
            DataFrame with matched pairs information
        """
        tissue1, tissue2 = tissues

        # Get samples for each tissue
        t1_samples = df[df[tissue_col] == tissue1][[individual_col, "sample_name"]]
        t2_samples = df[df[tissue_col] == tissue2][[individual_col, "sample_name"]]

        # Merge to find matches
        matched = t1_samples.merge(
            t2_samples,
            on=individual_col,
            suffixes=(f"_{tissue1.lower()}", f"_{tissue2.lower()}")
        )

        logger.info(f"Found {len(matched)} matched {tissue1}-{tissue2} pairs")
        return matched

    def encode_labels(
        self,
        df: pd.DataFrame,
        column: str,
        mapping: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Encode categorical labels to numeric.

        Args:
            df: DataFrame with categorical column
            column: Column name to encode
            mapping: Mapping from categories to integers

        Returns:
            DataFrame with new 'target' column
        """
        df = df.copy()
        df["target"] = df[column].map(mapping)

        unmapped = df["target"].isna().sum()
        if unmapped > 0:
            logger.warning(f"{unmapped} samples had unmapped {column} values")

        return df

    def get_sample_list(
        self,
        df: pd.DataFrame,
        sample_col: str = "sample_name"
    ) -> List[str]:
        """
        Get list of sample IDs.

        Args:
            df: Metadata DataFrame
            sample_col: Column containing sample IDs

        Returns:
            List of sample IDs
        """
        return df[sample_col].tolist()
