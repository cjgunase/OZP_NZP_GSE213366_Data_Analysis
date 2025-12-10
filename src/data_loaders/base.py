"""
Base data loader class providing common functionality.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """
    Abstract base class for data loaders.

    Provides common interface for loading different types of data
    (methylation matrices, metadata, probe annotations).
    """

    def __init__(self, config: Any):
        """
        Initialize data loader with configuration.

        Args:
            config: Configuration object with paths and parameters
        """
        self.config = config
        self._cache: Dict[str, pd.DataFrame] = {}

    @abstractmethod
    def load(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from file.

        Args:
            file_path: Path to data file
            **kwargs: Additional loading parameters

        Returns:
            Loaded data as DataFrame
        """
        pass

    def _resolve_path(self, file_path: Union[str, Path]) -> Path:
        """Resolve file path relative to base directory if not absolute."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.config.base_dir / path
        return path

    def _validate_file(self, file_path: Path) -> None:
        """Validate that file exists and is readable."""
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
        logger.debug("Data cache cleared")
