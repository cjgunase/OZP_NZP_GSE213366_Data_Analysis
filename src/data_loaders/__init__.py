"""
Data loading modules for methylation analysis.

This module provides flexible data loaders that can handle:
- Multiple GEO datasets
- Different probe sets (CoRSIV, Controls, EPIC full)
- Various metadata formats
"""

from .base import DataLoader
from .methylation import MethylationDataLoader
from .metadata import MetadataLoader

__all__ = ["DataLoader", "MethylationDataLoader", "MetadataLoader"]
