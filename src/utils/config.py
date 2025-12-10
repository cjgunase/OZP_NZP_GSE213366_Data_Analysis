"""
Configuration management for the methylation analysis pipeline.

Supports loading configurations from YAML files for:
- Dataset specifications
- Probe set definitions
- Model parameters
- Visualization settings
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """
    Central configuration class for the methylation analysis pipeline.

    Supports hierarchical configuration with dataset and probe set specifications
    that can be easily extended for new data sources.

    Attributes:
        base_dir: Root directory of the project
        data_dir: Directory containing input data
        output_dir: Directory for results and figures
        dataset: Current dataset configuration
        probe_set: Current probe set configuration
        model_params: Model hyperparameters
        viz_params: Visualization settings
    """

    def __init__(
        self,
        config_file: Optional[str] = None,
        dataset: str = "GSE213366",
        probe_set: str = "CoRSIV"
    ):
        """
        Initialize configuration.

        Args:
            config_file: Path to YAML config file. If None, uses defaults.
            dataset: Name of dataset to use (e.g., "GSE213366", "GSE51245")
            probe_set: Name of probe set (e.g., "CoRSIV", "Controls", "EPIC_full")
        """
        self.base_dir = Path(__file__).parent.parent.parent
        self.dataset_name = dataset
        self.probe_set_name = probe_set

        # Load base configuration
        self._init_defaults()

        # Override with config file if provided
        if config_file:
            self._load_yaml(config_file)

        # Load dataset-specific config
        self._load_dataset_config(dataset)

        # Load probe set config
        self._load_probe_set_config(probe_set)

    def _init_defaults(self):
        """Initialize default configuration values."""
        # Directories
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "results"
        self.figures_dir = self.output_dir / "plots"
        self.tables_dir = self.output_dir / "tables"

        # Model parameters
        self.model_params = {
            "random_state": 42,
            "cv_folds": 5,
            "correlation_threshold": 0.75,
            "logistic_regression": {
                "max_iter": 1000,
                "solver": "lbfgs"
            },
            "pls": {
                "n_components": 2
            }
        }

        # Visualization parameters
        self.viz_params = {
            "dpi": 300,
            "format": "pdf",
            "figure_sizes": {
                "single": (6, 5),
                "wide": (8, 6),
                "tall": (6, 8)
            },
            "font_sizes": {
                "title": 12,
                "label": 11,
                "tick": 10,
                "legend": 9
            },
            "colors": {
                "normozoospermia": "#2ecc71",
                "oligozoospermia": "#e74c3c",
                "blood": "#3498db",
                "sperm": "#9b59b6",
                "external": "#e67e22"
            }
        }

        # Class label mappings
        self.class_mapping = {
            "status": {
                "Normozoospermia": 0,
                "Oligozoospermia": 1
            },
            "external_status": {
                "fertile": 0,
                "infertile": 1
            }
        }

        # Column name mappings (for handling different datasets)
        self.column_mapping = {
            "sample_id": "sample_name",
            "tissue": "tissue",
            "status": "status",
            "individual": "Individual_ID"
        }

    def _load_yaml(self, config_file: str):
        """Load configuration from YAML file."""
        config_path = Path(config_file)
        if not config_path.is_absolute():
            config_path = self.base_dir / config_file

        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            self._update_from_dict(config_data)

    def _load_dataset_config(self, dataset: str):
        """Load dataset-specific configuration."""
        dataset_config_path = self.base_dir / "configs" / "datasets" / f"{dataset}.yaml"

        if dataset_config_path.exists():
            with open(dataset_config_path, 'r') as f:
                self.dataset = yaml.safe_load(f)
        else:
            # Use defaults for known datasets
            self.dataset = self._get_default_dataset_config(dataset)

    def _load_probe_set_config(self, probe_set: str):
        """Load probe set configuration."""
        probe_config_path = self.base_dir / "configs" / "probe_sets" / f"{probe_set}.yaml"

        if probe_config_path.exists():
            with open(probe_config_path, 'r') as f:
                self.probe_set = yaml.safe_load(f)
        else:
            # Use defaults for known probe sets
            self.probe_set = self._get_default_probe_set_config(probe_set)

    def _get_default_dataset_config(self, dataset: str) -> Dict[str, Any]:
        """Get default configuration for known datasets."""
        datasets = {
            "GSE213366": {
                "name": "GSE213366",
                "description": "Primary dataset with matched blood-sperm pairs",
                "geo_accession": "GSE213366",
                "n_samples": 70,
                "tissues": ["Blood", "Sperm"],
                "has_matched_pairs": True,
                "files": {
                    "methylation": "data/raw/GSE213366_matrix_processed.csv",
                    "metadata": "data/metadata/sample_tissue_individual_map.csv"
                },
                "metadata_columns": {
                    "sample_id": "sample_name",
                    "tissue": "tissue",
                    "status": "status",
                    "individual": "Individual_ID"
                }
            },
            "GSE51245": {
                "name": "GSE51245",
                "description": "External validation blood cohort",
                "geo_accession": "GSE51245",
                "n_samples": 39,
                "tissues": ["Blood"],
                "has_matched_pairs": False,
                "files": {
                    "methylation": "classifier/data/external/GSE51245/GSE51245_CoRSIV_probes.csv",
                    "metadata": "classifier/data/external/GSE51245/metadata.csv"
                },
                "metadata_columns": {
                    "sample_id": "sample_id",
                    "status": "status"
                }
            },
            "GSE149318": {
                "name": "GSE149318",
                "description": "Blood-sperm correlation reference dataset",
                "geo_accession": "GSE149318",
                "n_samples": 179,
                "tissues": ["Blood", "Sperm"],
                "has_matched_pairs": True,
                "files": {
                    "methylation": "classifier/data/external/GSE149318/GSE149318_CoRSIV_probes.csv",
                    "metadata": "classifier/data/external/GSE149318/GSE149318_sample_tissue_type.csv"
                },
                "metadata_columns": {
                    "sample_id": "sample_name",
                    "tissue": "tissue_type",
                    "match_id": "match"
                }
            }
        }
        return datasets.get(dataset, {"name": dataset, "files": {}})

    def _get_default_probe_set_config(self, probe_set: str) -> Dict[str, Any]:
        """Get default configuration for known probe sets."""
        probe_sets = {
            "CoRSIV": {
                "name": "CoRSIV",
                "description": "Correlated Regions of Systemic Interindividual Variation",
                "file": "data/raw/GSE213366_CoRSIV_2023.csv",
                "n_probes": "~9000",
                "selection_criteria": "Systemic variation probes with cross-tissue correlation"
            },
            "Controls": {
                "name": "Controls",
                "description": "Control probes from EPIC array",
                "file": "data/raw/GSE213366_Controls.csv",
                "n_probes": "~850",
                "selection_criteria": "Array control probes"
            },
            "EPIC_full": {
                "name": "EPIC_full",
                "description": "Full Illumina EPIC array probe set",
                "file": "data/raw/GSE213366_matrix_processed.csv",
                "n_probes": "~850000",
                "selection_criteria": "All EPIC array probes"
            }
        }
        return probe_sets.get(probe_set, {"name": probe_set, "file": None})

    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        if "model_params" in config_dict:
            self.model_params.update(config_dict["model_params"])
        if "viz_params" in config_dict:
            self.viz_params.update(config_dict["viz_params"])
        if "class_mapping" in config_dict:
            self.class_mapping.update(config_dict["class_mapping"])

    def get_data_path(self, file_key: str) -> Path:
        """Get full path for a data file."""
        if file_key in self.dataset.get("files", {}):
            return self.base_dir / self.dataset["files"][file_key]
        return self.data_dir / file_key

    def get_probe_set_path(self) -> Path:
        """Get path to current probe set file."""
        return self.base_dir / self.probe_set.get("file", "")

    def get_output_path(self, filename: str, subdir: str = "plots") -> Path:
        """Get output path for results."""
        if subdir == "plots":
            return self.figures_dir / filename
        elif subdir == "tables":
            return self.tables_dir / filename
        return self.output_dir / subdir / filename

    def ensure_output_dirs(self):
        """Create output directories if they don't exist."""
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"Config(dataset='{self.dataset_name}', "
            f"probe_set='{self.probe_set_name}', "
            f"base_dir='{self.base_dir}')"
        )


def load_config(
    config_file: Optional[str] = None,
    dataset: str = "GSE213366",
    probe_set: str = "CoRSIV"
) -> Config:
    """
    Load configuration for the analysis pipeline.

    Args:
        config_file: Path to custom YAML configuration file
        dataset: Dataset name to use
        probe_set: Probe set name to use

    Returns:
        Config object with all settings loaded

    Example:
        >>> config = load_config(dataset="GSE213366", probe_set="CoRSIV")
        >>> config.get_data_path("methylation")
        PosixPath('/path/to/data/raw/GSE213366_CoRSIV_2023.csv')
    """
    return Config(config_file=config_file, dataset=dataset, probe_set=probe_set)
