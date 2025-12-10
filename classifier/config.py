"""
Configuration file for the DNA Methylation Classifier Pipeline.

This pipeline classifies male fertility status (Normozoospermia vs Oligozoospermia)
using DNA methylation data from CoRSIV probes.
"""

from pathlib import Path

# =============================================================================
# Directory Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / "data"
EXTERNAL_DATA_GSE51245_DIR = BASE_DIR / "data" / "external" / "GSE51245"
EXTERNAL_DATA_GSE149318_DIR = BASE_DIR / "data" / "external" / "GSE149318"

# Output directories
OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# =============================================================================
# Input Data Files
# =============================================================================
# Primary dataset (GSE213366)
METHYLATION_FILE = DATA_DIR / "GSE213366_CoRSIV_2023.csv"
METADATA_FILE = DATA_DIR / "sample_tissue_individual_map.csv"
CORRELATION_FILE = DATA_DIR / "blood_sperm_correlation_results.csv"

# External validation dataset 1 (GSE51245)
GSE51245_PROBES_FILE = EXTERNAL_DATA_GSE51245_DIR / "GSE51245_CoRSIV_probes.csv"
GSE51245_METADATA_FILE = EXTERNAL_DATA_GSE51245_DIR / "metadata.csv"

# External dataset 2 (GSE149318) - for blood-sperm correlation analysis
GSE149318_PROBES_FILE = EXTERNAL_DATA_GSE149318_DIR / "GSE149318_CoRSIV_probes.csv"
GSE149318_METADATA_FILE = EXTERNAL_DATA_GSE149318_DIR / "GSE149318_sample_tissue_type.csv"

# =============================================================================
# Output Files
# =============================================================================
# Results
PREDICTIONS_OUTPUT = RESULTS_DIR / "external_validation_predictions.csv"
MODEL_PERFORMANCE_OUTPUT = RESULTS_DIR / "model_performance_summary.csv"

# Figures (PDF for publication)
ROC_CURVE_OUTPUT = FIGURES_DIR / "figure1_roc_curves.pdf"
PCA_BATCH_OUTPUT = FIGURES_DIR / "figure2_batch_effect_pca.pdf"
PCA_INVESTIGATION_OUTPUT = FIGURES_DIR / "figure3_pca_class_separation.pdf"
CORRELATION_HISTOGRAM_GSE149318_OUTPUT = FIGURES_DIR / "figure4a_blood_sperm_correlation_GSE149318.pdf"
CORRELATION_HISTOGRAM_GSE213366_OUTPUT = FIGURES_DIR / "figure4b_blood_sperm_correlation_GSE213366.pdf"

# GSE213366 correlation results output
GSE213366_CORRELATION_OUTPUT = RESULTS_DIR / "GSE213366_blood_sperm_probe_correlations.csv"

# =============================================================================
# Model Parameters
# =============================================================================
# Feature selection
CORRELATION_THRESHOLD = 0.75  # Minimum blood-sperm correlation for feature selection

# Model training
RANDOM_STATE = 42
LOGISTIC_REGRESSION_MAX_ITER = 1000
PLS_N_COMPONENTS = 2
CV_FOLDS = 5

# =============================================================================
# Plotting Parameters (Publication-ready)
# =============================================================================
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"

# Figure sizes (width, height in inches)
FIGURE_SIZE_SINGLE = (6, 5)
FIGURE_SIZE_WIDE = (8, 6)
FIGURE_SIZE_TALL = (6, 8)

# Font sizes
FONT_SIZE_TITLE = 12
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 9

# Colors for consistent plotting
COLORS = {
    "normozoospermia": "#2ecc71",  # Green
    "oligozoospermia": "#e74c3c",  # Red
    "blood": "#3498db",            # Blue
    "sperm": "#9b59b6",            # Purple
    "gse213366_blood": "#3498db",
    "gse213366_sperm": "#9b59b6",
    "gse51245_blood": "#e67e22",   # Orange
}

# =============================================================================
# Class Labels
# =============================================================================
STATE_MAPPING = {"Normozoospermia": 0, "Oligozoospermia": 1}
STATE_LABELS = {0: "Normozoospermia", 1: "Oligozoospermia"}
EXTERNAL_STATUS_MAPPING = {"fertile": 0, "infertile": 1}
