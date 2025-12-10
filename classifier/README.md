# DNA Methylation Classifier for Male Fertility Status

A machine learning pipeline for classifying male fertility status (Normozoospermia vs Oligozoospermia) using DNA methylation data from CoRSIV (Correlated Regions of Systemic Interindividual Variation) probes.

## Overview

This classifier leverages the unique property of CoRSIV probes - methylation markers that show consistent patterns across different tissues within an individual. By training on easily accessible **blood** samples, the pipeline can predict fertility status that is typically assessed through **sperm** analysis, enabling potential non-invasive fertility screening.

### Key Features

- Train on blood DNA methylation, validate on sperm samples
- Feature selection based on blood-sperm correlation (r > 0.75)
- Multiple classification algorithms for robust predictions
- External validation on independent cohorts
- Publication-ready figure generation (300 DPI PDFs)

## Datasets

| Dataset | GEO Accession | Samples | Tissue Types | Purpose |
|---------|---------------|---------|--------------|---------|
| Primary | GSE213366 | 70 (35 matched pairs) | Blood, Sperm | Training & Internal Validation |
| External 1 | GSE51245 | 39 | Blood | External Validation |
| External 2 | GSE149318 | 179 | Blood, Sperm | Blood-Sperm Correlation Analysis |

## Directory Structure

```
classifier/
├── pipeline.py              # Main analysis pipeline
├── config.py                # Configuration parameters
├── requirements.txt         # Python dependencies
├── README.md                # This documentation
│
├── data/                    # Input data
│   ├── GSE213366_CoRSIV_2023.csv              # Primary methylation data
│   ├── sample_tissue_individual_map.csv       # Sample metadata
│   ├── blood_sperm_correlation_results.csv    # Pre-computed correlations
│   └── external/
│       ├── GSE51245/
│       │   ├── GSE51245_CoRSIV_probes.csv
│       │   └── metadata.csv
│       └── GSE149318/
│           ├── GSE149318_CoRSIV_probes.csv
│           └── GSE149318_sample_tissue_type.csv
│
└── output/                  # Generated outputs
    ├── figures/             # Publication-ready PDFs
    │   ├── figure1_roc_curves.pdf
    │   ├── figure2_batch_effect_pca.pdf
    │   ├── figure3_pca_class_separation.pdf
    │   ├── figure4a_blood_sperm_correlation_GSE149318.pdf
    │   └── figure4b_blood_sperm_correlation_GSE213366.pdf
    └── results/
        ├── model_performance_summary.csv
        ├── external_validation_predictions.csv
        └── GSE213366_blood_sperm_probe_correlations.csv
```

## Installation

### Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

### Setup

```bash
# Navigate to classifier directory
cd classifier

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Full Pipeline

```bash
python pipeline.py
```

This executes all analysis steps:
1. Load and preprocess GSE213366 data
2. Select features with blood-sperm correlation > 0.75
3. Train classifiers (Logistic Regression, Naive Bayes, PLS-DA)
4. Evaluate on sperm samples (internal validation)
5. Validate on GSE51245 (external validation)
6. Generate publication-ready figures

### Run Individual Steps

```bash
# Train models and generate ROC curves
python pipeline.py --step train

# External validation only
python pipeline.py --step validate

# Batch effect PCA analysis
python pipeline.py --step batch

# Blood-sperm correlation histograms
python pipeline.py --step correlation

# Class separation PCA
python pipeline.py --step pca
```

## Pipeline Details

### 1. Data Loading

The pipeline loads:
- **Methylation data**: CoRSIV probes (rows) x Samples (columns)
- **Metadata**: Sample annotations with tissue type and fertility status

Data is transposed so samples become rows, then merged with metadata.

### 2. Feature Selection

Probes are selected based on Pearson correlation between blood and sperm methylation values:
- Correlation computed across matched blood-sperm pairs
- Default threshold: r > 0.75
- Ensures biomarkers maintain consistent methylation across tissues

### 3. Classification Models

| Model | Description | Hyperparameters |
|-------|-------------|-----------------|
| Logistic Regression | L2-regularized linear model | max_iter=1000 |
| Naive Bayes | Gaussian Naive Bayes | Default |
| PLS-DA | Partial Least Squares Discriminant | n_components=2 |

All models use:
- StandardScaler normalization
- 5-fold stratified cross-validation
- Random state = 42 for reproducibility

### 4. Validation Strategy

**Internal Validation**:
- Train: Blood samples from GSE213366
- Test: Matched sperm samples from same individuals

**External Validation**:
- Train: Blood samples from GSE213366
- Test: Blood samples from GSE51245 (independent cohort)

### 5. Evaluation Metrics

- Accuracy
- Cross-validation accuracy (5-fold)
- ROC-AUC
- Classification report (precision, recall, F1)

## Configuration

Edit `config.py` to modify:

```python
# Feature selection threshold
CORRELATION_THRESHOLD = 0.75

# Model parameters
RANDOM_STATE = 42
LOGISTIC_REGRESSION_MAX_ITER = 1000
PLS_N_COMPONENTS = 2
CV_FOLDS = 5

# Visualization settings
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"
```

### Color Scheme

```python
COLORS = {
    "normozoospermia": "#2ecc71",  # Green
    "oligozoospermia": "#e74c3c",  # Red
    "blood": "#3498db",            # Blue
    "sperm": "#9b59b6",            # Purple
}
```

## Output Files

### Figures (PDF, 300 DPI)

| File | Description |
|------|-------------|
| `figure1_roc_curves.pdf` | ROC curves with AUC for all models |
| `figure2_batch_effect_pca.pdf` | PCA showing dataset batch effects |
| `figure3_pca_class_separation.pdf` | PCA showing Normo/Oligo separation |
| `figure4a_blood_sperm_correlation_GSE149318.pdf` | Sample-wise correlations |
| `figure4b_blood_sperm_correlation_GSE213366.pdf` | Probe-wise correlations |

### Results (CSV)

| File | Description |
|------|-------------|
| `model_performance_summary.csv` | Accuracy, CV accuracy, AUC per model |
| `external_validation_predictions.csv` | Predictions on GSE51245 samples |
| `GSE213366_blood_sperm_probe_correlations.csv` | Probe correlation statistics |

## Data Format

### Input Methylation Data

CSV format with:
- First column: Probe IDs (cg-prefixed)
- Subsequent columns: Sample IDs
- Values: Beta values (0-1)

```csv
,203219670028_R01C01,203219670028_R02C01,...
cg19568003,0.85,0.82,...
cg14817997,0.12,0.15,...
```

### Input Metadata

CSV format with columns:
- `Individual_ID`: Subject identifier
- `sample_name`: Array sample ID
- `tissue`: Blood or Sperm
- `status`: Normozoospermia or Oligozoospermia

## Troubleshooting

### Common Issues

**"No samples matched between probe file and metadata"**
- Check that sample IDs match between methylation matrix columns and metadata
- Some files may have "X" prefix on numeric sample names

**"Not enough common probes for PCA"**
- Verify probe IDs are consistent across datasets
- Check for probe naming convention differences

**Low correlation values**
- This is expected - CoRSIV probes are selected precisely because they show high correlation
- Verify you're using the correct probe subset

## Methods Reference

### Blood-Sperm Correlation

For each probe, Pearson correlation is computed:
```
r = corr(blood_values, sperm_values)
```
Where values are from matched blood-sperm pairs within individuals.

### PLS-DA Implementation

PLS-DA uses PLSRegression from scikit-learn:
- Binary encoding: Normozoospermia=0, Oligozoospermia=1
- Prediction threshold: 0.5
- Components retained: 2

## Citation

If you use this pipeline, please cite:

- Primary dataset: GEO Accession GSE213366
- External validation: GSE51245
- Correlation reference: GSE149318

## License

This project is for academic research purposes.

## Contact

Waterland Lab
Baylor College of Medicine
