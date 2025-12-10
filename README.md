# DNA Methylation Analysis for Male Fertility Classification (GSE213366)

A comprehensive, extensible analysis pipeline for studying DNA methylation patterns associated with male fertility status (Normozoospermia vs Oligozoospermia) using Illumina EPIC array data.

## Overview

This project provides two complementary analysis approaches:

1. **Differential Methylation Region (DMR) Analysis** - Statistical identification of differentially methylated regions between fertility groups
2. **Machine Learning Classification** - Predictive models for fertility status using CoRSIV (Correlated Regions of Systemic Interindividual Variation) probes

The key innovation is the ability to train classifiers on non-invasive **blood** samples that can predict **sperm**-related fertility status, enabling potential clinical applications for fertility assessment.

## Features

- **Modular Architecture**: Easily extensible for new datasets and probe sets
- **Multiple Probe Sets**: Support for CoRSIV, Control, and full EPIC array probes
- **Multiple Datasets**: Pre-configured for GSE213366, GSE51245, and GSE149318
- **YAML Configuration**: Simple configuration files for adding new data sources
- **Jupyter Notebook Tutorial**: Interactive step-by-step analysis guide
- **Publication-Ready Figures**: 300 DPI PDF outputs

## Project Structure

```
OZP_NZP_GSE213366_Data_Analysis/
├── main.py                     # Main orchestration script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore configuration
├── README.md                   # This file
│
├── src/                        # Core Python modules
│   ├── data_loaders/          # Data loading utilities
│   │   ├── methylation.py     # Methylation matrix loader
│   │   └── metadata.py        # Sample metadata loader
│   ├── preprocessing/         # Data preprocessing
│   │   ├── feature_selection.py  # Correlation-based selection
│   │   └── transformers.py    # Scaling and imputation
│   ├── models/                # ML models
│   │   ├── classifiers.py     # Model factory and evaluator
│   │   └── plsda.py          # PLS-DA implementation
│   ├── visualization/         # Plotting utilities
│   │   ├── plots.py          # Plot generators
│   │   └── style.py          # Publication style settings
│   └── utils/                 # Configuration and logging
│       ├── config.py         # Configuration management
│       └── logging_utils.py  # Logging setup
│
├── configs/                    # Configuration files
│   ├── datasets/              # Dataset specifications
│   │   ├── GSE213366.yaml    # Primary dataset
│   │   ├── GSE51245.yaml     # External validation
│   │   └── GSE149318.yaml    # Correlation reference
│   └── probe_sets/            # Probe set definitions
│       ├── CoRSIV.yaml       # CoRSIV probes
│       ├── Controls.yaml     # Control probes
│       └── EPIC_full.yaml    # Full EPIC array
│
├── classifier/                 # ML classification pipeline
│   ├── pipeline.py            # Unified classifier pipeline
│   ├── config.py              # Classifier-specific config
│   └── README.md              # Classifier documentation
│
├── scripts/                    # Analysis scripts
│   ├── R/
│   │   └── dmr_analysis.R     # DMRcate-based analysis
│   └── python/
│       ├── distribution_analysis.py  # Density & PCA plots
│       └── plot_methylation_line_combined.py
│
├── notebooks/                  # Jupyter notebooks
│   └── analysis_tutorial.ipynb  # Interactive tutorial
│
├── data/                       # Data directory
│   ├── raw/                   # Raw methylation matrices
│   ├── processed/             # Processed data
│   └── metadata/              # Sample annotations
│
└── results/                    # Analysis outputs
    ├── plots/                 # Generated figures
    └── tables/                # DMR results and statistics
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/OZP_NZP_GSE213366_Data_Analysis.git
cd OZP_NZP_GSE213366_Data_Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# Run complete analysis with default settings (CoRSIV probes)
python main.py

# Run with different probe set
python main.py --probe-set Controls

# Run specific analyses
python main.py --skip-dmr          # Skip DMR analysis
python main.py --skip-classifier   # Skip ML classification
```

### Interactive Analysis (Jupyter Notebook)

```bash
# Start Jupyter
jupyter notebook notebooks/analysis_tutorial.ipynb
```

The notebook provides step-by-step guidance through:
1. Loading and exploring data
2. Feature selection
3. Model training and evaluation
4. Visualization
5. Custom analysis parameters

### Run Classifier Only

```bash
cd classifier
python pipeline.py                    # Full pipeline
python pipeline.py --step train       # Train and evaluate
python pipeline.py --step validate    # External validation
python pipeline.py --step correlation # Correlation analysis
```

## Datasets

| Dataset | GEO Accession | Samples | Tissues | Purpose |
|---------|---------------|---------|---------|---------|
| Primary | GSE213366 | 70 (35 matched pairs) | Blood, Sperm | Training & Validation |
| External 1 | GSE51245 | 39 | Blood | External Validation |
| External 2 | GSE149318 | 179 | Blood, Sperm | Correlation Analysis |

## Probe Sets

| Probe Set | Description | Use Case |
|-----------|-------------|----------|
| CoRSIV | ~9,000 probes with systemic variation | Classification (default) |
| Controls | ~850 array control probes | Negative control comparison |
| EPIC_full | ~850,000 full array probes | DMR analysis, discovery |

## Adding New Datasets

1. Create a YAML configuration file in `configs/datasets/`:

```yaml
# configs/datasets/YOUR_DATASET.yaml
name: YOUR_DATASET
description: "Your dataset description"
geo_accession: GSEXXXXXX
n_samples: 100
tissues:
  - Blood
  - Sperm

files:
  methylation: path/to/methylation.csv
  metadata: path/to/metadata.csv

metadata_columns:
  sample_id: sample_name
  tissue: tissue
  status: status
```

2. Place data files in the appropriate directories

3. Run analysis:
```bash
python main.py --dataset YOUR_DATASET
```

## Adding New Probe Sets

1. Create a YAML configuration in `configs/probe_sets/`:

```yaml
# configs/probe_sets/YOUR_PROBES.yaml
name: YOUR_PROBES
description: "Your probe set description"
n_probes: ~1000

files:
  GSE213366: path/to/probe_data.csv

feature_selection:
  method: variance
  variance_threshold: 0.01
```

2. Run analysis with your probe set:
```bash
python main.py --probe-set YOUR_PROBES
```

## Methods

### DMR Identification

Uses the DMRcate package with:
- Lambda: 1000 bp (smoothing bandwidth)
- C: 2 (minimum CpGs per region)
- Statistical test: limma-based linear model

### Classification Models

| Model | Description |
|-------|-------------|
| Logistic Regression | L2-regularized linear classifier |
| Naive Bayes | Gaussian Naive Bayes |
| PLS-DA | Partial Least Squares Discriminant Analysis |

### Feature Selection

Probes selected based on blood-sperm correlation (r > 0.75), ensuring biomarkers are transferable across tissues.

### Validation Strategy

- **Internal**: Train on Blood samples, validate on matched Sperm samples
- **External**: Validate on independent blood cohort (GSE51245)

## Output Files

### Figures (PDF, 300 DPI)
- ROC curves with AUC scores
- PCA plots (batch effect, class separation)
- Correlation histograms
- Methylation density distributions

### Tables (CSV)
- Model performance metrics
- External validation predictions
- DMR results with genomic coordinates

## R Dependencies

Install R packages for DMR analysis:

```r
# From Bioconductor
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c(
    "DMRcate",
    "minfi",
    "limma",
    "GenomicRanges",
    "IlluminaHumanMethylationEPICanno.ilm10b5.hg38"
))
```

## Citation

If you use this pipeline, please cite:

- Primary dataset: GEO Accession GSE213366
- External datasets: GSE51245, GSE149318

## License

This project is for academic research purposes.

## Contact

Waterland Lab
Baylor College of Medicine

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request
