# DNA Methylation Analysis (GSE213366)

This repository contains the analysis code and results for the DNA methylation dataset GSE213366.

## Directory Structure

- **data/**: Contains input data for the analysis.
    - `raw/`: Raw large datasets (not tracked by git).
    - `processed/`: Processed data files.
    - `metadata/`: Sample maps, probe lists, and annotation files.
- **scripts/**: Source code for analysis.
    - `R/`: R scripts for DMR analysis and plotting.
    - `python/`: Python scripts and notebooks for visualization and data processing.
- **results/**: Output of the analysis.
    - `plots/`: Generated figures (PDFs, PNGs).
    - `tables/`: Output tables and DMR lists.
- **doc/**: Documentation files.

## Prerequisites

- R (v4.0+)
- Python (v3.8+)
- Required R packages: `DMRcate`, `minfi`, `ggplot2`, `dplyr` (add others as needed)
- Required Python packages: `pandas`, `matplotlib`, `seaborn`

## Usage

### Main Pipeline
To run the full analysis (DMR identification and plotting):
```bash
python main.py
```

Options:
- `--skip-dmr`: Skip DMR analysis step.
- `--skip-plots`: Skip distribution plotting.

### Individual Scripts
- **DMR Analysis (R)**: `scripts/R/dmr_analysis.R`
- **Distribution Plots (Python)**: `scripts/python/distribution_analysis.py`
- **Line Plots (Python)**: `scripts/python/plot_methylation_line_combined.py` (requires probe file)

## Data Availability
The raw data for this analysis is available from GEO (accession GSE213366) and is expected to be placed in `data/raw/`.
