#!/usr/bin/env python3
"""
Methylation Distribution Analysis

Generate density plots and PCA visualizations for DNA methylation data.
Compares methylation distributions between tissues and fertility status groups.

Usage:
    python distribution_analysis.py --methylation_file <path> --sample_map <path>
    python distribution_analysis.py --help
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent.parent / "data"
DEFAULT_METHYLATION_FILE = DEFAULT_DATA_DIR / "raw" / "GSE213366_matrix_processed.csv"
DEFAULT_SAMPLE_MAP = DEFAULT_DATA_DIR / "metadata" / "sample_tissue_individual_map.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent.parent / "results" / "plots"


def load_data(
    methylation_file: str,
    sample_map_file: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load methylation data and sample metadata.

    Args:
        methylation_file: Path to methylation matrix CSV (probes x samples)
        sample_map_file: Path to sample metadata CSV

    Returns:
        Tuple of (methylation_df, sample_map_df)
    """
    logger.info(f"Loading methylation data from {methylation_file}...")

    # Validate files exist
    if not os.path.exists(methylation_file):
        raise FileNotFoundError(f"Methylation file not found: {methylation_file}")

    if not os.path.exists(sample_map_file):
        raise FileNotFoundError(f"Sample map file not found: {sample_map_file}")

    # Load sample map
    sample_map = pd.read_csv(sample_map_file)
    logger.info(f"Loaded metadata for {len(sample_map)} samples")

    # Read methylation file header to get column names
    header_df = pd.read_csv(methylation_file, nrows=0)
    all_cols = header_df.columns.tolist()

    # Find columns that match samples in the metadata
    sample_names = sample_map['sample_name'].values
    valid_cols = [c for c in all_cols if c in sample_names]

    if not valid_cols:
        # Try with X prefix (R sometimes adds this)
        valid_cols_x = [c for c in all_cols
                       if c.startswith('X') and c[1:] in sample_names]
        if valid_cols_x:
            valid_cols = valid_cols_x
            logger.info("Using X-prefixed column names")

    if not valid_cols:
        logger.warning("No exact sample matches found. Loading all columns.")
        valid_cols = [c for c in all_cols[1:]]  # Skip index column

    # Load methylation data with only matching columns
    use_cols = [all_cols[0]] + valid_cols  # Include index column
    logger.info(f"Loading {len(valid_cols)} sample columns...")

    df = pd.read_csv(methylation_file, usecols=use_cols, index_col=0)
    logger.info(f"Loaded methylation matrix: {df.shape[0]} probes x {df.shape[1]} samples")

    return df, sample_map


def plot_density(
    df: pd.DataFrame,
    sample_map: pd.DataFrame,
    target_state: str,
    output_file: str,
    tissues: Optional[List[str]] = None
) -> None:
    """
    Generate density plot of methylation distributions.

    Args:
        df: Methylation matrix (probes x samples)
        sample_map: Sample metadata
        target_state: Fertility status to plot (e.g., "Normozoospermia")
        output_file: Path to save plot
        tissues: List of tissues to include (default: ["Blood", "Sperm"])
    """
    logger.info(f"Generating density plot for {target_state}...")

    if tissues is None:
        tissues = ["Blood", "Sperm"]

    # Get status column name (handle both 'status' and 'state')
    status_col = "status" if "status" in sample_map.columns else "state"

    # Filter metadata for target state and tissues
    subset_map = sample_map[
        (sample_map[status_col] == target_state) &
        (sample_map['tissue'].isin(tissues))
    ]

    if subset_map.empty:
        logger.warning(f"No samples found for {target_state} in tissues {tissues}")
        return

    # Color scheme
    colors = {
        'Blood': '#3498db',  # Blue
        'Sperm': '#9b59b6'   # Purple
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    plotted_tissues = set()
    for _, row in subset_map.iterrows():
        sample = row['sample_name']
        tissue = row['tissue']

        # Handle X prefix in column names
        if sample not in df.columns and f'X{sample}' in df.columns:
            sample = f'X{sample}'

        if sample in df.columns:
            data = df[sample].dropna()
            if len(data) > 0:
                color = colors.get(tissue, 'black')
                label = tissue if tissue not in plotted_tissues else None
                plotted_tissues.add(tissue)

                sns.kdeplot(data, color=color, linewidth=1, ax=ax, label=label)

    # Formatting
    ax.set_xlabel("Methylation Level (Beta)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_title(f"Methylation Distribution ({target_state})", fontsize=14)

    if plotted_tissues:
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')

    plt.tight_layout()

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Density plot saved to {output_file}")


def plot_pca(
    df: pd.DataFrame,
    sample_map: pd.DataFrame,
    output_file: str,
    n_components: int = 2
) -> None:
    """
    Generate PCA plot of methylation data.

    Args:
        df: Methylation matrix (probes x samples)
        sample_map: Sample metadata
        output_file: Path to save plot
        n_components: Number of PCA components (default: 2)
    """
    logger.info("Performing PCA analysis...")

    # Get status column name
    status_col = "status" if "status" in sample_map.columns else "state"

    # Find common samples between data and metadata
    sample_names = sample_map['sample_name'].values
    common_samples = [s for s in df.columns if s in sample_names]

    # Also check for X prefix
    if not common_samples:
        common_samples = [s for s in df.columns
                        if s.startswith('X') and s[1:] in sample_names]

    if len(common_samples) < 3:
        logger.error(f"Not enough samples for PCA (found {len(common_samples)})")
        return

    logger.info(f"Running PCA on {len(common_samples)} samples")

    # Transpose: samples as rows, probes as columns
    data = df[common_samples].T

    # Remove probes with any NA
    data_clean = data.dropna(axis=1)

    if data_clean.shape[1] == 0:
        logger.error("No probes left after NA removal")
        return

    logger.info(f"Using {data_clean.shape[1]} probes after NA removal")

    # Standardize and run PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_clean)

    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    # Create DataFrame with PCA coordinates
    pca_df = pd.DataFrame(coords, columns=['PC1', 'PC2'], index=data_clean.index)

    # Merge with metadata
    # Handle X prefix in index
    pca_df_reset = pca_df.reset_index()
    pca_df_reset.columns = ['sample_name'] + list(pca_df.columns)

    # Try direct merge
    merged = pca_df_reset.merge(
        sample_map[['sample_name', 'tissue', status_col]],
        on='sample_name',
        how='left'
    )

    # If merge failed, try without X prefix
    if merged['tissue'].isna().all():
        pca_df_reset['sample_name'] = pca_df_reset['sample_name'].str.lstrip('X')
        merged = pca_df_reset.merge(
            sample_map[['sample_name', 'tissue', status_col]],
            on='sample_name',
            how='left'
        )

    # Drop rows without metadata
    merged = merged.dropna(subset=['tissue', status_col])

    if len(merged) == 0:
        logger.error("No samples matched between PCA results and metadata")
        return

    # Color mapping based on status and tissue
    color_map = {
        ('Normozoospermia', 'Blood'): '#e74c3c',    # Red
        ('Normozoospermia', 'Sperm'): '#2ecc71',    # Green
        ('Oligozoospermia', 'Blood'): '#e67e22',    # Orange
        ('Oligozoospermia', 'Sperm'): '#3498db',    # Blue
    }

    colors = []
    for _, row in merged.iterrows():
        key = (row[status_col], row['tissue'])
        colors.append(color_map.get(key, '#333333'))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        merged['PC1'], merged['PC2'],
        c=colors, s=100, alpha=0.8, edgecolors='white', linewidths=0.5
    )

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               label='Normo-Blood', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
               label='Normo-Sperm', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e67e22',
               label='Oligo-Blood', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               label='Oligo-Sperm', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='best', frameon=True,
              fancybox=False, edgecolor='black')

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=12)
    ax.set_title("PCA of Methylation Data", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"PCA plot saved to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Methylation Distribution Analysis (Density & PCA)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--methylation_file",
        default=str(DEFAULT_METHYLATION_FILE),
        help="Path to methylation matrix CSV"
    )
    parser.add_argument(
        "--sample_map",
        default=str(DEFAULT_SAMPLE_MAP),
        help="Path to sample metadata CSV"
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save output plots"
    )
    parser.add_argument(
        "--state",
        default="Normozoospermia",
        choices=["Normozoospermia", "Oligozoospermia"],
        help="Fertility status for density plot"
    )
    parser.add_argument(
        "--skip-density",
        action="store_true",
        help="Skip density plot generation"
    )
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="Skip PCA plot generation"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load data
        df, sample_map = load_data(args.methylation_file, args.sample_map)

        # Generate density plot
        if not args.skip_density:
            density_output = output_dir / f"density_{args.state}_Blood_Sperm.pdf"
            plot_density(df, sample_map, args.state, str(density_output))

        # Generate PCA plot
        if not args.skip_pca:
            pca_output = output_dir / "pca_plot.pdf"
            plot_pca(df, sample_map, str(pca_output))

        logger.info("Analysis completed successfully!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
