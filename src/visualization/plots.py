"""
Plotting functions for methylation analysis visualization.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .style import get_color_palette, setup_publication_style

logger = logging.getLogger(__name__)


class PlotGenerator:
    """
    Generate publication-ready plots for methylation analysis.
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize plot generator.

        Args:
            config: Configuration object with visualization parameters
        """
        self.config = config
        self.colors = get_color_palette(config)

        # Default figure sizes
        self.fig_sizes = {
            "single": (6, 5),
            "wide": (8, 6),
            "tall": (6, 8)
        }

        if config is not None and hasattr(config, "viz_params"):
            if "figure_sizes" in config.viz_params:
                self.fig_sizes.update(config.viz_params["figure_sizes"])

        # Setup matplotlib style
        setup_publication_style(config)

    def plot_roc_curves(
        self,
        roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
        output_path: Union[str, Path],
        title: Optional[str] = None
    ) -> None:
        """
        Generate ROC curves for multiple models.

        Args:
            roc_data: Dictionary of model name -> (fpr, tpr, auc)
            output_path: Path to save figure
            title: Optional custom title
        """
        output_path = Path(output_path)
        logger.info(f"Generating ROC curves -> {output_path}")

        fig, ax = plt.subplots(figsize=self.fig_sizes["single"])

        colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#e67e22"]

        for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors):
            ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {auc:.3f})")

        # Diagonal line
        ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", alpha=0.7)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        if title is None:
            title = "ROC Curves: Classification Performance"
        ax.set_title(title)

        ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black")
        ax.grid(True, alpha=0.3, linestyle="--")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf", dpi=300)
        plt.close(fig)

        logger.info("  ROC curves saved")

    def plot_pca(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        output_path: Union[str, Path],
        label_colors: Optional[Dict[str, str]] = None,
        markers: Optional[Dict[str, str]] = None,
        title: Optional[str] = None
    ) -> None:
        """
        Generate PCA scatter plot.

        Args:
            X: Feature matrix (samples x features)
            labels: Array of labels for each sample
            output_path: Path to save figure
            label_colors: Mapping of label -> color
            markers: Mapping of label -> marker style
            title: Optional custom title
        """
        output_path = Path(output_path)
        logger.info(f"Generating PCA plot -> {output_path}")

        # Standardize and compute PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)
        exp_var = pca.explained_variance_ratio_

        # Create DataFrame for plotting
        pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
        pca_df["label"] = labels

        # Default colors
        if label_colors is None:
            unique_labels = np.unique(labels)
            default_colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#e67e22"]
            label_colors = dict(zip(unique_labels, default_colors[:len(unique_labels)]))

        # Plot
        fig, ax = plt.subplots(figsize=self.fig_sizes["single"])

        for label in np.unique(labels):
            mask = pca_df["label"] == label
            marker = markers.get(label, "o") if markers else "o"
            ax.scatter(
                pca_df.loc[mask, "PC1"],
                pca_df.loc[mask, "PC2"],
                c=label_colors.get(label, "#333333"),
                marker=marker,
                s=60,
                alpha=0.7,
                label=label,
                edgecolors="white",
                linewidths=0.5,
            )

        ax.set_xlabel(f"PC1 ({exp_var[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({exp_var[1]*100:.1f}%)")

        if title is None:
            title = "PCA Analysis"
        ax.set_title(title)

        ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="black")
        ax.grid(True, alpha=0.3, linestyle="--")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf", dpi=300)
        plt.close(fig)

        logger.info("  PCA plot saved")

    def plot_correlation_histogram(
        self,
        correlations: np.ndarray,
        output_path: Union[str, Path],
        threshold: Optional[float] = None,
        title: Optional[str] = None,
        xlabel: str = "Pearson Correlation Coefficient (r)"
    ) -> None:
        """
        Generate histogram of correlation values.

        Args:
            correlations: Array of correlation coefficients
            output_path: Path to save figure
            threshold: Optional threshold line to display
            title: Optional custom title
            xlabel: X-axis label
        """
        output_path = Path(output_path)
        logger.info(f"Generating correlation histogram -> {output_path}")

        fig, ax = plt.subplots(figsize=self.fig_sizes["single"])

        # Histogram
        n, bins, patches = ax.hist(
            correlations, bins=50,
            color=self.colors["sperm"],
            alpha=0.7, edgecolor="white"
        )

        # Mean line
        mean_corr = np.nanmean(correlations)
        ax.axvline(mean_corr, color="red", linestyle="--", lw=2,
                   label=f"Mean = {mean_corr:.3f}")

        # Threshold line
        if threshold is not None:
            ax.axvline(threshold, color="green", linestyle="-", lw=2,
                       label=f"Threshold = {threshold}")

            # Shade above threshold
            for patch, left_edge in zip(patches, bins[:-1]):
                if left_edge >= threshold:
                    patch.set_facecolor(self.colors["normozoospermia"])

            # Add count annotation
            n_above = np.sum(correlations > threshold)
            ax.text(0.95, 0.95, f"n > {threshold}: {n_above}",
                    transform=ax.transAxes, ha="right", va="top",
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")

        if title is None:
            title = "Distribution of Correlation Coefficients"
        ax.set_title(title)

        ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="black")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf", dpi=300)
        plt.close(fig)

        logger.info("  Correlation histogram saved")

    def plot_density(
        self,
        df: pd.DataFrame,
        sample_col: str,
        group_col: str,
        output_path: Union[str, Path],
        groups: Optional[List[str]] = None,
        colors: Optional[Dict[str, str]] = None,
        title: Optional[str] = None
    ) -> None:
        """
        Generate density plots for methylation distributions.

        Args:
            df: DataFrame with methylation values (samples as columns)
            sample_col: Column in metadata for sample IDs
            group_col: Column in metadata for grouping
            output_path: Path to save figure
            groups: List of groups to plot
            colors: Color mapping for groups
            title: Optional custom title
        """
        output_path = Path(output_path)
        logger.info(f"Generating density plot -> {output_path}")

        if colors is None:
            colors = {"Blood": "#3498db", "Sperm": "#9b59b6"}

        fig, ax = plt.subplots(figsize=self.fig_sizes["wide"])

        # Plot each sample's density
        plotted_groups = set()
        for sample, group in zip(df[sample_col], df[group_col]):
            if groups is not None and group not in groups:
                continue

            # Get sample data (assuming transposed format)
            if sample in df.columns:
                data = df[sample].dropna()
                color = colors.get(group, "#333333")
                label = group if group not in plotted_groups else None
                plotted_groups.add(group)

                sns.kdeplot(data, color=color, linewidth=1, ax=ax, label=label)

        ax.set_xlabel("Methylation Level (Beta)")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)

        if title is None:
            title = "Methylation Distribution"
        ax.set_title(title)

        ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="black")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf", dpi=300)
        plt.close(fig)

        logger.info("  Density plot saved")

    def plot_methylation_line(
        self,
        df: pd.DataFrame,
        positions: np.ndarray,
        status_col: str,
        output_path: Union[str, Path],
        title: Optional[str] = None
    ) -> None:
        """
        Generate line plots of methylation across genomic positions.

        Args:
            df: DataFrame with samples as rows, positions as columns
            positions: Array of genomic positions
            status_col: Column with status labels
            output_path: Path to save figure
            title: Optional custom title
        """
        output_path = Path(output_path)
        logger.info(f"Generating line plot -> {output_path}")

        fig, ax = plt.subplots(figsize=(12, 4), dpi=300)

        color_map = {
            "Normozoospermia": self.colors["normozoospermia"],
            "Oligozoospermia": self.colors["oligozoospermia"]
        }

        # Plot each sample
        for idx, row in df.iterrows():
            # Get methylation values (exclude metadata columns)
            probe_cols = [c for c in df.columns if c not in [status_col, "sample_name"]]
            y = row[probe_cols].values.astype(float)
            status = row.get(status_col, "Unknown")

            ax.plot(
                positions,
                y,
                color=color_map.get(status, "black"),
                alpha=0.6,
                linewidth=1
            )

        # Legend
        legend_handles = [
            Line2D([0], [0], color=self.colors["normozoospermia"], lw=2,
                   label="Normozoospermia"),
            Line2D([0], [0], color=self.colors["oligozoospermia"], lw=2,
                   label="Oligozoospermia")
        ]
        ax.legend(handles=legend_handles, frameon=False, fontsize=11)

        ax.set_xlabel("Genomic Position", fontsize=12)
        ax.set_ylabel("Methylation Level", fontsize=12)

        if title is None:
            title = "Methylation Line Plot"
        ax.set_title(title, fontsize=14)

        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf", dpi=300)
        plt.close(fig)

        logger.info("  Line plot saved")
