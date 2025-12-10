"""
Publication-ready plotting style configuration.
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt


def setup_publication_style(config: Optional[Any] = None) -> None:
    """
    Configure matplotlib for publication-quality figures.

    Args:
        config: Optional configuration object with viz_params
    """
    # Default parameters
    params = {
        "dpi": 300,
        "font_sizes": {
            "title": 12,
            "label": 11,
            "tick": 10,
            "legend": 9
        }
    }

    # Override with config if provided
    if config is not None and hasattr(config, "viz_params"):
        params.update(config.viz_params)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": params["font_sizes"]["tick"],
        "axes.titlesize": params["font_sizes"]["title"],
        "axes.labelsize": params["font_sizes"]["label"],
        "xtick.labelsize": params["font_sizes"]["tick"],
        "ytick.labelsize": params["font_sizes"]["tick"],
        "legend.fontsize": params["font_sizes"]["legend"],
        "figure.dpi": params["dpi"],
        "savefig.dpi": params["dpi"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def get_color_palette(config: Optional[Any] = None) -> Dict[str, str]:
    """
    Get consistent color palette for plots.

    Args:
        config: Optional configuration object

    Returns:
        Dictionary mapping category names to hex colors
    """
    default_colors = {
        "normozoospermia": "#2ecc71",
        "oligozoospermia": "#e74c3c",
        "blood": "#3498db",
        "sperm": "#9b59b6",
        "external": "#e67e22",
        "gse213366_blood": "#3498db",
        "gse213366_sperm": "#9b59b6",
        "gse51245_blood": "#e67e22",
    }

    if config is not None and hasattr(config, "viz_params"):
        colors = config.viz_params.get("colors", {})
        default_colors.update(colors)

    return default_colors
