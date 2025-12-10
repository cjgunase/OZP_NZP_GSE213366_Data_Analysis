#!/usr/bin/env python3
"""
DNA Methylation Analysis Pipeline

Main entry point for running DMR analysis and visualization.
Supports multiple datasets and probe sets through configuration.

Usage:
    python main.py                          # Run full pipeline
    python main.py --skip-dmr               # Skip DMR analysis
    python main.py --skip-plots             # Skip distribution plots
    python main.py --dataset GSE213366      # Specify dataset
    python main.py --probe-set CoRSIV       # Specify probe set
    python main.py --tissue Sperm           # Analyze specific tissue

Examples:
    # Run full analysis on CoRSIV probes
    python main.py --probe-set CoRSIV

    # Run analysis with control probes for comparison
    python main.py --probe-set Controls

    # Run only DMR analysis for Blood tissue
    python main.py --tissue Blood --skip-plots
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger

# Setup logger
logger = setup_logger("main", level=logging.INFO)


def run_dmr_analysis(
    tissue: str,
    beta_file: Path,
    sample_map: Path,
    output_file: Path,
    r_script: Path
) -> bool:
    """
    Run the R DMR analysis script.

    Args:
        tissue: Tissue type to analyze (e.g., "Sperm", "Blood")
        beta_file: Path to methylation beta values matrix
        sample_map: Path to sample metadata CSV
        output_file: Path to save DMR results
        r_script: Path to R analysis script

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Starting DMR analysis for {tissue}...")
    logger.info(f"  Beta file: {beta_file}")
    logger.info(f"  Sample map: {sample_map}")
    logger.info(f"  Output: {output_file}")

    if not r_script.exists():
        logger.error(f"R script not found: {r_script}")
        return False

    if not beta_file.exists():
        logger.error(f"Beta file not found: {beta_file}")
        return False

    if not sample_map.exists():
        logger.error(f"Sample map not found: {sample_map}")
        return False

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "Rscript",
        str(r_script),
        str(beta_file),
        str(sample_map),
        tissue,
        str(output_file)
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"DMR analysis completed. Results: {output_file}")
        if result.stdout:
            logger.debug(result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"DMR analysis failed: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False

    except FileNotFoundError:
        logger.error("Rscript not found. Please ensure R is installed and in PATH.")
        return False


def run_distribution_analysis(
    methylation_file: Path,
    sample_map: Path,
    output_dir: Path,
    target_state: str = "Normozoospermia"
) -> bool:
    """
    Run Python distribution analysis (density plots and PCA).

    Args:
        methylation_file: Path to methylation matrix
        sample_map: Path to sample metadata
        output_dir: Directory for output plots
        target_state: State to analyze for density plot

    Returns:
        True if successful, False otherwise
    """
    logger.info("Starting distribution analysis...")
    logger.info(f"  Methylation file: {methylation_file}")
    logger.info(f"  Sample map: {sample_map}")
    logger.info(f"  Output directory: {output_dir}")

    if not methylation_file.exists():
        logger.error(f"Methylation file not found: {methylation_file}")
        return False

    if not sample_map.exists():
        logger.error(f"Sample map not found: {sample_map}")
        return False

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Import the analysis module
        scripts_dir = PROJECT_ROOT / "scripts" / "python"
        sys.path.insert(0, str(scripts_dir))

        import distribution_analysis

        # Load data
        df, sm = distribution_analysis.load_data(
            str(methylation_file),
            str(sample_map)
        )

        # Generate density plot
        density_output = output_dir / f"density_{target_state}_Blood_Sperm.pdf"
        distribution_analysis.plot_density(
            df, sm, target_state, str(density_output)
        )

        # Generate PCA plot
        pca_output = output_dir / "pca_plot.pdf"
        distribution_analysis.plot_pca(df, sm, str(pca_output))

        logger.info("Distribution analysis completed.")
        return True

    except ImportError as e:
        logger.error(f"Failed to import distribution_analysis: {e}")
        return False

    except Exception as e:
        logger.error(f"Distribution analysis failed: {e}")
        return False


def run_classification_pipeline(
    config,
    output_dir: Path
) -> bool:
    """
    Run the ML classification pipeline.

    Args:
        config: Configuration object
        output_dir: Directory for output files

    Returns:
        True if successful, False otherwise
    """
    logger.info("Starting classification pipeline...")

    try:
        # Import pipeline components
        from src.data_loaders import MethylationDataLoader, MetadataLoader
        from src.preprocessing import CorrelationSelector, DataTransformer
        from src.models import ClassifierFactory, ClassifierEvaluator
        from src.visualization import PlotGenerator

        # Initialize components
        meth_loader = MethylationDataLoader(config)
        meta_loader = MetadataLoader(config)
        transformer = DataTransformer(random_state=config.model_params["random_state"])
        factory = ClassifierFactory(config)
        evaluator = ClassifierEvaluator(
            cv_folds=config.model_params["cv_folds"],
            random_state=config.model_params["random_state"]
        )
        plotter = PlotGenerator(config)

        # Load data
        meth_path = config.get_probe_set_path()
        meta_path = config.get_data_path("metadata")

        logger.info(f"Loading data from {meth_path.name}...")
        merged_df = meth_loader.load_with_metadata(meth_path, meta_path)

        # Encode labels
        merged_df = meta_loader.encode_labels(
            merged_df, "status", config.class_mapping["status"]
        )

        # Feature selection
        selector = CorrelationSelector(
            threshold=config.model_params["correlation_threshold"]
        )

        corr_file = config.dataset.get("files", {}).get("blood_sperm_correlation")
        if corr_file:
            selector.fit_from_file(str(config.base_dir / corr_file))
        else:
            logger.warning("No correlation file found, using all features")
            selector.selected_features_ = [
                c for c in merged_df.columns
                if c not in ["Individual_ID", "geo_accession", "sample_name",
                           "tissue", "status", "state", "target"]
            ]

        features = selector.get_selected_features()
        available_features = [f for f in features if f in merged_df.columns]

        logger.info(f"Using {len(available_features)} features")

        # Prepare data
        X_train, y_train, X_test, y_test = transformer.prepare_for_training(
            merged_df,
            available_features,
            train_tissue="Blood",
            test_tissue="Sperm"
        )

        # Train and evaluate models
        models = factory.create_all()
        results = evaluator.evaluate_all(models, X_train, y_train, X_test, y_test)

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)

        # Performance summary
        perf_df = evaluator.to_dataframe(results)
        perf_df.to_csv(output_dir / "model_performance_summary.csv", index=False)
        logger.info(f"Saved performance summary to {output_dir / 'model_performance_summary.csv'}")

        # ROC curves
        roc_data = evaluator.get_roc_data(results, y_test)
        plotter.plot_roc_curves(
            roc_data,
            output_dir / "roc_curves.pdf",
            title="ROC Curves: Classification of Oligozoospermia\n(Training: Blood, Test: Sperm)"
        )

        logger.info("Classification pipeline completed.")
        return True

    except Exception as e:
        logger.error(f"Classification pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GSE213366 DNA Methylation Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--dataset",
        default="GSE213366",
        help="Dataset to analyze (default: GSE213366)"
    )
    parser.add_argument(
        "--probe-set",
        default="CoRSIV",
        choices=["CoRSIV", "Controls", "EPIC_full"],
        help="Probe set to use (default: CoRSIV)"
    )
    parser.add_argument(
        "--tissue",
        default="Sperm",
        choices=["Blood", "Sperm"],
        help="Tissue type for DMR analysis (default: Sperm)"
    )
    parser.add_argument(
        "--skip-dmr",
        action="store_true",
        help="Skip DMR analysis"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip distribution plots"
    )
    parser.add_argument(
        "--skip-classifier",
        action="store_true",
        help="Skip ML classification pipeline"
    )
    parser.add_argument(
        "--config",
        help="Path to custom configuration file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    logger.info("=" * 60)
    logger.info("DNA Methylation Analysis Pipeline")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Probe set: {args.probe_set}")

    config = load_config(
        config_file=args.config,
        dataset=args.dataset,
        probe_set=args.probe_set
    )

    # Setup paths
    base_dir = config.base_dir
    data_raw = base_dir / "data" / "raw"
    data_meta = base_dir / "data" / "metadata"
    results_dir = base_dir / "results"
    scripts_r = base_dir / "scripts" / "R"

    # Ensure output directories exist
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)

    success = True

    # Run DMR Analysis
    if not args.skip_dmr:
        logger.info("")
        logger.info("[Step 1] DMR Analysis")
        logger.info("-" * 40)

        # Get probe set file
        probe_set_file = config.probe_set.get("files", {}).get(args.dataset)
        if probe_set_file:
            beta_file = base_dir / probe_set_file
        else:
            beta_file = data_raw / "GSE213366_CoRSIV_2023.csv"

        sample_map = data_meta / "sample_tissue_individual_map.csv"
        output_file = results_dir / "tables" / f"dmr_results_{args.tissue}_{args.probe_set}.csv"
        r_script = scripts_r / "dmr_analysis.R"

        if not run_dmr_analysis(args.tissue, beta_file, sample_map, output_file, r_script):
            success = False
            logger.warning("DMR analysis failed, continuing with other steps...")

    # Run Distribution Plots
    if not args.skip_plots:
        logger.info("")
        logger.info("[Step 2] Distribution Analysis")
        logger.info("-" * 40)

        # Use full matrix for distribution analysis
        meth_file = data_raw / "GSE213366_matrix_processed.csv"
        sample_map = data_meta / "sample_tissue_individual_map.csv"
        output_dir = results_dir / "plots"

        if not run_distribution_analysis(meth_file, sample_map, output_dir):
            success = False
            logger.warning("Distribution analysis failed, continuing...")

    # Run Classification Pipeline
    if not args.skip_classifier:
        logger.info("")
        logger.info("[Step 3] Classification Pipeline")
        logger.info("-" * 40)

        classifier_output = results_dir / "classifier"
        if not run_classification_pipeline(config, classifier_output):
            success = False
            logger.warning("Classification pipeline failed.")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    if success:
        logger.info("Pipeline completed successfully!")
    else:
        logger.warning("Pipeline completed with some errors.")
    logger.info("=" * 60)

    logger.info(f"Results saved to: {results_dir}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
