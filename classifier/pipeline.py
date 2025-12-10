#!/usr/bin/env python3
"""
DNA Methylation Classifier Pipeline for Male Fertility Classification

This pipeline classifies male fertility status (Normozoospermia vs Oligozoospermia)
using DNA methylation data from CoRSIV probes. The classifier is trained on blood
samples and validated on both sperm samples (internal) and external blood datasets.

Dataset: GSE213366 (primary), GSE51245 (external validation), GSE149318 (correlation)

Usage:
    python pipeline.py                    # Run full pipeline
    python pipeline.py --step train       # Run specific step
    python pipeline.py --help             # Show help

This refactored version uses the modular src/ components for better
maintainability and extensibility.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from src modules
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.data_loaders import MethylationDataLoader, MetadataLoader
from src.preprocessing import CorrelationSelector, DataTransformer
from src.models import ClassifierFactory, ClassifierEvaluator
from src.visualization import PlotGenerator, setup_publication_style

# Local config for classifier-specific settings
import config as local_config

# Setup logging
logger = setup_logger("classifier", level=logging.INFO)


class ClassifierPipeline:
    """
    Main pipeline class for fertility classification.

    Encapsulates all steps: data loading, preprocessing, training,
    validation, and visualization.
    """

    def __init__(self, dataset: str = "GSE213366", probe_set: str = "CoRSIV"):
        """
        Initialize pipeline with configuration.

        Args:
            dataset: Name of primary dataset
            probe_set: Name of probe set to use
        """
        self.config = load_config(dataset=dataset, probe_set=probe_set)
        self.local_config = local_config

        # Initialize components
        self.meth_loader = MethylationDataLoader(self.config)
        self.meta_loader = MetadataLoader(self.config)
        self.transformer = DataTransformer(
            random_state=self.config.model_params["random_state"]
        )
        self.factory = ClassifierFactory(self.config)
        self.evaluator = ClassifierEvaluator(
            cv_folds=self.config.model_params["cv_folds"],
            random_state=self.config.model_params["random_state"]
        )
        self.plotter = PlotGenerator(self.config)

        # Setup matplotlib style
        setup_publication_style(self.config)

        # Ensure output directories exist
        local_config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        local_config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def load_primary_data(self) -> pd.DataFrame:
        """
        Load and preprocess the primary methylation dataset (GSE213366).

        Returns:
            Merged dataframe with samples as rows and probes as columns.
        """
        logger.info("Loading primary methylation data (GSE213366)...")

        # Load methylation data
        meth_df = pd.read_csv(local_config.METHYLATION_FILE, index_col=0)

        # Load metadata
        meta_df = pd.read_csv(local_config.METADATA_FILE)

        # Transpose methylation data (samples as rows)
        meth_df = meth_df.T
        meth_df.index.name = "sample_name"

        # Merge metadata with methylation data
        merged_df = meta_df.merge(meth_df, left_on="sample_name", right_index=True)

        # Encode target: Normozoospermia = 0, Oligozoospermia = 1
        merged_df["target"] = merged_df["status"].map(local_config.STATE_MAPPING)

        logger.info(f"Loaded {merged_df.shape[0]} samples with {merged_df.shape[1] - 6} probes")

        return merged_df

    def load_correlation_data(self) -> pd.DataFrame:
        """Load pre-computed blood-sperm correlation data."""
        logger.info("Loading blood-sperm correlation data...")
        corr_df = pd.read_csv(local_config.CORRELATION_FILE)
        logger.info(f"Loaded correlations for {len(corr_df)} probes")
        return corr_df

    def load_external_gse51245(self):
        """Load external validation dataset (GSE51245)."""
        logger.info("Loading external validation data (GSE51245)...")

        ext_df = pd.read_csv(local_config.GSE51245_PROBES_FILE, index_col=0)
        meta_df = pd.read_csv(local_config.GSE51245_METADATA_FILE)

        # Clean column names (remove .AVG_Beta suffix)
        ext_df.columns = [c.replace(".AVG_Beta", "") for c in ext_df.columns]

        # Transpose to samples as rows
        ext_df = ext_df.T
        ext_df.index.name = "sample_id"

        logger.info(f"Loaded {ext_df.shape[0]} samples with {ext_df.shape[1]} probes")

        return ext_df, meta_df

    def load_gse149318_data(self):
        """Load GSE149318 data for blood-sperm correlation analysis."""
        logger.info("Loading GSE149318 data for correlation analysis...")

        meth_df = pd.read_csv(local_config.GSE149318_PROBES_FILE, index_col=0)
        sample_df = pd.read_csv(local_config.GSE149318_METADATA_FILE)

        # Standardize tissue types
        sample_df["tissue_type"] = sample_df["tissue_type"].str.lower().str.strip()

        return meth_df, sample_df

    def split_train_test(self, df: pd.DataFrame):
        """Split data into training (Blood) and test (Sperm) sets."""
        logger.info("Splitting data into Blood (train) and Sperm (test)...")

        train_df = df[df["tissue"] == "Blood"].copy()
        test_df = df[df["tissue"] == "Sperm"].copy()

        # Metadata columns to exclude from features
        meta_cols = ["Individual_ID", "geo_accession", "sample_name",
                    "tissue", "status", "target"]

        X_train = train_df.drop(columns=meta_cols)
        y_train = train_df["target"]

        X_test = test_df.drop(columns=meta_cols)
        y_test = test_df["target"]

        # Impute missing values with training set means
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_train.mean())

        logger.info(f"Training set: {X_train.shape[0]} samples, "
                   f"Test set: {X_test.shape[0]} samples")

        return X_train, y_train, X_test, y_test

    def select_correlated_features(self, X_train: pd.DataFrame,
                                   threshold: float = None):
        """Select features with high blood-sperm correlation."""
        if threshold is None:
            threshold = local_config.CORRELATION_THRESHOLD

        logger.info(f"Selecting features with correlation > {threshold}...")

        corr_df = self.load_correlation_data()
        selected_probes = corr_df[
            corr_df["Pearson_Correlation"] > threshold
        ]["ID_REF"].tolist()

        # Filter to available probes
        available = [p for p in selected_probes if p in X_train.columns]

        logger.info(f"Selected {len(available)} probes (threshold={threshold})")

        return available

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Train multiple models and evaluate on test set."""
        logger.info("Training and evaluating models...")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = self.factory.create_all()
        results = self.evaluator.evaluate_all(
            models, X_train_scaled, y_train.values,
            X_test_scaled, y_test.values
        )

        # Add scaler to results for external validation
        for name in results:
            results[name]["scaler"] = scaler

        return results

    def train_final_models(self, correlation_threshold=None):
        """Train final models on Blood data for external validation."""
        if correlation_threshold is None:
            correlation_threshold = local_config.CORRELATION_THRESHOLD

        logger.info("Training final models for external validation...")

        # Load and prepare data
        df = self.load_primary_data()
        train_df = df[df["tissue"] == "Blood"].copy()

        # Feature selection
        features = self.select_correlated_features(
            train_df, threshold=correlation_threshold
        )

        # Prepare training data
        meta_cols = ["Individual_ID", "geo_accession", "sample_name",
                    "tissue", "status", "target"]
        X_train = train_df.drop(columns=meta_cols)
        y_train = train_df["target"]

        # Compute imputation means
        imputer_means = X_train.mean()
        X_train = X_train.fillna(imputer_means)

        # Filter to selected features
        available = [f for f in features if f in X_train.columns]
        X_train_selected = X_train[available]

        logger.info(f"Training on {len(available)} features")

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)

        # Train models
        models = self.factory.create_all()
        trained_models = {}

        for name, model in models.items():
            logger.info(f"  Training final {name}...")
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model

        return trained_models, scaler, available, imputer_means

    def validate_external(self, models, scaler, features, imputer_means):
        """Validate trained models on external dataset (GSE51245)."""
        logger.info("Validating on external dataset (GSE51245)...")

        ext_df, meta_df = self.load_external_gse51245()

        # Merge with metadata
        merged_df = meta_df.merge(
            ext_df, left_on="sample_id", right_index=True, how="inner"
        )

        if merged_df.empty:
            logger.error("No samples matched between probe file and metadata")
            return None

        # Align features
        X_ext = merged_df.reindex(columns=features)
        X_ext = X_ext.apply(pd.to_numeric, errors="coerce")
        X_ext = X_ext.fillna(imputer_means[features])

        # Scale
        X_ext_scaled = scaler.transform(X_ext)

        # Predict with each model
        results_df = merged_df[["sample_id", "status"]].copy()

        for name, model in models.items():
            logger.info(f"  Predicting with {name}...")

            pred = model.predict(X_ext_scaled)
            pred_labels = ["Oligozoospermia" if p == 1 else "Normozoospermia"
                          for p in pred]
            results_df[f"{name}_prediction"] = pred_labels

            # Calculate accuracy
            y_true = results_df["status"].map(local_config.EXTERNAL_STATUS_MAPPING)
            valid_mask = ~y_true.isna()

            if valid_mask.sum() > 0:
                from sklearn.metrics import accuracy_score
                acc = accuracy_score(y_true[valid_mask], pred[valid_mask])
                logger.info(f"    External accuracy: {acc:.4f}")

        return results_df

    def plot_roc_curves(self, results, y_test, output_path=None):
        """Generate publication-ready ROC curves."""
        if output_path is None:
            output_path = local_config.ROC_CURVE_OUTPUT

        roc_data = self.evaluator.get_roc_data(results, y_test.values)
        self.plotter.plot_roc_curves(
            roc_data, output_path,
            title="ROC Curves: Classification of Oligozoospermia\n"
                  "(Training: Blood, Test: Sperm)"
        )

    def plot_pca_batch_effect(self, output_path=None):
        """Generate PCA plot showing batch effects between datasets."""
        if output_path is None:
            output_path = local_config.PCA_BATCH_OUTPUT

        logger.info(f"Generating batch effect PCA -> {output_path}")

        # Load internal data
        df_internal = self.load_primary_data()
        meta_cols = ["Individual_ID", "geo_accession", "sample_name",
                    "tissue", "status", "target"]
        internal_probes = [c for c in df_internal.columns if c not in meta_cols]

        # Load external data
        ext_df, _ = self.load_external_gse51245()
        external_probes = ext_df.columns.tolist()

        # Find common probes
        common_probes = list(set(internal_probes).intersection(set(external_probes)))
        logger.info(f"  Using {len(common_probes)} common probes")

        if len(common_probes) < 2:
            logger.error("Not enough common probes for PCA")
            return

        # Prepare data
        X_internal = df_internal[common_probes].copy()
        X_internal = X_internal.fillna(X_internal.mean())
        labels_internal = "GSE213366_" + df_internal["tissue"]

        X_external = ext_df[common_probes].copy()
        X_external = X_external.apply(pd.to_numeric, errors="coerce")
        X_external = X_external.fillna(X_external.mean())
        labels_external = pd.Series(
            ["GSE51245_Blood"] * len(X_external), index=X_external.index
        )

        # Combine
        X_combined = pd.concat([X_internal, X_external])
        labels_combined = pd.concat([labels_internal, labels_external])

        # Plot using visualization module
        label_colors = {
            "GSE213366_Blood": local_config.COLORS["gse213366_blood"],
            "GSE213366_Sperm": local_config.COLORS["gse213366_sperm"],
            "GSE51245_Blood": local_config.COLORS["gse51245_blood"],
        }

        self.plotter.plot_pca(
            X_combined.values,
            labels_combined.values,
            output_path,
            label_colors=label_colors,
            title=f"PCA: Batch Effect Analysis\n(n = {len(common_probes)} CoRSIV probes)"
        )

    def compute_gse213366_probe_correlations(self):
        """Compute probe-wise blood-sperm correlations for GSE213366 dataset."""
        logger.info("Computing GSE213366 probe-wise blood-sperm correlations...")

        # Load methylation data (probes as rows, samples as columns)
        meth_df = pd.read_csv(local_config.METHYLATION_FILE, index_col=0)
        meta_df = pd.read_csv(local_config.METADATA_FILE)

        # Identify matched blood-sperm pairs by Individual_ID
        blood_samples = meta_df[meta_df["tissue"] == "Blood"][
            ["Individual_ID", "sample_name"]
        ]
        sperm_samples = meta_df[meta_df["tissue"] == "Sperm"][
            ["Individual_ID", "sample_name"]
        ]

        # Merge to get matched pairs
        matched = blood_samples.merge(
            sperm_samples, on="Individual_ID",
            suffixes=("_blood", "_sperm")
        )
        logger.info(f"  Found {len(matched)} matched blood-sperm pairs")

        # Get sample names for matched pairs
        blood_cols = matched["sample_name_blood"].tolist()
        sperm_cols = matched["sample_name_sperm"].tolist()

        # Verify all samples exist in methylation data
        blood_cols = [c for c in blood_cols if c in meth_df.columns]
        sperm_cols = [c for c in sperm_cols if c in meth_df.columns]

        # Reorder to ensure matching
        matched_filtered = matched[
            (matched["sample_name_blood"].isin(blood_cols)) &
            (matched["sample_name_sperm"].isin(sperm_cols))
        ]
        blood_cols = matched_filtered["sample_name_blood"].tolist()
        sperm_cols = matched_filtered["sample_name_sperm"].tolist()

        logger.info(f"  Using {len(blood_cols)} matched pairs for correlation")

        # Calculate probe-wise correlations
        results = []
        probes = meth_df.index.tolist()

        for probe in probes:
            blood_values = meth_df.loc[probe, blood_cols].values.astype(float)
            sperm_values = meth_df.loc[probe, sperm_cols].values.astype(float)

            # Remove NaN pairs
            valid_mask = ~(np.isnan(blood_values) | np.isnan(sperm_values))
            blood_valid = blood_values[valid_mask]
            sperm_valid = sperm_values[valid_mask]

            if len(blood_valid) >= 3:
                corr, pval = pearsonr(blood_valid, sperm_valid)
            else:
                corr, pval = np.nan, np.nan

            results.append({
                "Probe_ID": probe,
                "Pearson_Correlation": corr,
                "P_Value": pval,
                "Mean_Blood": np.nanmean(blood_values),
                "Mean_Sperm": np.nanmean(sperm_values),
                "N_Pairs": len(blood_valid),
            })

        corr_df = pd.DataFrame(results)

        # Save results
        local_config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        corr_df.to_csv(local_config.GSE213366_CORRELATION_OUTPUT, index=False)
        logger.info(
            f"  Saved correlation results to {local_config.GSE213366_CORRELATION_OUTPUT}"
        )

        return corr_df

    def plot_correlation_histogram_gse213366(self, corr_df=None, output_path=None):
        """Generate histogram of probe-wise blood-sperm methylation correlations."""
        if output_path is None:
            output_path = local_config.CORRELATION_HISTOGRAM_GSE213366_OUTPUT

        logger.info(f"Generating GSE213366 correlation histogram -> {output_path}")

        # Load or compute correlations
        if corr_df is None:
            if local_config.GSE213366_CORRELATION_OUTPUT.exists():
                corr_df = pd.read_csv(local_config.GSE213366_CORRELATION_OUTPUT)
            else:
                corr_df = self.compute_gse213366_probe_correlations()

        correlations = corr_df["Pearson_Correlation"].dropna().values

        self.plotter.plot_correlation_histogram(
            correlations,
            output_path,
            threshold=local_config.CORRELATION_THRESHOLD,
            title="Distribution of Blood-Sperm Probe Correlations\n"
                  "(GSE213366, probe-wise across matched pairs)"
        )

    def plot_correlation_histogram_gse149318(self, output_path=None):
        """Generate histogram of sample-wise correlations for GSE149318."""
        if output_path is None:
            output_path = local_config.CORRELATION_HISTOGRAM_GSE149318_OUTPUT

        logger.info(f"Generating GSE149318 correlation histogram -> {output_path}")

        # Load GSE149318 data
        meth_df, sample_df = self.load_gse149318_data()

        # Filter for blood and sperm
        valid_tissues = ["whole blood", "sperm"]
        sample_df = sample_df[sample_df["tissue_type"].isin(valid_tissues)]

        # Find matched pairs
        matched_pairs = []
        grouped = sample_df.groupby("match")

        for match_id, group in grouped:
            if len(group) == 2:
                tissues = group["tissue_type"].values
                if "whole blood" in tissues and "sperm" in tissues:
                    blood_sample = group[
                        group["tissue_type"] == "whole blood"
                    ]["sample_name"].values[0]
                    sperm_sample = group[
                        group["tissue_type"] == "sperm"
                    ]["sample_name"].values[0]
                    matched_pairs.append((match_id, blood_sample, sperm_sample))

        logger.info(f"  Found {len(matched_pairs)} matched blood-sperm pairs")

        # Calculate correlations
        correlations = []
        for match_id, blood_sample, sperm_sample in matched_pairs:
            if blood_sample in meth_df.columns and sperm_sample in meth_df.columns:
                s1 = meth_df[blood_sample]
                s2 = meth_df[sperm_sample]
                corr = s1.corr(s2)
                correlations.append(corr)

        if not correlations:
            logger.error("No correlations calculated")
            return

        self.plotter.plot_correlation_histogram(
            np.array(correlations),
            output_path,
            title="Distribution of Blood-Sperm Methylation Correlations\n"
                  "(GSE149318, sample-wise across all probes)"
        )

    def export_model_performance(self, results, y_test, output_path=None):
        """Export model performance metrics to CSV."""
        if output_path is None:
            output_path = local_config.MODEL_PERFORMANCE_OUTPUT

        logger.info(f"Exporting model performance -> {output_path}")

        perf_df = self.evaluator.to_dataframe(results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        perf_df.to_csv(output_path, index=False)

        logger.info("  Performance summary saved")
        return perf_df

    def run_full_pipeline(self):
        """Run the complete analysis pipeline."""
        logger.info("=" * 60)
        logger.info("DNA Methylation Classifier Pipeline")
        logger.info("=" * 60)

        # Step 1: Load and preprocess data
        logger.info("\n[Step 1] Loading and preprocessing data...")
        df = self.load_primary_data()
        X_train, y_train, X_test, y_test = self.split_train_test(df)

        # Step 2: Feature selection
        logger.info("\n[Step 2] Feature selection...")
        features = self.select_correlated_features(X_train)

        # Subset data to selected features
        X_train_selected = X_train[features]
        X_test_selected = X_test[features]

        # Step 3: Train and evaluate models
        logger.info("\n[Step 3] Training and evaluating models...")
        results = self.train_and_evaluate(
            X_train_selected, y_train, X_test_selected, y_test
        )

        # Step 4: Export performance metrics
        logger.info("\n[Step 4] Exporting results...")
        self.export_model_performance(results, y_test)

        # Step 5: Generate ROC curves
        logger.info("\n[Step 5] Generating ROC curves...")
        self.plot_roc_curves(results, y_test)

        # Step 6: External validation
        logger.info("\n[Step 6] External validation...")
        models, scaler, features_final, imputer_means = self.train_final_models()
        ext_predictions = self.validate_external(
            models, scaler, features_final, imputer_means
        )

        if ext_predictions is not None:
            ext_predictions.to_csv(local_config.PREDICTIONS_OUTPUT, index=False)
            logger.info(f"  Predictions saved to {local_config.PREDICTIONS_OUTPUT}")

        # Step 7: Batch effect analysis
        logger.info("\n[Step 7] Batch effect analysis...")
        self.plot_pca_batch_effect()

        # Step 8: GSE213366 Blood-sperm correlation analysis
        logger.info("\n[Step 8] GSE213366 blood-sperm correlation analysis...")
        corr_df = self.compute_gse213366_probe_correlations()
        self.plot_correlation_histogram_gse213366(corr_df)

        # Step 9: GSE149318 Blood-sperm correlation analysis
        logger.info("\n[Step 9] GSE149318 blood-sperm correlation analysis...")
        self.plot_correlation_histogram_gse149318()

        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)

        # Print summary
        logger.info("\nOutput files generated:")
        logger.info(f"  Figures: {local_config.FIGURES_DIR}")
        logger.info(f"  Results: {local_config.RESULTS_DIR}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DNA Methylation Classifier Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--step",
        type=str,
        choices=["train", "validate", "batch", "correlation", "pca"],
        help="Run specific pipeline step"
    )

    args = parser.parse_args()

    pipeline = ClassifierPipeline()

    if args.step:
        if args.step == "train":
            df = pipeline.load_primary_data()
            X_train, y_train, X_test, y_test = pipeline.split_train_test(df)
            features = pipeline.select_correlated_features(X_train)
            X_train_selected = X_train[features]
            X_test_selected = X_test[features]
            results = pipeline.train_and_evaluate(
                X_train_selected, y_train, X_test_selected, y_test
            )
            pipeline.export_model_performance(results, y_test)
            pipeline.plot_roc_curves(results, y_test)

        elif args.step == "validate":
            models, scaler, features, imputer_means = pipeline.train_final_models()
            ext_predictions = pipeline.validate_external(
                models, scaler, features, imputer_means
            )
            if ext_predictions is not None:
                ext_predictions.to_csv(local_config.PREDICTIONS_OUTPUT, index=False)

        elif args.step == "batch":
            pipeline.plot_pca_batch_effect()

        elif args.step == "correlation":
            corr_df = pipeline.compute_gse213366_probe_correlations()
            pipeline.plot_correlation_histogram_gse213366(corr_df)
            pipeline.plot_correlation_histogram_gse149318()

        elif args.step == "pca":
            df = pipeline.load_primary_data()
            X_train, y_train, X_test, y_test = pipeline.split_train_test(df)
            features = pipeline.select_correlated_features(X_train)
            # Note: PCA class separation would need to be implemented
            logger.info("PCA class separation - use full pipeline for this step")
    else:
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
