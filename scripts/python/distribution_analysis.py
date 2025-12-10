
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse
import os
import sys

# Default Configuration
DEFAULT_METHYLATION_FILE = "../../data/raw/GSE213366_matrix_processed.csv"
DEFAULT_SAMPLE_MAP = "../../data/metadata/sample_tissue_individual_map.csv"

def load_data(methylation_file, sample_map_file):
    """Loads methylation data and sample map."""
    print(f"Loading methylation data from {methylation_file}...")
    
    if not os.path.exists(methylation_file):
        print(f"Error: Methylation file not found at {methylation_file}")
        sys.exit(1)
        
    if not os.path.exists(sample_map_file):
        print(f"Error: Sample map file not found at {sample_map_file}")
        sys.exit(1)

    # Load Sample Map
    sample_map = pd.read_csv(sample_map_file)
    
    # Load Methylation Matrix
    # Using chunking or optimization if needed, but standard read_csv for now
    # The R script filtered columns: seq(1, 142, 2). This means it took columns 1, 3, 5... (1-based in R)
    # Python is 0-based. So 0, 2, 4...
    # However, usually column 0 is ID_REF.
    # Let's read the header first to understand keys.
    header_df = pd.read_csv(methylation_file, nrows=5)
    
    # Assuming column 0 is ID, and data starts from col 1.
    # If the R script used seq(1, 142, 2) on the data.table output, it might have been specific columns.
    # But usually we want all samples or specific samples.
    # Let's load the whole thing or relevant columns matching sample map.
    
    # Optimization: Read only columns present in sample_map
    # But files might have different naming conventions (X prefix etc).
    # Let's read the full file for now as in the original R loop logic (fread), 
    # but filter columns to match sample map immediately.
    
    try:
        # Check if file is huge; if so, maybe minimal columns?
        # The user said "large files", so mapped read might be better, but 'usecols' works well.
        # First read one row to get columns
        cols = pd.read_csv(methylation_file, nrows=0).columns.tolist()
        
        # Identify sample columns
        # R script: beta <- beta[paste0("X",case_control_stats$sample_name)] (in dmr_analysis)
        # But in plot_dist.R it used indices? No, it used exact match "sample_name".
        
        valid_cols = [c for c in cols if c in sample_map['sample_name'].values]
        if not valid_cols:
             # Try X prefix
            valid_cols_x = [c for c in cols if c[1:] in sample_map['sample_name'].values and c.startswith('X')]
            # If we found X-prefixed columns, we need to handle mapping.
            # But let's stick to simple first.
        
        # If we have valid_cols, use them + index col (usually first one)
        use_cols = [cols[0]] + valid_cols
        
        df = pd.read_csv(methylation_file, usecols=use_cols, index_col=0)
        
    except Exception as e:
        print(f"Error reading methylation file: {e}")
        sys.exit(1)
        
    return df, sample_map

def plot_density(df, sample_map, target_state, output_file):
    """Plots density of methylation values."""
    print(f"Generating density plot for state: {target_state}...")
    
    # Filter map for target state
    # Tissues: Blood and Sperm
    subset_map = sample_map[
        (sample_map['state'] == target_state) & 
        (sample_map['tissue'].isin(['Blood', 'Sperm']))
    ]
    
    if subset_map.empty:
        print("No samples found for density plot criteria.")
        return

    plt.figure(figsize=(10, 8))
    
    colors = {'Blood': 'red', 'Sperm': 'green'}
    
    for _, row in subset_map.iterrows():
        sample = row['sample_name']
        tissue = row['tissue']
        
        if sample in df.columns:
            data = df[sample].dropna()
            sns.kdeplot(data, color=colors.get(tissue, 'black'), linewidth=1, label=tissue)
    
    # Legend deduplication
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.title(f"Density of Methylation ({target_state} - Blood & Sperm)")
    plt.xlabel("Methylation Level")
    plt.ylabel("Density")
    plt.xlim(0, 1) # Generally methylation is 0-1
    
    plt.savefig(output_file)
    plt.close()
    print(f"Density plot saved to {output_file}")

def plot_pca(df, sample_map, output_file):
    """Performs PCA and plots 2D results."""
    print("Performing PCA...")
    
    # Transpose: Samples as rows, Probes as columns
    # Filter to common samples
    common_samples = [s for s in df.columns if s in sample_map['sample_name'].values]
    
    if len(common_samples) < 3:
        print("Not enough samples for PCA.")
        return
        
    data = df[common_samples].T
    
    # Remove columns (probes) with NA
    data_clean = data.dropna(axis=1)
    
    if data_clean.shape[1] == 0:
        print("No probes left after NA removal for PCA.")
        return

    # PCA
    pca = PCA(n_components=2)
    # Standardize features? Usually yes for PCA, though beta values are already 0-1.
    # In R script: prcomp(scale. = TRUE) -> yes.
    X_std = StandardScaler().fit_transform(data_clean)
    coords = pca.fit_transform(X_std)
    
    pca_df = pd.DataFrame(coords, columns=['PC1', 'PC2'], index=data_clean.index)
    
    # Add metadata
    pca_df = pca_df.merge(sample_map[['sample_name', 'tissue', 'state']], left_index=True, right_on='sample_name')
    
    # Calculate Variance
    var_exp = pca.explained_variance_ratio_ * 100
    
    # Plot
    plt.figure(figsize=(8, 8))
    
    # Colors: 
    # Normo-Blood: Red, Normo-Sperm: Green
    # Oligo-Blood: Orange, Oligo-Sperm: Blue
    
    colors = []
    for _, row in pca_df.iterrows():
        if row['state'] == 'Normozoospermia':
            colors.append('red' if row['tissue'] == 'Blood' else 'green')
        else: # Oligo
            colors.append('orange' if row['tissue'] == 'Blood' else 'blue')
            
    pca_df['color'] = colors
    
    plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['color'], s=100, alpha=0.8, edgecolors='w')
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Normo-Blood', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Normo-Sperm', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Oligo-Blood', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Oligo-Sperm', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    plt.ylabel(f"PC2 ({var_exp[1]:.1f}%)")
    plt.title("PCA of Methylation Data")
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"PCA plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Methylation Distribution Analysis (Density & PCA)")
    parser.add_argument("--methylation_file", default=DEFAULT_METHYLATION_FILE, help="Path to processed methylation matrix")
    parser.add_argument("--sample_map", default=DEFAULT_SAMPLE_MAP, help="Path to sample map CSV")
    parser.add_argument("--output_dir", default="../../results/plots", help="Directory to save plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    df, sample_map = load_data(args.methylation_file, args.sample_map)
    
    # Density Plot (Normozoospermia Blood vs Sperm)
    plot_density(df, sample_map, "Normozoospermia", os.path.join(args.output_dir, "density_Normozoospermia_Blood_Sperm.pdf"))
    
    # PCA Plot (All samples)
    plot_pca(df, sample_map, os.path.join(args.output_dir, "pca_plot.pdf"))
