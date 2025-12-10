
import os
import subprocess
import argparse
import sys

# Define Paths relative to this script (assuming it's in project root or readable from there)
# If script is in root:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_META = os.path.join(BASE_DIR, "data", "metadata")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SCRIPTS_R = os.path.join(BASE_DIR, "scripts", "R")
SCRIPTS_PY = os.path.join(BASE_DIR, "scripts", "python")

def run_dmr_analysis(tissue, beta_file, sample_map):
    """Runs the R DMR analysis script."""
    print(f"\n--- Starting DMR Analysis for {tissue} ---")
    
    r_script = os.path.join(SCRIPTS_R, "dmr_analysis.R")
    output_file = os.path.join(RESULTS_DIR, "tables", f"dmr_results_{tissue}.csv")
    
    if not os.path.exists(r_script):
        print(f"Error: R script not found at {r_script}")
        return

    cmd = [
        "Rscript", 
        r_script, 
        beta_file, 
        sample_map, 
        tissue, 
        output_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"DMR analysis completed. Results: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running DMR analysis: {e}")

def run_distribution_analysis(methylation_file, sample_map):
    """Runs the Python distribution analysis (Density & PCA)."""
    print("\n--- Starting Distribution Analysis (Density & PCA) ---")
    
    # Import dynamically or run as subprocess if dependencies issues, 
    # but importing is better if in same env.
    sys.path.append(SCRIPTS_PY)
    try:
        import distribution_analysis
        
        # Load data
        df, sm = distribution_analysis.load_data(methylation_file, sample_map)
        
        # Plot
        output_dir = os.path.join(RESULTS_DIR, "plots")
        distribution_analysis.plot_density(df, sm, "Normozoospermia", 
                                           os.path.join(output_dir, "density_Normozoospermia_Blood_Sperm.pdf"))
        distribution_analysis.plot_pca(df, sm, os.path.join(output_dir, "pca_plot.pdf"))
        
        print("Distribution analysis completed.")
        
    except ImportError:
        print("Error: Could not import distribution_analysis.py")
    except Exception as e:
        print(f"Error in distribution analysis: {e}")

def main():
    parser = argparse.ArgumentParser(description="GSE213366 Analysis Pipeline")
    parser.add_argument("--skip-dmr", action="store_true", help="Skip DMR analysis")
    parser.add_argument("--skip-plots", action="store_true", help="Skip distribution plots")
    
    args = parser.parse_args()
    
    # Check Directories
    if not os.path.exists(os.path.join(RESULTS_DIR, "tables")):
        os.makedirs(os.path.join(RESULTS_DIR, "tables"))
    if not os.path.exists(os.path.join(RESULTS_DIR, "plots")):
        os.makedirs(os.path.join(RESULTS_DIR, "plots"))

    # File Paths
    # Note: dmr_analysis.R originally used GSE213366_CoRSIV_2023.csv
    beta_file_dmr = os.path.join(DATA_RAW, "GSE213366_CoRSIV_2023.csv")
    
    beta_file_dist = os.path.join(DATA_RAW, "GSE213366_matrix_processed.csv")
    
    sample_map = os.path.join(DATA_META, "sample_tissue_individual_map.csv")
    
    # Run DMR Analysis
    if not args.skip_dmr:
        run_dmr_analysis("Sperm", beta_file_dmr, sample_map)
        # Add other tissues if needed, e.g. run_dmr_analysis("Blood", ...)
        
    # Run Plots
    if not args.skip_plots:
        run_distribution_analysis(beta_file_dist, sample_map)

if __name__ == "__main__":
    main()
