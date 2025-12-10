import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

# -----------------------------
# Script Configuration & Arguments
# -----------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot methylation line plots for specified tissue.")
    parser.add_argument("probe_file", help="Path to the probe CSV file (e.g., spatc1l_probes.csv)")
    parser.add_argument("--tissue", default="Blood", choices=["Blood", "Sperm"], help="Tissue type to plot (Blood or Sperm)")
    parser.add_argument("--output", "-o", help="Optional output filename for the PDF plot. If not provided, a default name based on the probe file and tissue will be used.")
    return parser.parse_args()

# -----------------------------
# Data Loading & Processing
# -----------------------------
def load_and_process_data(probe_file_path, tissue_type):
    # Hardcoded file paths as per requirements
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    # Check if files exist
    files = {
        "epic": "../../data/metadata/EPIC.hg38.txt",
        "matrix": "../../data/raw/GSE213366_matrix_cleaned.csv",
        "sample_map": "../../data/metadata/sample_tissue_individual_map.csv"
    }
    
    resolved_paths = {}
    
    for key, filename in files.items():
        # Resolve path relative to script
        file_path = os.path.join(base_dir, filename)
        
        if not os.path.exists(file_path):
             print(f"Error: Required file '{filename}' not found at {file_path}")
             sys.exit(1)
        resolved_paths[key] = file_path

    print(f"Loading data for {tissue_type}...")

    # 1. Load EPIC coordinates
    # Format: chr start end probe_id (tab separated, no header or header check needed?)
    # Based on `head` output: chr1 10524 10526 cg14817997
    epic = pd.read_csv(resolved_paths["epic"], sep="\t", header=None, names=["chr", "start", "end", "ID_REF"])
    
    # 2. Load Probes of Interest
    # Based on `head` output of spatc1l_probes.csv: 
    # chr21,46161490,46161492,+,cg11766577,...
    # The probe ID is in the *5th* column (index 4).
    try:
        probes = pd.read_csv(probe_file_path, header=None)
        target_probe_ids = probes[3].unique()
    except Exception as e:
        print(f"Error reading probe file: {e}")
        sys.exit(1)

    # 3. Load Methylation Matrix
    # This file might be large, so we could optimize, but loading full for now as per R script behavior.
    # We only need rows matching our target probes.
    print("Loading methylation matrix...")
    matrix_iter = pd.read_csv(resolved_paths["matrix"], chunksize=10000)
    matrix_list = []
    
    found_any = False
    for chunk in matrix_iter:
        filtered_chunk = chunk[chunk['ID_REF'].isin(target_probe_ids)]
        if not filtered_chunk.empty:
            matrix_list.append(filtered_chunk)
            found_any = True
    
    if not found_any:
        print("No matching probes found in methylation matrix.")
        sys.exit(1)
        
    data = pd.concat(matrix_list)

    # 4. Merge with EPIC to get coordinates
    # R: data <- merge(EPIC_data,data,by.x="V4",by.y="ID_REF")
    # EPIC V4 is ID_REF
    data = pd.merge(epic, data, on="ID_REF", how="inner")

    # 5. Order by coordinate (V2 in R -> start in our DF)
    data = data.sort_values(by="start")

    # 6. Create custom identifiers for columns if needed, but Python script logic
    # uses columns like "chr21.46161490_cg11766577"
    # R: paste0(data$V1,".",data$V2,"_",data$V4) -> chr.start_probeID
    # We will construct these identifiers to be the index or just use them for mapping later.
    # The python plotting script expects columns in the dataframe to be these identifiers.
    
    # Let's pivot/reshape. 
    # Current `data` structure:
    # chr, start, end, ID_REF, Sample1, Sample2, ...
    
    # We need to filter for samples of the specific tissue.
    sample_map = pd.read_csv(resolved_paths["sample_map"])
    
    # Extract list of sample IDs for the requested tissue
    # R: sample_map[sample_map$tissue=="Blood",3] -> sample_name column
    # The matrix columns usually have "X" prefix in R if they start with numbers, 
    # but in Python/Pandas read_csv, they stay as is unless they contain special chars.
    # `head` of matrix shows: 203219670028_R01C01 (no X).
    # `head` of sample_map shows: 203219670028_R01C01 (no X).
    
    target_samples = sample_map[sample_map['tissue'] == tissue_type]['sample_name'].tolist()
    
    # Verify these samples exist in our data columns
    valid_samples = [s for s in target_samples if s in data.columns]
    
    if not valid_samples:
        print(f"No samples found for tissue {tissue_type} in the matrix.")
        sys.exit(1)

    # Subset data to ID_REF + valid samples + coordinates for constructing headers
    # We need to construct the column headers for the final transposed DF: "chr.start_probeID"
    # The plotting script parses: c.split(".")[1].split("_")[0] -> start position
    
    # Create the new column names
    # Format: chr.start_probeID
    # e.g. chr21.46161490_cg11766577
    new_index_names = data['chr'] + "." + data['start'].astype(str) + "_" + data['ID_REF']
    
    # Subset just the sample data
    df_samples = data[valid_samples].copy()
    
    # Set the index to the new names so when we transpose they become columns
    df_samples.index = new_index_names
    
    # Transpose: Samples become rows, Probes become columns
    df_transposed = df_samples.T
    
    # 7. Merge with Metadata (Status)
    # We need to add 'status' to the rows.
    # R: merge(data_filtered_t, sample_map, by.x="sample_id", by.y="sample_name")
    
    # df_transposed index is sample_name
    df_transposed.index.name = 'sample_name'
    df_transposed = df_transposed.reset_index()
    
    # Merge
    # We only need 'sample_name' and 'status' from map
    meta = sample_map[['sample_name', 'status']]
    final_df = pd.merge(df_transposed, meta, on='sample_name')
    
    return final_df

# -----------------------------
# Plotting Logic
# -----------------------------
def plot_data(df, tissue_type, output_file):
    print(f"Plotting data for {tissue_type}...")
    
    # Identify methylation columns (those containing the probe data)
    # They follow the pattern chr.pos_id
    # We can exclude 'sample_name', 'status' and any other non-probe columns
    feature_cols = [c for c in df.columns if c not in ['sample_name', 'status', 'Individual_ID', 'tissue', 'geo_accession']]
    
    if not feature_cols:
        print("No probe data columns found for plotting.")
        return

    # Extract positions for sorting
    # Format: chr.start_probeID -> extract start
    try:
        positions = np.array([int(c.split(".")[1].split("_")[0]) for c in feature_cols])
    except Exception as e:
        print(f"Error parsing genomic positions from column names: {e}")
        # Fallback or debug
        print(f"First few columns: {feature_cols[:5]}")
        sys.exit(1)

    # Sort columns by position
    order = np.argsort(positions)
    positions = positions[order]
    sorted_cols = [feature_cols[i] for i in order]

    # Color mapping
    color_map = {
        "Normozoospermia": "green",
        "Oligozoospermia": "red"
    }

    plt.figure(figsize=(12, 4), dpi=300)

    # Plot each sample line
    for idx, row in df.iterrows():
        # Handle potential string/numeric mix. Ensure floats.
        try:
            y = row[sorted_cols].values.astype(float)
        except ValueError:
             # Handle cases where data might be missing or non-numeric
             # Coerce errors
             y = pd.to_numeric(row[sorted_cols], errors='coerce').values

        status = row.get("status", "Unknown")
        
        # Plot
        # Using simple plot (no smoothing function mentioned in this specific request, 
        # though original script had a helper. The original plotting loop didn't use the helper `smooth` in the main call?
        # Let's check the original script content...
        # Original script:
        # def smooth(y, window=3): ...
        # plt.plot(positions, y, ...) <- It used raw 'y' in the loop, NOT 'smooth(y)'.
        # I will stick to raw 'y' to match original behavior accurately.
        
        plt.plot(
            positions,
            y,
            color=color_map.get(status, "black"),
            alpha=0.6,
            linewidth=1
        )

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="green", lw=2, label="Normozoospermia"),
        Line2D([0], [0], color="red", lw=2, label="Oligozoospermia")
    ]
    plt.legend(handles=legend_handles, frameon=False, fontsize=11)

    plt.tight_layout()
    plt.xlabel("Genomic Position", fontsize=16)
    plt.ylabel("Methylation Level", fontsize=16)
    plt.title(f"Methylation Line Plot ({tissue_type})", fontsize=14)
    plt.tight_layout()

    # output_file logic moved to main or passed in
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    args = parse_arguments()
    
    # Process
    df = load_and_process_data(args.probe_file, args.tissue)
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        # Default: line_methylation_[probe_basename]_[tissue].pdf
        probe_basename = os.path.splitext(os.path.basename(args.probe_file))[0]
        output_file = f"line_methylation_{probe_basename}_{args.tissue}.pdf"
    
    # Plot
    plot_data(df, args.tissue, output_file)
