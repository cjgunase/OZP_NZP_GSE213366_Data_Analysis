# DMR Analysis Script (Refactored)
# Usage: Rscript dmr_analysis.R [beta_file] [sample_map] [tissue] [output_file]

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop("Usage: Rscript dmr_analysis.R <beta_file> <sample_map> <tissue> <output_file>")
}

BETA_FILE <- args[1]
SAMPLE_MAP_FILE <- args[2]
TARGET_TISSUE <- args[3] # e.g., "Sperm"
OUTPUT_FILE <- args[4]

message(paste("Running DMR analysis on tissue:", TARGET_TISSUE))
message(paste("Input Beta:", BETA_FILE))
message(paste("Sample Map:", SAMPLE_MAP_FILE))
message(paste("Output:", OUTPUT_FILE))

# Load libraries silently
suppressPackageStartupMessages({
  library(IlluminaHumanMethylationEPICanno.ilm10b5.hg38)
  library(DMRcate)
  library(limma)
  library(minfi)
  library(GenomicRanges)
})

# 1. Load Sample Map
case_control_stats <- read.csv(SAMPLE_MAP_FILE)

# Filter by tissue
case_control_stats <- case_control_stats[case_control_stats$tissue == TARGET_TISSUE, ]

if (nrow(case_control_stats) == 0) {
  stop(paste("No samples found for tissue:", TARGET_TISSUE))
}

# 2. Load Beta Matrix
# Assuming row.names = 1 as per original script
beta <- read.csv(file = BETA_FILE, row.names = 1)

# Filter columns
# Original script: beta <- beta[paste0("X",case_control_stats$sample_name)]
# We need to robustly specific columns.
# Check if columns start with X or not.
first_col <- colnames(beta)[1]
target_cols <- paste0("X", case_control_stats$sample_name)

# If standard colnames don't have X, try without.
if (!all(target_cols %in% colnames(beta))) {
  if (all(case_control_stats$sample_name %in% colnames(beta))) {
    target_cols <- case_control_stats$sample_name
  } else {
    # Fallback or error
    # Try matching numeric part?
    # For now, stick to original logic if possible, or warn.
    message("Warning: Column names might not match sample map entries perfectly. Attempting strict match.")
    missing <- target_cols[!target_cols %in% colnames(beta)]
    if (length(missing) > 0) message(paste("Missing samples:", head(missing)))
  }
}

beta <- beta[, target_cols, drop = FALSE]
beta <- as.matrix(beta)

stopifnot(all(beta >= 0 & beta <= 1, na.rm = TRUE))

# 3. Prepare for DMRcate
Mval <- log2(beta / (1 - beta))

# Design Matrix
# Assuming comparing 'status' (Normozoospermia vs Oligozoospermia)
# Adjust specific comparison if needed.
group <- factor(case_control_stats$status)
design <- model.matrix(~group)

# Annotation
annotation <- getAnnotation(IlluminaHumanMethylationEPICanno.ilm10b5.hg38)
annotation <- annotation[rownames(Mval), ]

# GenomicRatioSet (Optional for DMRcate input depending on method, but used in original)
# Original used makeGenomicRatioSetFromMatrix but DMRcate often takes matrix object directly if annotated?
# Original: cpg.annotate takes 'object' which is Mval matrix.

# 4. Run Analysis
message("Annotating CpGs...")
# datatype = "array" for EPIC
myAnnotation <- cpg.annotate(
  object = Mval, datatype = "array", what = "M",
  analysis.type = "differential", design = design,
  coef = 2, arraytype = "EPIC", pcutoff = 1
)

message("Running DMRcate...")
dmrcoutput <- dmrcate(myAnnotation, lambda = 1000, C = 2, pcutoff = 1)

# Extract Ranges
results.ranges <- extractRanges(dmrcoutput, genome = "hg38")

# Map Probes to DMRs
cpg_gr <- myAnnotation@ranges
hits <- findOverlaps(results.ranges, cpg_gr)

dmr_probes <- split(names(cpg_gr)[subjectHits(hits)], queryHits(hits))
names(dmr_probes) <- paste0("DMR_", names(dmr_probes))

# Add probes column
mcols(results.ranges)$probes <- sapply(dmr_probes, function(x) paste(x, collapse = ";"))

# 5. Save Results
output_results <- data.frame(results.ranges)
write.csv(output_results, file = OUTPUT_FILE, row.names = FALSE, quote = FALSE)

message(paste("DMR analysis complete. Results saved to", OUTPUT_FILE))
