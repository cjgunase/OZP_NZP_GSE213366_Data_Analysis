#!/usr/bin/env Rscript

# Compare DMRs between Blood and Sperm tissues
# Uses bedtools for genomic overlap analysis

library(data.table)
library(ggplot2)
library(VennDiagram)

# ============================================================================
# PARAMETERS - Adjust these cutoffs as needed
# ============================================================================
FDR_CUTOFF <- 0.05
MEANDIFF_CUTOFF <- 0.05  # Absolute value threshold

# ============================================================================
# 1. Load DMR data
# ============================================================================
cat("Loading DMR data...\n")
blood_dmr <- fread("CoRSIVs/CORSIV_normalvsOZS_DMR_in_Blood.txt")
sperm_dmr <- fread("CoRSIVs/CORSIV_normalvsOZS_DMR_in_Sperm.txt")

cat("Total DMRs before filtering:\n")
cat("  Blood:", nrow(blood_dmr), "\n")
cat("  Sperm:", nrow(sperm_dmr), "\n\n")

# ============================================================================
# 2. Filter DMRs based on FDR and mean difference cutoffs
# ============================================================================
cat("Filtering DMRs with FDR <", FDR_CUTOFF, "and |meandiff| >", MEANDIFF_CUTOFF, "\n")

blood_filt <- blood_dmr[min_smoothed_fdr < FDR_CUTOFF & abs(meandiff) > MEANDIFF_CUTOFF]
sperm_filt <- sperm_dmr[min_smoothed_fdr < FDR_CUTOFF & abs(meandiff) > MEANDIFF_CUTOFF]

cat("Filtered DMRs:\n")
cat("  Blood:", nrow(blood_filt), "\n")
cat("  Sperm:", nrow(sperm_filt), "\n\n")

# ============================================================================
# 3. Create BED files for bedtools
# ============================================================================
cat("Creating BED files for bedtools...\n")

# BED format: chr start end name score strand
blood_bed <- blood_filt[, .(seqnames, start, end,
                             name = paste0("blood_", 1:.N),
                             score = meandiff,
                             strand)]
sperm_bed <- sperm_filt[, .(seqnames, start, end,
                             name = paste0("sperm_", 1:.N),
                             score = meandiff,
                             strand)]

# Write BED files
fwrite(blood_bed, "blood_dmr_filtered.bed", sep = "\t", col.names = FALSE)
fwrite(sperm_bed, "sperm_dmr_filtered.bed", sep = "\t", col.names = FALSE)

cat("BED files created:\n")
cat("  blood_dmr_filtered.bed\n")
cat("  sperm_dmr_filtered.bed\n\n")

# ============================================================================
# 4. Use bedtools to find overlaps
# ============================================================================
cat("Running bedtools intersect...\n")

# Find DMRs that overlap between blood and sperm
system("bedtools intersect -a blood_dmr_filtered.bed -b sperm_dmr_filtered.bed -wa -wb > blood_sperm_overlap.bed")

# Find blood-specific DMRs (no overlap with sperm)
system("bedtools intersect -a blood_dmr_filtered.bed -b sperm_dmr_filtered.bed -v > blood_specific.bed")

# Find sperm-specific DMRs (no overlap with blood)
system("bedtools intersect -a sperm_dmr_filtered.bed -b blood_dmr_filtered.bed -v > sperm_specific.bed")

# ============================================================================
# 5. Read overlap results and calculate statistics
# ============================================================================
cat("\nReading overlap results...\n")

# Count overlaps
overlap <- fread("blood_sperm_overlap.bed", header = FALSE)
blood_specific <- fread("blood_specific.bed", header = FALSE)
sperm_specific <- fread("sperm_specific.bed", header = FALSE)

# Get unique overlapping regions
n_overlap_blood <- length(unique(overlap$V4))  # Unique blood DMR names in overlap
n_blood_specific <- nrow(blood_specific)
n_sperm_specific <- nrow(sperm_specific)

cat("\n============================================================================\n")
cat("RESULTS SUMMARY\n")
cat("============================================================================\n")
cat("Filtering criteria:\n")
cat("  FDR <", FDR_CUTOFF, "\n")
cat("  |Mean Difference| >", MEANDIFF_CUTOFF, "\n\n")

cat("DMR counts:\n")
cat("  Total Blood DMRs (filtered):", nrow(blood_filt), "\n")
cat("  Total Sperm DMRs (filtered):", nrow(sperm_filt), "\n\n")

cat("Overlap analysis:\n")
cat("  Blood DMRs overlapping with Sperm:", n_overlap_blood, "\n")
cat("  Blood-specific DMRs:", n_blood_specific, "\n")
cat("  Sperm-specific DMRs:", n_sperm_specific, "\n\n")

# ============================================================================
# 6. Create Venn Diagram
# ============================================================================
cat("Creating Venn diagram...\n")

venn.plot <- draw.pairwise.venn(
  area1 = nrow(blood_filt),
  area2 = nrow(sperm_filt),
  cross.area = n_overlap_blood,
  category = c("Blood", "Sperm"),
  fill = c("#E41A1C", "#377EB8"),
  lty = "blank",
  cex = 2,
  cat.cex = 2,
  cat.pos = c(330, 30),
  cat.dist = 0.05,
  main = paste0("DMR Overlap (FDR<", FDR_CUTOFF, ", |meandiff|>", MEANDIFF_CUTOFF, ")")
)

pdf("dmr_venn_diagram.pdf", width = 8, height = 8)
grid.draw(venn.plot)
dev.off()

cat("Venn diagram saved to: dmr_venn_diagram.pdf\n\n")

# ============================================================================
# 7. Detailed overlap table with annotations
# ============================================================================
cat("Creating detailed overlap table...\n")

if (nrow(overlap) > 0) {
  # Parse overlap file (12 columns: 6 from blood + 6 from sperm)
  colnames(overlap) <- c("blood_chr", "blood_start", "blood_end", "blood_name",
                          "blood_meandiff", "blood_strand",
                          "sperm_chr", "sperm_start", "sperm_end", "sperm_name",
                          "sperm_meandiff", "sperm_strand")

  # Extract indices from names
  overlap[, blood_idx := as.integer(gsub("blood_", "", blood_name))]
  overlap[, sperm_idx := as.integer(gsub("sperm_", "", sperm_name))]

  # Add gene annotations
  overlap[, blood_genes := blood_filt$overlapping.genes[blood_idx]]
  overlap[, sperm_genes := sperm_filt$overlapping.genes[sperm_idx]]
  
  overlap[, blood_probes := blood_filt$probes[blood_idx]]
  overlap[, sperm_probes := sperm_filt$probes[sperm_idx]]
  
  overlap[, blood_fdr := blood_filt$min_smoothed_fdr[blood_idx]]
  overlap[, sperm_fdr := sperm_filt$min_smoothed_fdr[sperm_idx]]

  # Calculate concordance (same direction of methylation change)
  overlap[, concordant := sign(blood_meandiff) == sign(sperm_meandiff)]

  cat("Concordance analysis:\n")
  cat("  Concordant DMRs (same direction):", sum(overlap$concordant), "\n")
  cat("  Discordant DMRs (opposite direction):", sum(!overlap$concordant), "\n\n")

  # Save detailed overlap table
  fwrite(overlap, "dmr_overlap_detailed.txt", sep = "\t")
  cat("Detailed overlap table saved to: dmr_overlap_detailed.txt\n\n")
}

# ============================================================================
# 8. Summary statistics
# ============================================================================
cat("Creating summary statistics...\n")

summary_stats <- data.frame(
  Tissue = c("Blood", "Sperm", "Overlap"),
  Total_DMRs = c(nrow(blood_filt), nrow(sperm_filt), n_overlap_blood),
  Hypermethylated = c(
    sum(blood_filt$meandiff > 0),
    sum(sperm_filt$meandiff > 0),
    if(nrow(overlap) > 0) sum(overlap$blood_meandiff > 0) else 0
  ),
  Hypomethylated = c(
    sum(blood_filt$meandiff < 0),
    sum(sperm_filt$meandiff < 0),
    if(nrow(overlap) > 0) sum(overlap$blood_meandiff < 0) else 0
  ),
  Mean_MeanDiff = c(
    mean(blood_filt$meandiff),
    mean(sperm_filt$meandiff),
    if(nrow(overlap) > 0) mean(overlap$blood_meandiff) else NA
  ),
  Median_Width = c(
    median(blood_filt$width),
    median(sperm_filt$width),
    if(nrow(overlap) > 0) median(overlap$blood_end - overlap$blood_start) else NA
  )
)

print(summary_stats)
fwrite(summary_stats, "dmr_summary_statistics.txt", sep = "\t")
cat("\nSummary statistics saved to: dmr_summary_statistics.txt\n\n")

# ============================================================================
# 9. Visualizations
# ============================================================================
cat("Creating visualizations...\n")

# Plot 1: Distribution of mean differences
pdf("dmr_meandiff_distribution.pdf", width = 10, height = 6)
par(mfrow = c(1, 2))
hist(blood_filt$meandiff, breaks = 50, main = "Blood DMR Mean Difference",
     xlab = "Mean Difference", col = "#E41A1C", border = "white")
hist(sperm_filt$meandiff, breaks = 50, main = "Sperm DMR Mean Difference",
     xlab = "Mean Difference", col = "#377EB8", border = "white")
dev.off()

# Plot 2: DMR width distributions
pdf("dmr_width_distribution.pdf", width = 10, height = 6)
par(mfrow = c(1, 2))
hist(log10(blood_filt$width), breaks = 50, main = "Blood DMR Width (log10)",
     xlab = "log10(Width in bp)", col = "#E41A1C", border = "white")
hist(log10(sperm_filt$width), breaks = 50, main = "Sperm DMR Width (log10)",
     xlab = "log10(Width in bp)", col = "#377EB8", border = "white")
dev.off()

# Plot 3: Scatter plot of overlapping DMRs (if any)
if (nrow(overlap) > 0) {
  pdf("dmr_overlap_scatter.pdf", width = 8, height = 8)
  plot(overlap$blood_meandiff, overlap$sperm_meandiff,
       xlab = "Blood Mean Difference", ylab = "Sperm Mean Difference",
       main = "Overlapping DMRs: Blood vs Sperm",
       pch = 19, col = ifelse(overlap$concordant, "#4DAF4A", "#FF7F00"),
       cex = 0.8)
  abline(h = 0, v = 0, lty = 2, col = "gray")
  abline(a = 0, b = 1, lty = 2, col = "blue")
  legend("topleft", c("Concordant", "Discordant"),
         col = c("#4DAF4A", "#FF7F00"), pch = 19)
  dev.off()
  cat("Overlap scatter plot saved to: dmr_overlap_scatter.pdf\n")
}

cat("\n============================================================================\n")
cat("Analysis complete!\n")
cat("============================================================================\n")
cat("\nOutput files created:\n")
cat("  1. blood_dmr_filtered.bed - Filtered blood DMRs in BED format\n")
cat("  2. sperm_dmr_filtered.bed - Filtered sperm DMRs in BED format\n")
cat("  3. blood_sperm_overlap.bed - Overlapping DMRs\n")
cat("  4. blood_specific.bed - Blood-specific DMRs\n")
cat("  5. sperm_specific.bed - Sperm-specific DMRs\n")
cat("  6. dmr_overlap_detailed.txt - Detailed annotation of overlaps\n")
cat("  7. dmr_summary_statistics.txt - Summary statistics table\n")
cat("  8. dmr_venn_diagram.pdf - Venn diagram visualization\n")
cat("  9. dmr_meandiff_distribution.pdf - Distribution plots\n")
cat(" 10. dmr_width_distribution.pdf - DMR width distributions\n")
if (nrow(overlap) > 0) {
  cat(" 11. dmr_overlap_scatter.pdf - Scatter plot of overlapping DMRs\n")
}
cat("\n")
