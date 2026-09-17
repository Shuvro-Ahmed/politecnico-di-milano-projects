# =========================================================
# GENERATE THRESHOLD-BASED PCA EXPLAINED VARIANCE TABLE
# One row per year:
# - PCs needed to reach 70%
# - PCs needed to reach 80%
# - Breakdown of explained variance by PC
# =========================================================

# -----------------------------
# 1. Packages
# -----------------------------
if (!require("readxl")) install.packages("readxl")
if (!require("writexl")) install.packages("writexl")
if (!require("dplyr")) install.packages("dplyr")
if (!require("stringr")) install.packages("stringr")
if (!require("purrr")) install.packages("purrr")

library(readxl)
library(writexl)
library(dplyr)
library(stringr)
library(purrr)

# -----------------------------
# 2. File paths
# -----------------------------
combined_file <- "outputs/pca/PCA_All_Years_Combined.xlsx"   # produced by R/02_pca.R

out_dir <- "outputs/pca/tables"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------
# 3. Read explained variance
# -----------------------------
explained_all <- read_excel(combined_file, sheet = "Explained_All_Years") %>%
  mutate(
    PC_num = as.numeric(str_remove(PC, "PC")),
    Variance_Explained_Pct = round(Variance_Explained * 100, 2),
    Cumulative_Variance_Pct = round(Cumulative_Variance * 100, 2)
  ) %>%
  arrange(Year, PC_num)

# -----------------------------
# 4. Helper function:
#    build one summary row for one year
# -----------------------------
build_threshold_row <- function(df_year) {
  
  df_year <- df_year %>% arrange(PC_num)
  
  # PCs needed to reach thresholds
  pc_70_row <- df_year %>% filter(Cumulative_Variance_Pct >= 70) %>% slice(1)
  pc_80_row <- df_year %>% filter(Cumulative_Variance_Pct >= 80) %>% slice(1)
  
  pc_70_num <- if (nrow(pc_70_row) > 0) pc_70_row$PC_num else NA
  pc_80_num <- if (nrow(pc_80_row) > 0) pc_80_row$PC_num else NA
  
  cum_70 <- if (nrow(pc_70_row) > 0) pc_70_row$Cumulative_Variance_Pct else NA
  cum_80 <- if (nrow(pc_80_row) > 0) pc_80_row$Cumulative_Variance_Pct else NA
  
  # Breakdown string up to threshold 70
  breakdown_70 <- if (!is.na(pc_70_num)) {
    df_year %>%
      filter(PC_num <= pc_70_num) %>%
      mutate(txt = paste0(PC, "=", Variance_Explained_Pct, "%")) %>%
      pull(txt) %>%
      paste(collapse = " | ")
  } else {
    NA_character_
  }
  
  # Breakdown string up to threshold 80
  breakdown_80 <- if (!is.na(pc_80_num)) {
    df_year %>%
      filter(PC_num <= pc_80_num) %>%
      mutate(txt = paste0(PC, "=", Variance_Explained_Pct, "%")) %>%
      pull(txt) %>%
      paste(collapse = " | ")
  } else {
    NA_character_
  }
  
  # Also keep first 6 PCs separately for cleaner viewing
  pc_vals <- df_year %>%
    select(PC_num, Variance_Explained_Pct) %>%
    slice(1:6)
  
  out <- tibble(
    Year = unique(df_year$Year),
    PCs_Needed_for_70pct = pc_70_num,
    Cumulative_at_70pct = cum_70,
    Breakdown_to_70pct = breakdown_70,
    PCs_Needed_for_80pct = pc_80_num,
    Cumulative_at_80pct = cum_80,
    Breakdown_to_80pct = breakdown_80
  )
  
  # add PC1...PC6 as columns
  for (i in 1:6) {
    val <- pc_vals %>% filter(PC_num == i) %>% pull(Variance_Explained_Pct)
    out[[paste0("PC", i, "_Pct")]] <- if (length(val) > 0) val else NA
  }
  
  out
}

# -----------------------------
# 5. Build final table
# -----------------------------
threshold_summary <- explained_all %>%
  group_by(Year) %>%
  group_split() %>%
  map_dfr(build_threshold_row) %>%
  ungroup()

# -----------------------------
# 6. Save
# -----------------------------
write_xlsx(
  list(
    Threshold_Summary = threshold_summary,
    Explained_Long = explained_all
  ),
  path = file.path(out_dir, "PCA_Threshold_Summary.xlsx")
)

# -----------------------------
# 7. Print preview
# -----------------------------
cat("\n====================================\n")
cat("PCA threshold summary table created.\n")
cat("Saved file:\n")
cat(file.path(out_dir, "PCA_Threshold_Summary.xlsx"), "\n")
cat("====================================\n\n")

print(threshold_summary)