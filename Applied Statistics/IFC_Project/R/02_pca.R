# =========================================================
# PCA FOR IFC INDICATORS - YEAR BY YEAR
# =========================================================

# -----------------------------
# 1. Load required packages
# -----------------------------
if (!require("readxl")) install.packages("readxl")
if (!require("writexl")) install.packages("writexl")
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("ggplot2")) install.packages("ggplot2")

library(readxl)
library(writexl)
library(tidyverse)
library(ggplot2)

# -----------------------------
# 2. Configuration
# -----------------------------
file_path <- "data/raw/ifc/composite_fragility_all_years.xlsx"
years_to_read <- c("2018", "2019", "2021", "2022")

# output folder (creates a single folder; mixed plots + xlsx tables)
out_dir <- "outputs/pca"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------
# 3. Function to read and clean one sheet
# -----------------------------
load_clean_data <- function(sheet_name) {
  message(paste("Reading sheet:", sheet_name))
  
  # Read all as text to avoid Excel format problems
  df <- read_excel(
    path = file_path,
    sheet = sheet_name,
    skip = 2,
    col_names = FALSE,
    col_types = "text"
  )
  
  # Keep only first 15 columns:
  # PRO_COM, Territory, MFI_Dec, Ind1...Ind12
  df <- df[, 1:15]
  
  colnames(df) <- c("PRO_COM", "Territory", "MFI_Dec", paste0("Ind", 1:12))
  
  # Clean data
  df_clean <- df %>%
    filter(!is.na(PRO_COM), !is.na(Territory)) %>%
    mutate(
      PRO_COM = as.character(PRO_COM),
      Territory = as.character(Territory),
      MFI_Dec = as.character(MFI_Dec)
    ) %>%
    mutate(
      across(
        .cols = starts_with("Ind"),
        .fns  = ~ as.numeric(gsub(",", ".", gsub("\\s+", "", .x)))
      )
    ) %>%
    mutate(Year = as.character(sheet_name)) %>%
    distinct(PRO_COM, Territory, Year, .keep_all = TRUE)
  
  return(df_clean)
}

# -----------------------------
# 4. Load all years together
# -----------------------------
data_all <- map_df(years_to_read, load_clean_data)

# -----------------------------
# 5. Check data structure
# -----------------------------
message("Data loaded successfully.")
message(paste("Total rows:", nrow(data_all)))
message(paste("Total columns:", ncol(data_all)))

# -----------------------------
# 6. Reverse polarity
# -----------------------------
# According to methodology / your IFC script:
# Ind3  = protected areas
# Ind9  = employment
# Ind10 = population growth / migration-related indicator
# Ind11 = business density
#
# These are the indicators where high value means less fragility,
# so we multiply by -1 to align all indicators in the same direction:
# high value = higher fragility

cols_to_invert <- c("Ind3", "Ind9", "Ind10", "Ind11")

data_trans <- data_all %>%
  mutate(across(all_of(cols_to_invert), ~ - .x))

# -----------------------------
# 7. Missing values check
# -----------------------------
missing_summary <- data_trans %>%
  summarise(across(starts_with("Ind"), ~ sum(is.na(.))))

write_xlsx(
  list(Missing_Values = missing_summary),
  file.path(out_dir, "missing_values_summary.xlsx")
)

# -----------------------------
# 8. PCA function for one year
# -----------------------------
run_pca_one_year <- function(df_year, year_label) {
  
  message(paste("Running PCA for year:", year_label))
  
  # Keep needed columns
  pca_input <- df_year %>%
    select(PRO_COM, Territory, starts_with("Ind"))
  
  # Remove rows with any NA in indicator columns
  pca_input_complete <- pca_input %>%
    drop_na(starts_with("Ind"))
  
  # Keep numeric indicator matrix only
  indicator_data <- pca_input_complete %>%
    select(starts_with("Ind"))
  
  # Safety check
  if (nrow(indicator_data) < 2) {
    stop(paste("Not enough complete rows for PCA in year", year_label))
  }
  
  # -----------------------------
  # PCA
  # -----------------------------
  # center = TRUE  -> subtract mean
  # scale. = TRUE  -> standardize variables
  pca_res <- prcomp(indicator_data, center = TRUE, scale. = TRUE)
  
  # -----------------------------
  # Explained variance
  # -----------------------------
  explained_var <- (pca_res$sdev^2) / sum(pca_res$sdev^2)
  explained_df <- data.frame(
    Year = year_label,
    PC = paste0("PC", seq_along(explained_var)),
    Variance_Explained = explained_var,
    Cumulative_Variance = cumsum(explained_var)
  )
  
  # -----------------------------
  # Loadings
  # -----------------------------
  loadings_df <- as.data.frame(pca_res$rotation)
  loadings_df$Indicator <- rownames(loadings_df)
  loadings_df$Year <- year_label
  
  # Reorder columns
  loadings_df <- loadings_df %>%
    select(Year, Indicator, everything())
  
  # -----------------------------
  # Scores
  # -----------------------------
  scores_df <- as.data.frame(pca_res$x)
  scores_df$PRO_COM <- pca_input_complete$PRO_COM
  scores_df$Territory <- pca_input_complete$Territory
  scores_df$Year <- year_label
  
  scores_df <- scores_df %>%
    select(Year, PRO_COM, Territory, everything())
  
  # -----------------------------
  # Keep only first 2 components summary
  # -----------------------------
  explained_first2 <- explained_df %>%
    filter(PC %in% c("PC1", "PC2"))
  
  loadings_first2 <- loadings_df %>%
    select(Year, Indicator, PC1, PC2)
  
  scores_first2 <- scores_df %>%
    select(Year, PRO_COM, Territory, PC1, PC2)
  
  # -----------------------------
  # Scree plot
  # -----------------------------
  scree_plot <- ggplot(explained_df, aes(x = PC, y = Variance_Explained)) +
    geom_col() +
    theme_minimal(base_size = 13) +
    labs(
      title = paste("Scree Plot -", year_label),
      x = "Principal Component",
      y = "Proportion of Variance Explained"
    )
  
  # -----------------------------
  # PC1 vs PC2 score plot
  # -----------------------------
  score_plot <- ggplot(scores_first2, aes(x = PC1, y = PC2)) +
    geom_point(alpha = 0.5) +
    theme_minimal(base_size = 13) +
    labs(
      title = paste("PCA Score Plot -", year_label),
      x = "PC1",
      y = "PC2"
    )
  
  # -----------------------------
  # Contribution table for PC1 and PC2
  # -----------------------------
  contribution_df <- loadings_first2 %>%
    mutate(
      Abs_PC1 = abs(PC1),
      Abs_PC2 = abs(PC2)
    ) %>%
    arrange(desc(Abs_PC1))
  
  # -----------------------------
  # Save plots
  # -----------------------------
  ggsave(
    filename = file.path(out_dir, paste0("Scree_", year_label, ".png")),
    plot = scree_plot,
    width = 8,
    height = 5,
    dpi = 300
  )
  
  ggsave(
    filename = file.path(out_dir, paste0("ScorePlot_PC1_PC2_", year_label, ".png")),
    plot = score_plot,
    width = 8,
    height = 5,
    dpi = 300
  )
  
  # -----------------------------
  # Save year-specific Excel
  # -----------------------------
  write_xlsx(
    list(
      Explained_Variance = explained_df,
      Explained_First2 = explained_first2,
      Loadings_All = loadings_df,
      Loadings_PC1_PC2 = loadings_first2,
      Scores_All = scores_df,
      Scores_PC1_PC2 = scores_first2,
      Contributions = contribution_df
    ),
    path = file.path(out_dir, paste0("PCA_", year_label, ".xlsx"))
  )
  
  return(list(
    pca_object = pca_res,
    explained_df = explained_df,
    loadings_df = loadings_df,
    scores_df = scores_df,
    explained_first2 = explained_first2,
    loadings_first2 = loadings_first2,
    scores_first2 = scores_first2
  ))
}

# -----------------------------
# 9. Run PCA for each year
# -----------------------------
pca_results <- list()

for (yr in years_to_read) {
  df_year <- data_trans %>% filter(Year == yr)
  pca_results[[yr]] <- run_pca_one_year(df_year, yr)
}

# -----------------------------
# 10. Combine all years outputs
# -----------------------------
all_explained <- bind_rows(lapply(pca_results, function(x) x$explained_df))
all_loadings  <- bind_rows(lapply(pca_results, function(x) x$loadings_df))
all_scores    <- bind_rows(lapply(pca_results, function(x) x$scores_df))

all_explained_first2 <- bind_rows(lapply(pca_results, function(x) x$explained_first2))
all_loadings_first2  <- bind_rows(lapply(pca_results, function(x) x$loadings_first2))
all_scores_first2    <- bind_rows(lapply(pca_results, function(x) x$scores_first2))

# Save combined file
write_xlsx(
  list(
    Explained_All_Years = all_explained,
    Explained_First2_All_Years = all_explained_first2,
    Loadings_All_Years = all_loadings,
    Loadings_PC1_PC2_All_Years = all_loadings_first2,
    Scores_All_Years = all_scores,
    Scores_PC1_PC2_All_Years = all_scores_first2
  ),
  path = file.path(out_dir, "PCA_All_Years_Combined.xlsx")
)

# -----------------------------
# 11. Print summary for each year
# -----------------------------
for (yr in years_to_read) {
  cat("\n====================================\n")
  cat("YEAR:", yr, "\n")
  cat("====================================\n")
  
  tmp_exp <- pca_results[[yr]]$explained_first2
  
  cat("PC1 explained variance:", round(tmp_exp$Variance_Explained[tmp_exp$PC == "PC1"] * 100, 2), "%\n")
  cat("PC2 explained variance:", round(tmp_exp$Variance_Explained[tmp_exp$PC == "PC2"] * 100, 2), "%\n")
  cat("Cumulative variance (PC1 + PC2):", round(max(tmp_exp$Cumulative_Variance) * 100, 2), "%\n")
}

cat("\n====================================\n")
cat("PCA completed successfully.\n")
cat("Outputs saved in folder:", out_dir, "\n")
cat("====================================\n")