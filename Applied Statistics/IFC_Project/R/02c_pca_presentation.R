# =========================================================
# PCA PRESENTATION OUTPUTS
# Uses already generated PCA files
# =========================================================

# -----------------------------
# 1. Packages
# -----------------------------
if (!require("readxl")) install.packages("readxl")
if (!require("writexl")) install.packages("writexl")
if (!require("dplyr")) install.packages("dplyr")
if (!require("tidyr")) install.packages("tidyr")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("stringr")) install.packages("stringr")
if (!require("forcats")) install.packages("forcats")
if (!require("purrr")) install.packages("purrr")

library(readxl)
library(writexl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(forcats)
library(purrr)

# -----------------------------
# 2. File paths
# -----------------------------
combined_file <- "outputs/pca/PCA_All_Years_Combined.xlsx"     # produced by R/02_pca.R
summary_file  <- "outputs/pca/PCA_Interpretation_Summary.xlsx" # (may be produced manually)
source_file   <- "data/raw/ifc/composite_fragility_all_years.xlsx"

out_dir <- "outputs/pca"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------
# 3. Indicator name mapping
# -----------------------------
indicator_map <- tibble(
  Indicator = paste0("Ind", 1:12),
  Indicator_Name = c(
    "Motorisation rate with high emissions",
    "Undifferentiated municipal waste generated",
    "Protected natural areas",
    "Areas at risk of landslides",
    "Land consumption",
    "Accessibility to essential services",
    "Population dependency index",
    "Population aged 25-64 with low education",
    "Employment rate (20-64 years)",
    "Incidence of net migration",
    "Density of local units in industry/services",
    "Persons employed in low-productivity local units"
  )
)

# -----------------------------
# 4. Read combined PCA outputs
# -----------------------------
explained_all <- read_excel(combined_file, sheet = "Explained_First2_All_Years")
loadings_all  <- read_excel(combined_file, sheet = "Loadings_PC1_PC2_All_Years")
scores_all    <- read_excel(combined_file, sheet = "Scores_PC1_PC2_All_Years")

# -----------------------------
# 5. Read interpretation summary
# -----------------------------
compact_summary <- read_excel(summary_file, sheet = "Compact_Summary_For_Slides")

# -----------------------------
# 6. Create explained variance table
# -----------------------------
variance_table <- explained_all %>%
  mutate(
    Variance_Explained_Pct = round(Variance_Explained * 100, 2),
    Cumulative_Variance_Pct = round(Cumulative_Variance * 100, 2)
  ) %>%
  select(Year, PC, Variance_Explained_Pct, Cumulative_Variance_Pct) %>%
  pivot_wider(
    names_from = PC,
    values_from = c(Variance_Explained_Pct, Cumulative_Variance_Pct)
  ) %>%
  transmute(
    Year,
    PC1_Explained_Pct = Variance_Explained_Pct_PC1,
    PC2_Explained_Pct = Variance_Explained_Pct_PC2,
    PC1_PC2_Cumulative_Pct = Cumulative_Variance_Pct_PC2
  )

write_xlsx(
  list(Variance_Summary = variance_table),
  path = file.path(out_dir, "PCA_Variance_Summary.xlsx")
)

# -----------------------------
# 7. Combined explained variance plot
# -----------------------------
variance_plot_df <- explained_all %>%
  mutate(Variance_Explained_Pct = Variance_Explained * 100) %>%
  filter(PC %in% c("PC1", "PC2"))

p_variance <- ggplot(variance_plot_df, aes(x = Year, y = Variance_Explained_Pct, fill = PC)) +
  geom_col(position = "dodge") +
  theme_minimal(base_size = 13) +
  labs(
    title = "Explained Variance of PC1 and PC2 Across Years",
    x = "Year",
    y = "Explained Variance (%)"
  )

ggsave(
  filename = file.path(out_dir, "ExplainedVariance_PC1_PC2_AllYears.png"),
  plot = p_variance,
  width = 8,
  height = 5,
  dpi = 300
)

# -----------------------------
# 8. Add readable names to loadings
# -----------------------------
loadings_named <- loadings_all %>%
  left_join(indicator_map, by = "Indicator") %>%
  mutate(
    Abs_PC1 = abs(PC1),
    Abs_PC2 = abs(PC2)
  )

# -----------------------------
# 9. Top 5 loadings for PC1 by year
# -----------------------------
top5_pc1 <- loadings_named %>%
  group_by(Year) %>%
  arrange(desc(Abs_PC1), .by_group = TRUE) %>%
  slice(1:5) %>%
  ungroup()

p_pc1_loadings <- ggplot(
  top5_pc1,
  aes(x = fct_reorder(Indicator_Name, Abs_PC1), y = PC1, fill = Year)
) +
  geom_col() +
  coord_flip() +
  facet_wrap(~ Year, scales = "free_y") +
  theme_minimal(base_size = 12) +
  labs(
    title = "Top 5 Loadings for PC1 by Year",
    x = "Indicator",
    y = "PC1 Loading"
  )

ggsave(
  filename = file.path(out_dir, "Top5_Loadings_PC1_ByYear.png"),
  plot = p_pc1_loadings,
  width = 12,
  height = 8,
  dpi = 300
)

# -----------------------------
# 10. Top 5 loadings for PC2 by year
# -----------------------------
top5_pc2 <- loadings_named %>%
  group_by(Year) %>%
  arrange(desc(Abs_PC2), .by_group = TRUE) %>%
  slice(1:5) %>%
  ungroup()

p_pc2_loadings <- ggplot(
  top5_pc2,
  aes(x = fct_reorder(Indicator_Name, Abs_PC2), y = PC2, fill = Year)
) +
  geom_col() +
  coord_flip() +
  facet_wrap(~ Year, scales = "free_y") +
  theme_minimal(base_size = 12) +
  labs(
    title = "Top 5 Loadings for PC2 by Year",
    x = "Indicator",
    y = "PC2 Loading"
  )

ggsave(
  filename = file.path(out_dir, "Top5_Loadings_PC2_ByYear.png"),
  plot = p_pc2_loadings,
  width = 12,
  height = 8,
  dpi = 300
)

# -----------------------------
# 11. Read original decile data for colouring score plot
# -----------------------------
load_decile_data <- function(sheet_name) {
  df <- read_excel(
    path = source_file,
    sheet = sheet_name,
    skip = 2,
    col_names = FALSE,
    col_types = "text"
  )
  
  df <- df[, 1:15]
  colnames(df) <- c("PRO_COM", "Territory", "MFI_Dec", paste0("Ind", 1:12))
  
  df_clean <- df %>%
    filter(!is.na(PRO_COM), !is.na(Territory)) %>%
    mutate(
      PRO_COM = as.character(PRO_COM),
      Territory = as.character(Territory),
      MFI_Dec = suppressWarnings(as.numeric(MFI_Dec)),
      Year = as.character(sheet_name)
    ) %>%
    distinct(PRO_COM, Territory, Year, .keep_all = TRUE)
  
  return(df_clean %>% select(PRO_COM, Territory, MFI_Dec, Year))
}

years_to_read <- c("2018", "2019", "2021", "2022")
decile_all <- map_df(years_to_read, load_decile_data)

# -----------------------------
# 12. Improved score plot with fragility decile colour
# -----------------------------
scores_colored <- scores_all %>%
  left_join(decile_all, by = c("Year", "PRO_COM", "Territory")) %>%
  filter(!is.na(MFI_Dec))

# Choose ONE year for presentation
score_year <- "2021"

score_plot_df <- scores_colored %>%
  filter(Year == score_year)

p_score <- ggplot(score_plot_df, aes(x = PC1, y = PC2, color = MFI_Dec)) +
  geom_point(alpha = 0.45, size = 1.1) +
  theme_minimal(base_size = 13) +
  labs(
    title = paste("PC1 vs PC2 Score Plot Colored by Fragility Decile -", score_year),
    x = "PC1",
    y = "PC2",
    color = "Fragility Decile"
  )

ggsave(
  filename = file.path(out_dir, paste0("ScorePlot_PC1_PC2_Colored_", score_year, ".png")),
  plot = p_score,
  width = 8,
  height = 5,
  dpi = 300
)

# -----------------------------
# 13. Optional: highlight only extreme municipalities
# -----------------------------
score_extreme_df <- score_plot_df %>%
  mutate(
    Extreme_Group = case_when(
      MFI_Dec >= 9 ~ "High Fragility (Decile 9-10)",
      MFI_Dec <= 2 ~ "Low Fragility (Decile 1-2)",
      TRUE ~ "Middle"
    )
  )

p_score_extreme <- ggplot(
  score_extreme_df %>% filter(Extreme_Group != "Middle"),
  aes(x = PC1, y = PC2, color = Extreme_Group)
) +
  geom_point(alpha = 0.6, size = 1.2) +
  theme_minimal(base_size = 13) +
  labs(
    title = paste("Extreme Municipalities on PC1-PC2 Plane -", score_year),
    x = "PC1",
    y = "PC2",
    color = "Group"
  )

ggsave(
  filename = file.path(out_dir, paste0("ScorePlot_Extremes_", score_year, ".png")),
  plot = p_score_extreme,
  width = 8,
  height = 5,
  dpi = 300
)

# -----------------------------
# 14. Save top loading tables
# -----------------------------
write_xlsx(
  list(
    Variance_Summary = variance_table,
    Top5_PC1 = top5_pc1,
    Top5_PC2 = top5_pc2,
    Compact_Summary = compact_summary
  ),
  path = file.path(out_dir, "PCA_Presentation_Tables.xlsx")
)

# -----------------------------
# 15. Console summary
# -----------------------------
cat("\n====================================\n")
cat("Presentation PCA outputs created.\n")
cat("Saved in folder:", out_dir, "\n")
cat("Files created:\n")
cat("- PCA_Variance_Summary.xlsx\n")
cat("- ExplainedVariance_PC1_PC2_AllYears.png\n")
cat("- Top5_Loadings_PC1_ByYear.png\n")
cat("- Top5_Loadings_PC2_ByYear.png\n")
cat("- ScorePlot_PC1_PC2_Colored_", score_year, ".png\n", sep = "")
cat("- ScorePlot_Extremes_", score_year, ".png\n", sep = "")
cat("- PCA_Presentation_Tables.xlsx\n")
cat("====================================\n")