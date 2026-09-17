# =========================================================
# OFFICIAL 2021 IFC DECILES vs GRINS TAXONOMY
# - Counts by detailed class x official decile
# - Counts by macroclass x official decile
# - Heatmap of counts
# - Heatmap of row percentages
# =========================================================

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(stringr)
  library(scales)
  library(forcats)
})

# ---------------------------------------------------------
# 1) FILE PATHS
# ---------------------------------------------------------
file_ifc <- "data/raw/ifc/final_analysis_sorted.xlsx"
file_tax <- "data/raw/grins/tassonomia_grins.xlsx"

OUT_DIR <- "outputs/confusion_matrix_2021"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------
# 2) READ IFC FILE (OFFICIAL DECILES)
# ---------------------------------------------------------
# If needed, change sheet = 1 to the correct sheet name
df_ifc <- read_excel(file_ifc, sheet = 1)

# Quick check of columns
cat("\nColumns in IFC file:\n")
print(names(df_ifc))

# Keep municipality code + OFFICIAL 2021 decile
ifc_2021 <- df_ifc %>%
  transmute(
    PRO_COM = as.numeric(PRO_COM),
    Decile_2021 = as.integer(MFI_Dec_2021)   # official deciles
  )

# ---------------------------------------------------------
# 3) READ TAXONOMY FILE
# ---------------------------------------------------------
# If needed, change sheet = 1 to the correct sheet name
df_tax <- read_excel(file_tax, sheet = 1)

cat("\nColumns in taxonomy file:\n")
print(names(df_tax))

taxonomy <- df_tax %>%
  transmute(
    PRO_COM = as.numeric(`Codice Istat del Comune 2021`),
    Macroclasse = `MACROCLASSE GRINS`,
    Classe = `CLASSE GRINS`
  )

# ---------------------------------------------------------
# 4) MERGE
# ---------------------------------------------------------
merged <- taxonomy %>%
  left_join(ifc_2021, by = "PRO_COM")

cat("\n========================================\n")
cat("CHECK MERGE\n")
cat("========================================\n")
cat("Taxonomy municipalities:         ", nrow(taxonomy), "\n")
cat("Municipalities with decile:      ", sum(!is.na(merged$Decile_2021)), "\n")
cat("Municipalities without decile:   ", sum(is.na(merged$Decile_2021)), "\n")
cat("Unique official deciles found:   ", paste(sort(unique(na.omit(merged$Decile_2021))), collapse = ", "), "\n")

# ---------------------------------------------------------
# 5) ORDER DETAILED CLASSES PROPERLY
# ---------------------------------------------------------
# This makes 1, 1.1, 1.2, 2, 2.1, 2.2, ... sort correctly
classe_ord_df <- tibble(Classe = unique(na.omit(merged$Classe))) %>%
  mutate(parts = str_split(Classe, "\\.")) %>%
  rowwise() %>%
  mutate(
    p1 = ifelse(length(parts) >= 1, as.numeric(parts[[1]]), NA_real_),
    p2 = ifelse(length(parts) >= 2, as.numeric(parts[[2]]), NA_real_),
    p3 = ifelse(length(parts) >= 3, as.numeric(parts[[3]]), NA_real_),
    p4 = ifelse(length(parts) >= 4, as.numeric(parts[[4]]), NA_real_)
  ) %>%
  ungroup() %>%
  arrange(p1, p2, p3, p4)

classe_levels_ordered <- classe_ord_df$Classe

merged <- merged %>%
  mutate(
    Classe = factor(Classe, levels = classe_levels_ordered),
    Decile_2021 = factor(Decile_2021, levels = 1:10)
  )

# ---------------------------------------------------------
# 6) CONTINGENCY TABLE - DETAILED CLASS x DECILE
# ---------------------------------------------------------
conf_class_long <- merged %>%
  filter(!is.na(Classe), !is.na(Decile_2021)) %>%
  count(Classe, Decile_2021, name = "n") %>%
  complete(Classe, Decile_2021, fill = list(n = 0))

conf_class_wide <- conf_class_long %>%
  mutate(Decile_2021 = paste0("Decile_", Decile_2021)) %>%
  pivot_wider(
    names_from = Decile_2021,
    values_from = n,
    values_fill = 0
  )

cat("\n========================================\n")
cat("CONTINGENCY TABLE: DETAILED CLASS x OFFICIAL 2021 DECILE\n")
cat("========================================\n")
print(conf_class_wide)

write.csv(
  conf_class_wide,
  file.path(OUT_DIR, "confusion_matrix_detailed_class_vs_official_decile_2021.csv"),
  row.names = FALSE
)

# ---------------------------------------------------------
# 7) CONTINGENCY TABLE - MACROCLASS x DECILE
# ---------------------------------------------------------
conf_macro_long <- merged %>%
  filter(!is.na(Macroclasse), !is.na(Decile_2021)) %>%
  count(Macroclasse, Decile_2021, name = "n") %>%
  complete(Macroclasse, Decile_2021, fill = list(n = 0))

conf_macro_wide <- conf_macro_long %>%
  mutate(Decile_2021 = paste0("Decile_", Decile_2021)) %>%
  pivot_wider(
    names_from = Decile_2021,
    values_from = n,
    values_fill = 0
  )

cat("\n========================================\n")
cat("CONTINGENCY TABLE: MACROCLASS x OFFICIAL 2021 DECILE\n")
cat("========================================\n")
print(conf_macro_wide)

write.csv(
  conf_macro_wide,
  file.path(OUT_DIR, "confusion_matrix_macroclass_vs_official_decile_2021.csv"),
  row.names = FALSE
)

# ---------------------------------------------------------
# 8) ROW PERCENTAGES - DETAILED CLASS x DECILE
# ---------------------------------------------------------
conf_class_rowpct <- conf_class_long %>%
  group_by(Classe) %>%
  mutate(
    row_total = sum(n),
    pct_row = ifelse(row_total > 0, n / row_total, 0)
  ) %>%
  ungroup()

conf_class_rowpct_wide <- conf_class_rowpct %>%
  mutate(Decile_2021 = paste0("Decile_", Decile_2021)) %>%
  select(Classe, Decile_2021, pct_row) %>%
  pivot_wider(
    names_from = Decile_2021,
    values_from = pct_row,
    values_fill = 0
  )

write.csv(
  conf_class_rowpct_wide,
  file.path(OUT_DIR, "row_percentages_detailed_class_vs_official_decile_2021.csv"),
  row.names = FALSE
)

# ---------------------------------------------------------
# 9) HEATMAP - COUNTS
# ---------------------------------------------------------
p_heat_counts <- ggplot(conf_class_long, aes(x = Decile_2021, y = Classe, fill = n)) +
  geom_tile(color = "white") +
  geom_text(aes(label = n), size = 3) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(
    title = "GRINS detailed class vs official IFC decile (2021) - counts",
    x = "Official IFC decile (2021)",
    y = "GRINS detailed class",
    fill = "Count"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )

ggsave(
  filename = file.path(OUT_DIR, "heatmap_counts_detailed_class_vs_official_decile_2021.png"),
  plot = p_heat_counts,
  width = 10,
  height = 8,
  dpi = 300
)

# ---------------------------------------------------------
# 10) HEATMAP - ROW PERCENTAGES
# ---------------------------------------------------------
p_heat_rowpct <- ggplot(conf_class_rowpct, aes(x = Decile_2021, y = Classe, fill = pct_row)) +
  geom_tile(color = "white") +
  geom_text(aes(label = percent(pct_row, accuracy = 0.1)), size = 3) +
  scale_fill_gradient(low = "white", high = "darkred", labels = percent_format(accuracy = 1)) +
  labs(
    title = "GRINS detailed class vs official IFC decile (2021) - row percentages",
    x = "Official IFC decile (2021)",
    y = "GRINS detailed class",
    fill = "Row %"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )

ggsave(
  filename = file.path(OUT_DIR, "heatmap_rowpercent_detailed_class_vs_official_decile_2021.png"),
  plot = p_heat_rowpct,
  width = 10,
  height = 8,
  dpi = 300
)

# ---------------------------------------------------------
# 11) OPTIONAL: MACROCLASS HEATMAP
# ---------------------------------------------------------
conf_macro_rowpct <- conf_macro_long %>%
  group_by(Macroclasse) %>%
  mutate(
    row_total = sum(n),
    pct_row = ifelse(row_total > 0, n / row_total, 0)
  ) %>%
  ungroup()

p_heat_macro <- ggplot(conf_macro_rowpct, aes(x = Decile_2021, y = Macroclasse, fill = pct_row)) +
  geom_tile(color = "white") +
  geom_text(aes(label = percent(pct_row, accuracy = 0.1)), size = 4) +
  scale_fill_gradient(low = "white", high = "forestgreen", labels = percent_format(accuracy = 1)) +
  labs(
    title = "GRINS macroclass vs official IFC decile (2021) - row percentages",
    x = "Official IFC decile (2021)",
    y = "GRINS macroclass",
    fill = "Row %"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank()
  )

ggsave(
  filename = file.path(OUT_DIR, "heatmap_rowpercent_macroclass_vs_official_decile_2021.png"),
  plot = p_heat_macro,
  width = 9,
  height = 5,
  dpi = 300
)

cat("\n========================================\n")
cat("ALL OUTPUTS SAVED IN:\n")
cat(normalizePath(OUT_DIR), "\n")
cat("========================================\n")