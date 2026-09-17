# =========================================================
# IFC Integrated Analysis Script
# Function: Combines discrepancy analysis, problem lists, and range analysis
# Output: 1 Integrated Excel file + 1 Slide-optimized range plot
# =========================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))

load_or_install <- function(package_name) {
  if (!requireNamespace(package_name, quietly = TRUE)) {
    install.packages(package_name)
  }
  library(package_name, character.only = TRUE)
}

load_or_install("readxl")
load_or_install("writexl")
load_or_install("dplyr")
load_or_install("tidyr")
load_or_install("ggplot2")
load_or_install("forcats")

# -------------------------------
# 1. Configuration and Data Loading
# -------------------------------
input_file   <- "data/raw/ifc/final_analysis_sorted.xlsx"
dir.create("outputs/integrated_analysis_report", recursive = TRUE, showWarnings = FALSE)
output_excel <- "outputs/integrated_analysis_report/IFC_Integrated_Analysis_Report.xlsx"
output_plot  <- "outputs/integrated_analysis_report/range_comparison.png"

df <- read_excel(input_file)

# -------------------------------
# 2. Difference Calculation and Classification (Original: IFC_DIFF_ANY.R)[cite: 1]
# -------------------------------
detailed_analysis <- df %>%
  mutate(
    Diff_2019 = abs(Calc_Decile_2019 - MFI_Dec_2019),
    Diff_2021 = abs(Calc_Decile_2021 - MFI_Dec_2021)
  )

classify_diff <- function(x) {
  case_when(
    is.na(x) ~ NA_character_,
    x == 0 ~ "Exact match",
    x == 1 ~ "Minor mismatch (±1 decile)",
    x >= 2 ~ "Serious mismatch (>=2 deciles)"
  )
}

detailed_analysis <- detailed_analysis %>%
  mutate(
    Diff_Class_2019 = classify_diff(Diff_2019),
    Diff_Class_2021 = classify_diff(Diff_2021)
  )

# -------------------------------
# 3. Identify Problematic Municipalities (Original: IFC_DIFF_ANY2.R)[cite: 2]
# -------------------------------
problem_both_years <- detailed_analysis %>%
  filter(Diff_2019 >= 2 | Diff_2021 >= 2) %>%
  select(
    PRO_COM, Territory, 
    IFC_2019, Calc_Decile_2019, MFI_Dec_2019, Diff_2019,
    IFC_2021, Calc_Decile_2021, MFI_Dec_2021, Diff_2021
  ) %>%
  arrange(desc(pmax(coalesce(Diff_2019, 0), coalesce(Diff_2021, 0))))

# -------------------------------
# 4. Range and Frequency Analysis (Original: IFC_MIN-MAX_ANY.R)[cite: 5]
# -------------------------------
analyze_ranges_and_freq <- function(data, year) {
  ifc_col  <- paste0("IFC_", year)
  calc_col <- paste0("Calc_Decile_", year)
  mfi_col  <- paste0("MFI_Dec_", year)
  
  temp <- data %>%
    transmute(
      IFC = .data[[ifc_col]],
      Calc_Decile = .data[[calc_col]],
      MFI_Dec = .data[[mfi_col]],
      Diff = abs(Calc_Decile - MFI_Dec)
    ) %>%
    filter(!is.na(IFC))
  
  # Calculate IFC Min/Max per Decile (Summary)
  ranges <- bind_rows(
    temp %>% 
      group_by(Decile = Calc_Decile) %>% 
      summarise(IFC_min = min(IFC), IFC_max = max(IFC), N = n()) %>% 
      mutate(Type = "Reconstructed decile"),
    temp %>% 
      group_by(Decile = MFI_Dec) %>% 
      summarise(IFC_min = min(IFC), IFC_max = max(IFC), N = n()) %>% 
      mutate(Type = "Official decile")
  ) %>% 
    mutate(Year = year)
  
  return(ranges)
}

range_stats <- bind_rows(
  analyze_ranges_and_freq(detailed_analysis, "2019"), 
  analyze_ranges_and_freq(detailed_analysis, "2021")
)

# -------------------------------
# 5. Export Integrated Excel Report
# -------------------------------
write_xlsx(
  list(
    "Discrepancy_Analysis" = detailed_analysis,
    "Problem_Municipalities" = problem_both_years,
    "Decile_Ranges" = range_stats
  ),
  output_excel
)

# -------------------------------
# 6. Generate Range Comparison Plot (Original: IFC_MIN-MAX_PLOT.R)[cite: 6]
# -------------------------------
plot_data <- range_stats %>%
  mutate(
    Type = factor(Type, levels = c("Reconstructed decile", "Official decile")),
    Decile = factor(Decile, levels = 1:10),
    Year = factor(Year, levels = c("2019", "2021"))
  )

p_range <- ggplot(plot_data, aes(x = Decile, color = Type)) +
  # Range Lines
  geom_linerange(
    aes(ymin = IFC_min, ymax = IFC_max), 
    position = position_dodge(width = 0.5), 
    linewidth = 1.4
  ) +
  # Min Points
  geom_point(aes(y = IFC_min), position = position_dodge(width = 0.5), size = 2) +
  # Max Points
  geom_point(aes(y = IFC_max), position = position_dodge(width = 0.5), size = 2) +
  facet_wrap(~Year) +
  labs(
    title = "IFC Ranges by Decile: Reconstructed vs Official", 
    x = "Decile", 
    y = "IFC Value", 
    color = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom", 
    plot.title = element_text(face = "bold", hjust = 0.5)
  )

ggsave(output_plot, p_range, width = 10, height = 6, dpi = 300, bg = "white")

message("--- Analysis Completed ---")
message("Generated Excel file: ", output_excel)
message("Generated Plot file: ", output_plot)