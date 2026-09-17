# ============================================================
# REGRESSION TASK (R VERSION)
# IFC Decile ~ GRINS Macroclass * Year
# Years: 2018, 2019, 2021, 2022
# ============================================================

# -----------------------------
# 1. Libraries
# -----------------------------
library(readxl)
library(dplyr)
library(ggplot2)
library(writexl)

# -----------------------------
# 2. File paths
# -----------------------------
ifc_file   <- "data/raw/ifc/composite_fragility_all_years.xlsx"
grins_file <- "data/raw/grins/tassonomia_grins.xlsx"
out_dir    <- "outputs/regression_basic"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

years <- c(2018, 2019, 2021, 2022)

# -----------------------------
# 3. Read IFC data for all years
# -----------------------------
all_data <- list()

for (year in years) {
  
  df <- read_excel(ifc_file, sheet = as.character(year))
  
  # Remove second header row
  df <- df[-1, ]
  
  # Rename columns
  colnames(df)[1] <- "PRO_COM"
  colnames(df)[2] <- "Municipality"
  colnames(df)[3] <- "IFC_decile"
  
  df <- df %>%
    select(PRO_COM, Municipality, IFC_decile)
  
  df <- df %>%
    mutate(
      PRO_COM = as.numeric(PRO_COM),
      IFC_decile = as.numeric(IFC_decile),
      Year = year
    ) %>%
    filter(!is.na(PRO_COM), !is.na(IFC_decile))
  
  all_data[[as.character(year)]] <- df
}

ifc_all <- bind_rows(all_data)

# -----------------------------
# 4. Read GRINS taxonomy
# -----------------------------
# The original Mac path pointed to a CSV export of the taxonomy.
# Here we read the xlsx version directly (sheet 1, same 3 columns).
grins <- as.data.frame(read_excel(grins_file, sheet = 1))

print(colnames(grins))
print(head(grins))

# Rename columns safely
grins <- grins %>%
  rename(
    PRO_COM = 1,
    GRINS_macroclass = 2,
    GRINS_class = 3
  ) %>%
  mutate(PRO_COM = as.numeric(PRO_COM)) %>%
  filter(!is.na(PRO_COM))

# -----------------------------
# 5. Merge datasets
# -----------------------------
data <- ifc_all %>%
  inner_join(grins, by = "PRO_COM")

# Pre/Post COVID variable
data <- data %>%
  mutate(
    Period = ifelse(Year %in% c(2018, 2019), "Pre-COVID", "Post-COVID"),
    Year = as.factor(Year),
    GRINS_macroclass = as.factor(GRINS_macroclass),
    Period = as.factor(Period)
  )

cat("Merged rows:", nrow(data), "\n")

# -----------------------------
# 6. Summary table
# -----------------------------
summary_table <- data %>%
  group_by(Year, Period, GRINS_macroclass) %>%
  summarise(
    count = n(),
    mean = mean(IFC_decile, na.rm = TRUE),
    sd = sd(IFC_decile, na.rm = TRUE)
  )

print(summary_table)

# -----------------------------
# 7. Regression 1: Year model
# -----------------------------
model_year <- lm(
  IFC_decile ~ GRINS_macroclass * Year,
  data = data
)

cat("\n===== YEAR REGRESSION =====\n")
summary(model_year)

# -----------------------------
# 8. Regression 2: Pre/Post COVID
# -----------------------------
model_period <- lm(
  IFC_decile ~ GRINS_macroclass * Period,
  data = data
)

cat("\n===== PRE/POST COVID REGRESSION =====\n")
summary(model_period)

# -----------------------------
# 9. Predictions by Year
# -----------------------------
pred_year <- data %>%
  select(GRINS_macroclass, Year) %>%
  distinct()

pred_year$predicted_IFC_decile <- predict(model_year, newdata = pred_year)

print(pred_year)

# -----------------------------
# 10. Predictions by Period
# -----------------------------
pred_period <- data %>%
  select(GRINS_macroclass, Period) %>%
  distinct()

pred_period$predicted_IFC_decile <- predict(model_period, newdata = pred_period)

print(pred_period)

# -----------------------------
# 11. Save outputs
# -----------------------------
write.csv(data,          file.path(out_dir, "merged_IFC_GRINS_all_years.csv"),          row.names = FALSE)
write.csv(summary_table, file.path(out_dir, "summary_IFC_by_year_macroclass.csv"),      row.names = FALSE)
write.csv(pred_year,     file.path(out_dir, "predicted_IFC_by_year_macroclass.csv"),    row.names = FALSE)
write.csv(pred_period,   file.path(out_dir, "predicted_IFC_by_period_macroclass.csv"),  row.names = FALSE)

# Save coefficients
write.csv(summary(model_year)$coefficients,   file.path(out_dir, "regression_coefficients_year.csv"))
write.csv(summary(model_period)$coefficients, file.path(out_dir, "regression_coefficients_period.csv"))

# -----------------------------
# 12. Plot
# -----------------------------
ggplot(plot_data, aes(
  x = Year, 
  y = mean_IFC, 
  group = GRINS_macroclass, 
  color = GRINS_macroclass   # <-- THIS FIXES IT
)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Mean IFC Decile by GRINS Macroclass Over Time",
    x = "Year",
    y = "Mean IFC Decile",
    color = "GRINS Macroclass"   # legend name
  ) +
  theme_minimal()

ggsave(file.path(out_dir, "mean_IFC_decile_plot.png"))