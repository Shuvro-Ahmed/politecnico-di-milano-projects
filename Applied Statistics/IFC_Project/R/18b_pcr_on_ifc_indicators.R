# ================================================================
# PCR ON THE 12 IFC INDICATORS (year by year)
# ----------------------------------------------------------------
# Companion to R/18_pcr_model.R, which runs PCR on the 326 GRINS V3
# predictors. This second script mirrors the descriptive PCA cascade
# in scripts 02 / 02b / 02c, which operates on the 12 IFC indicators
# (Ind1..Ind12) read from composite_fragility_all_years.xlsx, but
# adds the predictive step: it uses the leading principal components
# of the 12 indicators as regressors for the raw IFC values from
# IFC_Final_Analysis_Sorted.xlsx.
#
# This is a legitimate (non-circular) PCR because the raw IFC values
# are computed with a specific weighted scheme on the 12 indicators
# (and possibly other adjustments), so the PCs need not align with
# the IFC weighting direction. The trade-off curve R^2 vs K tells us
# how concentrated the IFC signal is along the leading PC axes.
# ================================================================

suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(tidyr); library(ggplot2)
})

set.seed(2026)

indicators_file <- "data/raw/ifc/composite_fragility_all_years.xlsx"
ifc_raw_file    <- "data/raw/ifc/final_analysis_sorted.xlsx"
out_dir         <- "outputs/pcr_on_ifc_indicators"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# OneDrive placeholder workaround
ensure_local <- function(path) {
  tmp <- file.path(tempdir(), basename(path))
  file.copy(path, tmp, overwrite = TRUE); tmp
}
indicators_file <- ensure_local(indicators_file)
ifc_raw_file    <- ensure_local(ifc_raw_file)

# ================================================================
# 1) Load the 12 IFC indicators for 2019 and 2021
#    (same years as our raw-IFC target)
# ================================================================
load_indicators <- function(year_str) {
  df <- read_excel(indicators_file, sheet = year_str, skip = 2,
                   col_names = FALSE, col_types = "text")
  df <- df[, 1:15]
  colnames(df) <- c("PRO_COM", "Territory", "MFI_Dec", paste0("Ind", 1:12))
  df %>%
    filter(!is.na(PRO_COM), !is.na(Territory)) %>%
    mutate(across(starts_with("Ind"),
                  ~ as.numeric(gsub(",", ".", gsub("\\s+", "", .x))))) %>%
    mutate(year = as.integer(year_str)) %>%
    select(PRO_COM, year, starts_with("Ind"))
}
ind_2019 <- load_indicators("2019")
ind_2021 <- load_indicators("2021")
ind_all  <- bind_rows(ind_2019, ind_2021) %>%
  mutate(PRO_COM = as.numeric(PRO_COM)) %>%
  filter(!is.na(PRO_COM))

# Reverse polarity of 4 indicators where "high value = less fragile",
# so that after the flip all indicators point in the same direction.
# This mirrors the choice made in R/02_pca.R.
to_flip <- c("Ind3", "Ind9", "Ind10", "Ind11")
ind_all <- ind_all %>%
  mutate(across(all_of(to_flip), ~ -.x))

# ================================================================
# 2) Load the raw IFC target (2019 / 2021)
# ================================================================
ifc <- read_excel(ifc_raw_file) %>%
  select(PRO_COM, IFC_2019, IFC_2021) %>%
  pivot_longer(c(IFC_2019, IFC_2021),
               names_to = "year", values_to = "IFC") %>%
  mutate(year    = as.integer(sub("IFC_", "", year)),
         PRO_COM = as.numeric(PRO_COM)) %>%
  filter(!is.na(IFC), !is.na(PRO_COM))

# Deduplicate before the merge. A small number of municipalities
# appear twice in the raw IFC file with conflicting values; the
# GRINS-based scripts mask this implicitly via inner_join with
# unique GRINS rows. Here we keep the first occurrence of each
# (PRO_COM, year) pair on both sides.
ind_all <- ind_all %>%
  distinct(PRO_COM, year, .keep_all = TRUE)
ifc <- ifc %>%
  distinct(PRO_COM, year, .keep_all = TRUE)

data <- ind_all %>%
  inner_join(ifc, by = c("PRO_COM", "year"))
cat("Observations after merge (deduplicated):", nrow(data), "\n")

# Median-impute residual NAs in the indicators
ind_cols <- paste0("Ind", 1:12)
for (j in ind_cols) {
  if (any(is.na(data[[j]])))
    data[[j]][is.na(data[[j]])] <- median(data[[j]], na.rm = TRUE)
}

# ================================================================
# 3) PCA on the 12 indicators
# ----------------------------------------------------------------
# year is kept OUT of the PCA and added back as a separate fixed
# effect in the regression below: a binary dummy distorts the
# principal directions and is conceptually a control variable,
# not a content variable.
# ================================================================
X    <- as.matrix(data[, ind_cols])
Xstd <- scale(X)
pca  <- prcomp(Xstd, center = FALSE, scale. = FALSE)
year_dummy <- as.integer(data$year == 2021)

var_explained <- pca$sdev^2 / sum(pca$sdev^2)
cum_var       <- cumsum(var_explained)

scree <- data.frame(PC = seq_along(var_explained),
                    var_explained = round(var_explained, 4),
                    cumulative    = round(cum_var, 4))
cat("\n--- Scree on the 12 IFC indicators ---\n")
print(scree, row.names = FALSE)
write.csv(scree, file.path(out_dir, "scree_table.csv"), row.names = FALSE)

p_scree <- ggplot(scree, aes(x = PC)) +
  geom_col(aes(y = var_explained), fill = "#377eb8", alpha = 0.7) +
  geom_line(aes(y = cumulative), colour = "firebrick", linewidth = 1) +
  geom_point(aes(y = cumulative), colour = "firebrick", size = 2) +
  geom_hline(yintercept = c(0.7, 0.8, 0.9),
             linetype = "dashed", colour = "grey50") +
  scale_x_continuous(breaks = 1:12) +
  labs(title = "Scree plot — PCA on the 12 IFC indicators",
       subtitle = "Bars: variance per PC. Line: cumulative variance.",
       x = "PC", y = "Variance share") +
  theme_minimal()
ggsave(file.path(out_dir, "scree_plot.png"),
       p_scree, width = 9, height = 5, dpi = 150)

# ================================================================
# 4) PCR trade-off: R^2 with K = 1..12 PCs
#    Panel-aware 5x5 CV, same protocol as the rest of the pipeline.
# ================================================================
scores <- as.data.frame(pca$x)
model_df <- data.frame(IFC = data$IFC, PRO_COM = data$PRO_COM,
                       year_dummy = year_dummy, scores)

municipalities <- unique(model_df$PRO_COM)
n_repeats <- 5; n_folds <- 5

cv_records <- data.frame()
for (K in 1:12) {
  pc_terms <- paste(paste0("PC", seq_len(K)), collapse = " + ")
  form     <- as.formula(paste("IFC ~", pc_terms, "+ year_dummy"))
  for (rep in 1:n_repeats) {
    set.seed(2026 + rep)
    shuffled <- sample(municipalities)
    fold_of  <- (seq_along(shuffled) - 1) %% n_folds + 1
    for (k in 1:n_folds) {
      te_munis <- shuffled[fold_of == k]
      te_idx   <- which(model_df$PRO_COM %in% te_munis)
      tr_idx   <- setdiff(seq_len(nrow(model_df)), te_idx)
      m <- lm(form, data = model_df[tr_idx, ])
      p <- predict(m, newdata = model_df[te_idx, ])
      yt <- model_df$IFC[te_idx]
      cv_records <- rbind(cv_records,
        data.frame(K = K, rep = rep, fold = k,
                   RMSE = sqrt(mean((yt - p)^2)),
                   MAE  = mean(abs(yt - p)),
                   R2   = 1 - sum((yt - p)^2) / sum((yt - mean(yt))^2)))
    }
  }
  cat(sprintf("PCR(IFC indicators) K = %2d done\n", K))
}

cv_summary <- cv_records %>%
  group_by(K) %>%
  summarise(R2_mean   = mean(R2),   R2_sd   = sd(R2),
            RMSE_mean = mean(RMSE), RMSE_sd = sd(RMSE),
            MAE_mean  = mean(MAE),  MAE_sd  = sd(MAE),
            .groups = "drop")
cat("\n--- PCR trade-off (12 IFC indicators) ---\n")
print(as.data.frame(cv_summary), row.names = FALSE, digits = 4)
write.csv(cv_summary, file.path(out_dir, "pcr_tradeoff.csv"), row.names = FALSE)

# Trade-off plot
p_trade <- ggplot(cv_summary, aes(x = K, y = R2_mean)) +
  geom_line() +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = R2_mean - R2_sd, ymax = R2_mean + R2_sd),
                width = 0.2) +
  geom_hline(yintercept = 0.825, linetype = "dashed", colour = "firebrick") +
  annotate("text", x = 12, y = 0.832,
           label = "LASSO benchmark (R² = 0.825)",
           hjust = 1, colour = "firebrick") +
  geom_hline(yintercept = 0.784, linetype = "dashed", colour = "steelblue") +
  annotate("text", x = 12, y = 0.790,
           label = "Parsimonious LM (R² = 0.784)",
           hjust = 1, colour = "steelblue") +
  scale_x_continuous(breaks = 1:12) +
  labs(title = "PCR R² vs K on the 12 IFC indicators",
       x = "Number of PCs used as regressors",
       y = "Out-of-sample R² (mean ± SD across 25 folds)") +
  theme_minimal()
ggsave(file.path(out_dir, "pcr_tradeoff.png"),
       p_trade, width = 9, height = 5, dpi = 150)

# ================================================================
# 5) Loadings of PC1..PC3 (which indicators drive each PC)
# ================================================================
ind_names <- c(
  Ind1 = "Motorisation rate (high emissions)",
  Ind2 = "Undifferentiated waste",
  Ind3 = "Protected areas (flipped)",
  Ind4 = "Landslide risk",
  Ind5 = "Land consumption",
  Ind6 = "Accessibility to services",
  Ind7 = "Population dependency",
  Ind8 = "Low-education share 25-64",
  Ind9 = "Employment rate (flipped)",
  Ind10 = "Net migration (flipped)",
  Ind11 = "Local-units density (flipped)",
  Ind12 = "Low-productivity employment"
)

loadings_top <- function(pc_idx) {
  v <- pca$rotation[, pc_idx]
  df <- data.frame(indicator = names(v),
                   name      = ind_names[names(v)],
                   loading   = round(v, 3))
  df[order(-abs(df$loading)), ]
}
cat("\n--- Top loadings of PC1..PC3 ---\n")
top_pcs <- lapply(1:3, loadings_top)
for (i in seq_along(top_pcs)) {
  cat(sprintf("\nPC%d:\n", i))
  print(top_pcs[[i]], row.names = FALSE)
}
saveRDS(top_pcs, file.path(out_dir, "pc_top_loadings.rds"))

# ================================================================
# 6) Final comparison block
# ================================================================
best_K <- cv_summary$K[which.max(cv_summary$R2_mean)]
best_R2 <- max(cv_summary$R2_mean)

cat("\n=== HEAD-TO-HEAD ===\n")
cat(sprintf("LASSO benchmark (191 GRINS cov):       R² = 0.826\n"))
cat(sprintf("M_geo_nested LMM (25 GRINS cov):       R² = 0.818\n"))
cat(sprintf("PCR on GRINS V3 (script 18, K=100):    R² = 0.787\n"))
cat(sprintf("PCR on 12 IFC indicators (this script, K=%d): R² = %.4f\n",
            best_K, best_R2))
cat(sprintf("Parsimonious LM (25 GRINS cov):        R² = 0.784\n"))

cat("\nOutputs in:", out_dir, "\n")
