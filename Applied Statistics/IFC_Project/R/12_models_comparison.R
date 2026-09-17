# ================================================================
# MODELS COMPARISON — Three layers, two covariate sets
# ----------------------------------------------------------------
# This script doesn't fit anything; it loads CV-summary CSVs from
# earlier scripts and produces a single comparison table + plot.
#
# Two coherent "stories" are reported side by side:
#
#  STORY 1 - "Full" 191-covariate set (from the LASSO benchmark):
#    - LASSO benchmark (script 10):  outputs/final_benchmark/repeated_cv_summary.csv
#
#  STORY 2 - "Parsimonious" 25-covariate set (from script 14, after
#  iterative VIF pruning + top-N by standardised effect):
#    - parsimonious linear model: row from the trade-off summary
#    - parsimonious linear mixed models (script 11, now fit on the
#      25-covariate set): outputs/mixed_models/mixed_models_cv_summary.csv
#
# The mixed-model summary always refers to the CURRENT covariate set
# used in script 11 (parsimonious by default).
# ================================================================

suppressPackageStartupMessages({
  library(dplyr); library(ggplot2)
})

lasso_file       <- "outputs/final_benchmark/repeated_cv_summary.csv"
tradeoff_file    <- "outputs/parsimonious_model/tradeoff_summary.csv"
lmm_file         <- "outputs/mixed_models/mixed_models_cv_summary.csv"
out_dir          <- "outputs/mixed_models"
n_parsimonious   <- 25
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ================================================================
# Load the three sources
# ================================================================
lasso <- read.csv(lasso_file, stringsAsFactors = FALSE)
lmm   <- read.csv(lmm_file,   stringsAsFactors = FALSE)
trade <- if (file.exists(tradeoff_file))
           read.csv(tradeoff_file, stringsAsFactors = FALSE) else NULL

# ================================================================
# Build a unified table with a clear "covariate set" annotation
# ================================================================
to_row <- function(model, R2m, R2s, RMSEm, RMSEs, MAEm, MAEs, class, covset) {
  data.frame(model = model, covariate_set = covset, class = class,
             R2_mean = R2m, R2_sd = R2s,
             RMSE_mean = RMSEm, RMSE_sd = RMSEs,
             MAE_mean = MAEm,   MAE_sd  = MAEs)
}

rows <- list()
for (i in seq_len(nrow(lasso))) {
  rows[[length(rows) + 1]] <- to_row(
    model = paste0("LASSO ", lasso$model[i]),
    R2m = lasso$R2_mean[i], R2s = lasso$R2_sd[i],
    RMSEm = lasso$RMSE_mean[i], RMSEs = lasso$RMSE_sd[i],
    MAEm = lasso$MAE_mean[i],   MAEs  = lasso$MAE_sd[i],
    class = "LASSO benchmark", covset = "Full (~191 cov)"
  )
}
if (!is.null(trade)) {
  pars_row <- trade[trade$n_covariates == n_parsimonious, ]
  if (nrow(pars_row) == 1) {
    rows[[length(rows) + 1]] <- to_row(
      model = sprintf("LM parsimonious (%d cov)", n_parsimonious),
      R2m = pars_row$R2_mean, R2s = pars_row$R2_sd,
      RMSEm = pars_row$RMSE_mean, RMSEs = pars_row$RMSE_sd,
      MAEm = pars_row$MAE_mean,   MAEs  = pars_row$MAE_sd,
      class = "Linear model", covset = "Parsimonious (25 cov)"
    )
  }
}
for (i in seq_len(nrow(lmm))) {
  rows[[length(rows) + 1]] <- to_row(
    model = lmm$model[i],
    R2m = lmm$R2_mean[i], R2s = lmm$R2_sd[i],
    RMSEm = lmm$RMSE_mean[i], RMSEs = lmm$RMSE_sd[i],
    MAEm = lmm$MAE_mean[i],   MAEs  = lmm$MAE_sd[i],
    class = "Mixed model", covset = sprintf("Parsimonious (%d cov)", n_parsimonious)
  )
}
comparison <- do.call(rbind, rows) %>% arrange(desc(R2_mean))

cat("--- All models, sorted by R^2 ---\n")
print(comparison, row.names = FALSE, digits = 4)
write.csv(comparison, file.path(out_dir, "models_comparison.csv"), row.names = FALSE)

# ================================================================
# Plot
# ================================================================
p <- ggplot(comparison,
            aes(x = reorder(model, R2_mean), y = R2_mean, fill = covariate_set)) +
  geom_col(width = 0.6) +
  geom_errorbar(aes(ymin = R2_mean - R2_sd, ymax = R2_mean + R2_sd), width = 0.2) +
  coord_flip() +
  labs(title = "Out-of-fold R² across all model variants",
       subtitle = "Two covariate sets compared: full (191 LASSO-selected) vs parsimonious (25 after VIF pruning)",
       x = NULL, y = "R² (mean ± SD across folds)",
       fill = "Covariate set") +
  scale_fill_manual(values = c("Full (~191 cov)" = "#377eb8",
                               "Parsimonious (25 cov)" = "#e41a1c")) +
  theme_minimal()
ggsave(file.path(out_dir, "models_comparison_R2.png"),
       p, width = 11, height = 6, dpi = 150)

# ================================================================
# Fair gains: LMM vs the LM with the SAME covariate set
# ================================================================
cat("\n--- Fair gains within each covariate set ---\n")
lm_full        <- max(lasso$R2_mean)
lm_parsim_R2   <- if (!is.null(trade) && any(trade$n_covariates == n_parsimonious))
                    trade$R2_mean[trade$n_covariates == n_parsimonious] else NA_real_
lmm_parsim_R2  <- max(lmm$R2_mean)

cat(sprintf("Full set (~191 cov):       LASSO best R^2 = %.4f\n", lm_full))
cat(sprintf("Parsimonious (25 cov):     LM    R^2     = %.4f\n", lm_parsim_R2))
cat(sprintf("Parsimonious (25 cov):     LMM best R^2 = %.4f  ->  +%.4f over the parsimonious LM\n",
            lmm_parsim_R2, lmm_parsim_R2 - lm_parsim_R2))
cat(sprintf("Drop from full LASSO -> parsimonious LM: %+.4f R^2 (interpretability cost)\n",
            lm_parsim_R2 - lm_full))
cat(sprintf("Net of LMM gain:                          %+.4f R^2 (still vs full LASSO)\n",
            lmm_parsim_R2 - lm_full))
