# ================================================================
# CALIBRATION & UNCERTAINTY diagnostics for the parsimonious model
# ----------------------------------------------------------------
# Out-of-sample R^2 alone tells us how much variance the model
# explains, but it hides three things that matter for a defensible
# report:
#
#   1) CALIBRATION: are the predicted IFC values biased (e.g. always
#      a bit too low for fragile municipalities)?
#   2) ERROR HETEROGENEITY: does the model err uniformly, or is it
#      systematically worse on certain types of municipality (very
#      fragile, very small, far from the mean)?
#   3) UNCERTAINTY: what is the *empirical* prediction interval
#      around a forecast? RMSE summarises this with a single
#      number; we want the full distribution.
#
# This script:
#  - Re-runs panel-aware 5-fold CV on the parsimonious 25-covariate
#    model, but this time STORES per-row (observed, predicted) pairs.
#  - Produces a calibration plot (predicted-decile means vs observed
#    means) with a y=x reference line.
#  - Produces a residuals-by-decile boxplot.
#  - Computes Pearson and Spearman correlations on the CV predictions.
#  - Builds the empirical distribution of out-of-sample residuals
#    and derives a 95% prediction interval.
#  - Re-reads AIC/BIC from the mixed-model in-sample summary and
#    presents a model-quality table that combines predictive (CV R^2)
#    and complexity-aware (AIC/BIC) criteria.
# ================================================================

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr)
})
source("R/_utils.R")

set.seed(2026)

in_sample_lmm <- "outputs/mixed_models/in_sample_summary.csv"
cv_lmm        <- "outputs/mixed_models/mixed_models_cv_summary.csv"
out_dir       <- "outputs/calibration_uncertainty"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ================================================================
# 1) Reproduce the data + parsimonious feature matrix
# ================================================================
prep <- pmu_prepare_data(with_taxonomy_filter = FALSE)
data <- prep$data; X_fe <- prep$X_fe; y <- prep$y
selected <- pmu_get_parsimonious_variables(X_fe)
cat("Parsimonious predictors:", length(selected), "\n")

X_std <- as.data.frame(scale(as.matrix(X_fe[, selected])))
model_df <- data.frame(IFC = y, PRO_COM = data$PRO_COM, X_std)
X_terms <- paste(selected, collapse = " + ")
form    <- as.formula(paste("IFC ~", X_terms))

# ================================================================
# 2) Panel-aware 5x5 CV with stored predictions
# ================================================================
municipalities <- unique(model_df$PRO_COM)
n_rep <- 5; n_fold <- 5
cv_preds <- data.frame()

for (rep in 1:n_rep) {
  set.seed(2026 + rep)
  shuffled <- sample(municipalities)
  fold_of  <- (seq_along(shuffled) - 1) %% n_fold + 1
  for (k in 1:n_fold) {
    te_munis <- shuffled[fold_of == k]
    te_idx   <- which(model_df$PRO_COM %in% te_munis)
    tr_idx   <- setdiff(seq_len(nrow(model_df)), te_idx)
    m <- lm(form, data = model_df[tr_idx,])
    p <- predict(m, newdata = model_df[te_idx,])
    cv_preds <- rbind(cv_preds,
      data.frame(rep = rep, fold = k,
                 PRO_COM = model_df$PRO_COM[te_idx],
                 observed = model_df$IFC[te_idx],
                 predicted = p))
  }
  cat(sprintf("  repeat %d done\n", rep))
}

# Average predictions per (PRO_COM, year position): each row may be
# tested up to n_rep times. We collapse to one prediction per row by
# averaging across repetitions.
cv_preds_collapsed <- cv_preds %>%
  group_by(PRO_COM, observed) %>%
  summarise(predicted = mean(predicted), .groups = "drop")
write.csv(cv_preds_collapsed,
          file.path(out_dir, "cv_predictions.csv"), row.names = FALSE)

# ================================================================
# 3) Calibration plot: decile of predicted vs mean of observed
# ================================================================
cv_preds_collapsed$pred_decile <- cut(cv_preds_collapsed$predicted,
                                      breaks = quantile(cv_preds_collapsed$predicted,
                                                        probs = seq(0,1,0.1), na.rm = TRUE),
                                      include.lowest = TRUE, labels = 1:10)
calib <- cv_preds_collapsed %>%
  group_by(pred_decile) %>%
  summarise(mean_predicted = mean(predicted),
            mean_observed  = mean(observed),
            sd_observed    = sd(observed),
            n              = n(),
            .groups = "drop")
write.csv(calib, file.path(out_dir, "calibration_table.csv"), row.names = FALSE)

p_calib <- ggplot(calib, aes(x = mean_predicted, y = mean_observed)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  geom_errorbar(aes(ymin = mean_observed - sd_observed/sqrt(n),
                    ymax = mean_observed + sd_observed/sqrt(n)),
                width = 0.1, alpha = 0.6) +
  geom_point(size = 3, colour = "#377eb8") +
  geom_line(colour = "#377eb8", alpha = 0.7) +
  labs(title = "Calibration: observed vs predicted IFC, by predicted decile",
       subtitle = "Bins are deciles of predicted IFC. Error bars: 1 SE of the observed mean.\nDashed line is the y = x ideal calibration.",
       x = "Mean predicted IFC (within decile)",
       y = "Mean observed IFC (within decile)") +
  theme_minimal()
ggsave(file.path(out_dir, "calibration_plot.png"),
       p_calib, width = 8, height = 6, dpi = 150)

# ================================================================
# 4) Residual distribution by decile of OBSERVED IFC
# ================================================================
cv_preds_collapsed$residual <- cv_preds_collapsed$observed - cv_preds_collapsed$predicted
cv_preds_collapsed$obs_decile <- cut(cv_preds_collapsed$observed,
                                     breaks = quantile(cv_preds_collapsed$observed,
                                                       probs = seq(0,1,0.1), na.rm = TRUE),
                                     include.lowest = TRUE, labels = 1:10)

p_res <- ggplot(cv_preds_collapsed,
                aes(x = factor(obs_decile), y = residual)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  geom_boxplot(fill = "#377eb8", alpha = 0.4, outlier.size = 0.5) +
  labs(title = "Residual distribution by decile of observed IFC",
       subtitle = "Negative residual = model over-predicts | Positive = under-predicts.\nIf the model is well-behaved, all boxes should be centred on zero.",
       x = "Decile of observed IFC (1 = least fragile, 10 = most fragile)",
       y = "Residual (observed − predicted)") +
  theme_minimal()
ggsave(file.path(out_dir, "residual_by_decile.png"),
       p_res, width = 9, height = 6, dpi = 150)

residual_by_decile <- cv_preds_collapsed %>%
  group_by(obs_decile) %>%
  summarise(mean_residual   = mean(residual),
            median_residual = median(residual),
            sd_residual     = sd(residual),
            n               = n(),
            .groups = "drop")
write.csv(residual_by_decile,
          file.path(out_dir, "residual_by_decile.csv"), row.names = FALSE)

# ================================================================
# 5) Correlations and concordance metrics
# ================================================================
pearson  <- cor(cv_preds_collapsed$observed, cv_preds_collapsed$predicted, method = "pearson")
spearman <- cor(cv_preds_collapsed$observed, cv_preds_collapsed$predicted, method = "spearman")
# Lin's concordance correlation coefficient
mu_o <- mean(cv_preds_collapsed$observed); mu_p <- mean(cv_preds_collapsed$predicted)
s2_o <- var (cv_preds_collapsed$observed); s2_p <- var (cv_preds_collapsed$predicted)
s_op <- cov (cv_preds_collapsed$observed,  cv_preds_collapsed$predicted)
ccc <- 2 * s_op / (s2_o + s2_p + (mu_o - mu_p)^2)
cat(sprintf("\nPearson r:   %.4f\nSpearman r:  %.4f\nLin's CCC:   %.4f\n",
            pearson, spearman, ccc))

# ================================================================
# 6) Empirical 95% prediction interval from CV residuals
# ================================================================
res <- cv_preds_collapsed$residual
q025 <- quantile(res, 0.025); q975 <- quantile(res, 0.975)
qpi <- data.frame(metric = c("Pearson r", "Spearman r", "Lin's CCC",
                             "Empirical 95% PI lower (residual)",
                             "Empirical 95% PI upper (residual)",
                             "Empirical 95% PI width (IFC units)"),
                  value  = c(round(pearson,4), round(spearman,4), round(ccc,4),
                             round(q025,3), round(q975,3),
                             round(q975 - q025,3)))
print(qpi, row.names = FALSE)
write.csv(qpi, file.path(out_dir, "uncertainty_summary.csv"), row.names = FALSE)

p_pi <- ggplot(data.frame(residual = res), aes(x = residual)) +
  geom_histogram(bins = 60, fill = "#377eb8", alpha = 0.7, colour = "white") +
  geom_vline(xintercept = c(q025, q975), linetype = "dashed", colour = "firebrick") +
  geom_vline(xintercept = 0, linetype = "dotted") +
  labs(title = "Empirical distribution of out-of-sample residuals",
       subtitle = sprintf("95%% prediction interval (dashed): [%.2f, %.2f] -> width %.2f IFC units",
                          q025, q975, q975 - q025),
       x = "Residual (observed − predicted)", y = "Count") +
  theme_minimal()
ggsave(file.path(out_dir, "residual_distribution.png"),
       p_pi, width = 9, height = 5, dpi = 150)

# ================================================================
# 7) AIC / BIC table for all model variants
# ================================================================
fit_lm  <- lm(form, data = model_df)
aic_lm  <- AIC(fit_lm)
bic_lm  <- BIC(fit_lm)

aicbic <- data.frame(model = "LM parsimonious (25 cov)",
                     AIC = aic_lm, BIC = bic_lm, R2_CV = NA_real_)

if (file.exists(in_sample_lmm)) {
  ins <- read.csv(in_sample_lmm, stringsAsFactors = FALSE)
  aicbic <- rbind(aicbic,
                  data.frame(model = ins$model,
                             AIC = ins$AIC, BIC = ins$BIC,
                             R2_CV = NA_real_))
}
if (file.exists(cv_lmm)) {
  cv_tbl <- read.csv(cv_lmm, stringsAsFactors = FALSE)
  for (i in seq_len(nrow(cv_tbl))) {
    j <- grep(cv_tbl$model[i], aicbic$model, fixed = TRUE)
    if (length(j) == 1) aicbic$R2_CV[j] <- round(cv_tbl$R2_mean[i], 4)
  }
}
# Compute parsimonious LM CV R2 from our CV predictions
r2_lm_cv <- 1 - sum((cv_preds_collapsed$observed - cv_preds_collapsed$predicted)^2) /
                  sum((cv_preds_collapsed$observed - mean(cv_preds_collapsed$observed))^2)
aicbic$R2_CV[aicbic$model == "LM parsimonious (25 cov)"] <- round(r2_lm_cv, 4)
aicbic <- aicbic %>% arrange(AIC)
cat("\n--- Model quality table (lower AIC/BIC = better, higher R2_CV = better) ---\n")
print(aicbic, row.names = FALSE)
write.csv(aicbic, file.path(out_dir, "model_quality_table.csv"), row.names = FALSE)

cat("\nOutputs in:", out_dir, "\n")
