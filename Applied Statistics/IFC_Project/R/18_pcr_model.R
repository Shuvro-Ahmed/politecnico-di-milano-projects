# ================================================================
# PRINCIPAL COMPONENT REGRESSION (PCR) on the GRINS V3 features
# ----------------------------------------------------------------
# Question raised by the instructor: "PCR not PCA?" The existing
# scripts 02 / 02b / 02c run a PCA on the 12 IFC indicators
# themselves -- a descriptive analysis of how IFC is constructed.
# This script does something different and complementary: a
# Principal Component Regression of IFC on the GRINS V3 predictors.
#
# PCR pipeline:
#   1) Standardise the 326 engineered GRINS features
#   2) Compute PCA on the standardised matrix
#   3) Take the first K principal components as regressors
#   4) Fit lm(IFC ~ PC1 + ... + PCK)
#   5) Evaluate by panel-aware 5x5 cross-validation
#
# We test several K values to find the sweet spot, exactly like the
# trade-off curve we did for the parsimonious model, and we report
# how PCR compares with the LASSO benchmark (R^2 = 0.826) and with
# the parsimonious 25-covariate model (R^2 = 0.784).
# ================================================================

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr)
})
source("R/_utils.R")

set.seed(2026)

out_dir <- "outputs/pcr_model"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ================================================================
# 1) DATA LOADING + FEATURE ENGINEERING (shared utils)
# ================================================================
prep <- pmu_prepare_data(with_taxonomy_filter = FALSE)
data <- prep$data; X_fe <- prep$X_fe; y <- prep$y
cat("Observations:", length(y), "\n")
cat("Engineered features:", ncol(X_fe), "\n")

# Exclude the year2021 dummy from PCA: it is a binary control variable,
# not a content variable. We add it back as a separate fixed effect
# in the regression below.
year_dummy <- X_fe$year2021
X_fe_pca   <- X_fe[, setdiff(names(X_fe), "year2021"), drop = FALSE]
cat("PCA matrix (year2021 excluded):", ncol(X_fe_pca), "columns\n")

# Standardise (PCA requires it for meaningful comparison across vars)
X_std <- scale(as.matrix(X_fe_pca))
attr(X_std, "scaled:center") <- NULL
attr(X_std, "scaled:scale")  <- NULL

# ================================================================
# 2) PCA on the full feature matrix
# ----------------------------------------------------------------
# We use prcomp with already-standardised input. The scree plot
# shows the eigenvalues (variance carried by each PC).
# ================================================================
pca <- prcomp(X_std, center = FALSE, scale. = FALSE)
var_explained <- pca$sdev^2 / sum(pca$sdev^2)
cum_var       <- cumsum(var_explained)

# How many PCs are needed to reach 70%, 80%, 90% of variance?
n_for_70 <- which.max(cum_var >= 0.70)
n_for_80 <- which.max(cum_var >= 0.80)
n_for_90 <- which.max(cum_var >= 0.90)
cat(sprintf("PCs needed: %d (70%%), %d (80%%), %d (90%%)\n",
            n_for_70, n_for_80, n_for_90))

scree_df <- data.frame(PC = seq_along(var_explained),
                       var_explained = var_explained,
                       cum_var       = cum_var)
write.csv(scree_df, file.path(out_dir, "scree_table.csv"), row.names = FALSE)

p_scree <- ggplot(scree_df[1:50, ], aes(x = PC)) +
  geom_col(aes(y = var_explained), fill = "#377eb8", alpha = 0.7) +
  geom_line(aes(y = cum_var), colour = "firebrick", linewidth = 1) +
  geom_point(aes(y = cum_var), colour = "firebrick", size = 1) +
  geom_hline(yintercept = c(0.7, 0.8, 0.9),
             linetype = "dashed", colour = "grey50") +
  labs(title = "Scree plot of the GRINS feature PCA",
       subtitle = "Blue bars: variance per PC. Red line: cumulative variance.",
       x = "Principal Component", y = "Variance share") +
  theme_minimal()
ggsave(file.path(out_dir, "scree_plot.png"),
       p_scree, width = 10, height = 5, dpi = 150)

# ================================================================
# 3) PCR trade-off: how many PCs do we need?
# ----------------------------------------------------------------
# For each candidate K, build a model_df with PC1..PCK + PRO_COM
# and run the same 5x5 panel-aware CV as the rest of the pipeline.
# Then look at the curve R^2 vs K and pick the elbow.
# ================================================================
scores <- as.data.frame(pca$x)
model_df_full <- data.frame(IFC = y, PRO_COM = data$PRO_COM,
                            year_dummy = year_dummy, scores)

candidate_K <- c(5, 10, 15, 20, 25, 30, 50, 100,
                 n_for_70, n_for_80, n_for_90)
candidate_K <- sort(unique(pmin(candidate_K, ncol(scores))))

municipalities <- unique(data$PRO_COM)
n_repeats <- 5
n_folds   <- 5

cv_records <- data.frame()
for (K in candidate_K) {
  pc_terms <- paste(paste0("PC", seq_len(K)), collapse = " + ")
  form     <- as.formula(paste("IFC ~", pc_terms, "+ year_dummy"))
  for (rep in 1:n_repeats) {
    set.seed(2026 + rep)
    shuffled <- sample(municipalities)
    fold_of  <- (seq_along(shuffled) - 1) %% n_folds + 1
    for (k in 1:n_folds) {
      te_munis <- shuffled[fold_of == k]
      te_idx   <- which(model_df_full$PRO_COM %in% te_munis)
      tr_idx   <- setdiff(seq_len(nrow(model_df_full)), te_idx)
      m <- lm(form, data = model_df_full[tr_idx, ])
      p <- predict(m, newdata = model_df_full[te_idx, ])
      yt <- model_df_full$IFC[te_idx]
      rmse <- sqrt(mean((yt - p)^2))
      mae  <- mean(abs(yt - p))
      r2   <- 1 - sum((yt - p)^2) / sum((yt - mean(yt))^2)
      cv_records <- rbind(cv_records,
        data.frame(K = K, rep = rep, fold = k,
                   RMSE = rmse, MAE = mae, R2 = r2))
    }
  }
  cat(sprintf("PCR with K = %3d done\n", K))
}

cv_summary <- cv_records %>%
  group_by(K) %>%
  summarise(R2_mean   = mean(R2),   R2_sd   = sd(R2),
            RMSE_mean = mean(RMSE), RMSE_sd = sd(RMSE),
            MAE_mean  = mean(MAE),  MAE_sd  = sd(MAE),
            .groups   = "drop") %>%
  arrange(K)
cat("\n--- PCR trade-off curve ---\n")
print(as.data.frame(cv_summary), row.names = FALSE, digits = 4)
write.csv(cv_summary, file.path(out_dir, "pcr_tradeoff.csv"), row.names = FALSE)
write.csv(cv_records,  file.path(out_dir, "pcr_per_fold.csv"), row.names = FALSE)

# Trade-off plot with the LASSO and parsimonious benchmark lines
p_trade <- ggplot(cv_summary, aes(x = K, y = R2_mean)) +
  geom_line() +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = R2_mean - R2_sd, ymax = R2_mean + R2_sd),
                width = 1.0) +
  geom_hline(yintercept = 0.825, linetype = "dashed", colour = "firebrick") +
  annotate("text", x = max(candidate_K), y = 0.830,
           label = "LASSO benchmark (191 cov, R² = 0.825)",
           hjust = 1, colour = "firebrick") +
  geom_hline(yintercept = 0.784, linetype = "dashed", colour = "steelblue") +
  annotate("text", x = max(candidate_K), y = 0.789,
           label = "Parsimonious (25 cov, R² = 0.784)",
           hjust = 1, colour = "steelblue") +
  labs(title = "PCR — R² out-of-sample as a function of K (number of PCs)",
       x = "Number of principal components used",
       y = "Out-of-sample R² (mean ± SD over 25 folds)") +
  theme_minimal()
ggsave(file.path(out_dir, "pcr_tradeoff.png"),
       p_trade, width = 10, height = 6, dpi = 150)

# ================================================================
# 4) Pick the best K and refit on full data
# ================================================================
best_K <- cv_summary$K[which.max(cv_summary$R2_mean)]
cat(sprintf("\nBest K by CV R^2: %d  (R^2 = %.4f)\n",
            best_K, max(cv_summary$R2_mean)))

pc_terms <- paste(paste0("PC", seq_len(best_K)), collapse = " + ")
fit_best <- lm(as.formula(paste("IFC ~", pc_terms, "+ year_dummy")),
               data = model_df_full)
coef_best <- summary(fit_best)$coefficients
write.csv(coef_best, file.path(out_dir, "pcr_coefficients.csv"))

# ================================================================
# 5) Top loadings for the leading PCs (which original variables
#    drive each PC?). Useful for interpretation.
# ================================================================
top_loadings <- function(loadings_vec, n = 5) {
  ord <- order(-abs(loadings_vec))[1:n]
  data.frame(variable = names(loadings_vec)[ord],
             loading  = loadings_vec[ord])
}
loadings <- pca$rotation
top_5_each <- list()
for (k in 1:min(5, ncol(loadings))) {
  top_5_each[[paste0("PC", k)]] <- top_loadings(loadings[, k], n = 5)
}
cat("\n--- Top 5 loadings for PC1..PC5 ---\n")
for (k in seq_along(top_5_each)) {
  cat(sprintf("\n%s:\n", names(top_5_each)[k]))
  print(top_5_each[[k]], row.names = FALSE, digits = 3)
}
saveRDS(top_5_each, file.path(out_dir, "pc_top_loadings.rds"))

# ================================================================
# 6) Summary line comparing PCR with the other models
# ================================================================
cat("\n=== HEAD-TO-HEAD COMPARISON ===\n")
cat(sprintf("LASSO benchmark (191 cov):        R^2 = 0.826\n"))
cat(sprintf("Parsimonious LM (25 cov):         R^2 = 0.784\n"))
cat(sprintf("PCR (best K = %d):                 R^2 = %.4f\n",
            best_K, max(cv_summary$R2_mean)))
cat(sprintf("M_geo_nested LMM (25 cov):        R^2 = 0.818\n"))

cat("\nOutputs in:", out_dir, "\n")
