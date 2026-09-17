# ================================================================
# PLS vs PCR head-to-head on the GRINS V3 predictors
# ----------------------------------------------------------------
# Background: PCR's components are chosen to maximise variance in
# X. PLS (Partial Least Squares) chooses components that maximise
# the COVARIANCE between X and Y. Both reduce dimension; only PLS
# uses the target in the projection. When the predictive direction
# does not coincide with the maximum-variance direction (as we
# observed in script 18 with the "jump at K=5"), PLS should
# dominate PCR.
#
# This script fits PCR and PLS at the same K values on the same
# data and reports the gap. It then puts the two against the LASSO
# benchmark and the parsimonious model.
# ================================================================

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(pls); library(tidyr)
})
source("R/_utils.R")

set.seed(2026)

out_dir <- "outputs/pls_vs_pcr"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ================================================================
# 1) DATA
# ================================================================
prep <- pmu_prepare_data(with_taxonomy_filter = FALSE)
data <- prep$data; X_fe <- prep$X_fe; y <- prep$y

year_dummy <- X_fe$year2021
X_cont     <- X_fe[, setdiff(names(X_fe), "year2021"), drop = FALSE]
cat("PLS/PCR matrix:", ncol(X_cont), "columns\n")

# Standardise predictors
X_std <- scale(as.matrix(X_cont))
attr(X_std, "scaled:center") <- NULL
attr(X_std, "scaled:scale")  <- NULL

# ================================================================
# 2) Common CV protocol: panel-aware 5-fold, single repeat to keep
#    the runtime down (PLS with K up to 50 on n=15k is heavy).
# ================================================================
municipalities <- unique(data$PRO_COM)
shuffled <- sample(municipalities)
fold_of  <- (seq_along(shuffled) - 1) %% 5 + 1

candidate_K <- c(5, 10, 15, 20, 25, 30, 50, 75, 100)

cv_records <- data.frame()
for (k in 1:5) {
  te_munis <- shuffled[fold_of == k]
  te_idx   <- which(data$PRO_COM %in% te_munis)
  tr_idx   <- setdiff(seq_len(nrow(X_std)), te_idx)

  Xtr <- X_std[tr_idx, ]; Xte <- X_std[te_idx, ]
  ytr <- y[tr_idx];        yte <- y[te_idx]
  yr_tr <- year_dummy[tr_idx]; yr_te <- year_dummy[te_idx]

  # --- PCR --- principal components of Xtr, regress y on them + year
  pca_tr <- prcomp(Xtr, center = FALSE, scale. = FALSE)
  scores_tr <- pca_tr$x
  scores_te <- Xte %*% pca_tr$rotation

  # --- PLS --- using pls::plsr with center=FALSE because we already scaled
  K_max <- max(candidate_K)
  pls_fit <- plsr(ytr ~ Xtr, ncomp = K_max, validation = "none", scale = FALSE)

  for (K in candidate_K) {
    # PCR: lm of y on first K PCs + year_dummy
    df_tr <- data.frame(y = ytr, yr = yr_tr,
                        scores_tr[, seq_len(K), drop = FALSE])
    df_te <- data.frame(yr = yr_te,
                        scores_te[, seq_len(K), drop = FALSE])
    m_pcr <- lm(y ~ ., data = df_tr)
    p_pcr <- predict(m_pcr, newdata = df_te)
    r2_pcr <- 1 - sum((yte - p_pcr)^2) / sum((yte - mean(yte))^2)

    # PLS: predict with K components + add year as a fixed effect post-hoc
    p_pls_raw <- as.vector(predict(pls_fit, newdata = Xte,
                                   ncomp = K, type = "response"))
    # Residual model on year (avoid leaking year through PLS components)
    res_tr <- ytr - as.vector(predict(pls_fit, newdata = Xtr,
                                      ncomp = K, type = "response"))
    year_lm <- lm(res_tr ~ yr_tr)
    p_pls <- p_pls_raw + predict(year_lm,
                                 newdata = data.frame(yr_tr = yr_te))
    r2_pls <- 1 - sum((yte - p_pls)^2) / sum((yte - mean(yte))^2)

    cv_records <- rbind(cv_records,
      data.frame(method = "PCR", K = K, fold = k, R2 = r2_pcr),
      data.frame(method = "PLS", K = K, fold = k, R2 = r2_pls))
  }
  cat(sprintf("Fold %d done\n", k))
}

cv_summary <- cv_records %>%
  group_by(method, K) %>%
  summarise(R2_mean = mean(R2), R2_sd = sd(R2), .groups = "drop") %>%
  arrange(method, K)

cat("\n--- PCR vs PLS trade-off ---\n")
print(as.data.frame(cv_summary), row.names = FALSE, digits = 4)
write.csv(cv_summary, file.path(out_dir, "pls_vs_pcr_summary.csv"),
          row.names = FALSE)
write.csv(cv_records, file.path(out_dir, "pls_vs_pcr_per_fold.csv"),
          row.names = FALSE)

# ================================================================
# 3) Plot
# ================================================================
p <- ggplot(cv_summary, aes(x = K, y = R2_mean, colour = method)) +
  geom_line() + geom_point(size = 2) +
  geom_errorbar(aes(ymin = R2_mean - R2_sd, ymax = R2_mean + R2_sd),
                width = 0.8) +
  geom_hline(yintercept = 0.825, linetype = "dashed", colour = "firebrick") +
  annotate("text", x = max(candidate_K), y = 0.830,
           label = "LASSO benchmark (R² = 0.825)",
           hjust = 1, colour = "firebrick") +
  geom_hline(yintercept = 0.784, linetype = "dashed", colour = "steelblue") +
  annotate("text", x = max(candidate_K), y = 0.790,
           label = "Parsimonious LM (R² = 0.784)",
           hjust = 1, colour = "steelblue") +
  scale_colour_manual(values = c(PCR = "#377eb8", PLS = "#e41a1c")) +
  labs(title = "PCR vs PLS on GRINS V3 features (panel-aware 5-fold CV)",
       x = "Number of components K",
       y = "Out-of-sample R² (mean ± SD across 5 folds)",
       colour = "Method") +
  theme_minimal()
ggsave(file.path(out_dir, "pls_vs_pcr.png"),
       p, width = 10, height = 6, dpi = 150)

# Final head-to-head
cat("\n=== HEAD-TO-HEAD AT K = 25 ===\n")
sub_25 <- cv_summary[cv_summary$K == 25, ]
print(sub_25, row.names = FALSE, digits = 4)
cat(sprintf("Gap PLS - PCR at K = 25: %+.4f R^2\n",
            sub_25$R2_mean[sub_25$method == "PLS"] -
            sub_25$R2_mean[sub_25$method == "PCR"]))

best_pls <- cv_summary[cv_summary$method == "PLS", ]
cat(sprintf("\nBest PLS: K = %d -> R^2 = %.4f\n",
            best_pls$K[which.max(best_pls$R2_mean)],
            max(best_pls$R2_mean)))
best_pcr <- cv_summary[cv_summary$method == "PCR", ]
cat(sprintf("Best PCR: K = %d -> R^2 = %.4f\n",
            best_pcr$K[which.max(best_pcr$R2_mean)],
            max(best_pcr$R2_mean)))

cat("\nOutputs in:", out_dir, "\n")
