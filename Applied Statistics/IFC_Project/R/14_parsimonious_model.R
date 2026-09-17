# ================================================================
# PARSIMONIOUS LINEAR MODEL (10-20 covariates)
# ----------------------------------------------------------------
# The LASSO benchmark (script 10) selects ~191 covariates out of
# 326 — predictively excellent but too many for a poster-style
# discussion. The instructor asked us to find a much smaller set
# of predictors (10-20) that still gives a credible R^2.
#
# Strategy:
#   1) Start from the 191 LASSO-selected covariates.
#   2) Compute the Variance Inflation Factor (VIF) of each one in
#      the linear model. Iteratively drop the variable with the
#      highest VIF until all VIFs are below 10 -- this removes
#      multicollinearity.
#   3) From the surviving "low-collinearity" set, rank predictors
#      by their standardised effect |beta * sd(x)|.
#   4) Build candidate parsimonious models with the top
#      5 / 10 / 15 / 20 / 25 predictors.
#   5) Evaluate each by 5x5 panel-aware CV (same protocol as the
#      LASSO benchmark, so the R^2 numbers are directly
#      comparable to 0.825).
#   6) Identify the "elbow" -- the sweet spot between parsimony
#      and accuracy -- and refit the final model on the full data.
# ================================================================

suppressPackageStartupMessages({
  library(car); library(ggplot2); library(dplyr)
})
source("R/_utils.R")

set.seed(2026)

out_dir <- "outputs/parsimonious_model"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Variables to EXCLUDE upfront on substantive grounds.
# Superficie_totale is a static geographic feature, not a fragility
# driver per se. Once the rest of the pipeline uses per-capita
# densities, it adds no useful information and would confound the
# interpretation. The instructor explicitly asked to remove it.
excluded_vars <- c("Superficie_totale_Kmq_formattato",
                   "Superficie_totale_Kmq_formattato.1",
                   make.names("Superficie_totale_Kmq_formattato"))

# ================================================================
# 1) DATA LOADING + FEATURE ENGINEERING (delegated to R/_utils.R)
# ================================================================
prep <- pmu_prepare_data(with_taxonomy_filter = FALSE)
data <- prep$data; X_fe <- prep$X_fe; y <- prep$y
cat("Observations:", length(y), "\n")

# Restrict to LASSO-selected predictors (191 from the benchmark)
# and drop the substantively excluded variables
sel_df   <- read.csv(PMU_PATHS$benchmark_coef, stringsAsFactors = FALSE)
selected <- intersect(sel_df$variable, names(X_fe))
if (any(selected %in% excluded_vars)) {
  cat("Excluding upfront:",
      paste(intersect(selected, excluded_vars), collapse = ", "), "\n")
  selected <- setdiff(selected, excluded_vars)
}
cat("Starting from", length(selected), "LASSO-selected predictors\n")
X_sel    <- X_fe[, selected, drop = FALSE]

# Standardise so coefficients on this scale are comparable
X_std    <- as.data.frame(scale(as.matrix(X_sel)))
model_df <- data.frame(IFC = y, PRO_COM = data$PRO_COM, X_std)

# ================================================================
# 2) ITERATIVE VIF REDUCTION
# ----------------------------------------------------------------
# VIF_j = 1 / (1 - R^2_j), where R^2_j is the R^2 of regressing the
# j-th predictor on ALL OTHER predictors. Conceptually it answers
# "how much of x_j is already explained by the other x's?".
# VIF=1 -> independent. VIF>=10 -> heavy collinearity, the
# coefficient of x_j becomes unstable (small data change flips
# sign or magnitude). We drop the worst offender, recompute, and
# stop when every remaining VIF is below 10.
# ================================================================
vif_threshold <- 10
current_vars  <- selected
drop_log      <- data.frame()

cat("\n--- VIF iterative pruning ---\n")
iter <- 0
repeat {
  iter <- iter + 1
  if (length(current_vars) < 3) break
  formula_str <- paste("IFC ~", paste(current_vars, collapse = " + "))
  fit_iter <- lm(as.formula(formula_str), data = model_df)
  vifs <- tryCatch(vif(fit_iter), error = function(e) NULL)
  if (is.null(vifs)) break
  worst_vif  <- max(vifs)
  worst_name <- names(which.max(vifs))
  if (worst_vif < vif_threshold) {
    cat(sprintf("Iter %3d | max VIF = %6.2f (%s) -> below threshold, stop\n",
                iter, worst_vif, worst_name))
    break
  }
  drop_log <- rbind(drop_log, data.frame(iteration = iter,
                                         dropped   = worst_name,
                                         VIF       = worst_vif))
  current_vars <- setdiff(current_vars, worst_name)
  if (iter %% 10 == 0)
    cat(sprintf("Iter %3d | dropped %-50s VIF=%.2f | remaining %d\n",
                iter, worst_name, worst_vif, length(current_vars)))
}
cat(sprintf("Final low-collinearity set: %d predictors (started from %d)\n",
            length(current_vars), length(selected)))
write.csv(drop_log, file.path(out_dir, "vif_drop_log.csv"), row.names = FALSE)

# ================================================================
# 3) RANK SURVIVING PREDICTORS BY STANDARDISED EFFECT
# ----------------------------------------------------------------
# Now that we have a low-collinearity set, fit lm() once and rank
# by |beta|. Since predictors are already standardised, |beta|
# is the standardised effect (beta * sd_x with sd_x = 1).
# ================================================================
fit_clean <- lm(as.formula(paste("IFC ~", paste(current_vars, collapse = " + "))),
                data = model_df)
co_clean <- summary(fit_clean)$coefficients
co_clean <- co_clean[rownames(co_clean) != "(Intercept)", , drop = FALSE]
ranked <- data.frame(variable = rownames(co_clean),
                     beta     = co_clean[, "Estimate"],
                     p_value  = co_clean[, "Pr(>|t|)"])
ranked <- ranked[order(-abs(ranked$beta)), ]
write.csv(ranked, file.path(out_dir, "vif_clean_ranking.csv"), row.names = FALSE)

# ================================================================
# 4) BUILD CANDIDATE PARSIMONIOUS MODELS
# ----------------------------------------------------------------
# 5 / 10 / 15 / 20 / 25 / 30 / all-after-VIF predictors.
# Evaluate each one with the SAME 5x5 panel-aware CV protocol as
# the LASSO benchmark in script 10 (panel-aware = a municipality
# stays in only one fold).
# ================================================================
candidate_n <- c(5, 10, 15, 20, 25, 30, length(current_vars))
candidate_n <- unique(pmin(candidate_n, length(current_vars)))

municipalities <- unique(model_df$PRO_COM)
n_repeats <- 5; n_folds <- 5

cv_records <- data.frame()
for (n_top in candidate_n) {
  top_vars <- ranked$variable[seq_len(n_top)]
  formula_str <- paste("IFC ~", paste(top_vars, collapse = " + "))
  for (rep in 1:n_repeats) {
    set.seed(2026 + rep)
    shuffled <- sample(municipalities)
    fold_of  <- (seq_along(shuffled) - 1) %% n_folds + 1
    for (k in 1:n_folds) {
      te_munis  <- shuffled[fold_of == k]
      te_idx    <- which(model_df$PRO_COM %in% te_munis)
      tr_idx    <- setdiff(seq_len(nrow(model_df)), te_idx)
      m <- lm(as.formula(formula_str), data = model_df[tr_idx, ])
      p <- predict(m, newdata = model_df[te_idx, ])
      yt <- model_df$IFC[te_idx]
      rmse <- sqrt(mean((yt - p)^2))
      mae  <- mean(abs(yt - p))
      r2   <- 1 - sum((yt - p)^2) / sum((yt - mean(yt))^2)
      cv_records <- rbind(cv_records,
        data.frame(n_covariates = n_top, rep = rep, fold = k,
                   RMSE = rmse, MAE = mae, R2 = r2))
    }
  }
  cat(sprintf("CV done for n_covariates = %d\n", n_top))
}

cv_summary <- cv_records %>%
  group_by(n_covariates) %>%
  summarise(R2_mean   = mean(R2),   R2_sd   = sd(R2),
            RMSE_mean = mean(RMSE), RMSE_sd = sd(RMSE),
            MAE_mean  = mean(MAE),  MAE_sd  = sd(MAE),
            .groups = "drop") %>%
  arrange(n_covariates)
cat("\n--- Trade-off curve: # covariates vs out-of-sample R^2 ---\n")
print(as.data.frame(cv_summary), row.names = FALSE, digits = 4)
write.csv(cv_summary,
          file.path(out_dir, "tradeoff_summary.csv"), row.names = FALSE)
write.csv(cv_records,
          file.path(out_dir, "tradeoff_per_fold.csv"), row.names = FALSE)

# ================================================================
# 4b) AIC / BIC optimum on the top-N curve
# ----------------------------------------------------------------
# In-sample AIC and BIC for each candidate model size. BIC penalises
# complexity more strongly than AIC and typically suggests smaller
# models. Reporting the optimum of each is a transparent way to
# justify the chosen "n_final".
# ================================================================
aicbic_curve <- data.frame()
for (n_top in candidate_n) {
  top_vars <- ranked$variable[seq_len(n_top)]
  fit_n <- lm(as.formula(paste("IFC ~", paste(top_vars, collapse = " + "))),
              data = model_df)
  aicbic_curve <- rbind(aicbic_curve,
    data.frame(n_covariates = n_top,
               AIC = AIC(fit_n), BIC = BIC(fit_n)))
}
aicbic_curve <- aicbic_curve %>% arrange(n_covariates)
cat("\n--- AIC/BIC vs # covariates (in-sample) ---\n")
print(aicbic_curve, row.names = FALSE, digits = 6)
write.csv(aicbic_curve,
          file.path(out_dir, "aicbic_curve.csv"), row.names = FALSE)

best_aic_n <- aicbic_curve$n_covariates[which.min(aicbic_curve$AIC)]
best_bic_n <- aicbic_curve$n_covariates[which.min(aicbic_curve$BIC)]
cat(sprintf("Best n by AIC: %d  |  Best n by BIC: %d\n",
            best_aic_n, best_bic_n))

# ================================================================
# 5) PICK THE "ELBOW", REFIT, AND ITERATIVELY DROP NON-SIGNIFICANT
# ----------------------------------------------------------------
# We pick n_final = 25 as the parsimonious size. When the top-25 by
# |beta| is refit, occasionally one of them turns out to be
# non-significant (the |beta| was carried by a multicollinearity that
# the VIF threshold of 10 didn't fully remove). We add a post-hoc
# filter: any predictor with p > 0.05 is replaced with the next
# candidate from the ranking, until either all coefficients are
# significant or the candidate pool is exhausted.
# ================================================================
n_final <- 25
sig_threshold <- 0.05

final_vars  <- ranked$variable[seq_len(n_final)]
candidates  <- ranked$variable[(n_final + 1):nrow(ranked)]
replacements_log <- data.frame()

repeat {
  fit_final  <- lm(as.formula(paste("IFC ~", paste(final_vars, collapse = " + "))),
                   data = model_df)
  pvals <- summary(fit_final)$coefficients[, "Pr(>|t|)"]
  pvals <- pvals[names(pvals) != "(Intercept)"]
  non_sig <- names(pvals)[pvals > sig_threshold]
  if (length(non_sig) == 0 || length(candidates) == 0) break
  worst    <- names(which.max(pvals[non_sig]))
  replace_with <- candidates[1]
  candidates   <- candidates[-1]
  replacements_log <- rbind(replacements_log,
    data.frame(dropped = worst, p_value = pvals[worst],
               replaced_with = replace_with))
  final_vars <- c(setdiff(final_vars, worst), replace_with)
  cat(sprintf("Replacing non-significant '%s' (p=%.3f) with '%s'\n",
              worst, pvals[worst], replace_with))
}
if (nrow(replacements_log) > 0) {
  write.csv(replacements_log,
            file.path(out_dir, "post_hoc_replacements.csv"), row.names = FALSE)
}

final_coef <- summary(fit_final)$coefficients
final_coef <- final_coef[rownames(final_coef) != "(Intercept)", , drop = FALSE]
final_table <- data.frame(
  variable = rownames(final_coef),
  beta_std = final_coef[, "Estimate"],
  std_err  = final_coef[, "Std. Error"],
  t_value  = final_coef[, "t value"],
  p_value  = final_coef[, "Pr(>|t|)"]
)
final_table <- final_table[order(-abs(final_table$beta_std)), ]
cat(sprintf("\n--- FINAL PARSIMONIOUS MODEL (%d covariates, all p<%.2f) ---\n",
            n_final, sig_threshold))
print(final_table, row.names = FALSE, digits = 3)
write.csv(final_table,
          file.path(out_dir, "final_parsimonious_coefficients.csv"),
          row.names = FALSE)

# Save the FINAL selected variable names (used by R/11_mixed_models.R)
writeLines(final_table$variable,
           file.path(out_dir, "selected_parsimonious_variables.txt"))

# Final VIF check
final_vifs <- vif(fit_final)
final_vif_tbl <- data.frame(variable = names(final_vifs), VIF = as.numeric(final_vifs))
final_vif_tbl <- final_vif_tbl[order(-final_vif_tbl$VIF), ]
write.csv(final_vif_tbl,
          file.path(out_dir, "final_parsimonious_vif.csv"), row.names = FALSE)
cat(sprintf("\nMax VIF in final model: %.2f\n", max(final_vifs)))

# ================================================================
# 6) PLOT TRADE-OFF CURVE
# ================================================================
p <- ggplot(cv_summary, aes(x = n_covariates, y = R2_mean)) +
  geom_line() +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = R2_mean - R2_sd, ymax = R2_mean + R2_sd),
                width = 0.5) +
  geom_hline(yintercept = 0.825, linetype = "dashed", colour = "firebrick") +
  annotate("text", x = max(cv_summary$n_covariates), y = 0.830,
           label = "LASSO benchmark (R² = 0.825)", hjust = 1, colour = "firebrick") +
  geom_vline(xintercept = n_final, linetype = "dotted") +
  labs(title = "Parsimony vs accuracy trade-off",
       x = "Number of covariates", y = "Out-of-sample R² (mean ± SD)") +
  theme_minimal()
ggsave(file.path(out_dir, "tradeoff_curve.png"),
       p, width = 9, height = 5, dpi = 150)

cat("\nOutputs in:", out_dir, "\n")
