# ================================================================
# LINEAR MIXED MODELS — geographic and taxonomic hierarchies
# ----------------------------------------------------------------
# Question: does explicitly modelling region/province/GRINS hierarchy
# as a random effect add information beyond what the fixed-effect
# benchmark already captures?
#
# Fixed-effects source (chosen in this order, first one available):
#   1) outputs/parsimonious_model/selected_parsimonious_variables.txt
#      -> the 25-covariate set selected by R/14_parsimonious_model.R
#         (VIF iterative pruning + top-N ranking + significance filter)
#   2) outputs/final_benchmark/final_coefficients_standardised.csv
#      -> the 191 LASSO-selected covariates of R/10_final_benchmark.R
#
# Four random-effects variants are fit on the same fixed-effects
# matrix, so the comparison is "apples to apples":
#
#   M_geo           : IFC ~ X + (1 | region)
#   M_geo_nested    : IFC ~ X + (1 | region / province)
#   M_tax           : IFC ~ X + (1 | GRINS_macroclass)
#   M_geo_plus_tax  : IFC ~ X + (1 | region) + (1 | GRINS_macroclass)
#
# Reported per model:
#   - R^2 marginal  (fixed effects only)
#   - R^2 conditional (fixed + random)
#   - ICC (share of residual variance that lies between groups)
#   - 5-fold panel-aware test R^2 / RMSE / MAE (same split logic as
#     the LASSO benchmark, for direct comparison)
# ================================================================

suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(MuMIn); library(dplyr)
})

set.seed(2026)

source("R/_utils.R")

out_dir <- "outputs/mixed_models"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ================================================================
# 1-2) DATA LOADING + FEATURE ENGINEERING (delegated to R/_utils.R)
# ================================================================
prep <- pmu_prepare_data(with_taxonomy_filter = TRUE)
data <- prep$data; X_fe <- prep$X_fe; y <- prep$y
cat("Rows after merge + group filter:", nrow(data), "\n")
cat("Regions:",    length(unique(data$nome_regione)),
    " | Provinces:", length(unique(data$nome_provincia)),
    " | GRINS macroclasses:", length(unique(data$GRINS_macroclass)), "\n")
cat("Feature matrix:", dim(X_fe), "\n")

# ================================================================
# 3) FIXED-EFFECTS SET — parsimonious 25 (with documented fallback)
# ================================================================
selected <- pmu_get_parsimonious_variables(X_fe, n_fallback = 25)
cat(sprintf("Fixed-effects predictors: %d\n", length(selected)))

X_sel <- X_fe[, selected, drop = FALSE]
# Standardise predictors so that lmer coefficients are comparable
X_sel <- scale(as.matrix(X_sel))

model_data <- data.frame(
  IFC              = y,
  region           = factor(data$nome_regione),
  province         = factor(data$nome_provincia),
  GRINS_macroclass = factor(data$GRINS_macroclass),
  PRO_COM          = data$PRO_COM,
  year             = data$year
)
model_data <- cbind(model_data, X_sel)

# Build the fixed-effects part of the formula
X_terms <- paste(selected, collapse = " + ")

# ================================================================
# 4) FIT THE FOUR MIXED MODELS ON FULL DATA
# ----------------------------------------------------------------
# Helper that captures any convergence warnings emitted by lme4
# so we can flag silently misbehaving fits at the end of the script.
# ================================================================
fit_lmm_safe <- function(formula_str, data) {
  warns <- character()
  fit <- withCallingHandlers(
    lmer(as.formula(formula_str), data = data, REML = FALSE,
         control = lmerControl(check.conv.singular = .makeCC(action = "ignore", tol = 1e-4))),
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  attr(fit, "warnings") <- warns
  attr(fit, "singular") <- isSingular(fit, tol = 1e-4)
  fit
}

cat("\n[Fitting M_geo]\n")
M_geo <- fit_lmm_safe(paste("IFC ~", X_terms, "+ (1 | region)"), model_data)

cat("[Fitting M_geo_nested]\n")
M_geo_nested <- fit_lmm_safe(paste("IFC ~", X_terms, "+ (1 | region/province)"), model_data)

cat("[Fitting M_tax]\n")
M_tax <- fit_lmm_safe(paste("IFC ~", X_terms, "+ (1 | GRINS_macroclass)"), model_data)

# Improvement #2: combined geography + taxonomy
# Tests whether GRINS_macroclass still contributes anything once region
# is already in the model. If the taxonomy is redundant with the
# geography (as we expect from the M_tax-alone result), the macroclass
# variance component should collapse to ~0 here too.
cat("[Fitting M_geo_plus_tax]\n")
M_geo_plus_tax <- fit_lmm_safe(
  paste("IFC ~", X_terms, "+ (1 | region) + (1 | GRINS_macroclass)"),
  model_data
)

# Improvement #4: convergence sanity check
convergence_report <- data.frame(
  model = c("M_geo", "M_geo_nested", "M_tax", "M_geo_plus_tax"),
  singular = sapply(list(M_geo, M_geo_nested, M_tax, M_geo_plus_tax),
                    function(f) attr(f, "singular")),
  n_warnings = sapply(list(M_geo, M_geo_nested, M_tax, M_geo_plus_tax),
                      function(f) length(attr(f, "warnings")))
)
cat("\n--- Convergence check ---\n")
print(convergence_report, row.names = FALSE)
write.csv(convergence_report,
          file.path(out_dir, "convergence_report.csv"), row.names = FALSE)

# ================================================================
# 5) IN-SAMPLE METRICS: R^2 marginal / conditional + ICC + variances
# ================================================================
extract_metrics <- function(m, name) {
  r2 <- tryCatch(r.squaredGLMM(m), error = function(e) c(R2m = NA, R2c = NA))
  vc <- as.data.frame(VarCorr(m))[, c("grp", "vcov")]
  total_var <- sum(vc$vcov)
  random_var <- sum(vc$vcov[vc$grp != "Residual"])
  data.frame(
    model        = name,
    R2_marginal  = r2[1],
    R2_conditional = r2[2],
    ICC          = random_var / total_var,
    sigma2_resid = vc$vcov[vc$grp == "Residual"],
    sigma2_random = random_var,
    AIC          = AIC(m),
    BIC          = BIC(m)
  )
}

insample <- bind_rows(
  extract_metrics(M_geo,          "M_geo (1|region)"),
  extract_metrics(M_geo_nested,   "M_geo_nested (1|region/province)"),
  extract_metrics(M_tax,          "M_tax (1|GRINS_macroclass)"),
  extract_metrics(M_geo_plus_tax, "M_geo_plus_tax (1|region)+(1|macroclass)")
)
print(insample, row.names = FALSE, digits = 4)
write.csv(insample, file.path(out_dir, "in_sample_summary.csv"), row.names = FALSE)

# ================================================================
# 6) PANEL-AWARE 5-FOLD CV (single repeat, for runtime reasons)
# ----------------------------------------------------------------
# Each municipality stays in a single fold (same split logic as
# the LASSO benchmark in script 10).
# ================================================================
municipalities <- unique(model_data$PRO_COM)
shuffled <- sample(municipalities)
fold_of  <- (seq_along(shuffled) - 1) %% 5 + 1

cv_records <- data.frame()
for (k in 1:5) {
  test_munis <- shuffled[fold_of == k]
  test_idx   <- which(model_data$PRO_COM %in% test_munis)
  train_idx  <- setdiff(seq_len(nrow(model_data)), test_idx)

  for (cfg in list(
    list(name = "M_geo",          formula = paste("IFC ~", X_terms, "+ (1 | region)")),
    list(name = "M_geo_nested",   formula = paste("IFC ~", X_terms, "+ (1 | region/province)")),
    list(name = "M_tax",          formula = paste("IFC ~", X_terms, "+ (1 | GRINS_macroclass)")),
    list(name = "M_geo_plus_tax", formula = paste("IFC ~", X_terms, "+ (1 | region) + (1 | GRINS_macroclass)"))
  )) {
    fit <- tryCatch(
      lmer(as.formula(cfg$formula), data = model_data[train_idx, ], REML = FALSE,
           control = lmerControl(check.conv.singular = .makeCC(action = "ignore", tol = 1e-4))),
      error = function(e) NULL
    )
    if (is.null(fit)) next
    # allow.new.levels handles regions/provinces present only in test
    pred <- tryCatch(
      predict(fit, newdata = model_data[test_idx, ], allow.new.levels = TRUE),
      error = function(e) rep(NA_real_, length(test_idx))
    )
    yt <- model_data$IFC[test_idx]
    rmse <- sqrt(mean((yt - pred)^2, na.rm = TRUE))
    mae  <- mean(abs(yt - pred), na.rm = TRUE)
    r2   <- 1 - sum((yt - pred)^2, na.rm = TRUE) / sum((yt - mean(yt))^2)
    cv_records <- rbind(cv_records,
      data.frame(model = cfg$name, fold = k, RMSE = rmse, MAE = mae, R2 = r2))
    cat(sprintf("  fold %d / %-15s done (R2=%.3f)\n", k, cfg$name, r2))
  }
}

cv_summary <- cv_records %>%
  group_by(model) %>%
  summarise(
    R2_mean   = mean(R2),   R2_sd   = sd(R2),
    RMSE_mean = mean(RMSE), RMSE_sd = sd(RMSE),
    MAE_mean  = mean(MAE),  MAE_sd  = sd(MAE),
    n_folds   = n(), .groups = "drop"
  )

cat("\n--- CV summary (5-fold panel-aware, 1 repeat) ---\n")
print(as.data.frame(cv_summary), row.names = FALSE, digits = 4)
write.csv(cv_summary,
          file.path(out_dir, "mixed_models_cv_summary.csv"), row.names = FALSE)
write.csv(cv_records,
          file.path(out_dir, "mixed_models_cv_per_fold.csv"), row.names = FALSE)

# ================================================================
# 7) RANDOM-EFFECT ESTIMATES (BLUPs) for the best model
# ----------------------------------------------------------------
# These are interpretable: how much each region/province/macroclass
# differs from the overall mean once X has been controlled for.
# ================================================================
saveRDS(M_geo,          file.path(out_dir, "fit_M_geo.rds"))
saveRDS(M_geo_nested,   file.path(out_dir, "fit_M_geo_nested.rds"))
saveRDS(M_tax,          file.path(out_dir, "fit_M_tax.rds"))
saveRDS(M_geo_plus_tax, file.path(out_dir, "fit_M_geo_plus_tax.rds"))

# Regional offsets from M_geo
ranef_geo <- as.data.frame(ranef(M_geo)$region)
ranef_geo$region <- rownames(ranef_geo)
names(ranef_geo)[1] <- "intercept_offset"
ranef_geo <- ranef_geo[order(-abs(ranef_geo$intercept_offset)), ]
write.csv(ranef_geo, file.path(out_dir, "ranef_region.csv"), row.names = FALSE)
cat("\nTop 10 regional offsets (M_geo) — higher = more fragile:\n")
print(head(ranef_geo, 10), row.names = FALSE, digits = 3)

# Improvement #3: province offsets from M_geo_nested (the best model)
# lme4 stores nested random effects under the name "province:region".
# Each row is "<province>:<region>", with the offset relative to the
# regional baseline (i.e. additional fragility on top of the region).
ranef_prov_raw <- ranef(M_geo_nested)
prov_key <- grep("province", names(ranef_prov_raw), value = TRUE)
ranef_prov <- as.data.frame(ranef_prov_raw[[prov_key]])
ranef_prov$province_region <- rownames(ranef_prov)
names(ranef_prov)[1] <- "intercept_offset"

# Polish B: fix UTF-8 mojibake in region/province names (Vallée d'Aoste etc.)
fix_encoding <- function(x) {
  # The source RDS stores some names with double-encoded UTF-8
  # (UTF-8 bytes mis-interpreted as Latin1 then re-encoded). Reversing
  # the Latin1->UTF-8 step recovers the original characters.
  out <- iconv(x, from = "UTF-8", to = "latin1")
  out <- iconv(out, from = "UTF-8", to = "UTF-8", sub = "?")
  Encoding(out) <- "UTF-8"
  out
}
ranef_prov$province_region <- fix_encoding(ranef_prov$province_region)

# Polish A: present BOTH "largest absolute offsets" AND "signed-sorted"
#  - sort by |offset| -> most extreme provinces (in either direction)
#  - sort by raw offset, top of head() and tail() -> most fragile / least fragile
ranef_prov_by_abs <- ranef_prov[order(-abs(ranef_prov$intercept_offset)), ]
ranef_prov_by_sign <- ranef_prov[order(-ranef_prov$intercept_offset), ]
write.csv(ranef_prov_by_abs, file.path(out_dir, "ranef_province.csv"), row.names = FALSE)

cat("\nTop 10 province offsets — most extreme |offset| (M_geo_nested):\n")
print(head(ranef_prov_by_abs, 10), row.names = FALSE, digits = 3)

cat("\nTop 10 MOST FRAGILE provinces (signed, beyond regional baseline):\n")
print(head(ranef_prov_by_sign, 10), row.names = FALSE, digits = 3)

cat("\nTop 10 LEAST FRAGILE provinces (signed, beyond regional baseline):\n")
print(head(ranef_prov_by_sign[order(ranef_prov_by_sign$intercept_offset), ], 10),
      row.names = FALSE, digits = 3)

# Same encoding fix on the regional table
ranef_geo$region <- fix_encoding(ranef_geo$region)
write.csv(ranef_geo, file.path(out_dir, "ranef_region.csv"), row.names = FALSE)

cat("\nOutputs in:", out_dir, "\n")
