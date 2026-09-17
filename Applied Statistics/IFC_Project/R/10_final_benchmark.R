# ================================================================
# FINAL BENCHMARK SCRIPT
# Target  : IFC raw values (95% decile match, IFC_Final_Analysis_Sorted)
# X       : GRINS V3 municipal indicators
# Method  : LASSO for variable selection + linear model on selected vars
# Output  : Honest test-set metrics, top predictors, stability, diagnostics
# ================================================================
# What this script does, end-to-end:
#   1) Load IFC raw values (2019 and 2021, pooled long-format)
#   2) Load GRINS V3 municipal panel
#   3) Merge on (PRO_COM, year)
#   4) Variable classification (count / count_neg / rate / size / skip)
#   5) Feature engineering (log1p, per-capita, winsorising)
#   6) Repeated 5x5 panel-aware cross-validation -> honest R^2
#   7) Final LASSO on full data + standardized coefficients
#   8) Stability selection (selection frequency across 25 folds)
#   9) Residual diagnostics on the final linear model
# ================================================================

suppressPackageStartupMessages({
  library(readxl);  library(dplyr);  library(tidyr)
  library(glmnet);  library(ggplot2)
})

set.seed(2026)

# -----------------------------
# Paths and output folder
# -----------------------------
ifc_file   <- "data/raw/ifc/final_analysis_sorted.xlsx"
grins_file <- "data/processed/grins_v3/comunale_v3.rds"
out_dir    <- "outputs/final_benchmark"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# OneDrive may keep the file as a cloud-only placeholder; copy locally
if (file.exists(ifc_file)) {
  tmp <- file.path(tempdir(), "IFC_tmp.xlsx")
  file.copy(ifc_file, tmp, overwrite = TRUE)
  ifc_file <- tmp
}

# ================================================================
# 1) LOAD IFC TARGET (long format: one row per municipality x year)
# ================================================================
ifc_long <- read_excel(ifc_file) %>%
  select(PRO_COM, IFC_2019, IFC_2021) %>%
  pivot_longer(c(IFC_2019, IFC_2021), names_to = "year", values_to = "IFC") %>%
  mutate(year    = as.integer(sub("IFC_", "", year)),
         PRO_COM = as.numeric(PRO_COM)) %>%
  filter(!is.na(IFC), !is.na(PRO_COM))
cat("IFC rows (long):", nrow(ifc_long), "\n")

# ================================================================
# 2) LOAD GRINS V3 (municipal panel, ~100k rows x 549 cols)
# ================================================================
grins <- readRDS(grins_file) %>%
  filter(anno %in% c(2019, 2021)) %>%
  mutate(codice_comune = as.numeric(codice_comune))

# Keep only numeric predictor columns
key_cols <- c("codice_comune", "anno")
num_cols <- setdiff(names(grins)[sapply(grins, is.numeric)], key_cols)
cat("GRINS numeric columns:", length(num_cols), "\n")

# ================================================================
# 3) MERGE IFC <-> GRINS
# ================================================================
data <- ifc_long %>%
  inner_join(grins %>% select(all_of(c(key_cols, num_cols))),
             by = c("PRO_COM" = "codice_comune", "year" = "anno"))
cat("Merged rows:", nrow(data),
    " | unique municipalities:", length(unique(data$PRO_COM)), "\n")

# Robust population (used as denominator for per-capita transforms)
pop <- data$Popolazione
pop[is.na(pop) | pop <= 0] <- median(pop[pop > 0], na.rm = TRUE)
y <- data$IFC

# ================================================================
# 4) VARIABLE CLASSIFICATION
# ----------------------------------------------------------------
# Rationale for each class:
#   "count"      -> non-negative size-dependent counts (employees,
#                   establishments, taxpayers, deaths, ...). Strongly
#                   right-skewed and confounded with municipality size.
#                   Treatment: divide by population (per-capita) then
#                   apply log1p to compress the right tail.
#   "count_neg"  -> counts that can be negative (e.g. net migration).
#                   Treatment: per-capita only (log not defined).
#   "size"       -> population and area: needed as covariates but
#                   should not be divided by themselves. Treatment:
#                   log1p only.
#   "rate"       -> indices, percentages, ratios, means, already
#                   per-capita variables. Treatment: NONE (already
#                   on a meaningful scale).
#   "skip"       -> codes, names, "year_x/y" merge markers, method
#                   strings: dropped from the matrix.
# ================================================================
classify <- function(nm) {
  ln <- tolower(nm)
  if (grepl("_anno_x$|_anno_y$|^cod_|^den_|sigla|nome_|^stringa|backcast|recovery|tipo_na", ln))
    return("skip")
  if (ln %in% c("popolazione", "superficie_totale_kmq_formattato", "superfici kmq"))
    return("size")
  rate_patterns <- c(
    "indice", "incidenza", "mobilità", "mobilita", "percentuale",
    "pro capite", "procapite", "pro-capite",
    "per_addetto", "per_dipendente", "valori_percentuali",
    "_media_", "_media$", "^eta_", "^anzianita_",
    "media_donne", "media_uomini", "media_media",
    "coverage", "contribuenti_su_pop", "reddito_medio", "reddito_pc",
    "lacc_mean", "employee_services", "employee_concentration",
    "degree_stem", "degree_concentration",
    "^no2_", "^pm10_", "^pm25_", "^o3_", "_media_valori_annuali",
    "distanza_", "^elezion", "regionali",
    "verde urbano", "suolo consumato", "acqua potabile",
    "produzione pro", "sau", "ricettività", "ricettivita", "densità"
  )
  if (any(sapply(rate_patterns, function(p) grepl(p, ln)))) return("rate")
  if (grepl("saldo", ln)) return("count_neg")
  "count"
}

# ================================================================
# 5) FEATURE ENGINEERING
# ----------------------------------------------------------------
# Pipeline applied in order:
#   - drop columns with >20% NA (too sparse to be trustworthy)
#   - median imputation for remaining NAs
#   - per-class transforms (see classify() above)
#   - winsorise at 1st / 99th percentile to neutralise residual
#     extreme outliers (Rome, Milan, etc.) without dropping rows
#   - drop columns with zero variance after the above
# ================================================================
build_feature_matrix <- function(df_num, pop) {
  cls  <- sapply(names(df_num), classify)
  keep <- names(df_num)[cls != "skip"]
  X    <- df_num[, keep]; cls <- cls[keep]

  na_prop <- sapply(X, function(z) mean(is.na(z)))
  X <- X[, names(na_prop[na_prop <= 0.20])]; cls <- cls[names(X)]

  for (j in seq_along(X)) {
    if (any(is.na(X[[j]]))) X[[j]][is.na(X[[j]])] <- median(X[[j]], na.rm = TRUE)
  }

  for (nm in names(X)) {
    cc <- cls[nm]
    if (cc == "count") {
      X[[nm]] <- X[[nm]] / pop
      X[[nm]][!is.finite(X[[nm]])] <- 0
      X[[nm]] <- log1p(pmax(X[[nm]], 0))
    } else if (cc == "count_neg") {
      X[[nm]] <- X[[nm]] / pop
      X[[nm]][!is.finite(X[[nm]])] <- 0
    } else if (cc == "size") {
      X[[nm]] <- log1p(pmax(X[[nm]], 0))
    }
  }

  wins <- function(z) {
    q <- quantile(z, c(0.01, 0.99), na.rm = TRUE)
    pmin(pmax(z, q[1]), q[2])
  }
  X[] <- lapply(X, wins)
  X   <- X[, sapply(X, var) > 0]
  attr(X, "classes") <- cls[names(X)]
  X
}

df_num <- data %>% select(all_of(num_cols))
X_fe   <- build_feature_matrix(df_num, pop)
# Add a post-COVID indicator as control variable
X_fe$year2021 <- as.integer(data$year == 2021)

classes_full <- c(attr(X_fe, "classes"), year2021 = "rate")
names(X_fe)  <- make.names(names(X_fe), unique = TRUE)
X_mat <- as.matrix(X_fe)
cat("Feature matrix:", dim(X_mat), "\n")
cat("Variable classes:\n"); print(table(classes_full[colnames(X_mat)]))

# ================================================================
# 6) HONEST PERFORMANCE: REPEATED 5x5 PANEL-AWARE CV
# ----------------------------------------------------------------
# Why panel-aware: random splits put the same municipality in both
# train (2019) and test (2021), which leaks spatial information.
# Panel-aware splits keep all rows of a given PRO_COM in the same
# fold, so the test set contains municipalities the model has
# never seen. R^2 reported here is what we expect on truly new
# municipalities.
#
# We repeat 5 times to estimate variability of R^2 across splits.
# Inside each outer fold we use cv.glmnet (5-fold internal CV) to
# pick lambda. Both lambda.min and lambda.1se are evaluated.
# ================================================================
municipalities <- unique(data$PRO_COM)
n_repeats <- 5
n_folds   <- 5
lambda_grid <- 10^seq(5, -3, length.out = 100)

cv_records  <- data.frame()
sel_matrix  <- matrix(0,
                      nrow = ncol(X_mat),
                      ncol = n_repeats * n_folds,
                      dimnames = list(colnames(X_mat), NULL))
fold_counter <- 0

for (rep in 1:n_repeats) {
  set.seed(2026 + rep)
  shuffled <- sample(municipalities)
  fold_of  <- (seq_along(shuffled) - 1) %% n_folds + 1

  for (k in 1:n_folds) {
    fold_counter <- fold_counter + 1
    test_munis <- shuffled[fold_of == k]
    test_idx   <- which(data$PRO_COM %in% test_munis)
    train_idx  <- setdiff(seq_len(nrow(X_mat)), test_idx)

    cv  <- cv.glmnet(X_mat[train_idx, ], y[train_idx],
                     alpha = 1, nfolds = 5, lambda = lambda_grid)
    fit <- glmnet(X_mat[train_idx, ], y[train_idx],
                  alpha = 1, lambda = lambda_grid)

    pred_1se <- as.vector(predict(fit, s = cv$lambda.1se, newx = X_mat[test_idx, ]))
    pred_min <- as.vector(predict(fit, s = cv$lambda.min, newx = X_mat[test_idx, ]))

    metric <- function(p) c(
      RMSE = sqrt(mean((y[test_idx] - p)^2)),
      MAE  = mean(abs(y[test_idx] - p)),
      R2   = 1 - sum((y[test_idx] - p)^2) /
                 sum((y[test_idx] - mean(y[test_idx]))^2)
    )

    # Track selected variables for stability selection (8 below)
    selected <- rownames(predict(fit, s = cv$lambda.1se, type = "coefficients"))[
      as.vector(predict(fit, s = cv$lambda.1se, type = "coefficients")) != 0]
    selected <- setdiff(selected, "(Intercept)")
    sel_matrix[selected, fold_counter] <- 1

    cv_records <- rbind(
      cv_records,
      data.frame(rep = rep, fold = k, model = "Lasso.1se",
                 t(metric(pred_1se)),
                 n_sel = length(selected)),
      data.frame(rep = rep, fold = k, model = "Lasso.min",
                 t(metric(pred_min)),
                 n_sel = sum(as.vector(
                   predict(fit, s = cv$lambda.min, type = "coefficients")) != 0) - 1)
    )
    cat(sprintf("  rep %d / fold %d done\n", rep, k))
  }
}

cv_summary <- cv_records %>%
  group_by(model) %>%
  summarise(R2_mean   = mean(R2),    R2_sd   = sd(R2),
            RMSE_mean = mean(RMSE),  RMSE_sd = sd(RMSE),
            MAE_mean  = mean(MAE),   MAE_sd  = sd(MAE),
            n_sel_mean = mean(n_sel), .groups = "drop")
cat("\n--- Repeated 5x5 panel-aware CV ---\n")
print(as.data.frame(cv_summary))
write.csv(cv_summary,
          file.path(out_dir, "repeated_cv_summary.csv"), row.names = FALSE)
write.csv(cv_records,
          file.path(out_dir, "repeated_cv_per_fold.csv"), row.names = FALSE)

# ================================================================
# 7) FINAL LASSO ON FULL DATA + STANDARDISED COEFFICIENTS
# ----------------------------------------------------------------
# The "final" reported model is fit on the entire dataset using the
# same lambda search. We report standardised effects (beta * sd(x))
# because raw coefficients are on the engineered scale
# (log1p(x/pop) for counts) and not directly interpretable in
# original units. Standardised effects say: "a 1-SD shift of the
# engineered variable changes IFC by this much".
# ================================================================
cv_full  <- cv.glmnet(X_mat, y, alpha = 1, nfolds = 10, lambda = lambda_grid)
fit_full <- glmnet(X_mat, y, alpha = 1, lambda = lambda_grid)
coef_1se <- predict(fit_full, s = cv_full$lambda.1se, type = "coefficients")

coef_df <- data.frame(variable = rownames(coef_1se),
                      coef     = as.vector(coef_1se))
coef_df <- coef_df[coef_df$variable != "(Intercept)" & coef_df$coef != 0, ]
sd_x <- apply(X_mat[, coef_df$variable, drop = FALSE], 2, sd)
coef_df$sd_x        <- sd_x[coef_df$variable]
coef_df$beta_std    <- coef_df$coef * coef_df$sd_x
coef_df$variable_class <- classes_full[coef_df$variable]
coef_df <- coef_df[order(-abs(coef_df$beta_std)), ]

cat(sprintf("\nFinal LASSO (lambda.1se) selected %d / %d predictors\n",
            nrow(coef_df), ncol(X_mat)))
cat("\nTop 15 by |beta_std|:\n")
print(head(coef_df, 15), row.names = FALSE, digits = 3)
write.csv(coef_df,
          file.path(out_dir, "final_coefficients_standardised.csv"),
          row.names = FALSE)

p_top <- ggplot(head(coef_df, 15),
                aes(x = reorder(variable, beta_std),
                    y = beta_std, fill = variable_class)) +
  geom_col() + coord_flip() +
  labs(title = "Top 15 predictors — standardised effect (beta * sd)",
       x = NULL, y = "Expected change in IFC per +1 SD of x") +
  theme_minimal()
ggsave(file.path(out_dir, "top15_standardised.png"),
       p_top, width = 10, height = 7)

# ================================================================
# 8) STABILITY SELECTION
# ----------------------------------------------------------------
# For each predictor, fraction of the 25 CV folds in which it
# received a non-zero coefficient. Variables selected in all 25
# folds form the "robust core" -- their inclusion is not an
# artefact of one particular split.
# ================================================================
selection_freq <- rowMeans(sel_matrix)
stability_df <- data.frame(variable = names(selection_freq),
                           selection_frequency = selection_freq)
beta_lookup <- setNames(coef_df$beta_std, coef_df$variable)
stability_df$beta_std_full <- beta_lookup[stability_df$variable]
stability_df$beta_std_full[is.na(stability_df$beta_std_full)] <- 0
stability_df <- stability_df[order(-stability_df$selection_frequency,
                                   -abs(stability_df$beta_std_full)), ]
write.csv(stability_df,
          file.path(out_dir, "stability_selection.csv"),
          row.names = FALSE)

robust_core <- stability_df[stability_df$selection_frequency == 1, ]
cat(sprintf("\nRobust core (selected in 25/25 folds): %d variables\n",
            nrow(robust_core)))

p_stab <- ggplot(stability_df[stability_df$selection_frequency > 0, ],
                 aes(x = selection_frequency)) +
  geom_histogram(bins = 26, fill = "steelblue", color = "white") +
  labs(title = "Stability selection — frequency distribution",
       x = "Fraction of folds (out of 25) selecting the variable",
       y = "Number of variables") +
  theme_minimal()
ggsave(file.path(out_dir, "stability_histogram.png"),
       p_stab, width = 8, height = 5)

# ================================================================
# 9) FINAL LINEAR MODEL ON SELECTED VARIABLES + DIAGNOSTICS
# ----------------------------------------------------------------
# Refit an ordinary lm() on the LASSO-selected predictors. This is
# the "simple linear regression" deliverable the professor asked for.
# Diagnostics check linear-model assumptions (linearity,
# homoscedasticity, normality of residuals).
# ================================================================
selected_vars <- coef_df$variable
X_sel_df <- as.data.frame(X_mat[, selected_vars, drop = FALSE])
names(X_sel_df) <- make.names(names(X_sel_df))
fit_lm <- lm(y ~ ., data = data.frame(y = y, X_sel_df))

png(file.path(out_dir, "lm_diagnostics.png"),
    width = 1100, height = 850)
par(mfrow = c(2, 2)); plot(fit_lm); par(mfrow = c(1, 1))
dev.off()

residuals_lm <- residuals(fit_lm)
fitted_lm    <- fitted(fit_lm)

skewness <- function(x) mean((x - mean(x))^3) / sd(x)^3
kurtosis <- function(x) mean((x - mean(x))^4) / sd(x)^4
shapiro_p <- shapiro.test(sample(residuals_lm, 5000))$p.value
bp_aux    <- lm(I(residuals_lm^2) ~ fitted_lm)
bp_stat   <- summary(bp_aux)$r.squared * length(residuals_lm)
bp_p      <- pchisq(bp_stat, df = 1, lower.tail = FALSE)

diag_tbl <- data.frame(
  Metric = c("Residual SD", "Skewness", "Kurtosis",
             "Shapiro p (n=5000 sample)", "Breusch-Pagan p"),
  Value  = c(sd(residuals_lm), skewness(residuals_lm),
             kurtosis(residuals_lm), shapiro_p, bp_p)
)
cat("\n--- Residual diagnostics ---\n")
print(diag_tbl, row.names = FALSE, digits = 4)
write.csv(diag_tbl,
          file.path(out_dir, "residual_diagnostics.csv"),
          row.names = FALSE)

cat("\nAll outputs saved to:", out_dir, "\n")
