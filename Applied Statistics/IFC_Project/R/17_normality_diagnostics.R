# ================================================================
# RESIDUAL NORMALITY DIAGNOSTICS — extended
# ----------------------------------------------------------------
# Out-of-sample R^2 says nothing about whether the residuals are
# normally distributed. The previous report relied on a single
# Shapiro--Wilk p-value, which is unreliable at n = 15 000 (any
# tiny deviation from normality is detected as significant). This
# script adds:
#
#  1) QQ plot of the residuals of the parsimonious LM
#  2) Histogram of residuals overlaid with the theoretical N(0, sigma)
#  3) Multiple normality tests:
#       - Anderson-Darling   (robust at large n)
#       - Jarque-Bera        (sensitive to skewness + kurtosis)
#       - Shapiro-Wilk       (on a 5000-row sample, for reference)
#  4) QQ plot of the BLUPs (random intercepts) of M_geo_nested
#     -> mixed models also assume Normal random effects
#  5) Box-Cox check on IFC: would a transformation flatten the tails?
# ================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(nortest)   # Anderson-Darling
  library(tseries)   # Jarque-Bera
  library(MASS)      # Box-Cox -- must come BEFORE dplyr so that
  library(dplyr)     # dplyr::select wins the namespace race
  library(lme4)
})
source("R/_utils.R")

set.seed(2026)

fit_geo_nested <- "outputs/mixed_models/fit_M_geo_nested.rds"
out_dir        <- "outputs/normality_diagnostics"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ================================================================
# 1) Reproduce the parsimonious data
# ================================================================
prep <- pmu_prepare_data(with_taxonomy_filter = FALSE)
data <- prep$data; X_fe <- prep$X_fe; y <- prep$y
selected <- pmu_get_parsimonious_variables(X_fe)
X_std <- as.data.frame(scale(as.matrix(X_fe[, selected])))
model_df <- data.frame(IFC = y, X_std)
form <- as.formula(paste("IFC ~", paste(selected, collapse = " + ")))

# ================================================================
# 2) Fit parsimonious LM and extract residuals
# ================================================================
fit_lm <- lm(form, data = model_df)
res_lm <- residuals(fit_lm)
fit_v  <- fitted(fit_lm)

# Standardised residuals for the QQ-plot
res_std <- (res_lm - mean(res_lm)) / sd(res_lm)
n <- length(res_lm)

# ================================================================
# 3) Normality tests on the LM residuals
# ================================================================
ad_test <- ad.test(res_lm)
jb_test <- jarque.bera.test(res_lm)
sh_test <- shapiro.test(sample(res_lm, min(5000, n)))

skewness <- function(x) mean((x - mean(x))^3) / sd(x)^3
kurtosis <- function(x) mean((x - mean(x))^4) / sd(x)^4

normality_tbl <- data.frame(
  test = c("Mean", "SD", "Skewness", "Kurtosis (Normal = 3)",
           "Anderson-Darling A",  "Anderson-Darling p",
           "Jarque-Bera X2",      "Jarque-Bera p",
           "Shapiro-Wilk W (n=5k)", "Shapiro-Wilk p"),
  value = c(mean(res_lm), sd(res_lm),
            skewness(res_lm), kurtosis(res_lm),
            unname(ad_test$statistic), ad_test$p.value,
            unname(jb_test$statistic), jb_test$p.value,
            unname(sh_test$statistic), sh_test$p.value)
)
cat("--- Normality tests on LM residuals ---\n")
print(normality_tbl, row.names = FALSE, digits = 4)
write.csv(normality_tbl,
          file.path(out_dir, "normality_tests.csv"), row.names = FALSE)

# ================================================================
# 4) QQ plot of LM residuals (with the y = x reference)
# ================================================================
qq_df <- data.frame(theoretical = qnorm(ppoints(n)),
                    sample = sort(res_std))

p_qq <- ggplot(qq_df, aes(x = theoretical, y = sample)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "firebrick") +
  geom_point(alpha = 0.25, size = 0.4) +
  labs(title = "Q-Q plot of LM residuals against the Normal",
       subtitle = sprintf("Skewness = %.2f, excess kurtosis = %.2f (Normal: 0 / 3)",
                          skewness(res_lm), kurtosis(res_lm) - 3),
       x = "Theoretical normal quantiles",
       y = "Standardised residual quantiles") +
  theme_minimal()
ggsave(file.path(out_dir, "qq_plot_lm.png"), p_qq, width = 7, height = 7, dpi = 150)

# ================================================================
# 5) Histogram overlaid with the theoretical Normal density
# ================================================================
xx <- seq(min(res_lm), max(res_lm), length.out = 500)
gauss <- data.frame(x = xx,
                    density = dnorm(xx, mean = mean(res_lm), sd = sd(res_lm)))
p_hist <- ggplot(data.frame(r = res_lm), aes(x = r)) +
  geom_histogram(aes(y = after_stat(density)), bins = 60,
                 fill = "#377eb8", alpha = 0.6, colour = "white") +
  geom_line(data = gauss, aes(x = x, y = density),
            colour = "firebrick", linewidth = 1) +
  labs(title = "Histogram of LM residuals vs theoretical Normal",
       subtitle = "Heavier tails would show as bins above the red curve far from zero.",
       x = "Residual (observed - predicted)", y = "Density") +
  theme_minimal()
ggsave(file.path(out_dir, "histogram_lm.png"), p_hist, width = 9, height = 5, dpi = 150)

# ================================================================
# 6) Box-Cox check: would a transformation of IFC help?
# ================================================================
# Note: Box-Cox needs strictly positive y. IFC > 0 always so we can
# run it directly.
bc <- boxcox(form, data = model_df, lambda = seq(-2, 2, by = 0.1), plotit = FALSE)
lambda_best <- bc$x[which.max(bc$y)]
cat(sprintf("\nBest Box-Cox lambda: %.2f (lambda=1 means no transformation)\n",
            lambda_best))
write.csv(data.frame(lambda = bc$x, loglik = bc$y),
          file.path(out_dir, "boxcox_profile.csv"), row.names = FALSE)

p_bc <- ggplot(data.frame(lambda = bc$x, loglik = bc$y),
               aes(x = lambda, y = loglik)) +
  geom_line(colour = "#377eb8") +
  geom_vline(xintercept = lambda_best, linetype = "dashed", colour = "firebrick") +
  geom_vline(xintercept = 1, linetype = "dotted") +
  labs(title = "Box-Cox profile log-likelihood",
       subtitle = sprintf("Best lambda = %.2f. lambda = 1 means no transformation is needed.",
                          lambda_best),
       x = "lambda", y = "Profile log-likelihood") +
  theme_minimal()
ggsave(file.path(out_dir, "boxcox_profile.png"), p_bc, width = 8, height = 5, dpi = 150)

# ================================================================
# 7) QQ plot of random intercepts from M_geo_nested
# ================================================================
if (file.exists(fit_geo_nested)) {
  cat("\nQQ plot of M_geo_nested random intercepts...\n")
  fit_lmm <- readRDS(fit_geo_nested)
  re_list <- ranef(fit_lmm)

  qq_re_data <- list()
  for (grp_name in names(re_list)) {
    re_vec <- re_list[[grp_name]][, 1]
    n_grp  <- length(re_vec)
    re_std <- (re_vec - mean(re_vec)) / sd(re_vec)
    qq_re_data[[grp_name]] <- data.frame(
      group       = grp_name,
      theoretical = qnorm(ppoints(n_grp)),
      sample      = sort(re_std)
    )
  }
  qq_re <- do.call(rbind, qq_re_data)

  p_qq_re <- ggplot(qq_re, aes(x = theoretical, y = sample)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "firebrick") +
    geom_point(colour = "#377eb8") +
    facet_wrap(~ group, scales = "free") +
    labs(title = "Q-Q plot of M_geo_nested random intercepts (BLUPs)",
         subtitle = "lme4 assumes random intercepts are Normal. Points on the dashed line = ok.",
         x = "Theoretical normal quantiles",
         y = "Standardised BLUP quantiles") +
    theme_minimal()
  ggsave(file.path(out_dir, "qq_random_effects.png"),
         p_qq_re, width = 11, height = 5, dpi = 150)
}

cat("\nOutputs in:", out_dir, "\n")
