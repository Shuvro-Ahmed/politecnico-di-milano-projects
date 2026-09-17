# ================================================================
# RESIDUAL MAP + ELASTIC NET CHECK on the parsimonious model
# ----------------------------------------------------------------
# Two diagnostics on top of the 25-covariate parsimonious model
# produced by R/14_parsimonious_model.R:
#
# 1) Residual map:
#    Where in Italy does the parsimonious model under- or over-
#    predict IFC? The municipality-level residual is averaged across
#    2019/2021 and plotted on the ISTAT 2021 shapefile with a
#    diverging colour scale. This visualises the geographic pattern
#    of unexplained fragility -- and is the natural visual companion
#    to the LISA maps produced by another team member.
#
# 2) Elastic Net cross-check:
#    The professor asked whether using a "LASSO with a ridge penalty"
#    (i.e. Elastic Net, alpha < 1) would change the selection. With
#    the VIF iterative pruning we have already removed multicollinear
#    predictors, so we expect Elastic Net to converge to a similar
#    set. We test alpha values in {0.1, 0.25, 0.5, 0.75, 1.0} and
#    report selection size and CV error for each.
# ================================================================

suppressPackageStartupMessages({
  library(ggplot2); library(sf); library(glmnet); library(dplyr)
})
source("R/_utils.R")

set.seed(2026)

shape_file   <- "data/raw/istat/shapefile/Com2021.shp"
ranking_file <- PMU_PATHS$parsimonious_ranking
out_dir      <- "outputs/parsimonious_model"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ================================================================
# 1) Reproduce the parsimonious data set
# ================================================================
prep <- pmu_prepare_data(with_taxonomy_filter = FALSE)
data <- prep$data; X_fe <- prep$X_fe; y <- prep$y

# Take the 25 top-ranked parsimonious predictors (from the VIF-clean ranking)
ranking <- read.csv(ranking_file, stringsAsFactors = FALSE)
parsimonious_vars <- intersect(ranking$variable[1:25], names(X_fe))
cat("Parsimonious predictors used:", length(parsimonious_vars), "\n")

X_parsimonious <- as.data.frame(scale(as.matrix(X_fe[, parsimonious_vars])))
model_df <- data.frame(IFC = y, PRO_COM = data$PRO_COM, X_parsimonious)

# ================================================================
# 2) Fit final parsimonious model and compute residuals
# ================================================================
fit_pars <- lm(as.formula(paste("IFC ~", paste(parsimonious_vars, collapse = " + "))),
               data = model_df)
res <- residuals(fit_pars)
cat("Residual SD:", round(sd(res), 3),
    " | mean:", round(mean(res), 4), "\n")

# Aggregate to municipality level (average across 2019/2021)
muni_res <- model_df %>%
  mutate(residual = res) %>%
  group_by(PRO_COM) %>%
  summarise(mean_residual = mean(residual), .groups = "drop")

# ================================================================
# 3) RESIDUAL MAP
# ================================================================
cat("\nBuilding residual map...\n")
shapes <- st_read(shape_file, quiet = TRUE)
shapes$PRO_COM_num <- as.numeric(as.character(shapes$PRO_COM))
shapes_res <- shapes %>%
  inner_join(muni_res, by = c("PRO_COM_num" = "PRO_COM"))
cat("Matched municipalities for map:", nrow(shapes_res), "/", nrow(shapes), "\n")

# Cap residuals at the 1st / 99th percentile for the colour scale
q <- quantile(shapes_res$mean_residual, c(0.01, 0.99), na.rm = TRUE)
shapes_res$residual_capped <- pmin(pmax(shapes_res$mean_residual, q[1]), q[2])
lim <- max(abs(q))

p_map <- ggplot(shapes_res) +
  geom_sf(aes(fill = residual_capped), colour = NA) +
  scale_fill_gradient2(low = "#2166ac", mid = "white", high = "#b2182b",
                       midpoint = 0, limits = c(-lim, lim),
                       name = "Residual\n(observed -\npredicted IFC)") +
  labs(title = "Residuals of the parsimonious 25-covariate model",
       subtitle = "Red = under-predicted (model too optimistic) | Blue = over-predicted") +
  theme_void() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 10))

ggsave(file.path(out_dir, "residual_map.png"), p_map,
       width = 8, height = 10, dpi = 200)
cat("Residual map saved.\n")

# Save the data underlying the map
write.csv(muni_res, file.path(out_dir, "municipal_residuals.csv"), row.names = FALSE)

# ================================================================
# 4) ELASTIC NET CHECK across alpha values
# ----------------------------------------------------------------
# We use the 157 VIF-pruned predictors as the input matrix. For
# each alpha in the grid we run cv.glmnet (10-fold) and report:
#   - chosen lambda.1se
#   - number of selected predictors at lambda.1se
#   - CV MSE at lambda.1se
# A smaller selection at the same MSE would mean Elastic Net is
# more efficient; a similar selection means LASSO is fine.
# ================================================================
cat("\n--- Elastic Net comparison across alpha ---\n")
ranking_full <- read.csv(ranking_file, stringsAsFactors = FALSE)
vif_clean_vars <- intersect(ranking_full$variable, names(X_fe))
X_vif_clean <- as.matrix(scale(X_fe[, vif_clean_vars]))

lambda_grid <- 10^seq(2, -3, length.out = 60)
enet_summary <- data.frame()
for (a in c(0.10, 0.25, 0.50, 0.75, 1.00)) {
  cv_fit <- cv.glmnet(X_vif_clean, y, alpha = a, nfolds = 10, lambda = lambda_grid)
  lam <- cv_fit$lambda.1se
  fit  <- glmnet(X_vif_clean, y, alpha = a, lambda = lambda_grid)
  cf   <- predict(fit, s = lam, type = "coefficients")
  n_sel <- sum(as.vector(cf) != 0) - 1   # subtract intercept
  cv_err <- cv_fit$cvm[cv_fit$lambda == lam]
  enet_summary <- rbind(enet_summary,
    data.frame(alpha = a, n_selected = n_sel,
               lambda_1se = lam, CV_MSE = cv_err))
  cat(sprintf("alpha = %.2f | n_selected = %3d | lambda.1se = %.4f | CV MSE = %.4f\n",
              a, n_sel, lam, cv_err))
}
write.csv(enet_summary,
          file.path(out_dir, "elastic_net_summary.csv"), row.names = FALSE)

# ================================================================
# 5) Save a one-page summary plot: trade-off curve + alpha comparison
# ================================================================
p_alpha <- ggplot(enet_summary, aes(x = factor(alpha))) +
  geom_col(aes(y = n_selected), fill = "steelblue") +
  geom_text(aes(y = n_selected, label = n_selected), vjust = -0.4) +
  labs(title = "Elastic Net selection size at lambda.1se",
       subtitle = "alpha = 1 is pure LASSO, alpha = 0 is pure Ridge (kept here in [0.1, 1.0])",
       x = "alpha (mixing parameter)",
       y = "# selected covariates") +
  theme_minimal()
ggsave(file.path(out_dir, "elastic_net_alpha.png"),
       p_alpha, width = 8, height = 5, dpi = 150)

cat("\nOutputs in:", out_dir, "\n")
