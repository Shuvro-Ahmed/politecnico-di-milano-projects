# ================================================================
# Shared utilities for the parsimonious pipeline (scripts 11-17).
# ----------------------------------------------------------------
# This file contains the data-loading and feature-engineering code
# that scripts 11, 13, 14, 15, 16 and 17 all need.
#
# Source it from each script with:
#     source("R/_utils.R")
#
# Convention: any function defined here uses a `pmu_` prefix
# ("parsimonious municipal utilities") so it cannot accidentally
# clash with a built-in name.
# ================================================================

suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(tidyr)
})

# Standard project paths.
# All scripts assume the working directory is the project root.
PMU_PATHS <- list(
  ifc      = "data/raw/ifc/final_analysis_sorted.xlsx",
  grins    = "data/processed/grins_v3/comunale_v3.rds",
  tassonomia = "data/raw/grins/tassonomia_grins.xlsx",
  benchmark_coef        = "outputs/final_benchmark/final_coefficients_standardised.csv",
  parsimonious_selected = "outputs/parsimonious_model/selected_parsimonious_variables.txt",
  parsimonious_ranking  = "outputs/parsimonious_model/vif_clean_ranking.csv"
)

# OneDrive sometimes keeps Excel files as cloud placeholders; if so,
# copy the file into the session's temp directory first so read_excel
# can actually open it.
pmu_resolve_xlsx <- function(path) {
  if (!file.exists(path)) return(path)
  tmp <- file.path(tempdir(),
                   paste0("pmu_", basename(path)))
  file.copy(path, tmp, overwrite = TRUE)
  tmp
}

# ---------------------------------------------------------------- #
# Classify a GRINS column by its name into one of five categories.
# The category drives the per-class transformation in pmu_build_X.
#
#   "count"      -> non-negative size-dependent count: per-capita + log1p
#   "count_neg"  -> count that can be negative (net flow): per-capita only
#   "size"       -> population/area used as covariate: log1p only
#   "rate"       -> already a rate/index/percentage: untransformed
#   "skip"       -> ISTAT codes, names, join markers: dropped from X
# ---------------------------------------------------------------- #
pmu_classify <- function(nm) {
  ln <- tolower(nm)
  if (grepl("_anno_x$|_anno_y$|^cod_|^den_|sigla|nome_|^stringa|backcast|recovery|tipo_na", ln))
    return("skip")
  if (ln %in% c("popolazione",
                "superficie_totale_kmq_formattato",
                "superfici kmq"))
    return("size")
  rate_patterns <- c(
    "indice","incidenza","mobilità","mobilita","percentuale",
    "pro capite","procapite","pro-capite",
    "per_addetto","per_dipendente","valori_percentuali",
    "_media_","_media$","^eta_","^anzianita_",
    "media_donne","media_uomini","media_media",
    "coverage","contribuenti_su_pop","reddito_medio",
    "reddito_pc","lacc_mean","employee_services",
    "employee_concentration","degree_stem","degree_concentration",
    "^no2_","^pm10_","^pm25_","^o3_","_media_valori_annuali",
    "distanza_","^elezion","regionali",
    "verde urbano","suolo consumato","acqua potabile",
    "produzione pro","sau","ricettività","ricettivita","densità"
  )
  if (any(sapply(rate_patterns, function(p) grepl(p, ln)))) return("rate")
  if (grepl("saldo", ln)) return("count_neg")
  "count"
}

# ---------------------------------------------------------------- #
# Feature-engineering pipeline used by every script.
# Input:
#   df_num : data.frame with all numeric candidate predictors
#   pop    : vector of population values (one per row), same length as df_num
# Output:
#   data.frame with the engineered predictors (and
#   year2021 dummy if you choose to add it OUTSIDE this function).
# ---------------------------------------------------------------- #
pmu_build_X <- function(df_num, pop) {
  cls  <- sapply(names(df_num), pmu_classify)
  keep <- names(df_num)[cls != "skip"]
  X    <- df_num[, keep, drop = FALSE]
  cls  <- cls[keep]

  na_prop <- sapply(X, function(z) mean(is.na(z)))
  X <- X[, names(na_prop[na_prop <= 0.20]), drop = FALSE]
  cls <- cls[names(X)]

  for (j in seq_along(X)) {
    if (any(is.na(X[[j]])))
      X[[j]][is.na(X[[j]])] <- median(X[[j]], na.rm = TRUE)
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
  X <- X[, sapply(X, var) > 0, drop = FALSE]
  X
}

# ---------------------------------------------------------------- #
# Load the IFC target table in long format.
# Output columns: PRO_COM (numeric), year (2019 or 2021), IFC.
# ---------------------------------------------------------------- #
pmu_load_ifc_long <- function(ifc_path = PMU_PATHS$ifc) {
  read_excel(pmu_resolve_xlsx(ifc_path)) %>%
    select(PRO_COM, IFC_2019, IFC_2021) %>%
    pivot_longer(c(IFC_2019, IFC_2021),
                 names_to = "year", values_to = "IFC") %>%
    mutate(year    = as.integer(sub("IFC_", "", year)),
           PRO_COM = as.numeric(PRO_COM)) %>%
    filter(!is.na(IFC), !is.na(PRO_COM))
}

# ---------------------------------------------------------------- #
# Load the GRINS V3 panel, filtered to 2019/2021.
# Adds the geographic grouping columns if `with_groups = TRUE`.
# ---------------------------------------------------------------- #
pmu_load_grins <- function(grins_path = PMU_PATHS$grins,
                           with_groups = TRUE) {
  d <- readRDS(grins_path) %>%
    filter(anno %in% c(2019, 2021)) %>%
    mutate(codice_comune = as.numeric(codice_comune))
  if (with_groups) d else
    d %>% select(-any_of(c("nome_regione", "nome_provincia")))
}

# ---------------------------------------------------------------- #
# Load the GRINS taxonomy (PRO_COM -> macroclass).
# ---------------------------------------------------------------- #
pmu_load_taxonomy <- function(tax_path = PMU_PATHS$tassonomia) {
  read_excel(pmu_resolve_xlsx(tax_path), sheet = 1) %>%
    rename(PRO_COM = 1, GRINS_macroclass = 2, GRINS_class = 3) %>%
    mutate(PRO_COM = as.numeric(PRO_COM)) %>%
    filter(!is.na(PRO_COM)) %>%
    select(PRO_COM, GRINS_macroclass)
}

# ---------------------------------------------------------------- #
# End-to-end pipeline: load + merge + build engineered features.
# Returns a list with:
#   data     : merged data frame with PRO_COM, year, IFC and groupings
#   X_fe     : engineered feature matrix (with year2021 dummy)
#   y        : the IFC target (vector)
#   pop      : the population vector
#   num_cols : the numeric column names used to build X_fe
# ---------------------------------------------------------------- #
pmu_prepare_data <- function(with_taxonomy_filter = TRUE) {
  ifc   <- pmu_load_ifc_long()
  grins <- pmu_load_grins(with_groups = TRUE)
  tax   <- if (with_taxonomy_filter) pmu_load_taxonomy() else NULL

  num_cols <- setdiff(names(grins)[sapply(grins, is.numeric)],
                      c("codice_comune", "anno"))

  data <- ifc %>%
    inner_join(grins %>% select(all_of(c("codice_comune", "anno",
                                         num_cols, "nome_regione",
                                         "nome_provincia"))),
               by = c("PRO_COM" = "codice_comune", "year" = "anno"))
  if (!is.null(tax)) {
    data <- data %>%
      left_join(tax, by = "PRO_COM") %>%
      filter(!is.na(nome_regione),
             !is.na(nome_provincia),
             !is.na(GRINS_macroclass))
  } else {
    data <- data %>%
      filter(!is.na(nome_regione), !is.na(nome_provincia))
  }

  pop <- data$Popolazione
  pop[is.na(pop) | pop <= 0] <- median(pop[pop > 0], na.rm = TRUE)

  X_fe <- pmu_build_X(data %>% select(all_of(num_cols)), pop)
  X_fe$year2021 <- as.integer(data$year == 2021)
  names(X_fe) <- make.names(names(X_fe), unique = TRUE)

  list(data = data, X_fe = X_fe, y = data$IFC,
       pop = pop, num_cols = num_cols)
}

# ---------------------------------------------------------------- #
# Read the FINAL parsimonious variable list with a sensible fallback.
# ---------------------------------------------------------------- #
pmu_get_parsimonious_variables <- function(X_fe,
                                           n_fallback = 25) {
  if (file.exists(PMU_PATHS$parsimonious_selected)) {
    vars <- readLines(PMU_PATHS$parsimonious_selected)
    sel  <- intersect(vars, names(X_fe))
    if (length(sel) > 0) return(sel)
  }
  if (file.exists(PMU_PATHS$parsimonious_ranking)) {
    rk <- read.csv(PMU_PATHS$parsimonious_ranking, stringsAsFactors = FALSE)
    return(intersect(rk$variable[seq_len(n_fallback)], names(X_fe)))
  }
  sel_df <- read.csv(PMU_PATHS$benchmark_coef, stringsAsFactors = FALSE)
  intersect(sel_df$variable, names(X_fe))
}
