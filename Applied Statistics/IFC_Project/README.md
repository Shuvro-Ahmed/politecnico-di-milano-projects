# IFC Project — Applied Statistics

Analysis of the **Italian Fragility Composite Index (IFC)** at the
municipal level using the **GRINS** dataset. The project provides:

1. A **LASSO linear benchmark** that predicts raw IFC values from
   GRINS indicators (R² ≈ 0.825 panel-aware test).
2. A **linear mixed-models extension** that adds a region/province
   random intercept (R² ≈ 0.853).
3. A **spatial residual diagnostic** (Moran's I) that quantifies how
   much fine-scale spatial structure the mixed model still leaves
   unexplained.
4. Supporting analyses: PCA, k-means, GRINS taxonomy confusion
   matrix, integrated 2021 analysis.

## Folder structure

```
.
├── data/
│   ├── raw/              source data (ISTAT, GRINS, IFC)
│   └── processed/        derived datasets (GRINS V3)
│
├── R/                    analysis scripts (numerical order)
│   ├── 01_build_ifc_reference_2018.R
│   ├── 02_pca.R / 02b / 02c
│   ├── 03_kmeans.R
│   ├── 04_kmeans_vs_grins.R
│   ├── 05_methodology.R
│   ├── 06_confusion_matrix_taxonomy.R
│   ├── 08_ifc_integrated_analysis.R
│   ├── 09_simple_linear_regression.R
│   ├── 10_final_benchmark.R          ← LASSO benchmark (headline)
│   ├── 11_mixed_models.R             ← LMM extension
│   ├── 12_models_comparison.R        ← LASSO vs LMM comparison
│   └── 13_spatial_residual_check.R   ← Moran's I diagnostic
│
├── outputs/
│   ├── final_benchmark/      LASSO results (CSV + PNG)
│   ├── mixed_models/         LMM results + comparison + Moran
│   ├── pca/, clusters/, ...  outputs of earlier scripts
│
├── reports/
│   └── final_benchmark/
│       ├── report.tex        academic write-up (compile in Overleaf or pdflatex)
│       └── figures/          figures referenced in the report
│
└── docs/                     project notes
```

## How to reproduce

```r
# from project root
Rscript R/10_final_benchmark.R    # ~15 min; produces outputs/final_benchmark/
Rscript R/11_mixed_models.R       # ~15 min; produces outputs/mixed_models/
Rscript R/12_models_comparison.R  # <1 min;  produces comparison table + plot
Rscript R/13_spatial_residual_check.R   # ~3 min; Moran's I diagnostic
```

Scripts read all paths relative to the project root, with random seed
fixed (`set.seed(2026)`) so the headline numbers reproduce exactly.

## R dependencies

```r
install.packages(c(
  "readxl", "dplyr", "tidyr", "ggplot2",
  "glmnet",                    # LASSO / Ridge / Elastic Net
  "lme4", "lmerTest", "MuMIn", # linear mixed models
  "sf", "spdep"                # spatial diagnostic
))
```

Tested with R 4.5.3.

## Compiling the report

```bash
cd reports/final_benchmark
pdflatex report.tex
pdflatex report.tex   # second pass resolves \ref / \cite
```

Without a local LaTeX installation, upload `report.tex` and the
`figures/` folder to [Overleaf](https://www.overleaf.com).

## Notes on large files

Two files exceed GitHub's per-file size limit and are excluded by
`.gitignore`. They have to be obtained separately:

| File | Size | Where to find it |
|------|------|------------------|
| `data/processed/grins_v3/comunale_v3.csv` | ~300 MB | Regenerate from `comunale_v3.rds`: `write.csv(readRDS(...), ...)` |
| `data/raw/istat/shapefile/Com2021.shp`    | ~110 MB | Download from ISTAT: [Confini delle unità amministrative — Limiti2021](https://www.istat.it/it/archivio/222527) and copy the `Com2021.*` files into `data/raw/istat/shapefile/` |

The shapefile is needed only by `R/03_kmeans.R`,
`R/04_kmeans_vs_grins.R`, and `R/13_spatial_residual_check.R`. The
LASSO benchmark (script 10) and the mixed models (script 11) do not
need it.
