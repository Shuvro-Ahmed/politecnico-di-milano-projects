# Poster draft — IFC Fragility Prediction Project

> Bozza di contenuto per il poster finale.
> Pensato per layout A0 portrait (~84×119 cm), griglia 3 colonne.
> Ogni sezione qui sotto corrisponde a un blocco/box del poster.

---

## TITOLO (header del poster)

**Italian Fragility Composite Index — Predictive Modeling Pipeline**
*A LASSO benchmark, mixed-model extension, and dimensionality-reduction comparison*

Autori: [Mattia, Alessandro, Riccardo, Luca, Francesco]
Gruppo: [N°]
Tutors: Dott. Matteo Greco, Prof.ssa Francesca Ieva

---

## COL 1 — Box 1: GOAL OF THE ANALYSIS (alto sinistra)

**What we predict.** The raw Italian Fragility Composite Index (IFC)
at the municipal level for 2019 and 2021, from socio-economic
indicators of the GRINS V3 dataset.

**Why.** Build an interpretable linear benchmark against which any
future non-linear or spatial model can be compared.

**Three layers of analysis:**
- Layer 1 — LASSO benchmark (191 covariates, R² = 0.825)
- Layer 2 — Parsimonious model (25 covariates, R² = 0.784)
- Layer 3 — Mixed model + spatial diagnostic (R² = 0.818)

*Visual suggerito: mini-mappa Italia colorata per IFC + freccia "we predict this from socio-economic indicators"*

---

## COL 1 — Box 2: DATASET (sotto box 1)

**Target**: `IFC_Final_Analysis_Sorted.xlsx` — raw IFC values for 2019 and 2021. 95.6% match with official MFI deciles.

**Predictors**: GRINS V3 municipal panel — 549 columns × 7900 municipalities × multi-year. Filtered to 2019, 2021 → **15 782 obs**.

**Coverage**: 7886 unique municipalities × 2 years.

| Quantity | Value |
|---|---|
| Observations | 15 782 |
| Municipalities | 7 886 |
| Years | 2019, 2021 |
| Raw GRINS columns | 549 |
| Engineered features | 326 |

*Visual: piccola Italia colorata + nuvola di nomi indicatori (PM10, addetti, redditi…)*

---

## COL 1 — Box 3: METHODOLOGY (parte bassa col 1)

**Pipeline in 3 step:**

**1. Feature engineering** (326 features su 5 categorie):
  - count → /pop + log1p
  - count_neg → /pop only
  - size → log1p
  - rate → leave as-is
  - winsorize 1°/99° percentile

**2. Variable selection** (LASSO):
  - λ.1se via 10-fold CV
  - Yields 191 covariates

**3. Parsimony refinement** (VIF + ranking):
  - Drop variables with VIF > 10 iteratively (34 dropped)
  - Take top 25 by |β·SD|
  - Replace non-significant ones
  - Final: 25 covariates, all p < 10⁻⁶, max VIF = 6.4

*Visual: diagram con 3 box collegati da frecce*

---

## COL 2 — Box 4: LINEAR MODEL — KEY EQUATION (alto centro)

**Parsimonious benchmark:**

$$\text{IFC}_i = \beta_0 + \sum_{j=1}^{25} \beta_j X_{ji} + \varepsilon_i$$

**Mixed-model extension (best):**

$$\text{IFC}_i = \beta_0 + \sum_{j=1}^{25} \beta_j X_{ji} + u_{r(i)} + v_{p(i)\mid r(i)} + \varepsilon_i$$

dove $u_r \sim \mathcal{N}(0, \sigma_R^2)$, $v_{p\mid r} \sim \mathcal{N}(0, \sigma_P^2)$
(r = 20 regions, p = 107 provinces)

---

## COL 2 — Box 5: TOP 5 DRIVERS (centro col 2) ⭐ box importante

**Standardised effects (β·SD): change in IFC per +1 SD of predictor**

| # | Variable | β | Effetto |
|---|---|---|---|
| 1 | **Middle-income taxpayer mass** | −1.03 | ⬇️ less fragile |
| 2 | **Total taxpayer density** | −0.78 | ⬇️ less fragile |
| 3 | **Low-income declared mass** | +0.57 | ⬆️ more fragile |
| 4 | **Pension share of income** | +0.56 | ⬆️ more fragile |
| 5 | **Small construction firms density** | −0.53 | ⬇️ less fragile |

**Reading.** Fragility = balance between *economic dynamism*
(middle-class mass, taxpayer base, small-business density) and
*economic distress* (low-income mass, pension dependency).

*Visual: bar plot orizzontale con top 5 + colore rosso/blu*

---

## COL 2 — Box 6: RESULTS (sotto box 5)

**Out-of-sample R² (panel-aware 5×5 CV, 25 folds):**

| Model | # dims | R² | RMSE |
|---|---|---|---|
| LASSO benchmark | 191 | **0.826** | 1.74 |
| PLS @ K=75 | 75 | 0.825 | 1.74 |
| M_geo_nested LMM | 25 + RE | **0.818** | 1.76 |
| Parsimonious LM | 25 | 0.784 | 1.94 |
| PCR @ K=100 | 100 | 0.787 | 1.92 |

**Headline message.** Parsimonious 25-covariate LM + nested
geographic random intercept = LASSO benchmark accuracy with full interpretability.

*Visual: bar chart con colori per modello, linea orizzontale al benchmark*

---

## COL 3 — Box 7: RESIDUAL MAP (alto destra) — high impact

**Where the model errs.**

[INSERIRE: `outputs/parsimonious_model/residual_map.png`]

🔴 Red = under-predicted (more fragile than expected)
🔵 Blue = over-predicted (less fragile than expected)

**Pattern**: positive residuals in southern Italy + isolated Alpine valleys; negative in northern foothills + central Italy.

---

## COL 3 — Box 8: SPATIAL DIAGNOSTIC

**Global Moran's I on LMM residuals (kNN-5 weights):**

$$I = 0.234, \quad p < 0.001$$

→ **Spatial autocorrelation persists** even after region/province random intercepts. Neighbour municipalities still share unexplained fragility.

**Local confirmation (LISA, teammate's analysis).** Decomposing the
same global statistic locally on the same residuals gives
**I = 0.236** — independent confirmation of our number. The LISA
cluster map shows *where*: 598 municipalities are "High-High"
(red — under-predicted, surrounded by under-predicted neighbours),
554 are "Low-Low" (blue), out of 6418 not significant.

[INSERIRE: `outputs/LISA_Maps/LISA_Residual_Cluster_knn5.png`]

*Visual: la mappa cluster LISA (più informativa del solo numero I, mostra dove si concentra l'autocorrelazione)*

---

## COL 3 — Box 9: DIMENSIONALITY REDUCTION COMPARISON

**PCR vs PLS vs LASSO**

[INSERIRE: `outputs/pls_vs_pcr/pls_vs_pcr.png`]

**Findings:**
- PCR alone underperforms (R²=0.787 at K=100)
- PLS dominates PCR by ~5pp consistently
- PLS at K=75 **matches** LASSO benchmark
- LASSO wins for **interpretability** (named variables vs combinations)

*Visual: il plot pls_vs_pcr.png con linea LASSO benchmark di riferimento*

---

## COL 3 — Box 10: CONCLUSIONS (bottom right)

**Three substantive findings:**

1. **GRINS predicts IFC well**: R² = 0.83 out-of-sample with only 25 interpretable variables + geographic hierarchy.

2. **The narrative is socio-economic**: middle-class density, taxpayer base, small-business fabric reduce fragility; low-income concentration and pension dependency increase it.

3. **Geography matters beyond the indicators**: ICC = 36% (region/province random effects) + persistent Moran's I = 0.234 suggest spatial models as a natural next step.

**Methodological aside.** PCR on the 12 IFC indicators reaches only R² = 0.987 (not 1.0), revealing that the official IFC includes a non-linear step (likely percentile rank) beyond the linear weighted aggregation.

---

## SUGGESTED FOOTER

- Project repository: github.com/the-general-lee/IFC_Project_AS
- Tools: R 4.5.3, glmnet, lme4, spdep, pls
- Reports: `parsimonious_extension`, `pcr_extension`

---

## NOTE PER L'IMPLEMENTAZIONE IN CANVA

**Palette colori suggerita:**
- Blu intenso (#1F4E79) per i numeri "buoni" (R² alti, parsimonia)
- Rosso (#C00000) per warning/limitazioni
- Grigio (#7F7F7F) per testo secondario
- Verde (#548235) per success/conclusioni

**Gerarchia visiva consigliata:**
- Titolo poster: 80pt
- Titoli box: 36pt
- Sottotitoli: 28pt
- Body: 18-20pt
- Caption: 14pt

**Visualizzazioni minimum:**
1. Mappa Italia con residui (Box 7) — la più impattante
2. Bar chart top 5 covariate (Box 5)
3. Bar chart confronto modelli (Box 6)
4. PCR vs PLS plot (Box 9)
5. Equazioni del modello (Box 4)

**Cosa NON mettere nel poster:**
- Tabella completa delle 25 covariate (troppo lunga; in appendice)
- Dettagli su VIF iteration (in appendice)
- QQ plot di normalità (in report, non in poster)
- Box-Cox profile (irrilevante per poster)
- Tabella full di Elastic Net α (basta dirlo)

**Cose extra da considerare:**
- QR code linkante al GitHub
- QR code linkante al report PDF
- Logo del gruppo / Politecnico
- Linea temporale del progetto (PCA descrittiva → LASSO → LMM → PCR/PLS)

---

## TEXT BLOCKS PER COPY-PASTE DIRETTO IN CANVA

### Intro text (1 frase pitch)

> *"Can we predict the Italian Fragility Composite Index from independent socio-economic indicators? A 3-layer linear pipeline reaches R² = 0.82 with only 25 readable covariates."*

### Methods text (3 frasi)

> *"We engineer 326 features from the GRINS V3 municipal panel via per-capita normalization, log1p transforms and winsorization. LASSO selects 191 informative covariates; iterative VIF pruning + standardised-effect ranking reduce them to 25 significant predictors. A region/province nested mixed model adds geographic hierarchy."*

### Results text (3 frasi)

> *"The parsimonious linear model reaches R² = 0.784 out-of-sample; adding region/province random intercepts raises it to R² = 0.818 — within 0.005 of a 191-covariate LASSO benchmark. The strongest drivers are the local middle-class mass (β·SD = −1.03), the total taxpayer density (−0.78) and the share of low-income declared income (+0.57). After all controls, Moran's I on the LMM residuals remains 0.234 (p < 0.001), suggesting that explicit spatial models are the natural next step."*

### Conclusion text (1 frase)

> *"Italian municipal fragility is a socio-economic balance between dynamism and distress, modulated by a non-trivial geographic component that 25 readable indicators capture but do not exhaust."*
