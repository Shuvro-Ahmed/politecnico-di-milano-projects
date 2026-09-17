# ==============================================================================
# ANALISI INTEGRATA 2021: CLUSTERING MULTIDIMENSIONALE vs TASSONOMIA GRINS
# ==============================================================================

# --- 1. LIBRERIE ---
suppressPackageStartupMessages({
  library(readxl)
  library(cluster)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(openxlsx)
  library(gridExtra)
  library(sf)
  library(stringr)
  library(scales)
})

# --- 2. PARAMETRI E FILE ---
FILE_DATI      <- "data/raw/ifc/composite_fragility_all_years.xlsx"
FILE_TAX       <- "data/raw/grins/tassonomia_grins.xlsx"
SHAPEFILE      <- "data/raw/istat/shapefile/Com2021.shp"   # not currently in repo
OUT_DIR        <- "outputs/integrated_analysis_2021"

if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)
if (!dir.exists(file.path(OUT_DIR, "plots"))) dir.create(file.path(OUT_DIR, "plots"))

# Configurazione K-Means
K_MAX          <- 10
NSTART         <- 25
SET_SEED       <- 2026

# Dizionari Variabili
gruppi_variabili <- list(
  "Economia" = c("Employment_rate_20_64", "Density_local_units_ind_serv", "Persons_low_productivity_units"),
  "Sociale"  = c("Accessibility_essential_services", "Population_dependency_index", "Pop_low_education_25_64", "Incidence_net_migration"),
  "Ambiente" = c("Motorisation_rate_high_emissions", "Undiff_waste_generated", "Protected_natural_areas", "Areas_risk_landslides", "Land_consumption")
)

# --- 3. CARICAMENTO E PREPARAZIONE DATI 2021 ---
cat("\n[1/5] Caricamento dati 2021...\n")

# Carica foglio 2021
df_2021 <- read_excel(FILE_DATI, sheet = "2021", skip = 2, col_names = FALSE)
colnames(df_2021) <- c("PRO_COM", "Territory", "MFI_Decile", 
                       gruppi_variabili$Ambiente, gruppi_variabili$Sociale, gruppi_variabili$Economia)
df_2021 <- df_2021 %>% mutate(across(-c(Territory), as.numeric)) %>% na.omit()

# Carica Tassonomia GRINS
df_tax <- read_excel(FILE_TAX, sheet = 1) %>%
  transmute(
    PRO_COM = as.numeric(`Codice Istat del Comune 2021`),
    Macroclasse_GRINS = `MACROCLASSE GRINS`,
    Classe_GRINS = `CLASSE GRINS`
  )

# Funzione Silhouette per trovare il K ottimale
silhouette_campionato <- function(cluster_labels, dati_scalati) {
  mean(silhouette(cluster_labels, dist(dati_scalati))[, "sil_width"])
}

# --- 4. CLUSTERING K-MEANS (SOLO 2021) ---
cat("[2/5] Esecuzione K-Means per dimensione...\n")

risultati_cluster <- df_2021 %>% select(PRO_COM, Territory, MFI_Decile)
dizionario_cluster_livelli <- list() # Per normalizzare l'indice composito

for (nome_gruppo in names(gruppi_variabili)) {
  vars <- gruppi_variabili[[nome_gruppo]]
  dati_scalati <- scale(df_2021[, vars])
  
  # Trova K
  sil <- numeric(K_MAX)
  set.seed(SET_SEED)
  for (k in 2:K_MAX) {
    res <- kmeans(dati_scalati, centers = k, nstart = NSTART)
    sil[k] <- silhouette_campionato(res$cluster, dati_scalati)
  }
  k_ottimale <- which.max(sil[3:K_MAX]) + 2
  
  # Esegui modello finale
  set.seed(SET_SEED)
  mod <- kmeans(dati_scalati, centers = k_ottimale, nstart = NSTART)
  
  # Ordina cluster dal meno fragile (1) al più fragile (K) in base al MFI medio
  mfi_medio <- aggregate(df_2021$MFI_Decile, by = list(mod$cluster), mean)
  ordine <- order(mfi_medio$x)
  cluster_ordinati <- match(mod$cluster, mfi_medio$Group.1[ordine])
  
  risultati_cluster[[paste0("Cluster_", nome_gruppo)]] <- cluster_ordinati
  dizionario_cluster_livelli[[nome_gruppo]] <- k_ottimale
}

# --- 5. CALCOLO INDICE COMPOSITO ---
cat("[3/5] Calcolo Fragilità Composita...\n")
risultati_cluster <- risultati_cluster %>%
  mutate(
    Norm_Econ = (Cluster_Economia - 1) / (dizionario_cluster_livelli$Economia - 1),
    Norm_Soc  = (Cluster_Sociale - 1)  / (dizionario_cluster_livelli$Sociale - 1),
    Norm_Amb  = (Cluster_Ambiente - 1) / (dizionario_cluster_livelli$Ambiente - 1),
    Score_Composito = round((Norm_Econ + Norm_Soc + Norm_Amb) / 3, 3),
    Tipologia_Composita = case_when(
      Score_Composito <= 0.25 ~ "1. Bassa Fragilità",
      Score_Composito <= 0.50 ~ "2. Fragilità Moderata",
      Score_Composito <= 0.75 ~ "3. Alta Fragilità",
      TRUE                    ~ "4. Fragilità Critica"
    )
  )

# --- 6. MERGE FINALE: CLUSTER + IFC + GRINS ---
cat("[4/5] Fusione con Tassonomia GRINS...\n")
df_master <- risultati_cluster %>%
  left_join(df_tax, by = "PRO_COM") %>%
  filter(!is.na(Classe_GRINS))

# Ordinamento logico Classi GRINS (es. 1.1, 1.2, 2.1...)
classe_ord_df <- tibble(Classe = unique(df_master$Classe_GRINS)) %>%
  mutate(parts = str_split(Classe, "\\.")) %>% rowwise() %>%
  mutate(p1 = as.numeric(parts[[1]]), p2 = ifelse(length(parts)>=2, as.numeric(parts[[2]]), 0), p3 = ifelse(length(parts)>=3, as.numeric(parts[[3]]), 0)) %>%
  arrange(p1, p2, p3)
df_master$Classe_GRINS <- factor(df_master$Classe_GRINS, levels = classe_ord_df$Classe)

# --- 7. ANALISI INCROCIATA (HEATMAPS) ---
cat("[5/5] Generazione Grafici e Output...\n")

# A) Heatmap: Macroclasse GRINS vs Decile IFC Ufficiale
heat_ifc <- df_master %>% count(Macroclasse_GRINS, MFI_Decile) %>% group_by(Macroclasse_GRINS) %>% mutate(pct = n/sum(n))

p_ifc <- ggplot(heat_ifc, aes(x = factor(MFI_Decile), y = Macroclasse_GRINS, fill = pct)) +
  geom_tile(color = "white") +
  geom_text(aes(label = percent(pct, 0.1)), size = 4) +
  scale_fill_gradient(low = "white", high = "#1b9e77") +
  labs(title = "GRINS vs Decili IFC Ufficiali (2021)", x = "Decile IFC", y = "Macroclasse GRINS", fill = "% Riga") +
  theme_minimal()
ggsave(file.path(OUT_DIR, "plots", "Heatmap_GRINS_vs_IFC_Decile.png"), p_ifc, width = 10, height = 5)

# B) Heatmap: Macroclasse GRINS vs Tipologia Composita (Il nuovo valore aggiunto!)
heat_comp <- df_master %>% count(Macroclasse_GRINS, Tipologia_Composita) %>% group_by(Macroclasse_GRINS) %>% mutate(pct = n/sum(n))

p_comp <- ggplot(heat_comp, aes(x = Tipologia_Composita, y = Macroclasse_GRINS, fill = pct)) +
  geom_tile(color = "white") +
  geom_text(aes(label = percent(pct, 0.1)), size = 4) +
  scale_fill_gradient(low = "white", high = "#d95f02") +
  labs(title = "GRINS vs Fragilità Composita K-Means (2021)", x = "Fragilità Composita (Dai ns Cluster)", y = "Macroclasse GRINS", fill = "% Riga") +
  theme_minimal()
ggsave(file.path(OUT_DIR, "plots", "Heatmap_GRINS_vs_ClusterComposito.png"), p_comp, width = 10, height = 5)

# C) Heatmap Dettagliata: Classe GRINS vs Cluster Economia
heat_econ <- df_master %>% count(Classe_GRINS, Cluster_Economia) %>% group_by(Classe_GRINS) %>% mutate(pct = n/sum(n))
p_econ <- ggplot(heat_econ, aes(x = factor(Cluster_Economia), y = Classe_GRINS, fill = pct)) +
  geom_tile(color = "white") + geom_text(aes(label = percent(pct, 0.1)), size = 3) +
  scale_fill_gradient(low = "white", high = "#7570b3") +
  labs(title = "Classe GRINS Dettagliata vs Cluster Economia (Livello 1=Migliore)", x = "Cluster Economia", y = "Classe GRINS") + theme_minimal()
ggsave(file.path(OUT_DIR, "plots", "Heatmap_GRINS_vs_Econ.png"), p_econ, width = 12, height = 8)


# --- 8. SALVATAGGIO EXCEL MASTER ---
wb <- createWorkbook()
addWorksheet(wb, "Master_Dati_2021")
writeData(wb, "Master_Dati_2021", df_master)

# Aggiungi fogli con le tabelle pivot (matrici di confusione)
addWorksheet(wb, "Pivot_Macro_vs_IFC")
writeData(wb, "Pivot_Macro_vs_IFC", df_master %>% count(Macroclasse_GRINS, MFI_Decile) %>% pivot_wider(names_from = MFI_Decile, values_from = n, values_fill = 0))

addWorksheet(wb, "Pivot_Macro_vs_Composito")
writeData(wb, "Pivot_Macro_vs_Composito", df_master %>% count(Macroclasse_GRINS, Tipologia_Composita) %>% pivot_wider(names_from = Tipologia_Composita, values_from = n, values_fill = 0))

saveWorkbook(wb, file.path(OUT_DIR, "Database_Integrato_2021.xlsx"), overwrite = TRUE)

cat("\n✅ Analisi completata con successo! Tutto salvato nella cartella:", OUT_DIR, "\n")