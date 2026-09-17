###---------------------------------------------------###
### K-Means - v4.0: Transizioni, Stabilità, Boxplot  ###
### Cluster Composito, Campionamento Silhouette,     ###
### Profili in Excel                                 ###
###---------------------------------------------------###

# --- 1. Librerie ------------------------------------------------------------
library(readxl)
library(cluster)
library(factoextra)
library(ggplot2)
library(dplyr)
library(openxlsx)
library(gridExtra)
library(sf)
library(grid)
library(tidyr)      # [NUOVO] per pivot_longer nei boxplot

# --- 2. Parametri Globali ---------------------------------------------------
FILE_PATH           <- "data/raw/ifc/composite_fragility_all_years.xlsx"
PERCORSO_SHAPEFILE  <- "data/raw/istat/shapefile/Com2021.shp"   # not currently in repo
ANNI                <- c("2018", "2019", "2021", "2022")
K_MAX               <- 10
NSTART              <- 25
SET_SEED            <- 2026
CARTELLE_OUTPUT     <- "outputs/clusters"
N_CAMPIONE_SIL      <- 2000   # [PUNTO 6] Max comuni per calcolo silhouette

if (!dir.exists(CARTELLE_OUTPUT)) dir.create(CARTELLE_OUTPUT, recursive = TRUE)

# --- 3. DEFINIZIONE COLORI E NOMI DESCRITTIVI --------------------------------
colori_fissi <- c(
  "Livello 1"  = "#00429d", "Livello 2"  = "#2e8ad8", "Livello 3"  = "#00fa9a",
  "Livello 4"  = "#b8df29", "Livello 5"  = "#ffeb3b", "Livello 6"  = "#ff9800",
  "Livello 7"  = "#f44336", "Livello 8"  = "#c51b7d", "Livello 9"  = "#4a148c",
  "Livello 10" = "#000000"
)

etichette_cluster <- list(
  "Economia" = c(
    "Livello 1" = "Polo Produttivo (Alta occupazione e imprese)",
    "Livello 2" = "Residenziale Stabile",
    "Livello 3" = "Area Intermedia (Bassa densità produttiva)",
    "Livello 4" = "Area Depressa (Bassa produttività)"
  ),
  "Sociale" = c(
    "Livello 1" = "Centro Attrattivo (Servizi e migrazione +)",
    "Livello 2" = "Area in Spopolamento (Bassa istruzione)",
    "Livello 3" = "Aree Interne (Isolate e anziane)",
    "Livello 4" = "Aree Senili (Altissima dipendenza)"
  ),
  "Ambiente" = c(
    "Livello 1" = "Aree Urbanizzate (Alto consumo suolo)",
    "Livello 2" = "Aree Agricole o Miste",
    "Livello 3" = "Riserve Naturali (Aree protette)",
    "Livello 4" = "Pressione Ambientale (Picco rifiuti)",
    "Livello 5" = "Alto Rischio Idrogeologico (Frane)"
  )
)

gruppi_variabili <- list(
  "Economia" = c("Employment_rate_20_64", "Density_local_units_ind_serv", "Persons_low_productivity_units"),
  "Sociale"  = c("Accessibility_essential_services", "Population_dependency_index", "Pop_low_education_25_64", "Incidence_net_migration"),
  "Ambiente" = c("Motorisation_rate_high_emissions", "Undiff_waste_generated", "Protected_natural_areas", "Areas_risk_landslides", "Land_consumption")
)

plot_list        <- list()
profili_cluster  <- list()
reference_centroids <- list()

# [PUNTO 3+4] Strutture per transizioni e stabilità
cluster_per_comune  <- list()   # PRO_COM -> cluster assegnato, per ogni gruppo e anno
stabilita_risultati <- list()   # KPI di stabilità per dimensione

for (g in names(gruppi_variabili)) {
  plot_list[[g]]       <- list(Elbow = list(), Sil = list(), Mappa = list(), Boxplot = list())
  profili_cluster[[g]] <- list()
  cluster_per_comune[[g]] <- list()
}

# --- 4. Caricamento Mappa Base ----------------------------------------------
cat("\nCaricamento base cartografica...\n")
mappa_italia <- st_read(PERCORSO_SHAPEFILE, quiet = TRUE) %>%
  mutate(PRO_COM = as.character(PRO_COM))

# --- 5. Funzioni di Supporto ------------------------------------------------
carica_foglio <- function(file_path, anno) {
  df <- read_excel(file_path, sheet = anno, skip = 2, col_names = FALSE)
  colnames(df) <- c("PRO_COM", "Territory", "MFI_Decile",
                    "Motorisation_rate_high_emissions", "Undiff_waste_generated",
                    "Protected_natural_areas", "Areas_risk_landslides", "Land_consumption",
                    "Accessibility_essential_services", "Population_dependency_index",
                    "Pop_low_education_25_64", "Employment_rate_20_64", "Incidence_net_migration",
                    "Density_local_units_ind_serv", "Persons_low_productivity_units")
  return(df)
}

prepara_dati <- function(df) {
  indici       <- df %>% select(MFI_Decile:Persons_low_productivity_units) %>% mutate(across(everything(), as.numeric))
  indici_puliti <- na.omit(indici)
  meta         <- df[complete.cases(indici), c("PRO_COM", "Territory")] %>% mutate(PRO_COM = as.character(PRO_COM))
  list(indici = indici_puliti, meta = meta)
}

# [PUNTO 6] Silhouette su campione stratificato per cluster
silhouette_campionato <- function(cluster_labels, dati_scalati, n_camp, seed) {
  set.seed(seed)
  n <- length(cluster_labels)
  if (n <= n_camp) {
    idx <- seq_len(n)
  } else {
    # Campionamento proporzionale per cluster
    idx <- unlist(tapply(seq_len(n), cluster_labels, function(i) {
      m <- max(2, round(length(i) * n_camp / n))
      sample(i, min(m, length(i)))
    }))
  }
  mean(silhouette(cluster_labels[idx], dist(dati_scalati[idx, ]))[, "sil_width"])
}

# --- 6. ESECUZIONE ANALISI --------------------------------------------------
fogli_risultati <- list()

for (anno in ANNI) {
  cat(paste0("\nElaborazione Anno: ", anno, "...\n"))
  df_raw <- carica_foglio(FILE_PATH, anno)
  dati   <- prepara_dati(df_raw)
  risultati_anno <- dati$meta
  
  for (nome_gruppo in names(gruppi_variabili)) {
    vars         <- gruppi_variabili[[nome_gruppo]]
    dati_scalati <- scale(dati$indici[, vars])
    
    # --- Ricerca K dinamico (con silhouette campionato) ---
    w <- numeric(K_MAX); sil <- numeric(K_MAX)
    set.seed(SET_SEED)
    for (k in 1:K_MAX) {
      res  <- kmeans(dati_scalati, centers = k, nstart = NSTART)
      w[k] <- sum(res$withinss) / res$totss
      # [PUNTO 6] Usa funzione campionata invece di dist() sull'intero dataset
      if (k >= 2) sil[k] <- silhouette_campionato(res$cluster, dati_scalati, N_CAMPIONE_SIL, SET_SEED)
    }
    
    k_ottimale <- which.max(sil[3:K_MAX]) + 2
    cat(paste0("  -> ", nome_gruppo, ": K scelto = ", k_ottimale, "\n"))
    
    # Costruisce palette ed etichette sul k effettivo dell'anno.
    # Se il dizionario non copre un livello (es. k dinamico > voci definite),
    # usa un fallback generico "Livello N" per non perdere comuni nella mappa.
    dict_gruppo <- etichette_cluster[[nome_gruppo]]
    nomi_livelli_anno <- character(k_ottimale)
    for (lv in seq_len(k_ottimale)) {
      chiave <- paste("Livello", lv)
      nomi_livelli_anno[lv] <- if (!is.null(dict_gruppo[[chiave]]) && !is.na(dict_gruppo[[chiave]])) {
        dict_gruppo[[chiave]]
      } else {
        chiave   # fallback: mostra "Livello N" se non c'è etichetta descrittiva
      }
    }
    names(nomi_livelli_anno) <- paste("Livello", seq_len(k_ottimale))
    palette_gruppo <- setNames(colori_fissi[paste("Livello", seq_len(k_ottimale))], nomi_livelli_anno)
    
    set.seed(SET_SEED)
    mod_kmeans <- kmeans(dati_scalati, centers = k_ottimale, nstart = NSTART)
    
    # --- Logica di assegnazione e tracking centroids ---
    if (anno == ANNI[1]) {
      mfi_temp   <- aggregate(dati$indici$MFI_Decile, by = list(Cluster_Originale = mod_kmeans$cluster), FUN = mean)
      mfi_temp   <- mfi_temp[order(mfi_temp$x), ]
      mapping_cluster <- data.frame(Cluster_Num = mfi_temp$Cluster_Originale, Nome_Cluster = paste("Livello", 1:nrow(mfi_temp)))
      reference_centroids[[nome_gruppo]] <- mod_kmeans$centers[mfi_temp$Cluster_Originale, ]
      rownames(reference_centroids[[nome_gruppo]]) <- mapping_cluster$Nome_Cluster
    } else {
      old_centers <- reference_centroids[[nome_gruppo]]
      new_centers <- mod_kmeans$centers
      dist_mat    <- as.matrix(dist(rbind(new_centers, old_centers)))
      dist_block  <- dist_mat[1:nrow(new_centers), (nrow(new_centers) + 1):ncol(dist_mat)]
      mapping_cluster <- data.frame(Cluster_Num = integer(), Nome_Cluster = character())
      for (i in 1:nrow(new_centers)) {
        if (all(is.infinite(dist_block[i, ]))) {
          numeri_usati <- as.numeric(gsub("Livello ", "", mapping_cluster$Nome_Cluster))
          assegnato    <- paste("Livello", max(c(0, numeri_usati)) + 1)
        } else {
          closest_idx <- which.min(dist_block[i, ])
          assegnato   <- colnames(dist_block)[closest_idx]
          dist_block[, closest_idx] <- Inf
        }
        mapping_cluster <- rbind(mapping_cluster, data.frame(Cluster_Num = i, Nome_Cluster = assegnato))
      }
    }
    
    risultati_gruppo <- data.frame(PRO_COM = dati$meta$PRO_COM, Cluster_Num = mod_kmeans$cluster) %>%
      left_join(mapping_cluster, by = "Cluster_Num")
    
    # [PUNTO 3+4] Salva assegnazione per comune (Nome_Cluster = "Livello X")
    cluster_per_comune[[nome_gruppo]][[anno]] <- risultati_gruppo %>%
      select(PRO_COM, Nome_Cluster)
    
    # --- Profili cluster ---
    dati_reali_gruppo <- dati$indici[, vars, drop = FALSE]
    medie_cluster     <- aggregate(dati_reali_gruppo,
                                   by  = list(Tipologia_Cluster = risultati_gruppo$Nome_Cluster),
                                   FUN = function(x) round(mean(x), 2))
    medie_cluster$MFI_Medio <- round(
      aggregate(dati$indici$MFI_Decile, by = list(risultati_gruppo$Nome_Cluster), FUN = mean)$x, 2)
    profili_cluster[[nome_gruppo]][[anno]] <- medie_cluster
    
    # --- PLOTTING: Elbow ---
    plot_list[[nome_gruppo]]$Elbow[[anno]] <- ggplot(data.frame(k = 1:K_MAX, wss = w), aes(x = k, y = wss)) +
      geom_line(color = "steelblue") + geom_point() + labs(title = anno) + theme_minimal()
    
    # --- PLOTTING: Silhouette ---
    plot_list[[nome_gruppo]]$Sil[[anno]] <- ggplot(data.frame(k = 2:K_MAX, s = sil[2:K_MAX]), aes(x = k, y = s)) +
      geom_line(color = "darkorange") + geom_point() + labs(title = anno) + theme_minimal() +
      geom_vline(xintercept = k_ottimale, linetype = "dashed", color = "red")
    
    # --- PLOTTING: Mappa ---
    mappa_cluster <- mappa_italia %>%
      left_join(risultati_gruppo, by = "PRO_COM") %>%
      filter(!is.na(Nome_Cluster))
    
    # Usa il vettore nomi_livelli_anno (già costruito con fallback) per tradurre
    # Nome_Cluster -> Etichetta_Descrittiva. Nessun comune riceve NA anche quando
    # k dinamico supera le voci nel dizionario etichette_cluster.
    mappa_cluster$Etichetta_Descrittiva <- nomi_livelli_anno[as.character(mappa_cluster$Nome_Cluster)]
    mappa_cluster$Etichetta_Descrittiva <- factor(mappa_cluster$Etichetta_Descrittiva,
                                                  levels = nomi_livelli_anno)
    
    plot_list[[nome_gruppo]]$Mappa[[anno]] <- ggplot(mappa_cluster) +
      geom_sf(aes(fill = Etichetta_Descrittiva), color = NA) +
      scale_fill_manual(values = palette_gruppo, name = "Tipologia di Territorio:", drop = TRUE) +
      labs(title = paste(anno, "| K =", k_ottimale)) +
      theme_void() +
      theme(
        legend.position   = "bottom",
        legend.direction  = "vertical",
        plot.title        = element_text(hjust = 0.5, size = 11, face = "bold", color = "white"),
        plot.background   = element_rect(fill = "black", color = NA),
        panel.background  = element_rect(fill = "black", color = NA),
        legend.text       = element_text(color = "white", size = 9),
        legend.title      = element_text(color = "white", face = "bold"),
        legend.background = element_rect(fill = "black", color = NA),
        legend.key.size   = unit(0.5, "cm")
      )
    
    # [PUNTO 8] PLOTTING: Boxplot variabili chiave per cluster
    # Sceglie la variabile più discriminante (prima del gruppo) come esempio principale
    var_box <- vars[1]
    df_box  <- data.frame(
      Cluster   = risultati_gruppo$Nome_Cluster,
      Valore    = dati$indici[[var_box]],
      MFI       = dati$indici$MFI_Decile
    )
    # Ordine livelli per la legenda
    df_box$Cluster <- factor(df_box$Cluster,
                             levels = paste("Livello", 1:k_ottimale))
    colori_box <- colori_fissi[paste("Livello", 1:k_ottimale)]
    
    plot_list[[nome_gruppo]]$Boxplot[[anno]] <- ggplot(df_box, aes(x = Cluster, y = Valore, fill = Cluster)) +
      geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.3) +
      scale_fill_manual(values = colori_box, guide = "none") +
      labs(title = anno, x = NULL, y = var_box) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 8))
    
    risultati_anno[[paste0("Cluster_", nome_gruppo)]] <-
      as.character(mappa_cluster$Etichetta_Descrittiva[match(risultati_anno$PRO_COM, mappa_cluster$PRO_COM)])
  }
  
  fogli_risultati[[anno]] <- risultati_anno
}

# --- 7. SALVATAGGIO IMMAGINI ------------------------------------------------
cat("\nSalvataggio grafici in '", CARTELLE_OUTPUT, "'...\n")

for (nome_gruppo in names(gruppi_variabili)) {
  
  # Elbow
  g_elbow <- arrangeGrob(grobs = plot_list[[nome_gruppo]]$Elbow, ncol = 2,
                         top = textGrob(paste("Elbow -", nome_gruppo), gp = gpar(fontsize = 14, font = 2)))
  ggsave(file.path(CARTELLE_OUTPUT, paste0("Elbow_", nome_gruppo, ".png")), g_elbow, width = 10, height = 8)
  
  # Silhouette (Validazione)
  g_sil <- arrangeGrob(grobs = plot_list[[nome_gruppo]]$Sil, ncol = 2,
                       top = textGrob(paste("Silhouette -", nome_gruppo), gp = gpar(fontsize = 14, font = 2)))
  ggsave(file.path(CARTELLE_OUTPUT, paste0("Validazione_", nome_gruppo, ".png")), g_sil, width = 14, height = 6)
  
  # Mappe
  g_mappa <- arrangeGrob(grobs = plot_list[[nome_gruppo]]$Mappa, ncol = 2,
                         top = textGrob(paste("Mappe Territoriali -", nome_gruppo),
                                        gp = gpar(fontsize = 14, font = 2, col = "white")))
  ggsave(file.path(CARTELLE_OUTPUT, paste0("Mappe_", nome_gruppo, ".png")),
         g_mappa, width = 10, height = 12, bg = "black")
  
  # [PUNTO 8] Boxplot
  g_box <- arrangeGrob(grobs = plot_list[[nome_gruppo]]$Boxplot, ncol = 2,
                       top = textGrob(paste("Distribuzione per Cluster -", nome_gruppo),
                                      gp = gpar(fontsize = 14, font = 2)))
  ggsave(file.path(CARTELLE_OUTPUT, paste0("Boxplot_", nome_gruppo, ".png")), g_box, width = 12, height = 8)
  
  cat(paste0("\n--- PROFILI ORDINATI: ", toupper(nome_gruppo), " ---\n"))
  for (anno in ANNI) {
    cat(paste0("\n> Anno: ", anno, "\n"))
    print(profili_cluster[[nome_gruppo]][[anno]])
  }
}

# =============================================================================
# --- 8. ANALISI STABILITÀ E TRANSIZIONI [PUNTI 3 & 4] ----------------------
# =============================================================================
cat("\n\n=== ANALISI STABILITÀ TEMPORALE ===\n")

transizioni_output <- list()   # per Excel
stabilita_output   <- list()   # per Excel
kpi_stabilita      <- data.frame(Dimensione = character(), KPI_Stabilita_Pct = numeric())

for (nome_gruppo in names(gruppi_variabili)) {
  
  cat(paste0("\n--- ", toupper(nome_gruppo), " ---\n"))
  
  # Unisce tutte le assegnazioni annuali per comune
  df_tutti <- Reduce(
    function(a, b) full_join(a, b, by = "PRO_COM"),
    mapply(function(df, anno) {
      colnames(df)[2] <- paste0("Cluster_", anno)
      df
    }, cluster_per_comune[[nome_gruppo]], ANNI, SIMPLIFY = FALSE)
  )
  
  # [PUNTO 4] KPI stabilità: % comuni con stesso cluster in TUTTI gli anni
  anni_cols <- paste0("Cluster_", ANNI)
  df_tutti$tutti_uguali <- apply(df_tutti[, anni_cols], 1, function(r) {
    r_clean <- r[!is.na(r)]
    length(r_clean) >= 2 && length(unique(r_clean)) == 1
  })
  pct_stabile <- round(mean(df_tutti$tutti_uguali, na.rm = TRUE) * 100, 1)
  cat(paste0("  KPI Stabilità (stesso cluster in tutti gli anni): ", pct_stabile, "%\n"))
  kpi_stabilita <- rbind(kpi_stabilita,
                         data.frame(Dimensione = nome_gruppo, KPI_Stabilita_Pct = pct_stabile))
  
  # [PUNTO 3] Matrici di transizione tra coppie di anni consecutive
  coppie_anni <- list(
    c("2018", "2019"),
    c("2019", "2021"),
    c("2021", "2022")
  )
  
  for (coppia in coppie_anni) {
    anno_da  <- coppia[1]
    anno_a   <- coppia[2]
    col_da   <- paste0("Cluster_", anno_da)
    col_a    <- paste0("Cluster_", anno_a)
    
    df_pair <- df_tutti[, c("PRO_COM", col_da, col_a)] %>%
      filter(!is.na(.data[[col_da]]), !is.na(.data[[col_a]]))
    
    mat <- table(
      Da = df_pair[[col_da]],
      A  = df_pair[[col_a]]
    )
    
    label <- paste0(nome_gruppo, "_", anno_da, "_", anno_a)
    
    cat(paste0("\n  Transizione ", anno_da, " -> ", anno_a, ":\n"))
    print(mat)
    
    # % rimasti nello stesso livello
    pct_stabile_coppia <- round(sum(diag(mat)) / sum(mat) * 100, 1)
    cat(paste0("  Stabili: ", pct_stabile_coppia, "%\n"))
    
    # Salva per Excel: matrice come dataframe con colonna "Da"
    mat_df <- as.data.frame.matrix(mat)
    mat_df <- cbind(Da = rownames(mat_df), mat_df)
    mat_df$Pct_Stabili <- paste0(pct_stabile_coppia, "%")
    transizioni_output[[label]] <- mat_df
  }
  
  # Salva df_tutti per eventuale uso
  stabilita_output[[nome_gruppo]] <- df_tutti
}

# =============================================================================
# --- 9. CLUSTER COMPOSITO [PUNTO 7] -----------------------------------------
# =============================================================================
cat("\n\n=== CLUSTER COMPOSITO (Econ + Soc + Amb) ===\n")

# Unisce i cluster dei 3 gruppi per anno
composito_output <- list()

for (anno in ANNI) {
  df_econ <- cluster_per_comune[["Economia"]][[anno]] %>% rename(Econ = Nome_Cluster)
  df_soc  <- cluster_per_comune[["Sociale"]][[anno]]  %>% rename(Soc  = Nome_Cluster)
  df_amb  <- cluster_per_comune[["Ambiente"]][[anno]] %>% rename(Amb  = Nome_Cluster)
  
  df_comp <- df_econ %>%
    left_join(df_soc, by = "PRO_COM") %>%
    left_join(df_amb, by = "PRO_COM") %>%
    filter(!is.na(Econ), !is.na(Soc), !is.na(Amb))
  
  # Estrae numero livello e calcola punteggio medio (1=meno fragile, N=più fragile)
  estraglia_num <- function(x) as.numeric(gsub("Livello ", "", x))
  
  df_comp <- df_comp %>%
    mutate(
      Num_Econ  = estraglia_num(Econ),
      Num_Soc   = estraglia_num(Soc),
      Num_Amb   = estraglia_num(Amb),
      # Normalizza su scala 0-1 per ciascuna dimensione (rispetto al max k del gruppo)
      Norm_Econ = (Num_Econ - 1) / (max(Num_Econ, na.rm = TRUE) - 1 + 1e-9),
      Norm_Soc  = (Num_Soc  - 1) / (max(Num_Soc,  na.rm = TRUE) - 1 + 1e-9),
      Norm_Amb  = (Num_Amb  - 1) / (max(Num_Amb,  na.rm = TRUE) - 1 + 1e-9),
      # Punteggio composito medio normalizzato (0 = minima fragilità, 1 = massima)
      Score_Composito = round((Norm_Econ + Norm_Soc + Norm_Amb) / 3, 3),
      # Tipologia qualitativa: tripla/doppia/singola fragilità
      N_Dimensioni_Alte = (Num_Econ == max(Num_Econ)) +
        (Num_Soc  == max(Num_Soc))  +
        (Num_Amb  == max(Num_Amb)),
      Tipologia_Composita = case_when(
        Score_Composito <= 0.25                          ~ "Bassa Fragilità Complessiva",
        Score_Composito <= 0.50                          ~ "Fragilità Moderata",
        Score_Composito <= 0.75                          ~ "Alta Fragilità",
        TRUE                                             ~ "Fragilità Critica (Multi-dimensionale)"
      )
    )
  
  composito_output[[anno]] <- df_comp
  
  cat(paste0("\nAnno: ", anno, "\n"))
  print(table(df_comp$Tipologia_Composita))
  
  # Distribuzione score composito
  cat(paste0("  Score medio composito: ", round(mean(df_comp$Score_Composito), 3), "\n"))
  cat(paste0("  Score mediano:         ", round(median(df_comp$Score_Composito), 3), "\n"))
}

# =============================================================================
# --- 10. SALVATAGGIO EXCEL COMPLETO [PUNTO 5] --------------------------------
# =============================================================================
cat("\nSalvataggio Excel completo...\n")

wb <- createWorkbook()

# Fogli risultati cluster per anno (già presenti nella versione precedente)
for (anno in ANNI) {
  addWorksheet(wb, paste0("Risultati_", anno))
  writeData(wb, paste0("Risultati_", anno), fogli_risultati[[anno]])
}

# [PUNTO 5] Profili cluster per dimensione e anno
for (nome_gruppo in names(gruppi_variabili)) {
  for (anno in ANNI) {
    sheet_name <- paste0("Profili_", substr(nome_gruppo, 1, 3), "_", anno)
    addWorksheet(wb, sheet_name)
    writeData(wb, sheet_name, profili_cluster[[nome_gruppo]][[anno]])
  }
}

# [PUNTO 3] Matrici di transizione
for (nome_foglio in names(transizioni_output)) {
  # Trunca il nome a 31 caratteri (limite Excel)
  nome_troncato <- substr(nome_foglio, 1, 31)
  addWorksheet(wb, nome_troncato)
  writeData(wb, nome_troncato, transizioni_output[[nome_foglio]])
}

# [PUNTO 4] KPI di stabilità sintetici
addWorksheet(wb, "KPI_Stabilita")
writeData(wb, "KPI_Stabilita", kpi_stabilita)

# [PUNTO 7] Cluster composito per anno
for (anno in ANNI) {
  sheet_name <- paste0("Composito_", anno)
  addWorksheet(wb, sheet_name)
  writeData(wb, sheet_name, composito_output[[anno]])
}

saveWorkbook(wb, file = "Risultati_KMeans_v4.xlsx", overwrite = TRUE)
cat("Excel salvato: Risultati_KMeans_v4.xlsx\n")
cat("\nAnalisi v4.0 conclusa con successo!\n")