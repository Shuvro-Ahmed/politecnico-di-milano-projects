# ============================================================
# Build IFC Italy-2018 reference row
# Output: one Excel file only: IFC_Italy_2018_reference_points.xlsx
#
# Important notes preserved:
# - I3 is a simple municipal mean because accessibility has no additive numerator/denominator.
# - I6 is an area-weighted approximation; official protected-area numerator is from GIS overlay.
# - I10 uses 2011 resident population as denominator/weight, as in the IFC methodology.
# - I11 is exact from totals: 1000 * total local units / total population.
# - I12 is NOT exact: municipal values are ventiles/classes, so this script uses an
#   employment-weighted ventile average as a rough approximation.
# ============================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))
need <- c("readxl", "readr", "dplyr", "stringr", "tibble", "openxlsx")
invisible(lapply(need, function(p) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
  library(p, character.only = TRUE)
}))

# ------------------------- CONFIG --------------------------
ifc_workbook_path <- "data/raw/ifc/composite_fragility_all_years.xlsx"
ifc_sheet <- "2018"
indicator_cols <- 4:15                 # 12 IFC indicators; columns 1-2 are code/name and column 3 is MFI decile
indicator_names <- paste0("I", 1:12)

population_2018_csv <- "data/raw/istat/popolazione_full.csv"
population_2011_csv <- "data/raw/istat/popolazione_2011.csv"
area_csv <- "data/raw/istat/area_totale.csv"
area_data_type <- "TOTAREA2"           # km2; TOTAREA would be hectares
asia_ul_xlsx <- "data/raw/istat/asia_size_employees.xlsx"
output_xlsx <- "data/raw/ifc/italy_2018_reference.xlsx"

# ------------------------- HELPERS -------------------------
as_num <- function(x) {
  x <- stringr::str_replace_all(stringr::str_trim(as.character(x)), "[\\s\\u00A0]", "")
  has_comma <- stringr::str_detect(x, ",")
  x[has_comma] <- stringr::str_replace_all(x[has_comma], "\\.", "")
  x[has_comma] <- stringr::str_replace_all(x[has_comma], ",", ".")
  suppressWarnings(as.numeric(x))
}

pick_col <- function(nms, candidates, label) {
  out <- intersect(candidates, nms)[1]
  if (is.na(out)) stop("Missing ", label, " column. Tried: ", paste(candidates, collapse = ", "))
  out
}

wmean <- function(x, w) {
  ok <- !is.na(x) & !is.na(w) & w > 0
  if (!any(ok)) return(NA_real_)
  sum(x[ok] * w[ok]) / sum(w[ok])
}

mean_safe <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  mean(x, na.rm = TRUE)
}

per1000 <- function(num, den) {
  ok <- !is.na(num) & !is.na(den) & den > 0
  if (!any(ok)) return(NA_real_)
  1000 * sum(num[ok]) / sum(den[ok])
}

# Compact municipality-code bridge: old code(s) -> harmonised IFC code.
bridge_spec <- c(
  "001317:001005,001138,001182", "001318:001151,001277,001297", "003166:003071,003157",
  "005122:005079,005110", "006193:006064,006089", "012143:012028,012111",
  "012144:012009,012018,012095", "013255:013038,013215", "013256:013199,013228",
  "015251:015235,015246", "018193:018028,018132,018170", "019116:019042,019071",
  "020073:020006,020009", "022251:022126,022225", "022252:022046,022088,022111",
  "022253:022027,022030,022063,022152,022154", "022254:022041,022070,022211",
  "024125:024023,024031,024093,024114", "024126:024058,024059", "024127:024033,024054",
  "024128:024103", "025074:025028,025034,025061", "026096:026024,026054",
  "034051:034021,034037", "038029:038002,038020", "038030:038009,038024",
  "041071:041003,041059", "048054:048003,048045", "075098:075001,075062",
  "096087:096017,096051", "096088:096062,096070,096073,096084", "099030:041033",
  "099031:041060", "103079:103020,103027,103030",
  "016215:097080", "024124:024011,024069", "028107:028051,028074,028081",
  "030190:030038,030134", "030191:030050,030125", "078157:078044,078108"
)
bridge <- dplyr::bind_rows(lapply(bridge_spec, function(s) {
  z <- strsplit(s, ":")[[1]]
  tibble::tibble(old_PRO_COM = unlist(strsplit(z[2], ",")), new_PRO_COM = z[1])
}))

harmonise <- function(df, code_col = "old_PRO_COM") {
  df %>%
    mutate(old_PRO_COM = stringr::str_pad(as.character(.data[[code_col]]), 6, pad = "0")) %>%
    left_join(bridge, by = "old_PRO_COM") %>%
    mutate(PRO_COM = if_else(is.na(new_PRO_COM), old_PRO_COM, new_PRO_COM)) %>%
    select(-new_PRO_COM)
}

read_pop_first12 <- function(path) {
  lines <- readr::read_lines(path, locale = readr::locale(encoding = "UTF-8"))
  header <- stringr::str_split_fixed(lines[1], ",", 13)[1, 1:12]
  mat <- stringr::str_split_fixed(lines[-1], ",", 13)[, 1:12, drop = FALSE]
  out <- tibble::as_tibble(as.data.frame(mat, stringsAsFactors = FALSE)); names(out) <- header
  out
}

pop_weights <- function(path, year) {
  read_pop_first12(path) %>%
    filter(TIME_PERIOD == year, INDICATOR == "RESPOP_AV", GENDER == "T", str_detect(REF_AREA, "^[0-9]{6}$")) %>%
    transmute(old_PRO_COM = REF_AREA, AGE_CLASS, pop = as_num(Observation)) %>%
    harmonise() %>%
    group_by(PRO_COM) %>%
    summarise(
      pop_total = sum(pop[AGE_CLASS == "TOTAL"], na.rm = TRUE),
      pop_20_64 = sum(pop[AGE_CLASS %in% c("Y20-24","Y25-29","Y30-34","Y35-39","Y40-44","Y45-49","Y50-54","Y55-59","Y60-64")], na.rm = TRUE),
      pop_25_64 = sum(pop[AGE_CLASS %in% c("Y25-29","Y30-34","Y35-39","Y40-44","Y45-49","Y50-54","Y55-59","Y60-64")], na.rm = TRUE),
      .groups = "drop"
    )
}

# ------------------------- READ INPUTS ---------------------
# IFC target municipalities and municipal indicator values
ifc_raw <- readxl::read_excel(ifc_workbook_path, sheet = ifc_sheet, skip = 2, col_names = FALSE, col_types = "text")
if (ncol(ifc_raw) < max(indicator_cols)) stop("IFC workbook does not contain the expected 12 indicator columns. Expected indicators in columns 4:15 after code, territory and MFI decile.")
# Sanity check: column 3 is the official MFI decile, not an elementary indicator.
# Using 3:14 would shift all indicators and make the reference row wrong.

ifc_target <- ifc_raw %>%
  transmute(PRO_COM = str_pad(as.character(...1), 6, pad = "0"), Territory_ifc = as.character(...2)) %>%
  filter(!is.na(PRO_COM), !is.na(Territory_ifc)) %>%
  group_by(PRO_COM) %>% summarise(Territory_ifc = paste(unique(Territory_ifc), collapse = " / "), .groups = "drop")

ind_block <- tibble::as_tibble(lapply(ifc_raw[names(ifc_raw)[indicator_cols]], as_num)); names(ind_block) <- indicator_names
ifc_indicators <- ifc_raw %>%
  transmute(PRO_COM = str_pad(as.character(...1), 6, pad = "0"), Territory_ifc = as.character(...2)) %>%
  bind_cols(ind_block) %>%
  filter(!is.na(PRO_COM), !is.na(Territory_ifc)) %>%
  group_by(PRO_COM) %>%
  summarise(across(all_of(indicator_names), ~ mean(.x, na.rm = TRUE)), .groups = "drop")

# Population: 2018 for most denominators, 2011 for indicator 10
pop18 <- pop_weights(population_2018_csv, "2018")
pop11 <- pop_weights(population_2011_csv, "2011") %>% transmute(PRO_COM, pop_2011 = pop_total)

# Municipal area
area_raw <- readr::read_csv(area_csv, col_types = readr::cols(.default = readr::col_character()))
if ("DATA_TYPE" %in% names(area_raw)) area_raw <- area_raw %>% filter(DATA_TYPE == area_data_type)
area_code <- pick_col(names(area_raw), c("REF_AREA", "ITTER107", "PRO_COM", "COD_COM", "Code", "Territory code"), "area municipality code")
area_value <- pick_col(names(area_raw), c("Observation", "OBS_VALUE", "Value", "VALUE", "value", "obs_value"), "area value")
area <- area_raw %>%
  transmute(old_PRO_COM = str_pad(as.character(.data[[area_code]]), 6, pad = "0"), area_total = as_num(.data[[area_value]])) %>%
  filter(str_detect(old_PRO_COM, "^[0-9]{6}$"), !is.na(area_total)) %>%
  harmonise() %>%
  group_by(PRO_COM) %>% summarise(area_total = sum(area_total, na.rm = TRUE), .groups = "drop")

# ASIA-UL: local units and persons employed, downloaded with size=[TOTAL], activity=[0010] TOTAL
asia <- readxl::read_excel(asia_ul_xlsx, sheet = 1, skip = 6, col_names = FALSE, col_types = "text") %>%
  slice(-(1:2)) %>%
  transmute(
    old_PRO_COM = str_extract(as.character(...1), "(?<=\\[)[0-9]{6}(?=\\])"),
    local_units_2018 = as_num(...2),
    persons_employed_lu_2018 = as_num(...3)
  ) %>%
  filter(!is.na(old_PRO_COM)) %>%
  harmonise() %>%
  group_by(PRO_COM) %>%
  summarise(
    local_units_2018 = sum(local_units_2018, na.rm = TRUE),
    persons_employed_lu_2018 = sum(persons_employed_lu_2018, na.rm = TRUE),
    .groups = "drop"
  )

# ------------------------- JOIN + CHECK --------------------
df <- ifc_target %>%
  left_join(ifc_indicators, by = "PRO_COM") %>%
  left_join(area, by = "PRO_COM") %>%
  left_join(pop18, by = "PRO_COM") %>%
  left_join(pop11, by = "PRO_COM") %>%
  left_join(asia, by = "PRO_COM")

missing <- df %>% filter(is.na(area_total) | is.na(pop_total) | is.na(pop_2011) | is.na(local_units_2018) | is.na(persons_employed_lu_2018))
if (nrow(missing) > 0) {
  stop("Missing weights for ", nrow(missing), " IFC municipalities. No audit files are written by this concise script; inspect object `missing` interactively if needed.")
}

# ------------------------- REFERENCE ROW -------------------
reference_row <- df %>%
  summarise(
    reference_area = "Italy",
    reference_year = 2018,

    # Actual IFC workbook order:
    # I1 cars, I2 waste, I3 protected areas, I4 landslides, I5 soil consumption,
    # I6 accessibility, I7 dependency, I8 low education, I9 employment,
    # I10 net migration, I11 local-unit density ventile, I12 low-productivity ventile.
    I1_ref = wmean(I1, pop_total),             # high-emission motorisation rate
    I2_ref = wmean(I2, pop_total),             # undifferentiated waste per inhabitant
    I3_ref_approx = wmean(I3, area_total),     # protected natural areas, approximate
    I4_ref = wmean(I4, area_total),            # landslide-risk area
    I5_ref = wmean(I5, area_total),            # land consumption
    I6_ref = mean_safe(I6),                    # accessibility, no additive denominator
    I7_ref = wmean(I7, pop_20_64),             # dependency index; keep same denominator choice as previous script
    I8_ref = wmean(I8, pop_25_64),             # low education, ages 25-64
    I9_ref = wmean(I9, pop_20_64),             # employment rate, ages 20-64
    I10_ref = wmean(I10, pop_2011),            # net migration / 2011 population

    # Unit-consistent I11 reference:
    # Municipal Ind11 is on a 1-20 ventile/class scale, but we can reconstruct
    # the raw local-unit density from ASIA-UL and population. Therefore we:
    #   1) compute Italy raw density from totals;
    #   2) locate that raw Italy value inside the 2018 municipal raw-density distribution;
    #   3) use the corresponding ventile class as the reference.
    I11_ref = {
      raw_m <- 1000 * local_units_2018 / pop_total
      raw_it <- per1000(local_units_2018, pop_total)
      p_it <- mean(raw_m <= raw_it, na.rm = TRUE)
      max(1, min(20, ceiling(20 * p_it)))
    },
    I11_ref_raw_density_info_only = per1000(local_units_2018, pop_total),

    # Approximation only: municipal Ind12 is also a ventile/class, not raw low-productivity percentage.
    I12_ref_rough_employment_weighted_ventile = wmean(I12, persons_employed_lu_2018),

    note_order = "Workbook order: I1 cars, I2 waste, I3 protected, I4 landslides, I5 soil, I6 accessibility, I7 dependency, I8 education, I9 employment, I10 migration, I11 LU ventile, I12 low-productivity ventile.",
    note_I3 = "Approximation: area-weighted municipal protected-area value; official numerator is GIS protected-area overlay and may use different surfaces.",
    note_I6 = "Simple municipal mean; accessibility has no natural additive denominator.",
    note_I11 = "Municipal Ind11 is a ventile/class (1-20). The script computes Italy raw LU density from totals, then converts it to the corresponding 2018 municipal ventile class; raw density is reported only as info.",
    note_I12 = "Rough approximation: employment-weighted mean of municipal ventiles/classes; exact raw value needs Frame-SBS Territorial low-productivity numerator."
  )

# ------------------------- WRITE ONLY FINAL FILE -----------
wb <- openxlsx::createWorkbook()
openxlsx::addWorksheet(wb, "reference_2018")
openxlsx::writeData(wb, "reference_2018", reference_row)
openxlsx::freezePane(wb, "reference_2018", firstRow = TRUE)
openxlsx::setColWidths(wb, "reference_2018", cols = 1:ncol(reference_row), widths = "auto")
openxlsx::saveWorkbook(wb, output_xlsx, overwrite = TRUE)

cat("DONE. Wrote only:", output_xlsx, "\n")
