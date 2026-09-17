IFC Italy-2018 reference notes
================================

This script prepares denominator/weight variables and calculates a one-row Excel sheet
with approximate/exact Italy-2018 reference points from the municipal indicator values in the IFC workbook.

Reference output
----------------
The Excel file IFC_Italy_2018_reference_points.xlsx contains sheet reference_2018 with one data row.

Indicator 3 - accessibility to essential services
---------------------------------------------------
This script uses a simple mean for indicator 3 because no natural additive denominator is available.
If you want a resident-exposure interpretation instead, change I3_ref = mean_safe(I3) to
I3_ref = weighted_mean_safe(I3, pop_total).

Indicator 6 - protected natural areas
-------------------------------------
Official definition: percentage of municipal surface covered by terrestrial protected natural areas
included in EUAP and/or Natura 2000 over total municipal surface. Official source is ISPRA/MASE;
the numerator is produced by GIS overlay procedures net of overlaps.

Approximation used here:
  I6_Italy_2018_approx = weighted.mean(I6_municipality_raw_percent, area_total)
equivalently:
  100 * sum((I6_municipality_raw_percent / 100) * area_total) / sum(area_total)

NOTE: this is not exact if area_total comes from ISTAT municipal area, because the IFC note warns
that ISPRA total surfaces may differ from ISTAT total surfaces. Use this only with an explicit note.
Also remember that indicator 6 has inverse polarity in the IFC methodology; polarity is handled later
in normalization, not in this raw Italy reference calculation.

Indicator 10 - population increase / net migration rate
--------------------------------------------------------
Official denominator is resident population at 31 Dec 2011. The script uses pop_2011 as the aggregation weight:
  I10_Italy_2018 = weighted.mean(I10_municipality_raw_rate, pop_2011)

Indicator 11 - density of local units of industry/services per 1,000 inhabitants
-------------------------------------------------------------------------------
Exact aggregate formula using this file:
  I11_Italy_2018 = 1000 * sum(local_units_2018) / sum(pop_total)
Do not take the simple average of municipal densities.

Indicator 12 - low-productivity local units
-------------------------------------------
Exact indicator needs the raw percentage of persons employed in low-productivity local units
or the numerator: addetti in local units below the first quartile of nominal labour productivity
by ATECO division. Public IFC values are ventile classes (1-20), so they cannot reconstruct
the true raw national percentage.

Approximation used here:
  I12_Italy_2018_very_rough = weighted.mean(I12_municipality_ventile, persons_employed_lu_2018)
This is better than a simple mean because the definition is employment-based, but it is still
an average of ranks/classes, not the official raw IFC indicator 12 reference.

Generated columns in the CSV
----------------------------
area_total: area weight, used for indicator 6 approximation and area-based indicators
pop_total: 2018 total population
pop_0_19, pop_20_64, pop_25_64, pop_65_plus: demographic denominators
pop_2011: 2011 total population, needed for indicator 10
local_units_2018: ASIA-UL local units, needed for indicator 11 numerator
persons_employed_lu_2018: ASIA-UL persons employed, useful as indicator 12 weight
