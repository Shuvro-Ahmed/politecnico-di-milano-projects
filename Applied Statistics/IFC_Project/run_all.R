# Master wrapper: run every R/*.R script in order, capture status + time.
# Outputs: run_all_log.csv with one row per script.

scripts <- list.files("R", pattern = "^[0-9].*\\.R$", full.names = TRUE)
scripts <- sort(scripts)

results <- data.frame()
for (s in scripts) {
  cat("\n========================================\n")
  cat(">>> RUNNING:", s, "\n")
  cat("========================================\n")
  t0 <- Sys.time()
  ok <- tryCatch({
    err_file <- tempfile()
    out_file <- tempfile()
    cmd <- sprintf('"%s/bin/Rscript.exe" %s',
                   R.home(), shQuote(s))
    status <- system(cmd, ignore.stdout = FALSE, ignore.stderr = FALSE)
    status == 0
  }, error = function(e) FALSE)
  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  results <- rbind(results, data.frame(
    script    = basename(s),
    success   = ok,
    seconds   = round(dt, 1)
  ))
  cat(sprintf("\n[Result] %s : %s (%.1f sec)\n",
              basename(s), if (ok) "OK" else "FAIL", dt))
}

write.csv(results, "run_all_log.csv", row.names = FALSE)
cat("\n=== FINAL SUMMARY ===\n")
print(results, row.names = FALSE)
cat(sprintf("\nTotal: %d scripts, %d OK, %d FAIL, total time = %.1f min\n",
            nrow(results), sum(results$success),
            sum(!results$success), sum(results$seconds) / 60))
