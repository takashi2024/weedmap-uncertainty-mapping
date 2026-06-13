library(dplyr)
library(tidyr)
library(ggplot2)

# ---------------------------
# Settings
# ---------------------------
ROOT  <- file.path("/Users/takashi/LocalAnalysis/WeedMap/ForGithub", "Results_TabICL")
dates <- c(
  "data_20250414","data_20250424","data_20250430","data_20250506",
  "data_20250513","data_20250520","data_20250526","data_20250602"
)
models <- c("OK", "RK", "Vanilla", "KpR")

# ---------------------------
# QCP function (generic)
# ---------------------------
quantile_coverage_function <- function(test, quantile_predictions) {
  coverage <- sapply(seq_len(ncol(quantile_predictions)), function(j) {
    mean(test <= quantile_predictions[, j]) * 100
  })
  coverage.df <- as.data.frame(t(coverage))
  MPIW <- mean(quantile_predictions$q0.9 - quantile_predictions$q0.1) +
          mean(quantile_predictions$q0.8 - quantile_predictions$q0.2) +
          mean(quantile_predictions$q0.7 - quantile_predictions$q0.3) +
          mean(quantile_predictions$q0.6 - quantile_predictions$q0.4)
  coverage_MPIW.df <- cbind(coverage.df, MPIW)
  colnames(coverage_MPIW.df) <- c(
    "QCP0.001","QCP0.1","QCP0.2","QCP0.3","QCP0.4",
    "QCP0.5","QCP0.6","QCP0.7","QCP0.8","QCP0.9","QCP0.999","MPIW"
  )
  return(coverage_MPIW.df)
}

read_first_col <- function(path) read.csv(path)[[1]]

# ---------------------------
# Load raw test + quantile data for one date/model
# Returns list(test, qdf) in log1p scale, or NULL
# ---------------------------
load_raw <- function(date_tag, model_name, target) {
  nndm_dir <- file.path(ROOT, date_tag, "NNDM")

  if (model_name == "OK") {
    true_path <- file.path(nndm_dir, "Vanilla", paste0(target, "_test.csv"))
    q_path    <- file.path(nndm_dir, paste0(target, "_OK_quantiles.csv"))
    if (!file.exists(true_path) || !file.exists(q_path)) return(NULL)
    test_vec <- read_first_col(true_path)
    qdf      <- read.csv(q_path)
  } else if (model_name == "Vanilla") {
    sub_dir   <- file.path(nndm_dir, "Vanilla")
    true_path <- file.path(sub_dir, paste0(target, "_test.csv"))
    q_path    <- file.path(sub_dir, paste0(target, "_predictions_quantiles.csv"))
    if (!file.exists(true_path) || !file.exists(q_path)) return(NULL)
    test_vec <- read_first_col(true_path)
    qdf      <- read.csv(q_path)
  } else if (model_name == "KpR") {
    sub_dir   <- file.path(nndm_dir, "KpR")
    true_path <- file.path(sub_dir, paste0(target, "_test.csv"))
    q_path    <- file.path(sub_dir, paste0(target, "_predictions_quantiles.csv"))
    if (!file.exists(true_path) || !file.exists(q_path)) return(NULL)
    test_vec <- read_first_col(true_path)
    qdf      <- read.csv(q_path)
  } else if (model_name == "RK") {
    sub_dir   <- file.path(nndm_dir, "RK")
    pred_path <- file.path(sub_dir, paste0(target, "_RK_predictions.csv"))
    q_path    <- file.path(sub_dir, paste0(target, "_RK_quantiles.csv"))
    if (!file.exists(pred_path) || !file.exists(q_path)) return(NULL)
    pred_df  <- read.csv(pred_path)
    if (!"true" %in% names(pred_df)) return(NULL)
    test_vec <- pred_df[["true"]]
    qdf      <- read.csv(q_path)
  } else {
    return(NULL)
  }

  if (length(test_vec) != nrow(qdf)) return(NULL)
  list(test = test_vec, qdf = qdf)
}

# Display name mapping
model_label <- c(OK="OK", Vanilla="TabICL", KpR="TabICL-KpR", RK="TabICL-RK")

# ---------------------------
# Collect rows for three variants
# ---------------------------
rows_log  <- list()   # (A) original log1p scale  — same as main_5_QCP_NNDM.R
rows_nz   <- list()   # (B) non-zero observations only (log1p)
rows_bt   <- list()   # (C) back-transformed count scale (expm1, clamp >= 0)

for (date_tag in dates) {
  vanilla_dir <- file.path(ROOT, date_tag, "NNDM", "Vanilla")
  if (!dir.exists(vanilla_dir)) next
  test_files <- list.files(vanilla_dir, pattern = "_test\\.csv$", full.names = FALSE)
  targets    <- gsub("_test\\.csv$", "", test_files)

  for (target in targets) {
    for (m in models) {
      raw <- load_raw(date_tag, m, target)
      if (is.null(raw)) next

      lbl  <- model_label[m]
      test <- raw$test
      qdf  <- raw$qdf
      n    <- length(test)
      pct_zero <- mean(test == 0) * 100

      meta <- data.frame(
        target_soil_property = target,
        model   = lbl,
        dataset = date_tag,
        n_total = n,
        pct_zero = pct_zero
      )

      # (A) log1p — full dataset
      rows_log[[length(rows_log)+1]] <- cbind(meta, quantile_coverage_function(test, qdf))

      # (B) non-zero only (log1p)
      nz_idx <- test > 0
      if (sum(nz_idx) >= 10) {
        rows_nz[[length(rows_nz)+1]] <- cbind(
          meta,
          n_used = sum(nz_idx),
          quantile_coverage_function(test[nz_idx], qdf[nz_idx, ])
        )
      }

      # (C) back-transformed: expm1(), clamp negatives to 0
      test_bt <- pmax(expm1(test), 0)
      qdf_bt  <- as.data.frame(lapply(qdf, function(x) pmax(expm1(x), 0)))
      rows_bt[[length(rows_bt)+1]] <- cbind(meta, quantile_coverage_function(test_bt, qdf_bt))
    }
  }
}

res_log <- bind_rows(rows_log)
res_nz  <- bind_rows(rows_nz)
res_bt  <- bind_rows(rows_bt)

# ---------------------------
# QCP deviation summary
# ---------------------------
summarise_qcp <- function(res, label) {
  res %>%
    pivot_longer(starts_with("QCP"), names_to = "quantile", values_to = "empirical_coverage") %>%
    mutate(target_quantile = as.numeric(gsub("QCP", "", quantile)) * 100,
           deviation = abs(empirical_coverage - target_quantile)) %>%
    group_by(model) %>%
    summarise(
      mean_abs_dev   = round(mean(deviation, na.rm=TRUE), 2),
      median_abs_dev = round(median(deviation, na.rm=TRUE), 2),
      .groups = "drop"
    ) %>%
    mutate(scale = label)
}

dev_summary <- bind_rows(
  summarise_qcp(res_log, "log1p (all obs)"),
  summarise_qcp(res_nz,  "log1p (non-zero only)"),
  summarise_qcp(res_bt,  "count scale (all obs)")
)

cat("\n=== Mean absolute QCP deviation by model and scale ===\n")
print(dev_summary %>% arrange(model, scale), n=50)

# ---------------------------
# Build long format for plotting
# ---------------------------
to_long <- function(res) {
  res %>%
    pivot_longer(starts_with("QCP"), names_to = "quantile", values_to = "empirical_coverage") %>%
    mutate(target_quantile = as.numeric(gsub("QCP", "", quantile)) * 100,
           deviation = abs(empirical_coverage - target_quantile))
}

long_log <- to_long(res_log) %>% mutate(scale = "log1p (all)")
long_nz  <- to_long(res_nz)  %>% mutate(scale = "log1p (non-zero)")
long_bt  <- to_long(res_bt)  %>% mutate(scale = "count (back-transformed)")

long_all <- bind_rows(long_log, long_nz, long_bt)

# Mean deviation annotations
dev_ann <- long_all %>%
  group_by(model, scale) %>%
  summarise(mean_abs_dev = round(mean(deviation, na.rm=TRUE), 1), .groups="drop")

# ---------------------------
# Combined 3-panel QCP plot (one panel per scale)
# ---------------------------
col_map <- c(
  "OK"         = "#4477AA",
  "TabICL-KpR"  = "#cc6699",
  "TabICL"      = "#009999",
  "TabICL-RK"   = "#e6b800"
)

make_qcp_plot <- function(long_data, dev_data, title_str) {
  ggplot(long_data,
         aes(x = target_quantile, y = empirical_coverage,
             color = model, group = interaction(dataset, target_soil_property))) +
    geom_line(linewidth = 0.6, alpha = 0.4) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
    geom_text(data = dev_data,
              aes(x = 65, y = 8,
                  label = paste0("bar(delta)[QCP]==", mean_abs_dev),
                  color = model),
              parse = TRUE, inherit.aes = FALSE, size = 3.5) +
    scale_x_continuous(name = "Target quantile (%)", breaks = seq(0,100,25), limits = c(0,100)) +
    scale_y_continuous(name = "Empirical coverage (%)", breaks = seq(0,100,25), limits = c(0,100)) +
    scale_color_manual(values = col_map) +
    facet_wrap(~ model) +
    ggtitle(title_str) +
    theme_bw() +
    theme(
      panel.grid.minor = element_blank(),
      axis.title  = element_text(size = 14, color = "black"),
      axis.text   = element_text(size = 12, color = "black"),
      strip.text  = element_text(size = 12, face = "bold"),
      plot.title  = element_text(size = 14, face = "bold"),
      legend.position = "none"
    )
}

p_log <- make_qcp_plot(
  long_log,
  filter(dev_ann, scale == "log1p (all)"),
  "A: log1p scale — all observations"
)
p_nz  <- make_qcp_plot(
  long_nz,
  filter(dev_ann, scale == "log1p (non-zero)"),
  "B: log1p scale — non-zero observations only"
)
p_bt  <- make_qcp_plot(
  long_bt,
  filter(dev_ann, scale == "count (back-transformed)"),
  "C: count scale (back-transformed) — all observations"
)

# Save individual plots
out_dir <- file.path(ROOT, "figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ggsave(file.path(out_dir, "NNDM_QCP_logscale_all.tiff"),
       p_log, dpi=300, width=10, height=7, units="in", device="tiff")
ggsave(file.path(out_dir, "NNDM_QCP_logscale_nonzero.tiff"),
       p_nz,  dpi=300, width=10, height=7, units="in", device="tiff")
ggsave(file.path(out_dir, "NNDM_QCP_countscale.tiff"),
       p_bt,  dpi=300, width=10, height=7, units="in", device="tiff")

# Save combined 3-row panel
library(patchwork)
p_combined <- p_log / p_nz / p_bt +
  plot_annotation(
    title   = "QCP sensitivity: scale and zero-value handling",
    caption = paste0(
      "A: log1p, all obs (original analysis).\n",
      "B: log1p, non-zero obs only (removes zero-inflation artefact).\n",
      "C: count scale after expm1() back-transformation, negatives clamped to 0."
    ),
    theme = theme(
      plot.title   = element_text(size = 16, face = "bold"),
      plot.caption = element_text(size = 10, hjust = 0)
    )
  )

ggsave(file.path(out_dir, "NNDM_QCP_sensitivity_3panel.tiff"),
       p_combined, dpi=300, width=10, height=21, units="in", device="tiff")
ggsave(file.path(out_dir, "NNDM_QCP_sensitivity_3panel.png"),
       p_combined, dpi=150, width=10, height=21, units="in")

# Save summary CSV
write.csv(dev_summary,
          file.path(out_dir, "NNDM_QCP_sensitivity_deviation_summary.csv"),
          row.names = FALSE)

cat("\nSaved figures and summary to", out_dir, "\n")
