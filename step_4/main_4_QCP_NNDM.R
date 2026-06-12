library(dplyr)
library(tidyr)
library(ggplot2)

# ---------------------------
# Settings
# ---------------------------
ROOT <- file.path("/Users/takashi/LocalAnalysis/WeedMap", "Results_TabICL")

dates <- c(
  "data_20250414","data_20250424","data_20250430","data_20250506",
  "data_20250513","data_20250520","data_20250526","data_20250602"
)

# No IDWpR in NNDM pipeline
models <- c("OK", "RK", "Vanilla", "KpR")

# ---------------------------
# quantile_coverage_function
# (identical to Kriging_prior_Regression-main/3_Analyse/2_Fig4_FigB3.R)
# ---------------------------
quantile_coverage_function <- function(test, quantile_predictions) {
  coverage <- sapply(1:ncol(quantile_predictions), function(j) {
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

# ---------------------------
# Helpers
# ---------------------------
read_first_col <- function(path) {
  df <- read.csv(path)
  df[[1]]
}

# ---------------------------
# compute_one_nndm: one row per date/model/target
# ---------------------------
# NNDM file layout (all under Results/data_{date}/NNDM/):
#   OK      : {target}_OK_quantiles.csv            (q0.001..q0.999)
#             true values: Vanilla/{target}_test.csv
#   Vanilla : Vanilla/{target}_test.csv
#             Vanilla/{target}_predictions_quantiles.csv
#   KpR     : KpR/{target}_test.csv
#             KpR/{target}_predictions_quantiles.csv
#   RK      : RK/{target}_RK_predictions.csv       (col: true)
#             RK/{target}_RK_quantiles.csv          (q0.001..q0.999)

compute_one_nndm <- function(date_tag, model_name, target) {
  nndm_dir <- file.path(ROOT, date_tag, "NNDM")

  if (model_name == "OK") {
    true_path <- file.path(nndm_dir, "Vanilla", paste0(target, "_test.csv"))
    q_path    <- file.path(nndm_dir, paste0(target, "_OK_quantiles.csv"))
    if (!file.exists(true_path) || !file.exists(q_path)) return(NULL)
    test_vec <- read_first_col(true_path)
    qdf <- read.csv(q_path)
    if (length(test_vec) != nrow(qdf)) {
      warning(paste0("Row mismatch OK ", date_tag, ": test=", length(test_vec),
                     " q=", nrow(qdf)))
      return(NULL)
    }
    return(data.frame(
      target_soil_property = target,
      model   = "OK",
      dataset = date_tag,
      quantile_coverage_function(test_vec, qdf)
    ))
  }

  if (model_name == "Vanilla") {
    sub_dir   <- file.path(nndm_dir, "Vanilla")
    true_path <- file.path(sub_dir, paste0(target, "_test.csv"))
    q_path    <- file.path(sub_dir, paste0(target, "_predictions_quantiles.csv"))
    if (!file.exists(true_path) || !file.exists(q_path)) return(NULL)
    test_vec <- read_first_col(true_path)
    qdf <- read.csv(q_path)
    if (length(test_vec) != nrow(qdf)) {
      warning(paste0("Row mismatch Vanilla ", date_tag, ": test=", length(test_vec),
                     " q=", nrow(qdf)))
      return(NULL)
    }
    return(data.frame(
      target_soil_property = target,
      model   = "TabICL",
      dataset = date_tag,
      quantile_coverage_function(test_vec, qdf)
    ))
  }

  if (model_name == "KpR") {
    sub_dir   <- file.path(nndm_dir, "KpR")
    true_path <- file.path(sub_dir, paste0(target, "_test.csv"))
    q_path    <- file.path(sub_dir, paste0(target, "_predictions_quantiles.csv"))
    if (!file.exists(true_path) || !file.exists(q_path)) return(NULL)
    test_vec <- read_first_col(true_path)
    qdf <- read.csv(q_path)
    if (length(test_vec) != nrow(qdf)) {
      warning(paste0("Row mismatch KpR ", date_tag, ": test=", length(test_vec),
                     " q=", nrow(qdf)))
      return(NULL)
    }
    return(data.frame(
      target_soil_property = target,
      model   = "TabICL-KpR",
      dataset = date_tag,
      quantile_coverage_function(test_vec, qdf)
    ))
  }

  if (model_name == "RK") {
    sub_dir   <- file.path(nndm_dir, "RK")
    pred_path <- file.path(sub_dir, paste0(target, "_RK_predictions.csv"))
    q_path    <- file.path(sub_dir, paste0(target, "_RK_quantiles.csv"))
    if (!file.exists(pred_path) || !file.exists(q_path)) return(NULL)
    pred_df <- read.csv(pred_path)
    if (!"true" %in% names(pred_df)) {
      warning(paste0("No 'true' column in RK predictions: ", pred_path))
      return(NULL)
    }
    test_vec <- pred_df[["true"]]
    qdf <- read.csv(q_path)
    if (length(test_vec) != nrow(qdf)) {
      warning(paste0("Row mismatch RK ", date_tag, ": test=", length(test_vec),
                     " q=", nrow(qdf)))
      return(NULL)
    }
    return(data.frame(
      target_soil_property = target,
      model   = "TabICL-RK",
      dataset = date_tag,
      quantile_coverage_function(test_vec, qdf)
    ))
  }

  return(NULL)
}

# ---------------------------
# Main: collect all rows
# ---------------------------
all_rows <- list()

for (date_tag in dates) {
  # Targets discovered from Vanilla test files (shared true-value reference)
  vanilla_dir <- file.path(ROOT, date_tag, "NNDM", "Vanilla")
  if (!dir.exists(vanilla_dir)) next

  test_files <- list.files(vanilla_dir, pattern = "_test\\.csv$", full.names = FALSE)
  targets <- gsub("_test\\.csv$", "", test_files)

  for (target in targets) {
    for (m in models) {
      row <- compute_one_nndm(date_tag, m, target)
      if (!is.null(row)) all_rows[[length(all_rows) + 1]] <- row
    }
  }
}

Results_uncertainty <- bind_rows(all_rows)
Results_uncertainty

# ---------------------------
# MPIW normalization
# ---------------------------
MPIW_normalized <- Results_uncertainty %>%
  group_by(dataset, target_soil_property) %>%
  mutate(
    MPIW_norm = (MPIW - min(MPIW, na.rm = TRUE)) /
      (max(MPIW, na.rm = TRUE) - min(MPIW, na.rm = TRUE))
  ) %>%
  ungroup()

MPIW_avg_model <- MPIW_normalized %>%
  group_by(model) %>%
  summarise(
    MPIW_norm_mean   = mean(MPIW_norm, na.rm = TRUE),
    MPIW_norm_median = median(MPIW_norm, na.rm = TRUE)
  )

Results_uncertainty_long <- Results_uncertainty %>%
  pivot_longer(
    cols = starts_with("QCP"),
    names_to = "quantile",
    values_to = "empirical_coverage"
  ) %>%
  mutate(
    target_quantile = as.numeric(gsub("QCP", "", quantile)) * 100,
    deviation = abs(empirical_coverage - target_quantile)
  )

mean_deviation_per_model <- Results_uncertainty_long %>%
  group_by(model) %>%
  summarise(
    mean_absolute_deviation   = mean(deviation, na.rm = TRUE),
    median_absolute_deviation = median(deviation, na.rm = TRUE)
  ) %>%
  arrange(mean_absolute_deviation)

mean_deviation_MPIW_per_model <- mean_deviation_per_model %>%
  left_join(MPIW_avg_model, by = "model")
mean_deviation_MPIW_per_model

# ---------------------------
# QCP plot (NNDM LOO)
# ---------------------------
QCP_plot <- ggplot(
  Results_uncertainty_long,
  aes(x = target_quantile, y = empirical_coverage,
      color = model,
      group = interaction(dataset, target_soil_property))
) +
  geom_line(size = 0.6, alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
  scale_x_continuous(name = "Target quantile (%)", breaks = seq(0, 100, 25), limits = c(0, 100)) +
  scale_y_continuous(name = "QCP (%)",             breaks = seq(0, 100, 25), limits = c(0, 100)) +
  geom_text(
    data = mean_deviation_MPIW_per_model,
    aes(
      x = 70, y = 10,
      label = paste0(
        "paste(bar(delta)[QCP] == ", round(mean_absolute_deviation, 1),
        ", \", \", bar(PIW)[norm] == ", round(MPIW_norm_mean, 1), ")"
      ),
      color = model
    ),
    parse = TRUE, inherit.aes = FALSE, size = 3.7
  ) +
  geom_text(
    data = mean_deviation_MPIW_per_model,
    aes(
      x = 70, y = 25,
      label = paste0(
        "paste(widetilde(delta)[QCP] == ", round(median_absolute_deviation, 1),
        ", \", \", widetilde(PIW)[norm] == ", round(MPIW_norm_median, 1), ")"
      ),
      color = model
    ),
    parse = TRUE, inherit.aes = FALSE, size = 3.7
  ) +
  theme_bw() +
  theme(
    panel.grid.minor = element_blank(),
    axis.title  = element_text(size = 18, color = "black"),
    axis.text   = element_text(size = 16, color = "black"),
    strip.text  = element_text(size = 14, face = "bold"),
    legend.position = "none"
  ) +
  scale_color_manual(
    values = c(
      "OK"          = "#4477AA",
      "TabICL-KpR"  = "#cc6699",
      "TabICL"      = "#009999",
      "TabICL-RK"   = "#e6b800"
    )
  ) +
  facet_wrap(~ model)

QCP_plot

# ---------------------------
# Save
# ---------------------------
dir.create(file.path(ROOT, "figures"), showWarnings = FALSE, recursive = TRUE)
ggsave(
  file.path(ROOT, "figures", "NNDM_QCP_plot.tiff"),
  plot = QCP_plot, dpi = 300, width = 10, height = 7, units = "in", device = "tiff"
)
write.csv(Results_uncertainty,
          file.path(ROOT, "figures", "NNDM_QCP_Results_uncertainty.csv"),
          row.names = FALSE)
