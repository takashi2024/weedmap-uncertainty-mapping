library(dplyr)
library(tidyr)
library(ggplot2)

# ---------------------------
# Settings
# ---------------------------
ROOT <- "Results_TabICL"

dates <- c(
  "data_20250414","data_20250424","data_20250430","data_20250506",
  "data_20250513","data_20250520","data_20250526","data_20250602"
)

models <- c("OK", "RK", "Vanilla", "KpR")

# QCP evaluated at 0.1 .. 0.9 only (0.001 and 0.999 require n > 5000)
QCP_COLS <- c("q0.1","q0.2","q0.3","q0.4","q0.5","q0.6","q0.7","q0.8","q0.9")
QCP_LEVELS <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# ---------------------------
# QCP + MPIW function
# ---------------------------
quantile_coverage_function <- function(test, qdf) {
  qdf <- qdf[, QCP_COLS, drop = FALSE]
  coverage <- sapply(seq_along(QCP_COLS), function(j) {
    mean(test <= qdf[, j]) * 100
  })
  MPIW <- mean(qdf$q0.9 - qdf$q0.1) +
          mean(qdf$q0.8 - qdf$q0.2) +
          mean(qdf$q0.7 - qdf$q0.3) +
          mean(qdf$q0.6 - qdf$q0.4)
  out <- as.data.frame(t(c(coverage, MPIW)))
  colnames(out) <- c(paste0("QCP", QCP_LEVELS), "MPIW")
  out
}

read_first_col <- function(path) read.csv(path)[[1]]

# ---------------------------
# compute_one_nndm
# ---------------------------
compute_one_nndm <- function(date_tag, model_name, target) {
  nndm_dir <- file.path(ROOT, date_tag, "NNDM")

  if (model_name == "OK") {
    true_path <- file.path(nndm_dir, "Vanilla", paste0(target, "_test.csv"))
    q_path    <- file.path(nndm_dir, paste0(target, "_OK_quantiles.csv"))
    if (!file.exists(true_path) || !file.exists(q_path)) return(NULL)
    test_vec <- read_first_col(true_path)
    qdf <- read.csv(q_path)
    if (length(test_vec) != nrow(qdf)) return(NULL)
    return(data.frame(
      target_var = target, model = "OK", dataset = date_tag,
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
    if (length(test_vec) != nrow(qdf)) return(NULL)
    return(data.frame(
      target_var = target, model = "TabICL", dataset = date_tag,
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
    if (length(test_vec) != nrow(qdf)) return(NULL)
    return(data.frame(
      target_var = target, model = "TabICL-KpR", dataset = date_tag,
      quantile_coverage_function(test_vec, qdf)
    ))
  }

  if (model_name == "RK") {
    sub_dir   <- file.path(nndm_dir, "RK")
    pred_path <- file.path(sub_dir, paste0(target, "_RK_predictions.csv"))
    q_path    <- file.path(sub_dir, paste0(target, "_RK_quantiles.csv"))
    if (!file.exists(pred_path) || !file.exists(q_path)) return(NULL)
    pred_df <- read.csv(pred_path)
    if (!"true" %in% names(pred_df)) return(NULL)
    test_vec <- pred_df[["true"]]
    qdf <- read.csv(q_path)
    if (length(test_vec) != nrow(qdf)) return(NULL)
    return(data.frame(
      target_var = target, model = "TabICL-RK", dataset = date_tag,
      quantile_coverage_function(test_vec, qdf)
    ))
  }

  return(NULL)
}

# ---------------------------
# Collect all rows
# ---------------------------
all_rows <- list()

for (date_tag in dates) {
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
print(Results_uncertainty)

# ---------------------------
# MPIW normalization
# ---------------------------
MPIW_normalized <- Results_uncertainty %>%
  group_by(dataset, target_var) %>%
  mutate(
    MPIW_norm = (MPIW - min(MPIW, na.rm = TRUE)) /
      (max(MPIW, na.rm = TRUE) - min(MPIW, na.rm = TRUE))
  ) %>%
  ungroup()

MPIW_avg_model <- MPIW_normalized %>%
  group_by(model) %>%
  summarise(
    MPIW_norm_mean   = mean(MPIW_norm, na.rm = TRUE),
    MPIW_norm_median = median(MPIW_norm, na.rm = TRUE),
    .groups = "drop"
  )

# ---------------------------
# Long format for QCP plot
# ---------------------------
qcp_col_names <- paste0("QCP", QCP_LEVELS)

Results_uncertainty_long <- Results_uncertainty %>%
  pivot_longer(
    cols = all_of(qcp_col_names),
    names_to  = "quantile",
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
    median_absolute_deviation = median(deviation, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(mean_absolute_deviation)

mean_deviation_MPIW_per_model <- mean_deviation_per_model %>%
  left_join(MPIW_avg_model, by = "model")

print(mean_deviation_MPIW_per_model)

# ---------------------------
# QCP plot
# ---------------------------
model_colors <- c(
  "OK"         = "#4477AA",
  "TabICL-KpR" = "#cc6699",
  "TabICL"     = "#009999",
  "TabICL-RK"  = "#e6b800"
)

model_order <- c("OK", "TabICL", "TabICL-KpR", "TabICL-RK")
Results_uncertainty_long$model <- factor(Results_uncertainty_long$model, levels = model_order)
mean_deviation_MPIW_per_model$model <- factor(mean_deviation_MPIW_per_model$model, levels = model_order)

QCP_plot <- ggplot(
  Results_uncertainty_long,
  aes(x = target_quantile, y = empirical_coverage,
      color = model,
      group = interaction(dataset, target_var))
) +
  geom_line(linewidth = 0.6, alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
  geom_text(
    data = mean_deviation_MPIW_per_model,
    aes(
      x = 65, y = 8,
      label = paste0(
        "paste(bar(delta)[QCP] == ", round(mean_absolute_deviation, 1),
        ", \", \", bar(PIW)[norm] == ", round(MPIW_norm_mean, 2), ")"
      ),
      color = model
    ),
    parse = TRUE, inherit.aes = FALSE, size = 3.7
  ) +
  geom_text(
    data = mean_deviation_MPIW_per_model,
    aes(
      x = 65, y = 20,
      label = paste0(
        "paste(widetilde(delta)[QCP] == ", round(median_absolute_deviation, 1),
        ", \", \", widetilde(PIW)[norm] == ", round(MPIW_norm_median, 2), ")"
      ),
      color = model
    ),
    parse = TRUE, inherit.aes = FALSE, size = 3.7
  ) +
  scale_x_continuous(name = "Target quantile (%)", breaks = seq(0, 100, 25), limits = c(0, 100)) +
  scale_y_continuous(name = "QCP (%)",             breaks = seq(0, 100, 25), limits = c(0, 100)) +
  scale_color_manual(values = model_colors) +
  theme_bw() +
  theme(
    panel.grid.minor = element_blank(),
    axis.title  = element_text(size = 18, color = "black"),
    axis.text   = element_text(size = 16, color = "black"),
    strip.text  = element_text(size = 14, face = "bold"),
    legend.position = "none"
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
ggsave(
  file.path(ROOT, "figures", "NNDM_QCP_plot.png"),
  plot = QCP_plot, dpi = 150, width = 10, height = 7, units = "in"
)

write.csv(Results_uncertainty,
          file.path(ROOT, "figures", "NNDM_QCP_Results_uncertainty.csv"),
          row.names = FALSE)

cat("\nDone. Output in", file.path(ROOT, "figures"), "\n")
