# ============================================================
# Mixed-effect analysis for recall / precision — NNDM LOO CV
# Model (per budget): response ~ rule * algorithm + (1 | date)
# Connected EMM line plot across all budget levels
# ============================================================

library(tidyverse)
library(lme4)
library(lmerTest)
library(emmeans)

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------
ROOT <- "/Users/takashi/LocalAnalysis/WeedMap"

file_prevGT0 <- file.path(ROOT, "Results_TabICL/NNDM_prob",
  "NNDM_Chenopodium_T10_budget_metrics_for_R_LMM_tauGE1_prevGT0.csv")

out_dir <- file.path(ROOT, "Results_TabICL/NNDM_LMM_outputs")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------
# 2. Settings
# ------------------------------------------------------------
budgets <- c(0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

alg_levels  <- c("TabICL-KpR_NNDM", "TabICL-RK_NNDM", "TabICL_NNDM", "OK_NNDM")
alg_labels  <- c("TabICL-KpR",     "TabICL-RK",     "TabICL",     "OK")
rule_levels <- c("DET", "PROB")

ALG_COLORS <- c(
  "TabICL-KpR" = "#cc6699",
  "TabICL-RK"  = "#e6b800",
  "TabICL"     = "#009999",
  "OK"         = "#4477AA"
)

# Logit transform (TRUE recommended for boundary-safe residuals at extreme budgets)
USE_LOGIT <- FALSE

# ------------------------------------------------------------
# 3. Helpers
# ------------------------------------------------------------
safe_logit <- function(x, eps = 1e-4) {
  x <- pmin(pmax(x, eps), 1 - eps)
  qlogis(x)
}

read_and_clean <- function(filepath) {
  dat <- read.csv(filepath, stringsAsFactors = FALSE) %>%
    mutate(
      date      = as.factor(date),
      algorithm = factor(algorithm, levels = alg_levels, labels = alg_labels),
      rule      = factor(rule, levels = rule_levels),
      budget    = as.numeric(budget),
      recall    = as.numeric(recall),
      precision = as.numeric(precision),
      prevalence = if ("prevalence" %in% names(.)) as.numeric(prevalence) else NA_real_
    ) %>%
    filter(!is.na(algorithm), !is.na(rule), !is.na(budget))
  dat
}

fit_one_budget <- function(data, response_var, budget_value, use_logit = FALSE) {
  dsub <- data %>%
    filter(abs(budget - budget_value) < 1e-8) %>%
    filter(!is.na(.data[[response_var]]))

  if (nrow(dsub) == 0) return(NULL)

  if (use_logit) {
    dsub <- dsub %>% mutate(y = safe_logit(.data[[response_var]]))
  } else {
    dsub <- dsub %>% mutate(y = .data[[response_var]])
  }

  mod <- tryCatch(
    lmer(y ~ rule * algorithm + (1 | date), data = dsub, REML = TRUE),
    error = function(e) { message("Model failed at b=", budget_value, ": ", e$message); NULL }
  )
  if (is.null(mod)) return(NULL)

  list(model = mod, data = dsub, response = response_var,
       budget = budget_value, use_logit = use_logit)
}

extract_emm <- function(fit_obj) {
  if (is.null(fit_obj)) return(NULL)
  emm <- emmeans(fit_obj$model, ~ algorithm | rule,
                 type = if (fit_obj$use_logit) "response" else "lm")
  df  <- as.data.frame(emm)
  df$budget   <- fit_obj$budget
  df$response <- fit_obj$response
  df
}

# rename columns depending on whether emmeans back-transforms
# (type="response" gives 'prob'; type="lm" keeps 'emmean')
standardise_emm_cols <- function(df) {
  if ("prob" %in% names(df)) {
    df <- df %>% rename(emmean = prob, lower.CL = asymp.LCL, upper.CL = asymp.UCL)
  }
  df
}

save_model_report <- function(fit_obj, dataset_tag) {
  budget_tag <- sprintf("%02d", as.integer(round(fit_obj$budget * 100)))
  scale_tag  <- if (fit_obj$use_logit) "logit" else "raw"
  txt_file   <- file.path(out_dir,
    sprintf("%s_%s_budget%s_%s_summary.txt",
            dataset_tag, fit_obj$response, budget_tag, scale_tag))

  sink(txt_file)
  cat("====================================================\n")
  cat("Dataset:", dataset_tag, " | Response:", fit_obj$response,
      " | Budget:", fit_obj$budget, " | Scale:", scale_tag, "\n")
  cat("====================================================\n\n")
  cat("--- Model: y ~ rule * algorithm + (1 | date) ---\n")
  print(summary(fit_obj$model))
  cat("\n--- ANOVA ---\n")
  print(anova(fit_obj$model))
  sink()
}

# ------------------------------------------------------------
# 4. Load data
# ------------------------------------------------------------
dat <- read_and_clean(file_prevGT0)
cat("Rows:", nrow(dat), " | Dates:", nlevels(dat$date),
    " | Algorithms:", nlevels(dat$algorithm), "\n")
print(table(dat$budget, dat$rule))

# ------------------------------------------------------------
# 5. Fit LMM at every budget level for recall and precision
# ------------------------------------------------------------
fits_recall    <- lapply(budgets, fit_one_budget, data = dat,
                         response_var = "recall",    use_logit = USE_LOGIT)
fits_precision <- lapply(budgets, fit_one_budget, data = dat,
                         response_var = "precision", use_logit = USE_LOGIT)

# Save text summaries + per-budget CSV
dataset_tag <- "NNDM_prevGT0"
for (fit_obj in c(fits_recall, fits_precision)) {
  if (!is.null(fit_obj)) save_model_report(fit_obj, dataset_tag)
}

# ------------------------------------------------------------
# 6. Collect EMMs across all budgets
# ------------------------------------------------------------
emm_recall    <- lapply(fits_recall,    extract_emm) %>% bind_rows() %>%
  standardise_emm_cols() %>% mutate(budget_pct = budget * 100)
emm_precision <- lapply(fits_precision, extract_emm) %>% bind_rows() %>%
  standardise_emm_cols() %>% mutate(budget_pct = budget * 100)

emm_all <- bind_rows(emm_recall, emm_precision)

write.csv(emm_all,
  file.path(out_dir, "NNDM_prevGT0_emm_all_budgets.csv"),
  row.names = FALSE)
cat("Saved EMM table:", file.path(out_dir, "NNDM_prevGT0_emm_all_budgets.csv"), "\n")

# ------------------------------------------------------------
# 7. Connected EMM line plot across all budgets
# ------------------------------------------------------------
# Helper: enforce algorithm colour order
scale_alg_color <- scale_color_manual(
  name   = "Algorithm",
  values = ALG_COLORS
)
scale_alg_fill <- scale_fill_manual(
  name   = "Algorithm",
  values = ALG_COLORS
)

plot_emm_curve <- function(emm_df, response_name, use_logit) {
  y_label <- if (response_name == "recall") "Recall" else "Precision"

  ggplot(emm_df, aes(x = budget_pct, y = emmean,
                      color = algorithm, fill = algorithm)) +
    geom_ribbon(aes(ymin = lower.CL, ymax = upper.CL), alpha = 0.15, colour = NA) +
    geom_line(linewidth = 1.0) +
    geom_point(size = 2.2) +
    facet_wrap(~ rule, labeller = labeller(rule = c(DET = "Deterministic",
                                                     PROB = "Probabilistic"))) +
    scale_x_continuous(name = expression("Sprayed fraction \u2013 " * italic(b) * " (%)"),
                       breaks = budgets * 100) +
    scale_y_continuous(name = paste("Marginal mean", y_label)) +
    coord_cartesian(ylim = c(0, 1)) +
    scale_alg_color + scale_alg_fill +
    theme_bw(base_size = 12) +
    theme(
      panel.grid.minor  = element_blank(),
      strip.text        = element_text(size = 11, face = "bold"),
      axis.text.x       = element_text(angle = 35, hjust = 1),
      legend.position   = "bottom"
    ) +
    labs(
      title = paste0("NNDM LOO — Marginal mean ", y_label,
                     " vs spray budget (LMM, date random effect",
                     if (use_logit) ", logit scale back-transformed" else "", ")")
    )
}

p_recall    <- plot_emm_curve(emm_recall,    "recall",    USE_LOGIT)
p_precision <- plot_emm_curve(emm_precision, "precision", USE_LOGIT)

scale_tag <- if (USE_LOGIT) "logit" else "raw"

ggsave(file.path(out_dir, paste0("NNDM_emm_recall_curve_",    scale_tag, ".png")),
       plot = p_recall,    width = 9, height = 5, dpi = 300)
ggsave(file.path(out_dir, paste0("NNDM_emm_precision_curve_", scale_tag, ".png")),
       plot = p_precision, width = 9, height = 5, dpi = 300)
cat("Saved EMM curve plots.\n")

# ------------------------------------------------------------
# 8. Combined 2x2 panel (recall/precision x PROB/DET)
# ------------------------------------------------------------
emm_all_plot <- emm_all %>%
  mutate(response = factor(response,
    levels = c("recall", "precision"),
    labels = c("Recall", "Precision")))

p_combined <- ggplot(emm_all_plot,
  aes(x = budget_pct, y = emmean, color = algorithm, fill = algorithm)) +
  geom_ribbon(aes(ymin = lower.CL, ymax = upper.CL), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 2.0) +
  facet_grid(response ~ rule,
    labeller = labeller(rule = c(DET = "Deterministic", PROB = "Probabilistic"),
                        response = label_value)) +
  scale_x_continuous(name = expression("Sprayed fraction \u2013 " * italic(b) * " (%)"),
                     breaks = budgets * 100) +
  scale_y_continuous(name = "Marginal mean") +
  coord_cartesian(ylim = c(0, 1)) +
  scale_alg_color + scale_alg_fill +
  theme_bw(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text       = element_text(size = 11, face = "bold"),
    axis.text.x      = element_text(angle = 35, hjust = 1),
    legend.position  = "bottom"
  ) +
  labs(title = paste0(
    "NNDM LOO — LMM marginal means (95% CI) | date as random effect",
    if (USE_LOGIT) " | logit back-transformed" else ""
  ))

ggsave(file.path(out_dir, paste0("NNDM_emm_combined_2x2_", scale_tag, ".png")),
       plot = p_combined, width = 10, height = 7, dpi = 300)
cat("Saved combined 2x2 EMM plot.\n")

# ------------------------------------------------------------
# 9. Quick ANOVA summary across all budgets
# ------------------------------------------------------------
anova_rows <- lapply(c(fits_recall, fits_precision), function(fit_obj) {
  if (is.null(fit_obj)) return(NULL)
  av <- tryCatch(as.data.frame(anova(fit_obj$model)), error = function(e) NULL)
  if (is.null(av)) return(NULL)
  av$term     <- rownames(av)
  av$budget   <- fit_obj$budget
  av$response <- fit_obj$response
  av
}) %>% bind_rows()

write.csv(anova_rows,
  file.path(out_dir, "NNDM_prevGT0_anova_all_budgets.csv"),
  row.names = FALSE)
cat("Saved ANOVA table:", file.path(out_dir, "NNDM_prevGT0_anova_all_budgets.csv"), "\n")

# Quick view of F-values and p-values for 'algorithm' term
alg_anova <- anova_rows %>%
  filter(term == "algorithm") %>%
  rename(F_value = `F value`, p_value = `Pr(>F)`) %>%
  select(response, budget, F_value, p_value) %>%
  mutate(sig = case_when(p_value < 0.001 ~ "***",
                         p_value < 0.01  ~ "**",
                         p_value < 0.05  ~ "*",
                         TRUE            ~ "ns"))
cat("\n=== ANOVA 'algorithm' term across budgets ===\n")
print(as.data.frame(alg_anova))
