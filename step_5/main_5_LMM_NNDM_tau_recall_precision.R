# ============================================================
# LMM analysis for recall / precision vs. probability threshold τ
# Model (per τ level): response ~ algorithm + (1 | date)
# Connected EMM line plot across τ levels
# Mirrors: main_4_LMM_NNDM_recall_precision.R (budget/sprayed-fraction version)
# ============================================================

library(tidyverse)
library(lme4)
library(lmerTest)
library(emmeans)

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------
ROOT    <- "/Users/takashi/LocalAnalysis/WeedMap/ForGithub"
PROB_DIR <- file.path(ROOT, "Results_TabICL", "NNDM_prob")
out_dir  <- file.path(ROOT, "Results_TabICL", "NNDM_LMM_outputs")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

TARGET <- "log1p_Chenopodium_Count"
T_VAL  <- 10

# Taucurve files (one per algorithm)
alg_files <- c(
  "Vanilla_TabPFN_NNDM" = file.path(PROB_DIR, sprintf("NNDM_Vanilla_TabPFN_%s_taucurve_T%d.csv", TARGET, T_VAL)),
  "KpR_NNDM"            = file.path(PROB_DIR, sprintf("NNDM_KpR_%s_taucurve_T%d.csv",            TARGET, T_VAL)),
  "RK_NNDM"             = file.path(PROB_DIR, sprintf("NNDM_RK_%s_taucurve_T%d.csv",             TARGET, T_VAL)),
  "OK_NNDM"             = file.path(PROB_DIR, sprintf("NNDM_OK_%s_taucurve_T%d.csv",             TARGET, T_VAL))
)

# ------------------------------------------------------------
# 2. Settings
# ------------------------------------------------------------
# τ grid for LMM (one model fit per level)
tau_grid <- c(0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

alg_levels <- c("KpR_NNDM", "RK_NNDM", "Vanilla_TabPFN_NNDM", "OK_NNDM")
alg_labels <- c("TabICL-KpR", "TabICL-RK", "TabICL", "OK")

ALG_COLORS <- c(
  "TabICL-KpR" = "#cc6699",
  "TabICL-RK"  = "#e6b800",
  "TabICL"     = "#009999",
  "OK"         = "#4477AA"
)

USE_LOGIT <- FALSE

# ------------------------------------------------------------
# 3. Load and combine taucurve data
# ------------------------------------------------------------
dfs <- lapply(names(alg_files), function(alg) {
  fp <- alg_files[[alg]]
  if (!file.exists(fp)) {
    warning("Missing file: ", fp); return(NULL)
  }
  df <- read.csv(fp, stringsAsFactors = FALSE)
  df$algorithm_raw <- alg
  df
})
dat_raw <- bind_rows(dfs)

dat <- dat_raw %>%
  filter(prevalence > 0) %>%
  mutate(
    date      = as.factor(date),
    algorithm = factor(algorithm_raw, levels = alg_levels, labels = alg_labels),
    tau       = as.numeric(tau),
    recall    = as.numeric(recall),
    precision = as.numeric(precision)
  ) %>%
  filter(!is.na(algorithm))

cat("Rows:", nrow(dat),
    "| Dates:", nlevels(dat$date),
    "| Algorithms:", nlevels(dat$algorithm), "\n")

# Snap τ: for each target τ level, find the closest observed τ per (date, algorithm)
snap_tau <- function(data, tau_target, tol = 0.005) {
  data %>% filter(abs(tau - tau_target) <= tol)
}

# ------------------------------------------------------------
# 4. Helpers: fit LMM and extract EMM
# ------------------------------------------------------------
safe_logit <- function(x, eps = 1e-4) qlogis(pmin(pmax(x, eps), 1 - eps))

fit_one_tau <- function(data, response_var, tau_value, use_logit = FALSE) {
  dsub <- snap_tau(data, tau_value) %>%
    filter(!is.na(.data[[response_var]]))

  if (nrow(dsub) < 4) return(NULL)   # need at least 4 rows (dates × algs)

  if (use_logit) {
    dsub <- dsub %>% mutate(y = safe_logit(.data[[response_var]]))
  } else {
    dsub <- dsub %>% mutate(y = .data[[response_var]])
  }

  mod <- tryCatch(
    lmer(y ~ algorithm + (1 | date), data = dsub, REML = TRUE),
    error = function(e) {
      message("LMM failed at τ=", tau_value, ": ", e$message); NULL
    }
  )
  if (is.null(mod)) return(NULL)

  list(model = mod, data = dsub, response = response_var,
       tau = tau_value, use_logit = use_logit)
}

extract_emm_tau <- function(fit_obj) {
  if (is.null(fit_obj)) return(NULL)
  emm <- emmeans(fit_obj$model, ~ algorithm,
                 type = if (fit_obj$use_logit) "response" else "lm")
  df  <- as.data.frame(emm)
  # Standardise column names (logit back-transform uses 'prob', linear keeps 'emmean')
  if ("prob" %in% names(df))
    df <- df %>% rename(emmean = prob, lower.CL = asymp.LCL, upper.CL = asymp.UCL)
  df$tau      <- fit_obj$tau
  df$response <- fit_obj$response
  df
}

# ------------------------------------------------------------
# 5. Fit LMM at every τ level for recall and precision
# ------------------------------------------------------------
fits_recall    <- lapply(tau_grid, fit_one_tau, data = dat,
                         response_var = "recall",    use_logit = USE_LOGIT)
fits_precision <- lapply(tau_grid, fit_one_tau, data = dat,
                         response_var = "precision", use_logit = USE_LOGIT)

# ------------------------------------------------------------
# 6. Collect EMMs and ANOVA
# ------------------------------------------------------------
emm_recall    <- lapply(fits_recall,    extract_emm_tau) %>% bind_rows()
emm_precision <- lapply(fits_precision, extract_emm_tau) %>% bind_rows()

emm_all <- bind_rows(emm_recall, emm_precision)

write.csv(emm_all,
  file.path(out_dir, "NNDM_prevGT0_emm_tau_all.csv"),
  row.names = FALSE)
cat("Saved EMM table.\n")

# ANOVA table across all τ levels
anova_rows <- lapply(c(fits_recall, fits_precision), function(fit_obj) {
  if (is.null(fit_obj)) return(NULL)
  av <- tryCatch(as.data.frame(anova(fit_obj$model)), error = function(e) NULL)
  if (is.null(av)) return(NULL)
  av$term     <- rownames(av)
  av$tau      <- fit_obj$tau
  av$response <- fit_obj$response
  av
}) %>% bind_rows()

write.csv(anova_rows,
  file.path(out_dir, "NNDM_prevGT0_anova_tau_all.csv"),
  row.names = FALSE)
cat("Saved ANOVA table.\n")

# ------------------------------------------------------------
# 7. Combined 2-row plot (Recall / Precision) vs τ
# Mirrors the 2×2 reference figure but with a single "Probabilistic (τ)" column
# ------------------------------------------------------------
scale_alg_color <- scale_color_manual(name = "Algorithm", values = ALG_COLORS)
scale_alg_fill  <- scale_fill_manual( name = "Algorithm", values = ALG_COLORS)

emm_all_plot <- emm_all %>%
  mutate(
    response = factor(response,
                      levels = c("recall", "precision"),
                      labels = c("Recall", "Precision")),
    algorithm = factor(algorithm, levels = alg_labels)
  )

p_combined <- ggplot(emm_all_plot,
    aes(x = tau, y = emmean, color = algorithm, fill = algorithm)) +
  geom_ribbon(aes(ymin = lower.CL, ymax = upper.CL),
              alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 2.0) +
  facet_grid(response ~ .,
             labeller = labeller(response = label_value)) +
  scale_x_continuous(
    name   = "Probability threshold τ",
    breaks = tau_grid,
    labels = tau_grid
  ) +
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
  labs(title = "NNDM LOO — LMM marginal means (95% CI) | date as random effect | Probabilistic (τ)")

scale_tag <- if (USE_LOGIT) "logit" else "raw"

ggsave(file.path(out_dir, paste0("NNDM_emm_tau_combined_", scale_tag, ".png")),
       plot = p_combined, width = 7, height = 7, dpi = 300)
ggsave(file.path(out_dir, paste0("NNDM_emm_tau_combined_", scale_tag, ".tiff")),
       plot = p_combined, width = 7, height = 7, dpi = 300,
       compression = "lzw")

cat("Saved combined τ EMM plot.\n")
