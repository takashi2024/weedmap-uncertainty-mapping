# ============================================================
# LMM for NNDM LOO CV R² — adapted from main_4_LMM_for_R2.R
# Model: R2_log ~ model_group + (1 | date)
# No fold random effect (NNDM LOO gives one R² per date, not per fold)
# ============================================================

library(lme4)
library(lmerTest)
library(emmeans)
library(ggplot2)
library(dplyr)

ROOT    <- "/Users/takashi/LocalAnalysis/WeedMap"
out_dir <- file.path(ROOT, "Results_TabICL", "figures")
dir.create(out_dir, showWarnings = FALSE)

# ------------------------------------------------------------
# 1. Load NNDM R² summary
# ------------------------------------------------------------
d <- read.csv(
  file.path(ROOT, "Results_TabICL", "figures",
            "NNDM_best_models_by_date_20250414_20250602_R2_log.csv"),
  stringsAsFactors = FALSE
)

# Rename Vanilla_TabICL -> TabICL if present
d$model_group[d$model_group == "Vanilla_TabICL"] <- "TabICL"

# Keep the 4 NNDM models only
d <- d %>%
  filter(model_group %in% c("OK", "TabICL", "TabICL-KpR", "TabICL-RK")) %>%
  mutate(
    model_group = factor(model_group, levels = c("OK", "TabICL", "TabICL-RK", "TabICL-KpR")),
    date        = as.factor(date),
    R2_log      = as.numeric(R2_log)
  )

cat("Rows:", nrow(d), " | Dates:", nlevels(d$date),
    " | Models:", nlevels(d$model_group), "\n")
print(table(d$model_group, d$date))

targets <- sort(unique(d$target))

# ------------------------------------------------------------
# 2. Fit LMM per target, extract EMM
# ------------------------------------------------------------
res_emm <- list()
plots   <- list()

for (tgt in targets) {

  dt <- subset(d, target == tgt)

  # LMM: date as random effect only (no fold in NNDM LOO)
  m_t <- lmer(R2_log ~ model_group + (1 | date), data = dt, REML = TRUE)

  cat("\n====", tgt, "====\n")
  print(summary(m_t))
  cat("\n--- ANOVA ---\n")
  print(anova(m_t))

  emm_t  <- emmeans(m_t, ~ model_group)
  emm_df <- as.data.frame(emm_t)
  emm_df$target      <- tgt
  emm_df$model_group <- reorder(emm_df$model_group, emm_df$emmean)

  res_emm[[tgt]] <- list(model = m_t, emm = emm_df)

  # EMM dot-and-whisker plot
  plots[[tgt]] <- ggplot(emm_df, aes(x = model_group, y = emmean)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), width = 0.15) +
    geom_hline(yintercept = 0, linetype = "dashed",
               colour = "grey50", linewidth = 0.7) +
    coord_flip() +
    labs(
      x = NULL,
      y = expression("Estimated marginal mean " * R^2 * " (95% CI)")
    ) +
    theme_bw(base_size = 12) +
    theme(panel.grid.major.y = element_blank())
}

# ------------------------------------------------------------
# 3. Save plots
# ------------------------------------------------------------
for (tgt in names(plots)) {
  png_path <- file.path(out_dir, paste0("NNDM_LMM_R2_EMM_", tgt, ".png"))
  tif_path <- file.path(out_dir, paste0("NNDM_LMM_R2_EMM_", tgt, ".tiff"))
  ggsave(png_path, plot = plots[[tgt]], width = 6, height = 3.5, dpi = 300)
  ggsave(tif_path, plot = plots[[tgt]], width = 6, height = 3.5, dpi = 300,
         compression = "lzw")
  cat("Saved:", png_path, "\n")
}

# ------------------------------------------------------------
# 4. Save combined EMM table
# ------------------------------------------------------------
emm_all <- bind_rows(lapply(res_emm, `[[`, "emm"))

write.csv(emm_all, file.path(out_dir, "NNDM_LMM_R2_emmeans.csv"), row.names = FALSE)
cat("Saved EMM table:", file.path(out_dir, "NNDM_LMM_R2_emmeans.csv"), "\n")

# Show first plot
plots[[1]]
