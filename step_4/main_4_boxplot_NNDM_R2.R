library(ggplot2)
library(dplyr)

ROOT    <- "/Users/takashi/LocalAnalysis/WeedMap"
out_dir <- file.path(ROOT, "Results_TabICL", "figures")
dir.create(out_dir, showWarnings = FALSE)

# ---------------------------
# Load R² data
# ---------------------------
d <- read.csv(
  file.path(ROOT, "Results_TabICL", "figures",
            "NNDM_best_models_by_date_20250414_20250602_R2_log.csv"),
  stringsAsFactors = FALSE
)

# Rename model groups to manuscript labels
d$model_group[d$model_group == "Vanilla TabPFN"] <- "TabICL"
d$model_group[d$model_group == "KpR TabPFN"]     <- "TabICL-KpR"
d$model_group[d$model_group == "RK"]             <- "TabICL-RK"

model_order  <- c("OK", "TabICL", "TabICL-KpR", "TabICL-RK")
model_colors <- c(
  "OK"         = "#4477AA",
  "TabICL"     = "#009999",
  "TabICL-KpR" = "#cc6699",
  "TabICL-RK"  = "#e6b800"
)

targets <- sort(unique(d$target))

plots <- list()

for (tgt in targets) {

  dt <- d %>%
    filter(target == tgt) %>%
    mutate(
      model_group = factor(model_group, levels = model_order),
      R2_log      = as.numeric(R2_log)
    )

  plots[[tgt]] <- ggplot(dt, aes(x = model_group, y = R2_log, color = model_group)) +
    geom_boxplot(
      aes(fill = model_group),
      alpha = 0.15, outlier.shape = NA,
      width = 0.5, linewidth = 0.7
    ) +
    geom_jitter(
      width = 0.08, size = 2.5, alpha = 0.8
    ) +
    geom_hline(yintercept = 0, linetype = "dashed",
               colour = "grey50", linewidth = 0.7) +
    scale_color_manual(values = model_colors) +
    scale_fill_manual(values  = model_colors) +
    labs(
      x = NULL,
      y = expression(R^2 ~ "(log(1 + y) scale)")
    ) +
    theme_bw(base_size = 13) +
    theme(
      legend.position  = "none",
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      axis.text.x  = element_text(size = 12, color = "black"),
      axis.text.y  = element_text(size = 12, color = "black"),
      axis.title.y = element_text(size = 14, color = "black")
    )
}

# ---------------------------
# Save
# ---------------------------
for (tgt in names(plots)) {
  png_path <- file.path(out_dir, paste0("NNDM_R2_boxplot_", tgt, ".png"))
  tif_path <- file.path(out_dir, paste0("NNDM_R2_boxplot_", tgt, ".tiff"))
  ggsave(png_path, plot = plots[[tgt]], width = 5.5, height = 4.5, dpi = 150)
  ggsave(tif_path, plot = plots[[tgt]], width = 5.5, height = 4.5, dpi = 300,
         device = "tiff")
  cat("Saved:", png_path, "\n")
}

plots[[1]]
