library(dplyr)
library(tidyr)
library(ggplot2)

ROOT    <- "/Users/takashi/LocalAnalysis/WeedMap"
RESDIR  <- file.path(ROOT, "Results_TabICL")
TARGET  <- "log1p_Chenopodium_Count"
OUTDIR  <- file.path(RESDIR, "figures", "supp")
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

DATES <- c("20250414","20250424","20250430","20250506",
           "20250513","20250520","20250526","20250602")

DATE_LABELS <- c(
  "20250414" = "14 Apr", "20250424" = "24 Apr", "20250430" = "30 Apr",
  "20250506" = "06 May", "20250513" = "13 May", "20250520" = "20 May",
  "20250526" = "26 May", "20250602" = "02 Jun"
)

METHOD_ORDER <- c("OK", "TabICL", "TabICL-KpR", "TabICL-RK")

# Global coordinate offset
all_coords <- lapply(DATES, function(d) {
  fp <- file.path(RESDIR, paste0("data_", d), "NNDM", "RK",
                  paste0(TARGET, "_RK_predictions.csv"))
  if (!file.exists(fp)) return(NULL)
  read.csv(fp)[, c("x_25832", "y_25832")]
})
all_coords <- bind_rows(Filter(Negate(is.null), all_coords))
X_MIN <- min(all_coords$x_25832)
Y_MIN <- min(all_coords$y_25832)

for (date in DATES) {
  nndm_dir <- file.path(RESDIR, paste0("data_", date), "NNDM")
  rk_path  <- file.path(nndm_dir, "RK", paste0(TARGET, "_RK_predictions.csv"))
  if (!file.exists(rk_path)) { warning("Missing: ", date); next }

  rk     <- read.csv(rk_path)
  coords <- rk %>%
    select(obs_idx) %>%
    mutate(
      x = rk$x_25832 - X_MIN,
      y = rk$y_25832 - Y_MIN
    )

  # --- OK ---
  ok_pred <- read.csv(file.path(nndm_dir, "OK_predictions.csv")) %>% arrange(obs_idx)
  ok_q    <- read.csv(file.path(nndm_dir, paste0(TARGET, "_OK_quantiles.csv")))
  ok_iqr  <- coords %>%
    mutate(iqr = ok_q$q0.9 - ok_q$q0.1, method = "OK")

  # --- Vanilla ---
  van_q  <- read.csv(file.path(nndm_dir, "Vanilla", paste0(TARGET, "_predictions_quantiles.csv")))
  van_iqr <- coords %>%
    mutate(iqr = van_q$q0.9 - van_q$q0.1, method = "TabICL")

  # --- KpR ---
  kpr_q  <- read.csv(file.path(nndm_dir, "KpR", paste0(TARGET, "_predictions_quantiles.csv")))
  kpr_iqr <- coords %>%
    mutate(iqr = kpr_q$q0.9 - kpr_q$q0.1, method = "TabICL-KpR")

  # --- RK ---
  rk_q   <- read.csv(file.path(nndm_dir, "RK", paste0(TARGET, "_RK_quantiles.csv")))
  rk_iqr <- coords %>%
    mutate(iqr = rk_q$q0.9 - rk_q$q0.1, method = "TabICL-RK")

  all_pts <- bind_rows(ok_iqr, van_iqr, kpr_iqr, rk_iqr) %>%
    mutate(method = factor(method, levels = METHOD_ORDER))

  max_iqr <- max(all_pts$iqr, na.rm = TRUE)

  p <- ggplot(all_pts, aes(x = x, y = y, color = iqr)) +
    geom_point(size = 1.2, shape = 16) +
    facet_wrap(~ method, nrow = 1) +
    scale_color_viridis_c(
      name   = "80% PI width\n(log scale)",
      limits = c(0, max_iqr),
      option = "viridis"
    ) +
    coord_fixed() +
    labs(
      x = "Easting (m)",
      y = "Northing (m)"
    ) +
    theme_bw(base_size = 15) +
    theme(
      strip.background = element_blank(),
      strip.text       = element_text(size = 18, face = "bold"),
      legend.position  = "right",
      panel.grid.minor = element_blank(),
      axis.text        = element_text(size = 14),
      axis.title       = element_text(size = 15),
      plot.margin      = unit(c(1, 1, 1, 1), "mm")
    )

  ggsave(file.path(OUTDIR, paste0("unc_map_", date, ".png")),
         plot = p, width = 14, height = 3.2, dpi = 150)
  ggsave(file.path(OUTDIR, paste0("unc_map_", date, ".tiff")),
         plot = p, width = 14, height = 3.2, dpi = 300, device = "tiff")
  cat("Saved unc_map for", date, "\n")
}

cat("Done. Figures in", OUTDIR, "\n")
