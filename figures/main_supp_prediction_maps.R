library(dplyr)
library(ggplot2)
library(scales)
library(patchwork)

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

NORM_PPF_09 <- qnorm(0.9)

# ---------------------------
# Shared colour scale (identical to main_Fig1_spatial_distribution.R)
# ---------------------------
LO_VAL        <- log1p(0.5)
THR_VAL       <- log1p(10)
MAX_VAL       <- log1p(50)
COLOUR_VALUES <- scales::rescale(
  c(LO_VAL, THR_VAL, THR_VAL + 1e-3, MAX_VAL),
  from = c(LO_VAL, MAX_VAL)
)
COLOUR_STOPS <- c("#2D0057", "#C67BCA", "#E8641A", "#FDE725")

# ---------------------------
# Global coordinate offset
# ---------------------------
all_coords <- lapply(DATES, function(d) {
  fp <- file.path(RESDIR, paste0("data_", d), "NNDM", "RK",
                  paste0(TARGET, "_RK_predictions.csv"))
  if (!file.exists(fp)) return(NULL)
  read.csv(fp)[, c("x_25832", "y_25832")]
})
all_coords <- bind_rows(Filter(Negate(is.null), all_coords))
X_MIN <- min(all_coords$x_25832)
Y_MIN <- min(all_coords$y_25832)

sigma2_from_quantiles <- function(q09, q01) ((q09 - q01) / (2 * NORM_PPF_09))^2

# ---------------------------
# Single-panel ggplot helper
# ---------------------------
make_panel <- function(pts_zeros, pts_nonzero, title_str, show_x, show_y, pt_shape = 15) {
  ggplot() +
    geom_point(data = pts_zeros,
               aes(x = x, y = y),
               colour = "grey70", size = 1.5, alpha = 0.6, shape = pt_shape) +
    geom_point(data = pts_nonzero,
               aes(x = x, y = y, colour = log1p_cnt),
               size = 1.5, shape = pt_shape) +
    scale_colour_gradientn(
      colours = COLOUR_STOPS,
      values  = COLOUR_VALUES,
      limits  = c(LO_VAL, MAX_VAL),
      name    = expression(atop(italic(Chenopodium),
                                "density (plants m"^{-2}*")")),
      breaks  = log1p(c(1, 5, 10, 20, 50)),
      labels  = c("1", "5", "10", "20", "50"),
      oob     = scales::squish
    ) +
    coord_fixed() +
    labs(
      title = title_str,
      x     = if (show_x) "Easting (m)" else NULL,
      y     = if (show_y) "Northing (m)" else NULL
    ) +
    theme_bw(base_size = 15) +
    theme(
      plot.title       = element_text(size = 18, face = "bold", hjust = 0.5),
      legend.position  = "none",
      panel.grid.minor = element_blank(),
      axis.text        = element_text(size = 14),
      axis.title       = element_text(size = 15)
    )
}

# ---------------------------
# Main loop
# ---------------------------
for (date in DATES) {
  nndm_dir <- file.path(RESDIR, paste0("data_", date), "NNDM")
  rk_path  <- file.path(nndm_dir, "RK", paste0(TARGET, "_RK_predictions.csv"))
  if (!file.exists(rk_path)) { warning("Missing: ", date); next }

  rk     <- read.csv(rk_path)
  coords <- data.frame(
    obs_idx = rk$obs_idx,
    x = rk$x_25832 - X_MIN,
    y = rk$y_25832 - Y_MIN
  )

  # --- Observed ---
  obs_pts <- coords %>%
    mutate(pred_count = pmax(exp(rk$true) - 1, 0),
           log1p_cnt  = log1p(pred_count))

  # --- OK ---
  ok_df <- read.csv(file.path(nndm_dir, "OK_predictions.csv")) %>%
    arrange(obs_idx) %>%
    mutate(ok_psi_safe = ifelse(is.na(ok_psi), 0, ok_psi),
           pred_count  = pmax(exp(ok_pred + 0.5*pmax(ok_var,0) - ok_psi_safe) - 1, 0))
  ok_pts <- coords %>%
    left_join(ok_df %>% select(obs_idx, pred_count), by = "obs_idx") %>%
    mutate(log1p_cnt = log1p(pred_count))

  # --- Vanilla (TabICL) ---
  van_pred <- read.csv(file.path(nndm_dir, "Vanilla", paste0(TARGET, "_predictions.csv")))
  van_q    <- read.csv(file.path(nndm_dir, "Vanilla", paste0(TARGET, "_predictions_quantiles.csv")))
  van_pts  <- coords %>%
    mutate(pred_count = pmax(exp(van_pred[[paste0(TARGET,"_pred")]] +
                                   0.5*sigma2_from_quantiles(van_q$q0.9, van_q$q0.1)) - 1, 0),
           log1p_cnt = log1p(pred_count))

  # --- KpR ---
  kpr_pred <- read.csv(file.path(nndm_dir, "KpR", paste0(TARGET, "_predictions.csv")))
  kpr_q    <- read.csv(file.path(nndm_dir, "KpR", paste0(TARGET, "_predictions_quantiles.csv")))
  kpr_pts  <- coords %>%
    mutate(pred_count = pmax(exp(kpr_pred[[paste0(TARGET,"_pred")]] +
                                   0.5*sigma2_from_quantiles(kpr_q$q0.9, kpr_q$q0.1)) - 1, 0),
           log1p_cnt = log1p(pred_count))

  # --- RK ---
  rk_pts <- coords %>%
    mutate(pred_count = pmax(exp(rk$pred_rk + 0.5*pmax(rk$var_ok, 0)) - 1, 0),
           log1p_cnt = log1p(pred_count))

  # --- Split zeros / non-zeros per method ---
  split_pts <- function(pts) list(
    z  = pts %>% filter(pred_count <  0.5),
    nz = pts %>% filter(pred_count >= 0.5)
  )
  s_obs <- split_pts(obs_pts)
  s_ok  <- split_pts(ok_pts)
  s_van <- split_pts(van_pts)
  s_kpr <- split_pts(kpr_pts)
  s_rk  <- split_pts(rk_pts)

  # --- Build and save both circle and square versions ---
  for (cfg in list(list(shape=16, suffix=""), list(shape=15, suffix="_sq"))) {
    p_obs <- make_panel(s_obs$z, s_obs$nz, "Observed",   show_x=FALSE, show_y=TRUE,  pt_shape=cfg$shape)
    p_ok  <- make_panel(s_ok$z,  s_ok$nz,  "OK",         show_x=FALSE, show_y=FALSE, pt_shape=cfg$shape)
    p_van <- make_panel(s_van$z, s_van$nz, "TabICL",     show_x=FALSE, show_y=FALSE, pt_shape=cfg$shape)
    p_kpr <- make_panel(s_kpr$z, s_kpr$nz, "TabICL-KpR", show_x=TRUE,  show_y=FALSE, pt_shape=cfg$shape)
    p_rk  <- make_panel(s_rk$z,  s_rk$nz,  "TabICL-RK",  show_x=TRUE,  show_y=FALSE, pt_shape=cfg$shape)

    combined <- (p_obs | p_ok  | p_van) /
                (plot_spacer() | p_kpr | p_rk) +
      plot_layout(guides = "collect") &
      theme(legend.position = "right",
            plot.margin = unit(c(1, 2, 1, 2), "mm"))

    ggsave(file.path(OUTDIR, paste0("pred_map_", date, cfg$suffix, ".png")),
           plot = combined, width = 13, height = 5.5, dpi = 150)
    ggsave(file.path(OUTDIR, paste0("pred_map_", date, cfg$suffix, ".tiff")),
           plot = combined, width = 13, height = 5.5, dpi = 300, device = "tiff")
  }
  cat("Saved pred_map for", date, "\n")
}

cat("Done. Figures in", OUTDIR, "\n")
