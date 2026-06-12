library(dplyr)
library(ggplot2)
library(scales)

ROOT    <- "/Users/takashi/LocalAnalysis/WeedMap"
RESDIR  <- file.path(ROOT, "Results_TabICL")
TARGET  <- "log1p_Chenopodium_Count"
OUTDIR  <- file.path(RESDIR, "figures", "supp")
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

DATES <- c("20250414","20250424","20250430","20250506",
           "20250513","20250520","20250526","20250602")

DATE_LABELS <- c(
  "20250414" = "(a) 14 April", "20250424" = "(b) 24 April", "20250430" = "(c) 30 April",
  "20250506" = "(d) 06 May",   "20250513" = "(e) 13 May",   "20250520" = "(f) 20 May",
  "20250526" = "(g) 26 May",   "20250602" = "(h) 02 June"
)

METHOD_ORDER <- c("Observed", "OK", "TabICL", "TabICL-KpR", "TabICL-RK")
NORM_PPF_09  <- qnorm(0.9)

# --- Shared colour scale (identical to Fig 1) ---
LO_VAL        <- log1p(0.5)
THR_VAL       <- log1p(10)
MAX_VAL       <- log1p(50)
COLOUR_VALUES <- scales::rescale(
  c(LO_VAL, THR_VAL, THR_VAL + 1e-3, MAX_VAL),
  from = c(LO_VAL, MAX_VAL)
)
COLOUR_STOPS <- c("#2D0057", "#C67BCA", "#E8641A", "#FDE725")

# --- Global coordinate offset ---
all_coords <- lapply(DATES, function(d) {
  fp <- file.path(RESDIR, paste0("data_", d), "NNDM", "RK",
                  paste0(TARGET, "_RK_predictions.csv"))
  if (!file.exists(fp)) return(NULL)
  read.csv(fp)[, c("x_25832", "y_25832")]
})
X_MIN <- min(bind_rows(Filter(Negate(is.null), all_coords))$x_25832)
Y_MIN <- min(bind_rows(Filter(Negate(is.null), all_coords))$y_25832)

sigma2_from_quantiles <- function(q09, q01) ((q09 - q01) / (2 * NORM_PPF_09))^2

# ============================================================
# 1. PREDICTION MAP — all dates combined
# ============================================================
cat("Building combined prediction data...\n")
pred_all <- list()

for (date in DATES) {
  nndm_dir <- file.path(RESDIR, paste0("data_", date), "NNDM")
  rk_path  <- file.path(nndm_dir, "RK", paste0(TARGET, "_RK_predictions.csv"))
  if (!file.exists(rk_path)) next

  rk     <- read.csv(rk_path)
  coords <- data.frame(
    obs_idx = rk$obs_idx,
    x = rk$x_25832 - X_MIN,
    y = rk$y_25832 - Y_MIN
  )

  obs_pts <- coords %>% mutate(pred_count = pmax(exp(rk$true) - 1, 0), method = "Observed")

  ok_df <- read.csv(file.path(nndm_dir, "OK_predictions.csv")) %>%
    arrange(obs_idx) %>%
    mutate(ok_psi_safe = ifelse(is.na(ok_psi), 0, ok_psi),
           pred_count  = pmax(exp(ok_pred + 0.5*pmax(ok_var,0) - ok_psi_safe) - 1, 0))
  ok_pts <- coords %>%
    left_join(ok_df %>% select(obs_idx, pred_count), by = "obs_idx") %>%
    mutate(method = "OK")

  van_pred <- read.csv(file.path(nndm_dir, "Vanilla", paste0(TARGET, "_predictions.csv")))
  van_q    <- read.csv(file.path(nndm_dir, "Vanilla", paste0(TARGET, "_predictions_quantiles.csv")))
  van_pts  <- coords %>%
    mutate(pred_count = pmax(exp(van_pred[[paste0(TARGET,"_pred")]] +
                                   0.5*sigma2_from_quantiles(van_q$q0.9, van_q$q0.1)) - 1, 0),
           method = "TabICL")

  kpr_pred <- read.csv(file.path(nndm_dir, "KpR", paste0(TARGET, "_predictions.csv")))
  kpr_q    <- read.csv(file.path(nndm_dir, "KpR", paste0(TARGET, "_predictions_quantiles.csv")))
  kpr_pts  <- coords %>%
    mutate(pred_count = pmax(exp(kpr_pred[[paste0(TARGET,"_pred")]] +
                                   0.5*sigma2_from_quantiles(kpr_q$q0.9, kpr_q$q0.1)) - 1, 0),
           method = "TabICL-KpR")

  rk_pts <- coords %>%
    mutate(pred_count = pmax(exp(rk$pred_rk + 0.5*pmax(rk$var_ok, 0)) - 1, 0),
           method = "TabICL-RK")

  date_df <- bind_rows(obs_pts, ok_pts, van_pts, kpr_pts, rk_pts) %>%
    mutate(date_label = DATE_LABELS[date],
           log1p_cnt  = log1p(pred_count))

  pred_all[[date]] <- date_df
  cat("  Loaded", date, "\n")
}

pred_combined <- bind_rows(pred_all) %>%
  mutate(
    method     = factor(method, levels = METHOD_ORDER),
    date_label = factor(date_label, levels = DATE_LABELS)
  )

zeros_p     <- pred_combined %>% filter(pred_count <  0.5)
non_zeros_p <- pred_combined %>% filter(pred_count >= 0.5)

p_pred <- ggplot() +
  geom_point(data = zeros_p,
             aes(x = x, y = y),
             colour = "grey75", size = 0.4, alpha = 0.5, shape = 16) +
  geom_point(data = non_zeros_p,
             aes(x = x, y = y, colour = log1p_cnt),
             size = 0.4, shape = 16) +
  scale_colour_gradientn(
    colours = COLOUR_STOPS,
    values  = COLOUR_VALUES,
    limits  = c(LO_VAL, MAX_VAL),
    name    = expression(paste(italic("Chenopodium"), "\n(plants m"^{-2}, ")")),
    breaks  = log1p(c(1, 5, 10, 20, 50)),
    labels  = c("1", "5", "10", "20", "50"),
    oob     = scales::squish
  ) +
  facet_grid(date_label ~ method) +
  coord_fixed() +
  labs(x = "Easting (m)", y = "Northing (m)") +
  theme_bw(base_size = 9) +
  theme(
    strip.text.x     = element_text(size = 8, face = "bold"),
    strip.text.y     = element_text(size = 8, face = "bold", angle = 0),
    legend.position  = "right",
    panel.grid       = element_blank(),
    axis.text        = element_text(size = 5),
    axis.title       = element_text(size = 8),
    panel.spacing    = unit(0.3, "lines")
  )

ggsave(file.path(OUTDIR, "pred_maps_all_dates.png"),
       plot = p_pred, width = 14, height = 20, dpi = 150)
ggsave(file.path(OUTDIR, "pred_maps_all_dates.tiff"),
       plot = p_pred, width = 14, height = 20, dpi = 300, device = "tiff")
cat("Saved pred_maps_all_dates\n")

# ============================================================
# 2. UNCERTAINTY MAP — all dates combined
# ============================================================
cat("Building combined uncertainty data...\n")
unc_all <- list()

for (date in DATES) {
  nndm_dir <- file.path(RESDIR, paste0("data_", date), "NNDM")
  rk_path  <- file.path(nndm_dir, "RK", paste0(TARGET, "_RK_predictions.csv"))
  if (!file.exists(rk_path)) next

  rk     <- read.csv(rk_path)
  coords <- data.frame(
    obs_idx = rk$obs_idx,
    x = rk$x_25832 - X_MIN,
    y = rk$y_25832 - Y_MIN
  )

  ok_q  <- read.csv(file.path(nndm_dir, paste0(TARGET, "_OK_quantiles.csv")))
  van_q <- read.csv(file.path(nndm_dir, "Vanilla", paste0(TARGET, "_predictions_quantiles.csv")))
  kpr_q <- read.csv(file.path(nndm_dir, "KpR",     paste0(TARGET, "_predictions_quantiles.csv")))
  rk_q  <- read.csv(file.path(nndm_dir, "RK",      paste0(TARGET, "_RK_quantiles.csv")))

  unc_df <- bind_rows(
    coords %>% mutate(iqr = ok_q$q0.9  - ok_q$q0.1,  method = "OK"),
    coords %>% mutate(iqr = van_q$q0.9 - van_q$q0.1, method = "TabICL"),
    coords %>% mutate(iqr = kpr_q$q0.9 - kpr_q$q0.1, method = "TabICL-KpR"),
    coords %>% mutate(iqr = rk_q$q0.9  - rk_q$q0.1,  method = "TabICL-RK")
  ) %>%
    mutate(date_label = DATE_LABELS[date])

  unc_all[[date]] <- unc_df
  cat("  Loaded", date, "\n")
}

unc_combined <- bind_rows(unc_all) %>%
  mutate(
    method     = factor(method, levels = METHOD_ORDER[-1]),  # no Observed
    date_label = factor(date_label, levels = DATE_LABELS)
  )

# Shared colour scale across both figures
unc_limit <- quantile(unc_combined$iqr, 0.99, na.rm = TRUE)

make_unc_plot <- function(data) {
  ggplot(data, aes(x = x, y = y, colour = iqr)) +
    geom_point(size = 0.4, shape = 16) +
    facet_grid(date_label ~ method) +
    scale_colour_viridis_c(
      name   = "80% PI width\n(log scale)",
      option = "viridis",
      limits = c(0, unc_limit),
      oob    = scales::squish
    ) +
    coord_fixed() +
    labs(x = "Easting (m)", y = "Northing (m)") +
    theme_bw(base_size = 15) +
    theme(
      strip.background = element_blank(),
      strip.text.x     = element_text(size = 14, face = "bold"),
      strip.text.y     = element_text(size = 12, face = "bold", angle = 0),
      legend.position  = "right",
      panel.grid       = element_blank(),
      axis.text        = element_text(size = 9),
      axis.title       = element_text(size = 13),
      panel.spacing    = unit(0.3, "lines"),
      plot.margin      = unit(c(1, 1, 1, 1), "mm")
    )
}

DATES_EARLY <- DATES[1:4]
DATES_LATE  <- DATES[5:8]

unc_early <- unc_combined %>%
  filter(date_label %in% DATE_LABELS[DATES_EARLY]) %>%
  mutate(date_label = factor(date_label, levels = DATE_LABELS[DATES_EARLY]))

unc_late <- unc_combined %>%
  filter(date_label %in% DATE_LABELS[DATES_LATE]) %>%
  mutate(date_label = factor(date_label, levels = DATE_LABELS[DATES_LATE]))

p_unc_early <- make_unc_plot(unc_early)
p_unc_late  <- make_unc_plot(unc_late)

ggsave(file.path(OUTDIR, "unc_maps_dates1to4.png"),
       plot = p_unc_early, width = 13, height = 9, dpi = 150)
ggsave(file.path(OUTDIR, "unc_maps_dates1to4.tiff"),
       plot = p_unc_early, width = 13, height = 9, dpi = 300, device = "tiff")
cat("Saved unc_maps_dates1to4\n")

ggsave(file.path(OUTDIR, "unc_maps_dates5to8.png"),
       plot = p_unc_late, width = 13, height = 9, dpi = 150)
ggsave(file.path(OUTDIR, "unc_maps_dates5to8.tiff"),
       plot = p_unc_late, width = 13, height = 9, dpi = 300, device = "tiff")
cat("Saved unc_maps_dates5to8\n")

cat("Done. Output in", OUTDIR, "\n")
