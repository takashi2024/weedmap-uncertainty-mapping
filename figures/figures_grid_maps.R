library(dplyr)
library(ggplot2)
library(scales)
library(patchwork)

ROOT    <- "/Users/takashi/LocalAnalysis/WeedMap/ForGithub"
RESDIR  <- file.path(ROOT, "Results_TabICL")
OUTDIR  <- file.path(RESDIR, "figures", "supp")
GRDDIR  <- file.path(ROOT, "grid_predictions")
UAV_ROOT<- file.path(ROOT, "data_20260805_updated")
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

TARGET  <- "log1p_Chenopodium_Count"

DATES <- c("20250414","20250424","20250430","20250506",
           "20250513","20250520","20250526","20250603")
DATE_TAGS <- c(
  "20250414"="250414F3mRX","20250424"="250424F3mRX","20250430"="250430F3mRX",
  "20250506"="250506F3mRX","20250513"="250513F3mRX","20250520"="250520F3mRX",
  "20250526"="250526F3mRX","20250603"="250603F3mRX"
)
DATE_LABELS <- c(
  "20250414"="(a) 14 April","20250424"="(b) 24 April","20250430"="(c) 30 April",
  "20250506"="(d) 06 May",  "20250513"="(e) 13 May",  "20250520"="(f) 20 May",
  "20250526"="(g) 26 May",  "20250603"="(h) 02 June"
)
METHOD_ORDER <- c("Observed","OK","TabICL","TabICL-KpR","TabICL-RK")

LO_VAL       <- log1p(0.5)
THR_VAL      <- log1p(10)
COLOUR_STOPS <- c("#2D0057","#C67BCA","#E8641A","#FDE725")
.TABICL_PROBS <- c(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                   0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
.TABICL_QCOLS <- sprintf("q%.2f", .TABICL_PROBS)  # matches predict_grid_tabicl.py format

mean_count_trapz <- function(q_df) {
  q_log  <- as.matrix(q_df[, .TABICL_QCOLS])
  q_cnt  <- pmax(exp(q_log) - 1, 0)
  lower  <- matrix(0, nrow = nrow(q_df), ncol = 1)
  q_upp  <- pmax(exp(q_df$q0.95 + (q_df$q0.95 - q_df$q0.90)) - 1, 0)
  upper  <- matrix(q_upp, ncol = 1)
  all_q  <- cbind(lower, q_cnt, upper)
  all_p  <- c(0.0, .TABICL_PROBS, 1.0)
  widths <- diff(all_p)
  midpts <- (all_q[, -ncol(all_q)] + all_q[, -1]) / 2
  pmax(rowSums(sweep(midpts, 2, widths, "*")), 0)
}

# Per-date colour scale helpers
make_colour_params <- function(date_max_count) {
  max_val <- log1p(date_max_count)
  cv <- scales::rescale(c(LO_VAL, THR_VAL, THR_VAL + 1e-3, max_val),
                        from = c(LO_VAL, max_val))
  brks <- if (date_max_count <= 50)  c(1, 5, 10, 25, 50)
     else if (date_max_count <= 100) c(1, 10, 25, 50, 100)
     else if (date_max_count <= 200) c(1, 10, 50, 100, 200)
     else if (date_max_count <= 300) c(1, 10, 50, 150, 300)
     else                            c(1, 10, 50, 200, 400)
  list(max_val = max_val, colour_values = cv,
       breaks = log1p(brks), labels = as.character(brks))
}

# Global coordinate origin — min across all dates
cat("Computing global coordinate origin ...\n")
all_coords <- lapply(DATES, function(d) {
  ok <- file.path(GRDDIR, d, "OK_grid.csv")
  if (!file.exists(ok)) return(NULL)
  read.csv(ok)[, c("x_25832","y_25832")]
})
coord_pool <- bind_rows(Filter(Negate(is.null), all_coords))
X_MIN <- min(coord_pool$x_25832)
Y_MIN <- min(coord_pool$y_25832)
X_RANGE <- range(coord_pool$x_25832 - X_MIN)
Y_RANGE <- range(coord_pool$y_25832 - Y_MIN)
cat(sprintf("  Origin: (%.1f, %.1f)\n", X_MIN, Y_MIN))
cat(sprintf("  X range: %.1f – %.1f m\n", X_RANGE[1], X_RANGE[2]))
cat(sprintf("  Y range: %.1f – %.1f m\n\n", Y_RANGE[1], Y_RANGE[2]))

# =============================================================================
# 1.  PREDICTION MAP — all dates combined (rows=dates, cols=methods)
# =============================================================================
cat("Building prediction map data ...\n")
pred_all <- list()

for (date in DATES) {
  grd <- file.path(GRDDIR, date)
  ok_path  <- file.path(grd, "OK_grid.csv")
  van_path <- file.path(grd, "Vanilla_grid.csv")
  kpr_path <- file.path(grd, "KpR_grid.csv")
  rk_path  <- file.path(grd, "RK_grid.csv")
  obs_path <- file.path(UAV_ROOT, DATE_TAGS[date], "result.csv")
  if (!file.exists(ok_path)) { cat("  SKIP", date, "(no OK_grid.csv)\n"); next }

  ok   <- read.csv(ok_path)
  obs  <- read.csv(obs_path)

  label <- DATE_LABELS[date]

  # OK prediction (lognormal bias correction)
  ok_pts <- ok %>%
    mutate(x = x_25832 - X_MIN, y = y_25832 - Y_MIN,
           pred_count = pmax(exp(ok_pred + 0.5 * pmax(ok_var, 0)) - 1, 0),
           method = "OK", date_label = label)

  # Observed points
  obs_pts <- obs %>%
    mutate(x = x_25832 - X_MIN, y = y_25832 - Y_MIN,
           pred_count = pmax(exp(log1p_Chenopodium_Count) - 1, 0),
           method = "Observed", date_label = label) %>%
    select(x, y, pred_count, method, date_label)

  parts <- list(obs_pts, ok_pts[, c("x","y","pred_count","method","date_label")])

  if (file.exists(van_path)) {
    van <- read.csv(van_path)
    van_pts <- van %>%
      mutate(x = x_25832 - X_MIN, y = y_25832 - Y_MIN,
             pred_count = mean_count_trapz(van),
             method = "TabICL", date_label = label) %>%
      select(x, y, pred_count, method, date_label)
    parts <- c(parts, list(van_pts))
  }

  if (file.exists(kpr_path)) {
    kpr <- read.csv(kpr_path)
    kpr_pts <- kpr %>%
      mutate(x = x_25832 - X_MIN, y = y_25832 - Y_MIN,
             pred_count = mean_count_trapz(kpr),
             method = "TabICL-KpR", date_label = label) %>%
      select(x, y, pred_count, method, date_label)
    parts <- c(parts, list(kpr_pts))
  }

  if (file.exists(rk_path)) {
    rk <- read.csv(rk_path)
    rk_pts <- rk %>%
      mutate(x = x_25832 - X_MIN, y = y_25832 - Y_MIN,
             pred_count = pmax(exp(pred_rk + 0.5 * pmax(resid_var, 0)) - 1, 0),
             method = "TabICL-RK", date_label = label) %>%
      select(x, y, pred_count, method, date_label)
    parts <- c(parts, list(rk_pts))
  }

  pred_all[[date]] <- bind_rows(parts)
  cat("  Loaded", date, "\n")
}

pred_combined <- bind_rows(pred_all) %>%
  mutate(
    method     = factor(method, levels = METHOD_ORDER),
    date_label = factor(date_label, levels = DATE_LABELS),
    log1p_cnt  = log1p(pred_count)
  )

zeros_p     <- pred_combined %>% filter(pred_count <  0.5)
non_zeros_p <- pred_combined %>% filter(pred_count >= 0.5)

# Observed column uses points; grid columns use tiles
obs_data   <- pred_combined %>% filter(method == "Observed")
grid_data  <- pred_combined %>% filter(method != "Observed")
zeros_g    <- grid_data %>% filter(pred_count <  0.5)
nonzeros_g <- grid_data %>% filter(pred_count >= 0.5)

# Global colour params for combined plot (max across all dates)
global_max_count <- max(pred_combined$pred_count, na.rm = TRUE)
gcp <- make_colour_params(global_max_count)

p_pred <- ggplot() +
  # Grid layers: zero = grey tile, non-zero = coloured tile
  geom_tile(data = zeros_g,
            aes(x = x, y = y), fill = "grey85", width = 1, height = 1) +
  geom_tile(data = nonzeros_g,
            aes(x = x, y = y, fill = log1p_cnt), width = 1, height = 1) +
  # Observed column: scatter points
  geom_point(data = obs_data %>% filter(pred_count <  0.5),
             aes(x = x, y = y), colour = "grey75", size = 0.5, shape = 16) +
  geom_point(data = obs_data %>% filter(pred_count >= 0.5),
             aes(x = x, y = y, colour = log1p_cnt), size = 0.5, shape = 16) +
  scale_fill_gradientn(
    colours = COLOUR_STOPS, values = gcp$colour_values,
    limits = c(LO_VAL, gcp$max_val),
    name   = expression(paste(italic("Chenopodium"), " spp.\n(plants m"^{-2}, ")")),
    breaks = gcp$breaks, labels = gcp$labels,
    oob    = scales::squish
  ) +
  scale_colour_gradientn(
    colours = COLOUR_STOPS, values = gcp$colour_values,
    limits = c(LO_VAL, gcp$max_val), guide = "none",
    oob    = scales::squish
  ) +
  facet_grid(date_label ~ method) +
  coord_fixed() +
  labs(x = "Easting (m)", y = "Northing (m)") +
  theme_bw(base_size = 9) +
  theme(
    strip.text.x    = element_text(size = 8, face = "bold"),
    strip.text.y    = element_text(size = 8, face = "bold", angle = 0),
    legend.position = "right",
    panel.grid      = element_blank(),
    axis.text       = element_text(size = 5),
    axis.title      = element_text(size = 8),
    panel.spacing   = unit(0.3, "lines")
  )


# =============================================================================
# 2. UNCERTAINTY MAP — all dates combined (rows=dates, cols=methods, no Observed)
# =============================================================================
cat("\nBuilding uncertainty map data ...\n")
unc_all <- list()

for (date in DATES) {
  grd <- file.path(GRDDIR, date)
  ok_path  <- file.path(grd, "OK_grid.csv")
  van_path <- file.path(grd, "Vanilla_grid.csv")
  kpr_path <- file.path(grd, "KpR_grid.csv")
  rk_path  <- file.path(grd, "RK_grid.csv")
  if (!file.exists(ok_path)) next

  label <- DATE_LABELS[date]
  parts <- list()

  ok <- read.csv(ok_path)
  parts$ok <- ok %>%
    mutate(x=x_25832-X_MIN, y=y_25832-Y_MIN,
           # 80% PI width on log scale (OK variance → sd → 80% PI)
           iqr = 2 * qnorm(0.9) * sqrt(pmax(ok_var, 0)),
           method="OK", date_label=label) %>%
    select(x, y, iqr, method, date_label)

  if (file.exists(van_path)) {
    van <- read.csv(van_path)
    parts$van <- van %>%
      mutate(x=x_25832-X_MIN, y=y_25832-Y_MIN,
             iqr=q0.90-q0.10, method="TabICL", date_label=label) %>%
      select(x, y, iqr, method, date_label)
  }

  if (file.exists(kpr_path)) {
    kpr <- read.csv(kpr_path)
    parts$kpr <- kpr %>%
      mutate(x=x_25832-X_MIN, y=y_25832-Y_MIN,
             iqr=q0.90-q0.10, method="TabICL-KpR", date_label=label) %>%
      select(x, y, iqr, method, date_label)
  }

  if (file.exists(rk_path)) {
    rk <- read.csv(rk_path)
    parts$rk <- rk %>%
      mutate(x=x_25832-X_MIN, y=y_25832-Y_MIN,
             iqr = 2 * qnorm(0.9) * sqrt(pmax(resid_var, 0)),
             method="TabICL-RK", date_label=label) %>%
      select(x, y, iqr, method, date_label)
  }

  unc_all[[date]] <- bind_rows(parts)
  cat("  Loaded", date, "\n")
}

unc_combined <- bind_rows(unc_all) %>%
  mutate(
    method     = factor(method, levels = METHOD_ORDER[-1]),
    date_label = factor(date_label, levels = DATE_LABELS)
  )

unc_limit <- quantile(unc_combined$iqr, 0.99, na.rm = TRUE)

make_unc_plot <- function(data) {
  ggplot(data, aes(x = x, y = y, fill = iqr)) +
    geom_tile(width = 1, height = 1) +
    facet_grid(date_label ~ method) +
    scale_fill_viridis_c(
      name = "80% PI width\n(log scale)", option = "viridis",
      limits = c(0, unc_limit), oob = scales::squish
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
      plot.margin      = unit(c(1,1,1,1), "mm")
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


# Per-date single uncertainty maps
make_unc_single <- function(data) {
  ggplot(data, aes(x = x, y = y, fill = iqr)) +
    geom_tile(width = 1, height = 1) +
    facet_wrap(~method, nrow = 1) +
    scale_fill_viridis_c(
      name = "80% PI width\n(log scale)", option = "viridis",
      limits = c(0, unc_limit), oob = scales::squish
    ) +
    coord_fixed() +
    labs(x = "Easting (m)", y = "Northing (m)") +
    theme_bw(base_size = 18) +
    theme(
      strip.background  = element_blank(),
      strip.text.x      = element_text(size = 22, face = "bold"),
      legend.position   = "right",
      legend.title      = element_text(size = 16),
      legend.text       = element_text(size = 14),
      panel.grid        = element_blank(),
      axis.text         = element_text(size = 14),
      axis.title        = element_text(size = 18),
      panel.spacing     = unit(0.5, "lines"),
      plot.margin       = unit(c(2,2,2,2), "mm")
    )
}

for (date in DATES) {
  label  <- DATE_LABELS[date]
  d_data <- unc_combined %>%
    filter(date_label == label) %>%
    mutate(method = factor(method, levels = METHOD_ORDER[-1]))
  if (nrow(d_data) == 0) next
  p     <- make_unc_single(d_data)
  unc_prefix <- if (date %in% c("20250414","20250424","20250430","20250506")) "Fig_S12_" else "Fig_S13_"
  fname <- paste0(unc_prefix, "unc_map_", date)
  ggsave(file.path(OUTDIR, paste0(fname, ".png")),
         plot = p, width = 13, height = 4, dpi = 150)
  ggsave(file.path(OUTDIR, paste0(fname, ".tiff")),
         plot = p, width = 13, height = 4, dpi = 300, device = "tiff")
  cat("Saved", fname, "\n")
}

# =============================================================================
# 3. PER-DATE PREDICTION MAPS  (pred_map_{date}.png / _sq.png)
#    Layout mirrors main_supp_prediction_maps.R:
#    Row 1: Observed | OK | TabICL
#    Row 2: [spacer] | TabICL-KpR | TabICL-RK
# =============================================================================
cat("\nBuilding per-date prediction maps ...\n")

make_panel_tile <- function(pts_zeros, pts_nonzero, title_str, show_x, show_y, cp) {
  ggplot() +
    geom_tile(data = pts_zeros,
              aes(x = x, y = y), fill = "grey70", width = 1, height = 1) +
    geom_tile(data = pts_nonzero,
              aes(x = x, y = y, fill = log1p_cnt), width = 1, height = 1) +
    scale_fill_gradientn(
      colours = COLOUR_STOPS, values = cp$colour_values,
      limits  = c(LO_VAL, cp$max_val),
      name    = expression(atop(paste(italic("Chenopodium"), " spp."),
                                "density (plants m"^{-2}*")")),
      breaks  = cp$breaks, labels = cp$labels,
      oob     = scales::squish
    ) +
    coord_fixed(xlim = X_RANGE, ylim = Y_RANGE) +
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

make_panel_obs <- function(pts_zeros, pts_nonzero, title_str, show_x, show_y, pt_shape, cp) {
  ggplot() +
    geom_point(data = pts_zeros,
               aes(x = x, y = y),
               colour = "grey70", size = 1.5, alpha = 0.6, shape = pt_shape) +
    geom_point(data = pts_nonzero,
               aes(x = x, y = y, colour = log1p_cnt),
               size = 1.5, shape = pt_shape) +
    scale_colour_gradientn(
      colours = COLOUR_STOPS, values = cp$colour_values,
      limits  = c(LO_VAL, cp$max_val),
      name    = expression(atop(paste(italic("Chenopodium"), " spp."),
                                "density (plants m"^{-2}*")")),
      breaks  = cp$breaks, labels = cp$labels,
      oob     = scales::squish
    ) +
    coord_fixed(xlim = X_RANGE, ylim = Y_RANGE) +
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

split_pts <- function(pts) list(
  z  = pts %>% filter(pred_count <  0.5),
  nz = pts %>% filter(pred_count >= 0.5)
)

for (date in DATES) {
  grd     <- file.path(GRDDIR, date)
  ok_path <- file.path(grd, "OK_grid.csv")
  if (!file.exists(ok_path)) { cat("  SKIP", date, "\n"); next }

  ok  <- read.csv(ok_path)
  obs <- read.csv(file.path(UAV_ROOT, DATE_TAGS[date], "result.csv"))

  # Observed scatter
  obs_pts <- obs %>%
    mutate(x = x_25832 - X_MIN, y = y_25832 - Y_MIN,
           pred_count = pmax(exp(log1p_Chenopodium_Count) - 1, 0),
           log1p_cnt  = log1p(pred_count))

  # Grid tile data per method
  ok_pts <- ok %>%
    mutate(x = x_25832 - X_MIN, y = y_25832 - Y_MIN,
           pred_count = pmax(exp(ok_pred + 0.5 * pmax(ok_var, 0)) - 1, 0),
           log1p_cnt  = log1p(pred_count))

  van_path <- file.path(grd, "Vanilla_grid.csv")
  van_df   <- if (file.exists(van_path)) read.csv(van_path) else NULL
  van_pts  <- if (!is.null(van_df)) {
    van_df %>%
      mutate(x = x_25832 - X_MIN, y = y_25832 - Y_MIN,
             pred_count = mean_count_trapz(van_df),
             log1p_cnt  = log1p(pred_count))
  } else ok_pts  # fallback

  kpr_path <- file.path(grd, "KpR_grid.csv")
  kpr_df   <- if (file.exists(kpr_path)) read.csv(kpr_path) else NULL
  kpr_pts  <- if (!is.null(kpr_df)) {
    kpr_df %>%
      mutate(x = x_25832 - X_MIN, y = y_25832 - Y_MIN,
             pred_count = mean_count_trapz(kpr_df),
             log1p_cnt  = log1p(pred_count))
  } else ok_pts

  rk_path <- file.path(grd, "RK_grid.csv")
  rk_pts  <- if (file.exists(rk_path)) {
    read.csv(rk_path) %>%
      mutate(x = x_25832 - X_MIN, y = y_25832 - Y_MIN,
             pred_count = pmax(exp(pred_rk + 0.5 * pmax(resid_var, 0)) - 1, 0),
             log1p_cnt  = log1p(pred_count))
  } else ok_pts

  s_obs <- split_pts(obs_pts)
  s_ok  <- split_pts(ok_pts)
  s_van <- split_pts(van_pts)
  s_kpr <- split_pts(kpr_pts)
  s_rk  <- split_pts(rk_pts)

  # Per-date colour scale: max across all panels
  date_max_count <- max(
    obs_pts$pred_count, ok_pts$pred_count,
    van_pts$pred_count, kpr_pts$pred_count, rk_pts$pred_count,
    na.rm = TRUE
  )
  cp <- make_colour_params(date_max_count)
  cat(sprintf("    max=%.0f plants m-2 → colour scale max=%s\n",
              date_max_count, tail(cp$labels, 1)))

  p_obs <- make_panel_obs(s_obs$z, s_obs$nz, "Observed",    show_x = FALSE, show_y = TRUE,  pt_shape = 16, cp = cp)
  p_ok  <- make_panel_tile(s_ok$z,  s_ok$nz,  "OK",          show_x = FALSE, show_y = FALSE, cp = cp)
  p_van <- make_panel_tile(s_van$z, s_van$nz, "TabICL",      show_x = FALSE, show_y = FALSE, cp = cp)
  p_kpr <- make_panel_tile(s_kpr$z, s_kpr$nz, "TabICL-KpR",  show_x = TRUE,  show_y = FALSE, cp = cp)
  p_rk  <- make_panel_tile(s_rk$z,  s_rk$nz,  "TabICL-RK",   show_x = TRUE,  show_y = FALSE, cp = cp)

  combined <- (p_obs | p_ok | p_van) /
              (plot_spacer() | p_kpr | p_rk) +
    plot_layout(guides = "collect") &
    theme(legend.position = "right",
          plot.margin = unit(c(1, 2, 1, 2), "mm"))

  pred_num <- c("20250414"="S4","20250424"="S5","20250430"="S6","20250506"="S7",
                "20250513"="S8","20250520"="S9","20250526"="S10","20250603"="S11")[date]
  fname <- paste0("Fig_", pred_num, "_pred_map_", date)
  ggsave(file.path(OUTDIR, paste0(fname, ".png")),
         plot = combined, width = 13, height = 5.5, dpi = 150)
  ggsave(file.path(OUTDIR, paste0(fname, ".tiff")),
         plot = combined, width = 13, height = 5.5, dpi = 300, device = "tiff")
  cat("  Saved", fname, "\n")
}

cat("\nDone. Output in", OUTDIR, "\n")
