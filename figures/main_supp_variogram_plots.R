library(dplyr)
library(sp)
library(automap)
library(gstat)
library(ggplot2)
library(patchwork)

ROOT     <- "/Users/takashi/LocalAnalysis/WeedMap/ForGithub"
RESDIR   <- file.path(ROOT, "Results_TabICL")
TARGET   <- "log1p_Chenopodium_Count"
UAV_ROOT <- file.path(ROOT, "data_20260805_updated")
OUTDIR   <- file.path(RESDIR, "figures", "supp")
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

DATES <- c("20250414","20250424","20250430","20250506",
           "20250513","20250520","20250526","20250603")

DATE_LABELS <- c(
  "20250414" = "14 Apr", "20250424" = "24 Apr", "20250430" = "30 Apr",
  "20250506" = "06 May", "20250513" = "13 May", "20250520" = "20 May",
  "20250526" = "26 May", "20250603" = "02 Jun"
)

# Helper: build SpatialPointsDataFrame for one date from per-date result.csv
make_sp <- function(date_str) {
  uav_tag <- paste0(substr(date_str, 3, 8), "F3mRX")
  csv     <- file.path(UAV_ROOT, uav_tag, "result.csv")
  df      <- read.csv(csv, stringsAsFactors = FALSE)
  df$log1p_Count <- df$log1p_Chenopodium_Count

  # Convert to UTM EPSG:25832
  sf_obj <- sf::st_as_sf(df, coords = c("Longitude", "Latitude"), crs = 4326)
  sf_utm <- sf::st_transform(sf_obj, 25832)
  coords_utm <- sf::st_coordinates(sf_utm)
  df$x_25832 <- coords_utm[, 1]
  df$y_25832 <- coords_utm[, 2]

  sp::coordinates(df) <- ~x_25832 + y_25832
  sp::proj4string(df) <- sp::CRS("+init=epsg:25832")
  df
}

# Grid-search variogram fitting: Sph/Exp/Ste (no Gaussian), multiple starting
# points for nugget and range, selects the fit with the lowest WLS SSErr.
# No nugget floor — the optimizer is free to find nugget=0 if that is best.
fit_vgm_grid <- function(formula, data,
                          models       = c("Sph", "Exp", "Ste"),
                          nugget_fracs = c(0, 0.1, 0.3, 0.6),
                          range_fracs  = c(0.15, 0.4, 0.75),
                          kappa_values = c(1.0, 1.5, 2.0, 2.5)) {
  # kappa_values apply to Ste only; kappa=0.5 is excluded (identical to Exp)
  empvar   <- variogram(formula, data)
  emp_sill <- max(empvar$gamma)
  max_dist <- max(empvar$dist)
  best_fit <- NULL
  best_err <- Inf
  candidates <- do.call(rbind, lapply(models, function(mdl) {
    if (mdl == "Ste") data.frame(mdl = mdl, kap = kappa_values, stringsAsFactors = FALSE)
    else              data.frame(mdl = mdl, kap = NA_real_,     stringsAsFactors = FALSE)
  }))
  for (i in seq_len(nrow(candidates))) {
    mdl <- candidates$mdl[i]
    kap <- candidates$kap[i]
    for (nf in nugget_fracs) {
      for (rf in range_fracs) {
        nug  <- nf * emp_sill
        rng  <- rf * max_dist
        init <- if (mdl == "Ste")
          vgm(psill = max(emp_sill - nug, 1e-6), model = mdl,
              range = rng, nugget = nug, kappa = kap)
        else
          vgm(psill = max(emp_sill - nug, 1e-6), model = mdl,
              range = rng, nugget = nug)
        fit  <- tryCatch(fit.variogram(empvar, init), error = function(e) NULL)
        if (is.null(fit)) next
        err  <- tryCatch(attr(fit, "SSErr"), error = function(e) Inf)
        if (is.finite(err) && !is.na(err) && err < best_err) {
          best_err <- err; best_fit <- fit
        }
      }
    }
  }
  if (is.null(best_fit))
    best_fit <- autofitVariogram(formula, data, model = models)$var_model
  list(var_model = best_fit)
}

# Lag boundaries: 0.5 m bins up to 3 m for short-range resolution, then
# standard 2 m bins to the full default cutoff (~26 m). No cutoff restriction.
VGM_BOUNDARIES <- c(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 7.0, 10.0, 13.0, 17.0, 22.0, 27.0, 32.0)
# Display x-axis limit: crop the plot for readability without discarding data
VGM_XLIM_OK <- 30
VGM_XLIM_RK <- 15

VGM_NAMES <- c(Sph = "Spherical", Exp = "Exponential", Gau = "Gaussian",
               Ste = "Matérn", Mat = "Matérn", Cir = "Circular",
               Lin = "Linear", Bes = "Bessel", Pen = "Pentaspherical")

vgm_label <- function(model_str) {
  model_str <- as.character(model_str)
  lbl <- VGM_NAMES[model_str]
  if (is.na(lbl)) model_str else unname(lbl)
}

# Distance at which fitted model reaches 95% of total sill (practical range)
practical_range <- function(fit_vgm, level = 0.95) {
  total_sill <- sum(fit_vgm$psill)
  search_seq <- seq(0.01, fit_vgm$range[2] * 30, length.out = 5000)
  search_val <- variogramLine(fit_vgm, dist_vector = search_seq)$gamma
  idx <- which(search_val >= level * total_sill)[1]
  if (is.na(idx)) round(fit_vgm$range[2], 1) else round(search_seq[idx], 1)
}

# Helper: empirical variogram + fitted model → ggplot panel
vgm_panel <- function(emp_vgm, fit_vgm, title_str, xlim = VGM_XLIM_OK, ann_pos = "bottomright") {
  emp_df   <- as.data.frame(emp_vgm)
  emp_vis  <- emp_df[emp_df$dist <= xlim, ]
  dist_seq <- seq(0.01, xlim, length.out = 300)
  model_df <- variogramLine(fit_vgm, dist_vector = dist_seq)

  ann_txt <- paste0(
    vgm_label(fit_vgm$model[2]),           "\n",
    "Nugget = ", round(fit_vgm$psill[1], 3), "\n",
    "Sill = ",   round(sum(fit_vgm$psill), 3), "\n",
    "Range = ",  practical_range(fit_vgm), " m"
  )
  y_max <- max(max(emp_vis$gamma), max(model_df$gamma)) * 1.1
  ann_y   <- if (ann_pos == "topright") 0.96 * y_max else 0.04 * y_max
  ann_vjust <- if (ann_pos == "topright") 1 else 0

  ggplot() +
    geom_point(data = emp_df, aes(x = dist, y = gamma, size = np),
               color = "grey30", alpha = 0.8) +
    geom_line(data = model_df, aes(x = dist, y = gamma),
              color = "#cc3333", linewidth = 0.9) +
    scale_size_area(name = "n pairs", max_size = 8) +
    scale_y_continuous(limits = c(0, y_max), expand = c(0, 0)) +
    coord_cartesian(xlim = c(0, xlim)) +
    annotate("text", x = xlim, y = ann_y,
             label = ann_txt, hjust = 1, vjust = ann_vjust,
             size = 4.8, lineheight = 1.3, color = "grey20") +
    labs(title = title_str, x = "Distance (m)", y = "Semivariance") +
    theme_bw(base_size = 18) +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(size = 16, face = "bold")
    )
}

# ---------------------------
# Build panels for all dates
# ---------------------------
panels_ok  <- list()
panels_rk  <- list()

for (date in DATES) {
  cat("Processing variogram for", date, "...\n")
  label <- DATE_LABELS[date]

  pts_sp <- tryCatch(make_sp(date), error = function(e) {
    warning("Failed to build sp for ", date, ": ", e$message); NULL
  })
  if (is.null(pts_sp)) next

  # OK global variogram
  ok_vgm <- tryCatch({
    fit <- fit_vgm_grid(log1p_Count ~ 1, pts_sp)
    emp <- variogram(log1p_Count ~ 1, pts_sp, boundaries = VGM_BOUNDARIES)
    vgm_panel(emp, fit$var_model, label)
  }, error = function(e) { warning("OK vgm failed: ", date); NULL })

  # RK residual variogram (use pred_vanilla from RK_predictions.csv)
  rk_path <- file.path(RESDIR, paste0("data_", date), "NNDM", "RK",
                       paste0(TARGET, "_RK_predictions.csv"))
  rk_vgm <- tryCatch({
    rk_df  <- read.csv(rk_path)
    # Match pts_sp rows to rk_df by coordinates
    coords_sp <- as.data.frame(sp::coordinates(pts_sp))
    names(coords_sp) <- c("x_25832", "y_25832")
    coords_sp$log1p_Count <- pts_sp$log1p_Count

    # Join residuals: pred_vanilla is on the same obs in the same coord order
    # Use merge on rounded coordinates
    rk_df$x_r <- round(rk_df$x_25832, 3)
    rk_df$y_r <- round(rk_df$y_25832, 3)
    coords_sp$x_r <- round(coords_sp$x_25832, 3)
    coords_sp$y_r <- round(coords_sp$y_25832, 3)
    merged <- merge(coords_sp, rk_df[, c("x_r","y_r","pred_vanilla")],
                    by = c("x_r","y_r"), all.x = FALSE)
    if (nrow(merged) < 20) stop("Too few merged rows")
    merged$resid <- merged$log1p_Count - merged$pred_vanilla

    resid_sp <- merged
    sp::coordinates(resid_sp) <- ~x_25832 + y_25832
    sp::proj4string(resid_sp) <- sp::CRS("+init=epsg:25832")

    fit_rk  <- fit_vgm_grid(resid ~ 1, resid_sp)
    emp_rk  <- variogram(resid ~ 1, resid_sp, boundaries = VGM_BOUNDARIES)
    rk_ann  <- if (date == "20250526") "topright" else "bottomright"
    vgm_panel(emp_rk, fit_rk$var_model, label, xlim = VGM_XLIM_RK, ann_pos = rk_ann)
  }, error = function(e) { warning("RK vgm failed: ", date, " — ", e$message); NULL })

  panels_ok[[date]] <- ok_vgm
  panels_rk[[date]] <- rk_vgm
}

# ---------------------------
# Two separate 3×3 figures
# ---------------------------
fig_ok <- wrap_plots(Filter(Negate(is.null), panels_ok), ncol = 3) +
  plot_annotation(
    title = "Fitted variograms — OK (ordinary kriging)",
    theme = theme(plot.title = element_text(size = 12, face = "bold"))
  )
ggsave(file.path(OUTDIR, "Fig_S1_variogram_OK.png"),
       plot = fig_ok, width = 14, height = 14, dpi = 150)
ggsave(file.path(OUTDIR, "Fig_S1_variogram_OK.tiff"),
       plot = fig_ok, width = 14, height = 14, dpi = 300, device = "tiff")
cat("Saved Fig_S1_variogram_OK\n")

fig_rk <- wrap_plots(Filter(Negate(is.null), panels_rk), ncol = 3) +
  plot_annotation(
    title = "Fitted variograms — TabICL-RK residuals",
    theme = theme(plot.title = element_text(size = 12, face = "bold"))
  )
ggsave(file.path(OUTDIR, "Fig_S2_variogram_RK.png"),
       plot = fig_rk, width = 14, height = 14, dpi = 150)
ggsave(file.path(OUTDIR, "Fig_S2_variogram_RK.tiff"),
       plot = fig_rk, width = 14, height = 14, dpi = 300, device = "tiff")
cat("Saved Fig_S2_variogram_RK\n")

cat("Done. Output in", OUTDIR, "\n")
