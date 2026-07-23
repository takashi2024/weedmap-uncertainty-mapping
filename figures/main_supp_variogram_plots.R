library(dplyr)
library(sp)
library(automap)
library(gstat)
library(ggplot2)
library(patchwork)

ROOT     <- "/Users/takashi/LocalAnalysis/WeedMap/ForGithub"
RESDIR   <- file.path(ROOT, "Results_TabICL")
TARGET   <- "log1p_Chenopodium_Count"
UAV_ROOT <- file.path(ROOT, "data")
OUTDIR   <- file.path(RESDIR, "figures", "supp")
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

DATES <- c("20250414","20250424","20250430","20250506",
           "20250513","20250520","20250526","20250602")

DATE_LABELS <- c(
  "20250414" = "14 Apr", "20250424" = "24 Apr", "20250430" = "30 Apr",
  "20250506" = "06 May", "20250513" = "13 May", "20250520" = "20 May",
  "20250526" = "26 May", "20250602" = "02 Jun"
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

# Robust variogram fitting: exclude Gaussian (near-singular kriging matrix at
# zero nugget) and floor nugget at 5% of total sill to prevent zero-nugget fits.
fit_vgm_robust <- function(formula, data, min_nugget_frac = 0.05, ...) {
  fit        <- autofitVariogram(formula, data, model = c("Sph", "Exp", "Ste"), ...)
  total_sill <- sum(fit$var_model$psill)
  min_nug    <- min_nugget_frac * total_sill
  if (fit$var_model$psill[1] < min_nug) {
    deficit <- min_nug - fit$var_model$psill[1]
    fit$var_model$psill[1] <- min_nug
    fit$var_model$psill[2] <- max(fit$var_model$psill[2] - deficit, 0)
  }
  fit
}

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
vgm_panel <- function(emp_vgm, fit_vgm, title_str) {
  # Empirical points
  emp_df <- as.data.frame(emp_vgm)

  # Model line
  max_dist <- max(emp_df$dist)
  dist_seq <- seq(0.01, max_dist, length.out = 200)
  model_df <- variogramLine(fit_vgm, dist_vector = dist_seq)

  ggplot() +
    geom_point(data = emp_df, aes(x = dist, y = gamma, size = np),
               color = "grey30", alpha = 0.8) +
    geom_line(data = model_df, aes(x = dist, y = gamma),
              color = "#cc3333", linewidth = 0.9) +
    scale_size_continuous(name = "n pairs", range = c(2, 8)) +
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
    fit <- fit_vgm_robust(log1p_Count ~ 1, pts_sp)
    emp <- variogram(log1p_Count ~ 1, pts_sp)
    vgm_panel(emp, fit$var_model,
              paste0(label, "\n", vgm_label(fit$var_model$model[2]),
                     "  nugget=", round(fit$var_model$psill[1], 3),
                     "  range=", practical_range(fit$var_model), "m"))
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

    fit_rk  <- fit_vgm_robust(resid ~ 1, resid_sp)
    emp_rk  <- variogram(resid ~ 1, resid_sp)
    vgm_panel(emp_rk, fit_rk$var_model,
              paste0(label, "\n", vgm_label(fit_rk$var_model$model[2]),
                     "  nugget=", round(fit_rk$var_model$psill[1], 3),
                     "  range=", practical_range(fit_rk$var_model), "m"))
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
ggsave(file.path(OUTDIR, "variogram_OK.png"),
       plot = fig_ok, width = 14, height = 14, dpi = 150)
ggsave(file.path(OUTDIR, "variogram_OK.tiff"),
       plot = fig_ok, width = 14, height = 14, dpi = 300, device = "tiff")
cat("Saved variogram_OK\n")

fig_rk <- wrap_plots(Filter(Negate(is.null), panels_rk), ncol = 3) +
  plot_annotation(
    title = "Fitted variograms — TabICL-RK residuals",
    theme = theme(plot.title = element_text(size = 12, face = "bold"))
  )
ggsave(file.path(OUTDIR, "variogram_RK.png"),
       plot = fig_rk, width = 14, height = 14, dpi = 150)
ggsave(file.path(OUTDIR, "variogram_RK.tiff"),
       plot = fig_rk, width = 14, height = 14, dpi = 300, device = "tiff")
cat("Saved variogram_RK\n")

cat("Done. Output in", OUTDIR, "\n")
