library(gstat)
library(sf)
library(sp)
library(automap)

ROOT      <- "/Users/takashi/LocalAnalysis/WeedMap/ForGithub"
UAV_ROOT  <- file.path(ROOT, "data_20260805_updated")
GRID_GPKG <- file.path(ROOT, "spatial", "field_1m_grid.gpkg")
OUT_ROOT  <- file.path(ROOT, "grid_predictions")

DATE_TAGS <- c(
  "20250414" = "250414F3mRX",
  "20250424" = "250424F3mRX",
  "20250430" = "250430F3mRX",
  "20250506" = "250506F3mRX",
  "20250513" = "250513F3mRX",
  "20250520" = "250520F3mRX",
  "20250526" = "250526F3mRX",
  "20250603" = "250603F3mRX"
)

# Grid-search variogram fitting (verbatim from figures/main_supp_variogram_plots.R)
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

# Load 1m grid (same for all dates)
cat("Loading grid ...\n")
grid_sf   <- sf::st_read(GRID_GPKG, quiet = TRUE)
grid_centroids <- sf::st_centroid(grid_sf)
grid_coords    <- sf::st_coordinates(grid_centroids)

grid_df             <- data.frame(grid_id = grid_sf$grid_id,
                                  x_25832 = grid_coords[, 1],
                                  y_25832 = grid_coords[, 2],
                                  dummy   = 0)
grid_spdf           <- grid_df
sp::coordinates(grid_spdf) <- ~x_25832 + y_25832
sp::proj4string(grid_spdf) <- sp::CRS("+init=epsg:25832")
cat(sprintf("  %d grid cells\n\n", nrow(grid_df)))

for (date in names(DATE_TAGS)) {
  tag     <- DATE_TAGS[date]
  out_dir <- file.path(OUT_ROOT, date)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  cat("===", date, "(", tag, ") ===\n")

  csv_path <- file.path(UAV_ROOT, tag, "result.csv")
  if (!file.exists(csv_path)) {
    warning(sprintf("  Missing: %s — skipping\n", csv_path)); next
  }
  df <- read.csv(csv_path)
  cat(sprintf("  N observations: %d\n", nrow(df)))

  sp_df <- df
  sp::coordinates(sp_df) <- ~x_25832 + y_25832
  sp::proj4string(sp_df) <- sp::CRS("+init=epsg:25832")

  cat("  Fitting variogram ...\n")
  fit <- tryCatch(
    fit_vgm_grid(log1p_Chenopodium_Count ~ 1, sp_df),
    error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
  )
  if (is.null(fit)) next
  cat(sprintf("  Model: %s  nugget=%.4f  sill=%.4f  range=%.1f m\n",
              as.character(fit$var_model$model[2]),
              fit$var_model$psill[1],
              sum(fit$var_model$psill),
              fit$var_model$range[2]))

  cat("  Kriging at grid ...\n")
  kg <- tryCatch(
    gstat::krige(log1p_Chenopodium_Count ~ 1, sp_df, grid_spdf,
                 model = fit$var_model, debug.level = 0),
    error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
  )
  if (is.null(kg)) next

  ok_grid <- data.frame(
    grid_id = grid_df$grid_id,
    x_25832 = grid_df$x_25832,
    y_25832 = grid_df$y_25832,
    ok_pred = kg$var1.pred,
    ok_var  = kg$var1.var
  )
  write.csv(ok_grid, file.path(out_dir, "OK_grid.csv"), row.names = FALSE)
  cat(sprintf("  Saved OK_grid.csv  (pred range: %.3f – %.3f)\n",
              min(ok_grid$ok_pred), max(ok_grid$ok_pred)))

  # LOO kriging at training points (needed as kp_mean/kp_var for KpR training)
  cat("  LOO kriging at training points ...\n")
  loo <- tryCatch(
    gstat::krige.cv(log1p_Chenopodium_Count ~ 1, sp_df,
                    model = fit$var_model, nfold = nrow(df), verbose = FALSE),
    error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
  )
  if (is.null(loo)) next

  loo_coords <- sp::coordinates(loo)
  ok_train <- data.frame(
    ROI_ID  = df$ROI_ID,
    x_25832 = loo_coords[, 1],
    y_25832 = loo_coords[, 2],
    ok_loo_pred = loo$var1.pred,
    ok_loo_var  = loo$var1.var
  )
  write.csv(ok_train, file.path(out_dir, "OK_train_loo.csv"), row.names = FALSE)
  cat(sprintf("  Saved OK_train_loo.csv  (loo_pred range: %.3f – %.3f)\n\n",
              min(ok_train$ok_loo_pred), max(ok_train$ok_loo_pred)))
}

cat("Done. Output in", OUT_ROOT, "\n")
