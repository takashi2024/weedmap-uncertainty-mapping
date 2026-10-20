library(gstat)
library(sf)
library(sp)
library(automap)

ROOT     <- "/Users/takashi/LocalAnalysis/WeedMap/ForGithub"
OUT_ROOT <- file.path(ROOT, "grid_predictions")

DATES <- c("20250414","20250424","20250430","20250506",
           "20250513","20250520","20250526","20250603")

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

for (date in DATES) {
  cat("===", date, "===\n")
  out_dir <- file.path(OUT_ROOT, date)

  van_train_path <- file.path(out_dir, "Vanilla_train.csv")
  van_grid_path  <- file.path(out_dir, "Vanilla_grid.csv")

  if (!file.exists(van_train_path) || !file.exists(van_grid_path)) {
    cat("  SKIP — Vanilla_train.csv or Vanilla_grid.csv missing (run predict_grid_tabicl.py first)\n\n")
    next
  }

  van_train <- read.csv(van_train_path)
  van_train$resid <- van_train$y_true - van_train$pred_train
  cat(sprintf("  N=%d  resid range: %.3f – %.3f\n",
              nrow(van_train), min(van_train$resid), max(van_train$resid)))

  van_grid <- read.csv(van_grid_path)
  cat(sprintf("  Grid M=%d\n", nrow(van_grid)))

  resid_sp <- van_train
  sp::coordinates(resid_sp) <- ~x_25832 + y_25832
  sp::proj4string(resid_sp) <- sp::CRS("+init=epsg:25832")

  grid_sp <- van_grid[, c("x_25832", "y_25832")]
  grid_sp$dummy <- 0
  sp::coordinates(grid_sp) <- ~x_25832 + y_25832
  sp::proj4string(grid_sp) <- sp::CRS("+init=epsg:25832")

  cat("  Fitting residual variogram ...\n")
  fit_resid <- tryCatch(
    fit_vgm_grid(resid ~ 1, resid_sp),
    error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
  )
  if (is.null(fit_resid)) next
  cat(sprintf("  Model: %s  nugget=%.4f  sill=%.4f  range=%.1f m\n",
              as.character(fit_resid$var_model$model[2]),
              fit_resid$var_model$psill[1],
              sum(fit_resid$var_model$psill),
              fit_resid$var_model$range[2]))

  cat("  Kriging residuals at grid ...\n")
  kg_resid <- tryCatch(
    gstat::krige(resid ~ 1, resid_sp, grid_sp,
                 model = fit_resid$var_model, debug.level = 0),
    error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
  )
  if (is.null(kg_resid)) next

  rk_pred <- van_grid$pred + kg_resid$var1.pred

  rk_df <- data.frame(
    grid_id    = van_grid$grid_id,
    x_25832    = van_grid$x_25832,
    y_25832    = van_grid$y_25832,
    pred_rk    = rk_pred,
    resid_pred = kg_resid$var1.pred,
    resid_var  = kg_resid$var1.var,
    van_pred   = van_grid$pred
  )
  write.csv(rk_df, file.path(out_dir, "RK_grid.csv"), row.names = FALSE)
  cat(sprintf("  Saved RK_grid.csv  (pred_rk range: %.3f – %.3f)\n\n",
              min(rk_pred), max(rk_pred)))
}

cat("Done. Output in", OUT_ROOT, "\n")
