# =============================================================================
# datasets_NNDM_RK_tabICL.R
#
# Regression Kriging (RK) under NNDM LOO CV — TabICL version.
# Reads per-fold files produced by main_NNDM_tabICL_for_RK.py and
# kriging-corrects the vanilla TabICL predictions via residual kriging.
#
# Uses tabicl_loo_residuals.csv (not tabfpn_loo_residuals.csv) to preserve
# TabPFN V2.5 results independently.
#
# Mirrors: datasets_NNDM_RK.R (TabPFN V2.5 version)
# Reads  : Results_TabICL/data_{date}/NNDM/RK/{target}_train_residLOO_fold{i}.csv
#          Results_TabICL/data_{date}/NNDM/RK/{target}_test_true_pred_fold{i}.csv
# Writes : Results_TabICL/data_{date}/NNDM/RK/{target}_RK_predictions.csv
#          Results_TabICL/data_{date}/NNDM/RK/{target}_RK_quantiles.csv
# =============================================================================

library(automap)
library(gstat)
library(sf)
library(sp)

# Grid-search variogram fitting — same method as main_NNDM_preprocessing.R
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

# ---- User settings -----------------------------------------------------------
date   <- "20250603"
target <- "log1p_Chenopodium_Count"
.args <- commandArgs(trailingOnly = TRUE)
if (length(.args) >= 1) date <- .args[1]
crs_epsg <- 25832

# Option A: refit variogram for every fold (N fits, rigorous)
# Option B: fit once on pooled residuals, reuse (fast)
FIT_VARIOGRAM_PER_FOLD <- FALSE   # TRUE = Option A,  FALSE = Option B

ROOT     <- "/Users/takashi/LocalAnalysis/WeedMap/ForGithub"
base_dir <- file.path(ROOT, "Results_TabICL", paste0("data_", date), "NNDM", "RK")
qs       <- c(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
             0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

# ---- Helper ------------------------------------------------------------------
R2 <- function(y, y_hat) 1 - sum((y - y_hat)^2) / sum((y - mean(y))^2)

resid_col <- paste0(target, "_resid_LOO")
pred_col  <- paste0(target, "_pred")
true_col  <- paste0(target, "_true")
fml_resid <- as.formula(paste(resid_col, "~ 1"))

# ---- Determine N from file list ----------------------------------------------
fold_files <- list.files(base_dir,
                         pattern = sprintf("^%s_test_true_pred_fold\\d+\\.csv$", target),
                         full.names = FALSE)
N <- length(fold_files)
if (N == 0) stop(sprintf("No per-fold test files found in: %s\nRun main_NNDM_tabICL_for_RK.py with RUN_MODE='nndm' first.", base_dir))
message(sprintf("Found N=%d folds in: %s", N, base_dir))

# ---- Option B: fit global variogram on TabICL LOO residuals -----------------
# Uses tabicl_loo_residuals.csv (N unique points, proper LOO residuals)
if (!FIT_VARIOGRAM_PER_FOLD) {
  message("Option B: fitting global variogram on TabICL LOO residuals...")
  loo_path <- file.path(ROOT, "Dataset_for_python", paste0("data_", date),
                        "NNDM", "tabicl_loo_residuals.csv")
  if (!file.exists(loo_path))
    stop(sprintf("Missing LOO residuals file (run Step 1 first):\n  %s", loo_path))

  loo_df <- read.csv(loo_path)

  # Rename Python residual column to match fml_resid formula
  py_resid_col <- paste0(target, "_resid_loo")   # Python uses lowercase _loo
  if (!py_resid_col %in% names(loo_df))
    stop(sprintf("Column '%s' not found in %s", py_resid_col, loo_path))
  loo_df[[resid_col]] <- loo_df[[py_resid_col]]  # resid_col = {target}_resid_LOO

  loo_sp <- as(st_as_sf(loo_df, coords = c("x_25832", "y_25832"), crs = crs_epsg), "Spatial")
  vgm_fit   <- fit_vgm_grid(fml_resid, loo_sp)
  fixed_vgm <- vgm_fit$var_model
  message(sprintf("  Global variogram: %s  nugget=%.4f  sill=%.4f  range=%.2f m",
                  fixed_vgm$model[2], fixed_vgm$psill[1],
                  fixed_vgm$psill[2], fixed_vgm$range[2]))
}

# ---- Main fold loop ----------------------------------------------------------
rk_all <- vector("list", N)

for (fold_i in 0:(N - 1)) {
  tr_path <- file.path(base_dir, sprintf("%s_train_residLOO_fold%d.csv", target, fold_i))
  te_path <- file.path(base_dir, sprintf("%s_test_true_pred_fold%d.csv",  target, fold_i))

  if (!file.exists(tr_path)) stop(sprintf("Missing: %s", tr_path))
  if (!file.exists(te_path)) stop(sprintf("Missing: %s", te_path))

  tr <- read.csv(tr_path)
  te <- read.csv(te_path)

  # Guard: residual column must exist
  if (!resid_col %in% names(tr)) {
    if (true_col %in% names(tr) && paste0(target, "_pred_LOO") %in% names(tr)) {
      tr[[resid_col]] <- tr[[true_col]] - tr[[paste0(target, "_pred_LOO")]]
    } else {
      stop(sprintf("Fold %d: missing column '%s' and cannot compute it.", fold_i, resid_col))
    }
  }

  tr_sp <- as(st_as_sf(tr, coords = c("x_25832", "y_25832"), crs = crs_epsg), "Spatial")
  te_sp <- as(st_as_sf(te, coords = c("x_25832", "y_25832"), crs = crs_epsg), "Spatial")

  if (FIT_VARIOGRAM_PER_FOLD) {
    # Option A: fit variogram on this fold's training residuals
    if (length(unique(tr[[resid_col]])) < 2 || var(tr[[resid_col]]) < 1e-10) {
      resid_hat <- 0
      resid_var <- 0
    } else {
      tryCatch({
        fold_vgm  <- fit_vgm_grid(fml_resid, tr_sp)$var_model
        ok_pred   <- krige(fml_resid, tr_sp, te_sp, model = fold_vgm)
        resid_hat <- ok_pred$var1.pred
        resid_var <- ok_pred$var1.var
      }, error = function(e) {
        message(sprintf("  [WARN] fold %d krige failed: %s", fold_i, conditionMessage(e)))
        resid_hat <<- 0
        resid_var <<- 0
      })
    }
  } else {
    # Option B: use fixed global variogram
    tryCatch({
      ok_pred   <- krige(fml_resid, tr_sp, te_sp, model = fixed_vgm)
      resid_hat <- ok_pred$var1.pred
      resid_var <- ok_pred$var1.var
    }, error = function(e) {
      message(sprintf("  [WARN] fold %d krige failed: %s", fold_i, conditionMessage(e)))
      resid_hat <<- 0
      resid_var <<- 0
    })
  }

  pred_vanilla <- te[[pred_col]]
  pred_rk      <- pred_vanilla + resid_hat

  rk_all[[fold_i + 1]] <- data.frame(
    fold          = fold_i,
    obs_idx       = fold_i,
    x_25832       = te$x_25832,
    y_25832       = te$y_25832,
    Longitude     = if ("Longitude" %in% names(te)) te$Longitude else NA,
    Latitude      = if ("Latitude"  %in% names(te)) te$Latitude  else NA,
    true          = te[[true_col]],
    pred_vanilla  = pred_vanilla,
    resid_ok      = resid_hat,
    var_ok        = resid_var,
    pred_rk       = pred_rk,
    variogram_mode = ifelse(FIT_VARIOGRAM_PER_FOLD, "per_fold", "fixed")
  )

  if ((fold_i + 1) %% 50 == 0 || fold_i == 0)
    message(sprintf("  fold %d/%d done | RK pred=%.4f", fold_i + 1, N, pred_rk))
}

rk_df <- do.call(rbind, rk_all)

# ---- Save predictions --------------------------------------------------------
out_pred <- file.path(base_dir, sprintf("%s_RK_predictions.csv", target))
write.csv(rk_df, out_pred, row.names = FALSE)
message(sprintf("Saved: %s", out_pred))

# ---- Save quantiles (Normal approximation: mean=pred_rk, sd=sqrt(var_ok)) ---
quant_mat <- t(mapply(
  function(mu, v) qnorm(qs, mean = mu, sd = sqrt(pmax(v, 1e-12))),
  rk_df$pred_rk, rk_df$var_ok
))
quant_df <- as.data.frame(quant_mat)
colnames(quant_df) <- paste0("q", qs)
out_q <- file.path(base_dir, sprintf("%s_RK_quantiles.csv", target))
write.csv(quant_df, out_q, row.names = FALSE)
message(sprintf("Saved: %s", out_q))

# ---- Summary -----------------------------------------------------------------
cat(sprintf("\n=== RK Results (TabICL | NNDM LOO | variogram=%s) ===\n",
            ifelse(FIT_VARIOGRAM_PER_FOLD, "per_fold", "fixed")))
cat(sprintf("  Overall RK      R2 = %.4f\n", R2(rk_df$true, rk_df$pred_rk)))
cat(sprintf("  Overall Vanilla R2 = %.4f\n", R2(rk_df$true, rk_df$pred_vanilla)))
cat(sprintf("  N = %d\n", nrow(rk_df)))
