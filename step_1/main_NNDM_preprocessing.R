# =============================================================================
# main_NNDM_preprocessing.R  (restructured — model-specific NNDM LOO)
#
# Each model gets its own phi (from LOO residual variogram) and its own
# nndm() call following the NNDM GitHub demo recommendation.
#
# TWO-STAGE design:
#
# STAGE 1 (runs automatically):
#   1. Load data, CRS transform
#   2. Fit global variogram on full dataset (shared basis for LOO steps)
#   3. LOO kriging (N calls, global variogram):
#      - Saves ok_loo_errors.csv        → phi_ok
#      - Saves kpr_global_loo_features.csv → used by Python for LOO KpR phi
#   4. If Python LOO residuals missing → stop with instructions for user
#
# STAGE 2 (after Python produces tabicl_loo_residuals.csv + kpr_tabicl_loo_residuals.csv):
#   5. phi per model → nndm() per model → four model-specific fold CSVs
#   6. OK evaluation  (using ok_nndm_folds)  → OK_predictions.csv
#   7. KpR features   (using kpr_nndm_folds) → KpR_features.csv
#
# Reads (Python LOO residuals, produced by main_NNDM_tabICL_for_RK.py RUN_MODE="loo"):
#   Dataset_for_python/data_{date}/NNDM/tabicl_loo_residuals.csv
#   Dataset_for_python/data_{date}/NNDM/kpr_tabicl_loo_residuals.csv
#
# Writes (Stage 1):
#   Dataset_for_python/data_{date}/NNDM/ok_loo_errors.csv
#   Dataset_for_python/data_{date}/NNDM/kpr_global_loo_features.csv
#
# Writes (Stage 2 — fold CSVs):
#   Dataset_for_python/data_{date}/NNDM/tabicl_nndm_folds.csv
#   Dataset_for_python/data_{date}/NNDM/ok_nndm_folds.csv
#   Dataset_for_python/data_{date}/NNDM/kpr_nndm_folds.csv
#   Dataset_for_python/data_{date}/NNDM/rk_nndm_folds.csv
#
# Writes (Stage 2 — model outputs):
#   Results_TabICL/data_{date}/NNDM/OK_predictions.csv
#   Results_TabICL/data_{date}/NNDM/KpR_features.csv
# =============================================================================

library(NNDM)
library(automap)
library(sf)
library(sp)
library(gstat)
library(dplyr)
library(tidyr)

# ---- User settings -----------------------------------------------------------
date_id  <- 8
target   <- "log1p_Chenopodium_Count"

# Allow overriding from command line: Rscript main_NNDM_preprocessing.R <date_id> [target]
.args <- commandArgs(trailingOnly = TRUE)
if (length(.args) >= 1) date_id <- as.integer(.args[1])
if (length(.args) >= 2) target  <- .args[2]
crs_epsg <- 25832
qs       <- c(0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999)

DATES <- c("20250414","20250424","20250430","20250506","20250513","20250520","20250526","20250602")
ROOT  <- "/Users/takashi/LocalAnalysis/WeedMap/ForGithub"

# ------------------------------------------------------------------------------

# ---- 1. Load and prepare data ------------------------------------------------
path_date <- DATES[date_id]
uav_tag   <- paste0(substr(path_date, 3, 8), "F3mRX")
UAV_CSV   <- file.path(ROOT, "data", uav_tag, "result.csv")

message(sprintf("Processing date: %s  (date_id=%d)", path_date, date_id))
result <- read.csv(UAV_CSV)
# result.csv already contains: Longitude, Latitude, Chenopodium_Count,
# log1p_Chenopodium_Count, x_25832, y_25832, fold, ROI_ID

pts_wgs <- st_as_sf(result, coords = c("Longitude", "Latitude"), crs = 4326)
pts_utm <- st_transform(pts_wgs, crs_epsg)
utm_xy  <- st_coordinates(pts_utm)
result$x_25832 <- utm_xy[, "X"]
result$y_25832 <- utm_xy[, "Y"]

N      <- nrow(result)
y_true <- result[[target]]
message(sprintf("N observations: %d", N))

pts_sf     <- st_as_sf(result, coords = c("x_25832", "y_25832"), crs = crs_epsg)
pts_sp     <- as(pts_sf, "Spatial")
fml_target <- as.formula(paste(target, "~ 1"))

# ---- Output directories ------------------------------------------------------
nndm_data_dir <- file.path(ROOT, "Dataset_for_python", paste0("data_", path_date), "NNDM")
nndm_res_dir  <- file.path(ROOT, "Results_TabICL",      paste0("data_", path_date), "NNDM")
dir.create(nndm_data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(nndm_res_dir,  recursive = TRUE, showWarnings = FALSE)

# Paths for Stage 1 outputs
ok_loo_path         <- file.path(nndm_data_dir, "ok_loo_errors.csv")
kpr_global_loo_path <- file.path(nndm_data_dir, "kpr_global_loo_features.csv")

# Paths for Python LOO residuals (produced by main_NNDM_tabFPN_for_RK.py RUN_MODE="loo")
tabicl_loo_path <- file.path(nndm_data_dir, "tabicl_loo_residuals.csv")
kpr_loo_path    <- file.path(nndm_data_dir, "kpr_tabicl_loo_residuals.csv")

# =============================================================================
# STAGE 1: Global variogram + LOO kriging (OK phi + KpR global features)
# =============================================================================
message("\n=== STAGE 1: LOO kriging for phi estimation ===")

message("Fitting global variogram on full dataset...")
global_vgm_fit <- autofitVariogram(fml_target, pts_sp)
global_vgm     <- global_vgm_fit$var_model
message(sprintf("  Global variogram: %s  sill=%.4f  range=%.2f m",
                global_vgm$model[2], global_vgm$psill[2], global_vgm$range[2]))

# Helper: extract Lagrange multiplier psi from the OK system
# Solves [Gamma 1; 1^T 0] [lambda; psi] = [gamma0; 1] and returns psi.
# psi is used in the lognormal mean back-transform for OK:
#   E[N | x0] = exp(Y_hat + sigma^2/2 - psi) - 1   (W&O 2007, eq. 8.38)
extract_ok_psi <- function(training_sp, target_sp, vgm_model) {
  n         <- nrow(training_sp)
  coords_tr <- coordinates(training_sp)
  coords_tg <- coordinates(target_sp)

  d_train  <- as.matrix(dist(coords_tr))
  d_target <- sqrt(rowSums(sweep(coords_tr, 2, coords_tg[1L, ])^2))

  gv_mat <- matrix(
    variogramLine(vgm_model, dist_vector = as.vector(d_train))$gamma,
    nrow = n
  )
  diag(gv_mat) <- 0   # exact interpolation: gamma(xi, xi) = 0 in the kriging matrix

  gv_tg <- variogramLine(vgm_model, dist_vector = d_target)$gamma

  K   <- rbind(cbind(gv_mat, 1), c(rep(1, n), 0))
  rhs <- c(gv_tg, 1)

  sol <- tryCatch(solve(K, rhs), error = function(e) NULL)
  if (is.null(sol)) return(NA_real_)
  sol[n + 1L]
}

message(sprintf("Running LOO kriging (N=%d calls, global variogram)...", N))
message("  This produces: ok_loo_errors.csv  and  kpr_global_loo_features.csv")

ok_loo_rows  <- vector("list", N)
kpr_glob_rows <- vector("list", N)

for (i in seq_len(N)) {
  tryCatch({
    capture.output(
      ok_i <- krige(fml_target, pts_sp[-i, ], pts_sp[i, ], model = global_vgm),
      type = "output"
    )
    krig_pred_i <- ok_i$var1.pred
    krig_var_i  <- ok_i$var1.var
  }, error = function(e) {
    message(sprintf("  [WARN] LOO kriging fold %d failed: %s", i, conditionMessage(e)))
    krig_pred_i <<- mean(y_true[-i])
    krig_var_i  <<- 0
  })

  ok_loo_rows[[i]] <- data.frame(
    obs_idx     = i - 1L,
    x_25832     = result$x_25832[i],
    y_25832     = result$y_25832[i],
    ok_loo_resid = y_true[i] - krig_pred_i
  )
  kpr_glob_rows[[i]] <- data.frame(
    obs_idx   = i - 1L,
    x_25832   = result$x_25832[i],
    y_25832   = result$y_25832[i],
    krig_pred = krig_pred_i,
    krig_var  = krig_var_i
  )

  if (i %% 50 == 0 || i == N)
    message(sprintf("  LOO kriging %d/%d done", i, N))
}

ok_loo_df  <- do.call(rbind, ok_loo_rows)
kpr_glob_df <- do.call(rbind, kpr_glob_rows)

write.csv(ok_loo_df,   ok_loo_path,         row.names = FALSE)
write.csv(kpr_glob_df, kpr_global_loo_path, row.names = FALSE)
message(sprintf("Saved: %s", ok_loo_path))
message(sprintf("Saved: %s", kpr_global_loo_path))

# =============================================================================
# Check for Python LOO residuals — stop here if not yet produced
# =============================================================================
python_files_ready <- file.exists(tabicl_loo_path) && file.exists(kpr_loo_path)

if (!python_files_ready) {
  message("\n=== STAGE 1 complete. ===")
  message("Next step: run main_NNDM_tabICL_for_RK.py with  RUN_MODE = 'loo'")
  message("That will produce:")
  message(sprintf("  %s", tabicl_loo_path))
  message(sprintf("  %s", kpr_loo_path))
  message("Then re-run this script to execute Stage 2.")
  quit(save = "no", status = 0)
}

# =============================================================================
# STAGE 2: phi estimation per model → nndm() per model → evaluations
# =============================================================================
message("\n=== STAGE 2: phi estimation, nndm(), OK evaluation, KpR features ===")

# ---- Helper: fit variogram on residuals → phi → nndm() → fold CSV ----------
save_folds_csv <- function(nndm_out, fold_csv_path) {
  N_loc <- length(nndm_out$indx_train)
  rows_list <- vector("list", N_loc * 3)
  row_pos   <- 1L
  for (i in seq_len(N_loc)) {
    train_0 <- nndm_out$indx_train[[i]] - 1L
    excl_0  <- if (length(nndm_out$indx_excluded[[i]]) > 0)
                  nndm_out$indx_excluded[[i]] - 1L else integer(0)
    test_0  <- i - 1L
    rows_list[[row_pos]] <- data.frame(fold = i-1L, obs_idx = test_0,  role = "test",     stringsAsFactors = FALSE)
    row_pos <- row_pos + 1L
    if (length(train_0) > 0) {
      rows_list[[row_pos]] <- data.frame(fold = i-1L, obs_idx = train_0, role = "train",    stringsAsFactors = FALSE)
      row_pos <- row_pos + 1L
    }
    if (length(excl_0) > 0) {
      rows_list[[row_pos]] <- data.frame(fold = i-1L, obs_idx = excl_0,  role = "excluded", stringsAsFactors = FALSE)
      row_pos <- row_pos + 1L
    }
  }
  folds_df <- do.call(rbind, rows_list[seq_len(row_pos - 1L)])
  write.csv(folds_df, fold_csv_path, row.names = FALSE)
  stopifnot(sum(folds_df$role == "test") == N_loc)
  message(sprintf("  Saved fold CSV: %s  (%d folds)", basename(fold_csv_path), N_loc))
  invisible(nndm_out)
}

phi_and_nndm <- function(resid_df, resid_col, model_label, fold_csv_path) {
  resid_sf <- st_as_sf(resid_df, coords = c("x_25832", "y_25832"), crs = crs_epsg)
  resid_sp <- as(resid_sf, "Spatial")
  fml_r    <- as.formula(paste(resid_col, "~ 1"))
  empvar   <- variogram(fml_r, data = resid_sp)
  fitvar   <- fit.variogram(empvar, vgm(model = "Sph", nugget = TRUE))
  phi_val  <- fitvar$range[2]
  message(sprintf("  %s: phi = %.2f m", model_label, phi_val))

  # Option A: per-fold nndm() with single test point as ppoints.
  # When ppoints = tpoints (same set), Gij = 0 for all points so nothing
  # is ever excluded. Using ppoints = single test point gives the correct
  # nearest-neighbor distance and proper NNDM LOO exclusions.
  train_per_fold <- vector("list", N)
  excl_per_fold  <- vector("list", N)
  rows_list      <- vector("list", N * 3L)
  row_pos        <- 1L

  for (i in seq_len(N)) {
    tpoints_i <- pts_sf[-i, ]
    ppoints_i <- pts_sf[i,  ]
    nndm_i    <- nndm(tpoints = tpoints_i, ppoints = ppoints_i,
                      phi = phi_val, min_train = 0.5)

    # Map local indices (1..N-1) back to original global indices (1..N)
    orig_train  <- which(seq_len(N) != i)
    train_local <- nndm_i$indx_train[[1]]
    excl_local  <- nndm_i$indx_exclude[[1]]

    train_orig <- orig_train[train_local]
    excl_orig  <- if (length(excl_local) > 0) orig_train[excl_local] else integer(0)

    train_per_fold[[i]] <- train_orig   # 1-based global
    excl_per_fold[[i]]  <- excl_orig    # 1-based global

    # Fold CSV rows (0-based obs_idx for Python compatibility)
    rows_list[[row_pos]] <- data.frame(fold = i-1L, obs_idx = i-1L,
                                       role = "test", stringsAsFactors = FALSE)
    row_pos <- row_pos + 1L
    if (length(train_orig) > 0) {
      rows_list[[row_pos]] <- data.frame(fold = i-1L, obs_idx = train_orig - 1L,
                                         role = "train", stringsAsFactors = FALSE)
      row_pos <- row_pos + 1L
    }
    if (length(excl_orig) > 0) {
      rows_list[[row_pos]] <- data.frame(fold = i-1L, obs_idx = excl_orig - 1L,
                                         role = "excluded", stringsAsFactors = FALSE)
      row_pos <- row_pos + 1L
    }

    if (i %% 50 == 0 || i == N)
      message(sprintf("    nndm fold %d/%d", i, N))
  }

  folds_df <- do.call(rbind, rows_list[seq_len(row_pos - 1L)])
  write.csv(folds_df, fold_csv_path, row.names = FALSE)
  stopifnot(sum(folds_df$role == "test") == N)
  message(sprintf("  Saved fold CSV: %s  (%d folds)", basename(fold_csv_path), N))

  # Return structure compatible with downstream OK/KpR loops
  invisible(list(indx_train = train_per_fold, indx_exclude = excl_per_fold, phi = phi_val))
}

# ---- 2a. Vanilla TabICL -------------------------------------------------------
message("\n--- Vanilla TabICL ---")
tabicl_loo <- read.csv(tabicl_loo_path)
nndm_tabicl <- phi_and_nndm(
  tabicl_loo, paste0(target, "_resid_loo"), "Vanilla TabICL",
  file.path(nndm_data_dir, "tabicl_nndm_folds.csv")
)

# ---- 2b. OK -------------------------------------------------------------------
message("\n--- OK ---")
nndm_ok <- phi_and_nndm(
  ok_loo_df, "ok_loo_resid", "OK",
  file.path(nndm_data_dir, "ok_nndm_folds.csv")
)

# ---- 2c. KpR ------------------------------------------------------------------
message("\n--- KpR ---")
kpr_loo <- read.csv(kpr_loo_path)
nndm_kpr <- phi_and_nndm(
  kpr_loo, paste0(target, "_resid_loo"), "KpR",
  file.path(nndm_data_dir, "kpr_nndm_folds.csv")
)

# ---- 2d. RK -------------------------------------------------------------------
# RK LOO residual = tabicl_loo_resid - kriged_correction
# Fit variogram on tabicl LOO residuals, then LOO krige corrections.
message("\n--- RK ---")
fml_resid_tabicl <- as.formula(paste0(target, "_resid_loo ~ 1"))

tabicl_resid_sf <- st_as_sf(tabicl_loo, coords = c("x_25832", "y_25832"), crs = crs_epsg)
tabicl_resid_sp <- as(tabicl_resid_sf, "Spatial")

message("  Fitting variogram on TabICL LOO residuals for RK LOO kriging...")
rk_vgm_fit <- autofitVariogram(fml_resid_tabicl, tabicl_resid_sp)
rk_vgm     <- rk_vgm_fit$var_model

message("  Computing LOO kriging corrections for RK residuals...")
rk_loo_resid <- numeric(N)
for (i in seq_len(N)) {
  tryCatch({
    capture.output(
      ok_rk <- krige(fml_resid_tabicl, tabicl_resid_sp[-i, ], tabicl_resid_sp[i, ],
                     model = rk_vgm),
      type = "output"
    )
    kriged_correction <- ok_rk$var1.pred
  }, error = function(e) {
    kriged_correction <<- 0
  })
  # RK LOO resid = tabicl_loo_resid - kriged_correction
  rk_loo_resid[i] <- tabicl_loo[[paste0(target, "_resid_loo")]][i] - kriged_correction
}

rk_loo_df <- data.frame(
  obs_idx       = 0L:(N - 1L),
  x_25832       = result$x_25832,
  y_25832       = result$y_25832,
  rk_loo_resid  = rk_loo_resid
)
nndm_rk <- phi_and_nndm(
  rk_loo_df, "rk_loo_resid", "RK",
  file.path(nndm_data_dir, "rk_nndm_folds.csv")
)

# =============================================================================
# OK evaluation using ok_nndm_folds
# =============================================================================
message("\n--- OK evaluation (NNDM folds) ---")
ok_rows <- vector("list", N)

for (i in seq_len(N)) {
  train_idx <- nndm_ok$indx_train[[i]]
  test_idx  <- i
  z_train   <- y_true[train_idx]

  if (length(z_train) < 4 || var(z_train) < 1e-10) {
    ok_rows[[i]] <- data.frame(fold = i-1L, obs_idx = i-1L,
                                ok_pred = mean(z_train), ok_var = 0, ok_psi = NA_real_)
  } else {
    tryCatch({
      vgm_fit <- autofitVariogram(fml_target, pts_sp[train_idx, ])
      capture.output(
        ok_fit <- krige(fml_target, pts_sp[train_idx, ], pts_sp[test_idx, ],
                        model = vgm_fit$var_model),
        type = "output"
      )
      ok_psi_i <- extract_ok_psi(pts_sp[train_idx, ], pts_sp[test_idx, ], vgm_fit$var_model)
      ok_rows[[i]] <- data.frame(fold = i-1L, obs_idx = i-1L,
                                  ok_pred = ok_fit$var1.pred, ok_var = ok_fit$var1.var,
                                  ok_psi  = ok_psi_i)
    }, error = function(e) {
      message(sprintf("  [WARN] OK fold %d failed: %s", i, conditionMessage(e)))
      ok_rows[[i]] <<- data.frame(fold = i-1L, obs_idx = i-1L,
                                   ok_pred = mean(z_train), ok_var = 0, ok_psi = NA_real_)
    })
  }
  if (i %% 50 == 0 || i == N) message(sprintf("  OK fold %d/%d", i, N))
}

ok_df   <- do.call(rbind, ok_rows)
ok_path <- file.path(nndm_res_dir, "OK_predictions.csv")
write.csv(ok_df, ok_path, row.names = FALSE)
message(sprintf("Saved: %s", ok_path))


# Save OK quantiles (Normal approx: mean = ok_pred, sd = sqrt(ok_var))
# Same approach as RK in datasets_NNDM_RK.R
ok_quant_mat <- t(mapply(
  function(mu, v) qnorm(qs, mean = mu, sd = sqrt(pmax(v, 1e-12))),
  ok_df$ok_pred, ok_df$ok_var
))
ok_quant_df          <- as.data.frame(ok_quant_mat)
colnames(ok_quant_df) <- paste0("q", qs)
ok_q_path <- file.path(nndm_res_dir, sprintf("%s_OK_quantiles.csv", target))
write.csv(ok_quant_df, ok_q_path, row.names = FALSE)
message(sprintf("Saved: %s", ok_q_path))

# # =============================================================================
# # KpR features using kpr_nndm_folds (per-fold LOO kriging within NNDM training subset)
# # =============================================================================
# message("\n--- KpR features (NNDM folds) ---")
# kpr_rows <- vector("list", N)
# 
# for (i in seq_len(N)) {
#   train_idx <- nndm_kpr$indx_train[[i]]
#   test_idx  <- i
#   n_train   <- length(train_idx)
#   z_train   <- y_true[train_idx]
#   train_sp  <- pts_sp[train_idx, ]
#   test_sp   <- pts_sp[test_idx, ]
#   fold_rows <- vector("list", n_train + 1L)
# 
#   if (n_train < 4 || var(z_train) < 1e-10) {
#     mean_z <- mean(z_train)
#     for (j in seq_len(n_train)) {
#       fold_rows[[j]] <- data.frame(fold = i-1L, obs_idx = train_idx[j]-1L,
#                                    role = "train", krig_pred = mean_z, krig_var = 0)
#     }
#     fold_rows[[n_train+1L]] <- data.frame(fold = i-1L, obs_idx = i-1L,
#                                            role = "test", krig_pred = mean_z, krig_var = 0)
#   } else {
#     vgm_fit <- tryCatch(
#       autofitVariogram(fml_target, train_sp),
#       error = function(e) {
#         message(sprintf("  [WARN] KpR fold %d variogram failed: %s", i, conditionMessage(e)))
#         NULL
#       }
#     )
#     if (is.null(vgm_fit)) {
#       mean_z <- mean(z_train)
#       for (j in seq_len(n_train)) {
#         fold_rows[[j]] <- data.frame(fold = i-1L, obs_idx = train_idx[j]-1L,
#                                      role = "train", krig_pred = mean_z, krig_var = 0)
#       }
#       fold_rows[[n_train+1L]] <- data.frame(fold = i-1L, obs_idx = i-1L,
#                                              role = "test", krig_pred = mean_z, krig_var = 0)
#     } else {
#       for (j in seq_len(n_train)) {
#         tryCatch({
#           capture.output(
#             ok_j <- krige(fml_target, train_sp[-j, ], train_sp[j, ],
#                           model = vgm_fit$var_model),
#             type = "output"
#           )
#           fold_rows[[j]] <- data.frame(fold = i-1L, obs_idx = train_idx[j]-1L,
#                                        role = "train",
#                                        krig_pred = ok_j$var1.pred, krig_var = ok_j$var1.var)
#         }, error = function(e) {
#           fold_rows[[j]] <<- data.frame(fold = i-1L, obs_idx = train_idx[j]-1L,
#                                         role = "train",
#                                         krig_pred = mean(z_train), krig_var = 0)
#         })
#       }
#       tryCatch({
#         capture.output(
#           ok_test <- krige(fml_target, train_sp, test_sp, model = vgm_fit$var_model),
#           type = "output"
#         )
#         fold_rows[[n_train+1L]] <- data.frame(fold = i-1L, obs_idx = i-1L,
#                                                role = "test",
#                                                krig_pred = ok_test$var1.pred,
#                                                krig_var  = ok_test$var1.var)
#       }, error = function(e) {
#         fold_rows[[n_train+1L]] <- data.frame(fold = i-1L, obs_idx = i-1L,
#                                                role = "test",
#                                                krig_pred = mean(z_train), krig_var = 0)
#       })
#     }
#   }
# 
#   kpr_rows[[i]] <- do.call(rbind, fold_rows)
#   if (i %% 10 == 0 || i == N) message(sprintf("  KpR features fold %d/%d", i, N))
# }

# =============================================================================                        
# KpR features using kpr_nndm_folds                       
# Training features: reuse Stage 1 global LOO kriging (kpr_global_loo_features.csv)                    
# Test feature    : one krige() call per fold using NNDM training subset + global variogram            
# =============================================================================                        
message("\n--- KpR features (NNDM folds) ---")                                                         

kpr_global <- read.csv(kpr_global_loo_path)  # already loaded in Stage 1                               
kpr_rows   <- vector("list", N)                                                                        

for (i in seq_len(N)) {                                   
  train_idx <- nndm_kpr$indx_train[[i]]
  test_idx  <- i                                                                                       
  n_train   <- length(train_idx)
  z_train   <- y_true[train_idx]                                                                       
  train_sp  <- pts_sp[train_idx, ]                        
  test_sp   <- pts_sp[test_idx, ]
  
  # --- Training features: from Stage 1 global LOO (obs_idx is 0-based)                                
  tr_rows <- kpr_global[kpr_global$obs_idx %in% (train_idx - 1L), ]                                    
  tr_out  <- data.frame(                                                                               
    fold      = i - 1L,                                   
    obs_idx   = tr_rows$obs_idx,                                                                       
    role      = "train",                                  
    krig_pred = tr_rows$krig_pred,                                                                     
    krig_var  = tr_rows$krig_var                                                                       
  )
  
  # --- Test feature: krige test point from NNDM training subset
  if (n_train < 4 || var(z_train) < 1e-10) {
    kp_test <- mean(z_train); kv_test <- 0                                                             
  } else {                                                                                             
    tryCatch({                                                                                         
      capture.output(                                                                                  
        ok_test <- krige(fml_target, train_sp, test_sp, model = global_vgm),
        type = "output"                                                                                
      )
      kp_test <- ok_test$var1.pred                                                                     
      kv_test <- ok_test$var1.var                         
    }, error = function(e) {
      message(sprintf("  [WARN] KpR fold %d test krige failed: %s", i, conditionMessage(e)))
      kp_test <<- mean(z_train); kv_test <<- 0                                                         
    })                                                                                                 
  }                                                                                                    
  
  te_out <- data.frame(fold = i-1L, obs_idx = i-1L, role = "test",                                     
                       krig_pred = kp_test, krig_var = kv_test)
  
  kpr_rows[[i]] <- rbind(tr_out, te_out)                  
  if (i %% 50 == 0 || i == N) message(sprintf("  KpR features fold %d/%d", i, N))
}    


kpr_df   <- do.call(rbind, kpr_rows)
kpr_path <- file.path(nndm_res_dir, "KpR_features.csv")
write.csv(kpr_df, kpr_path, row.names = FALSE)
message(sprintf("Saved: %s", kpr_path))

message("\n=== Preprocessing complete ===")
message(sprintf("  tabicl_nndm_folds : %s", file.path(nndm_data_dir, "tabicl_nndm_folds.csv")))
message(sprintf("  ok_nndm_folds     : %s", file.path(nndm_data_dir, "ok_nndm_folds.csv")))
message(sprintf("  kpr_nndm_folds    : %s", file.path(nndm_data_dir, "kpr_nndm_folds.csv")))
message(sprintf("  rk_nndm_folds     : %s", file.path(nndm_data_dir, "rk_nndm_folds.csv")))
message(sprintf("  OK_predictions    : %s", ok_path))
message(sprintf("  KpR_features      : %s", kpr_path))
message("\nNext: run main_NNDM_tabICL_for_RK.py with  RUN_MODE = 'nndm'")
