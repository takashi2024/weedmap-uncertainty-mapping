"""
main_NNDM_tabICL_for_RK.py

TabICL version of main_NNDM_tabFPN_for_RK.py.
Writes LOO residuals to tabicl_loo_residuals.csv / kpr_tabicl_loo_residuals.csv
so that TabPFN 2.5 residuals (tabfpn_loo_residuals.csv) are preserved unchanged.

RUN_MODE = "loo"
    Stage 0 — phi estimation residuals (run BEFORE main_NNDM_preprocessing.R Stage 2):
    - LOO TabICL  → tabicl_loo_residuals.csv
    - LOO KpR     → kpr_tabicl_loo_residuals.csv

RUN_MODE = "nndm"
    Stage 2 — NNDM evaluation (run AFTER main_NNDM_preprocessing.R Stage 2):
    - Vanilla NNDM evaluation → Results_TabICL/data_{date}/NNDM/Vanilla/
    - RK data generation      → Results_TabICL/data_{date}/NNDM/RK/
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tabicl import TabICLRegressor


# =============================================================================
# User settings
# =============================================================================
ROOT      = r"/Users/takashi/LocalAnalysis/WeedMap/ForGithub"
DATA_ROOT = os.path.join(ROOT, "Dataset_for_python")
RESULTS   = os.path.join(ROOT, "Results_TabICL")
UAV_ROOT  = os.path.join(ROOT, "data")

date      = "20250602" # all dates
target    = "log1p_Chenopodium_Count"
TOL_M     = 1.0

# "loo"  → Stage 0: compute LOO residuals for phi estimation (run first)
# "nndm" → Stage 2: NNDM evaluation using model-specific fold CSVs (run after R Stage 2)
RUN_MODE  = "nndm" # loo or nndm

# RK residual option (only used in nndm mode):
# False (Option B): outer model predicts its own training points (fast, slight optimism)
# True  (Option A): nested LOO TabICL within training (no leakage, very slow)
USE_LOO_RESIDUALS = False

DEVICE           = "cpu"    # "cpu" | "mps" (Apple Silicon) | "cuda"
N_ESTIMATORS_LOO  = 1       # fast for LOO (fits N times)
N_ESTIMATORS_EVAL = 4       # evaluation ensemble size (TabICL default is 8; 4 matches TabPFN25)
quantiles = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]

def make_tabicl(n_estimators):
    return TabICLRegressor(n_estimators=n_estimators, device=DEVICE)

# =============================================================================
os.chdir(ROOT)

RAW_COUNT_COLS = ["Dicot_Count", "Monocot_Count", "Chenopodium_Count"]
COORD_COLS     = ["Longitude", "Latitude", "x_25832", "y_25832"]
GENERAL_COLS   = ["fold", "ROI_ID"]


def _ensure_roi_prefix(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.upper()
    needs = ~s.str.startswith("ROI_", na=False)
    s.loc[needs] = "ROI_" + s.loc[needs]
    return s


def load_result_csv(date: str) -> pd.DataFrame:
    """Load already-aggregated weed counts + coordinates from per-date result.csv."""
    out_tag = f"{date[2:]}F3mRX"
    path    = os.path.join(UAV_ROOT, out_tag, "result.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path, dtype={"ROI_ID": "string"})
    df["ROI_ID"] = _ensure_roi_prefix(df["ROI_ID"])
    print(f"[load_result_csv] date={date} | N={len(df)} observation points")
    return df


def load_roi_features(date: str) -> pd.DataFrame:
    """Load UAV spectral/texture features per ROI from ROI_features_stacked.csv."""
    out_tag = f"{date[2:]}F3mRX"
    path    = os.path.join(UAV_ROOT, out_tag, "ROI_features_stacked.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    roi = pd.read_csv(path, dtype={"ROI_ID": "string"})
    roi["ROI_ID"] = _ensure_roi_prefix(roi["ROI_ID"])
    roi = roi.drop_duplicates(subset=["ROI_ID"]).copy()
    roi = roi.loc[:, ~roi.columns.duplicated()]
    return roi


def merge_roi_features(df: pd.DataFrame, roi_map: pd.DataFrame) -> pd.DataFrame:
    """Merge UAV spectral features into observation DataFrame (ROI_ID already present)."""
    merged   = df.merge(roi_map, on="ROI_ID", how="left", validate="m:1")
    uav_cols = [c for c in roi_map.columns if c != "ROI_ID"]
    if uav_cols:
        print(f"[ROI merge] mean missing rate: {merged[uav_cols].isna().mean().mean():.3%}")
    return merged


def build_X(df: pd.DataFrame) -> pd.DataFrame:
    log1p_cols  = [c for c in df.columns if c.startswith("log1p_")  and c.endswith("_Count")]
    logeps_cols = [c for c in df.columns if c.startswith("logeps_") and c.endswith("_Count")]
    kpr_cols    = [c for c in df.columns if c.endswith("_feature") or c.endswith("_krig_var")]
    drop = set(RAW_COUNT_COLS + COORD_COLS + GENERAL_COLS + log1p_cols + logeps_cols + kpr_cols)
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    return X.select_dtypes(include=[np.number])


def nested_loo_preds_tabicl(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Option A: nested LOO TabICL — no-leakage residuals (slow)."""
    n   = len(y)
    loo = np.empty(n, dtype=float)
    for i in range(n):
        mask   = np.ones(n, dtype=bool); mask[i] = False
        m      = make_tabicl(N_ESTIMATORS_LOO); m.fit(X.iloc[mask], y[mask])
        loo[i] = float(m.predict(X.iloc[[i]])[0])
        if (i + 1) % 50 == 0:
            print(f"    nested LOO {i+1}/{n}")
    return loo


def nan_guard(X_tr: pd.DataFrame, X_te: pd.DataFrame):
    if X_tr.isna().any().any() or X_te.isna().any().any():
        med = X_tr.median()
        X_tr = X_tr.fillna(med)
        X_te = X_te.fillna(med)
    return X_tr, X_te


# =============================================================================
# Load full dataset (common to both modes)
# =============================================================================
full_df = load_result_csv(date)
roi_map = load_roi_features(date)
full_df = merge_roi_features(full_df, roi_map)

coord_df = full_df[["x_25832", "y_25832",
                     "Longitude" if "Longitude" in full_df.columns else "x_25832",
                     "Latitude"  if "Latitude"  in full_df.columns else "y_25832"]].copy()
coord_df.columns = ["x_25832", "y_25832", "Longitude", "Latitude"]

X_full = build_X(full_df)
y_full = full_df[target].to_numpy()
N      = len(full_df)

if X_full.isna().any().any():
    X_full = X_full.fillna(X_full.median())

if X_full.shape[1] == 0:
    raise ValueError("X_full has 0 columns after dropping. Check build_X().")

nndm_data_dir = os.path.join(DATA_ROOT, f"data_{date}", "NNDM")
os.makedirs(nndm_data_dir, exist_ok=True)

print(f"N={N} | Target: {target} | Date: {date} | RUN_MODE={RUN_MODE}")

# =============================================================================
# RUN_MODE = "loo"  —  Stage 0: LOO residuals for phi estimation
# =============================================================================
if RUN_MODE == "loo":

    # ---- LOO TabICL -----------------------------------------------------------
    print(f"\n[LOO TabICL] N={N} fits...")
    y_pred_tabicl_loo = np.empty(N, dtype=float)
    for i in range(N):
        mask = np.ones(N, dtype=bool); mask[i] = False
        m = make_tabicl(N_ESTIMATORS_LOO)
        m.fit(X_full.iloc[mask], y_full[mask])
        y_pred_tabicl_loo[i] = float(m.predict(X_full.iloc[[i]])[0])
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  TabICL LOO {i+1}/{N}")

    resid_tabicl_loo = y_full - y_pred_tabicl_loo

    tabicl_loo_df = coord_df.copy().reset_index(drop=True)
    tabicl_loo_df.insert(0, "obs_idx", np.arange(N))
    tabicl_loo_df[f"{target}_true"]     = y_full
    tabicl_loo_df[f"{target}_pred_loo"] = y_pred_tabicl_loo
    tabicl_loo_df[f"{target}_resid_loo"] = resid_tabicl_loo

    out_tabicl_loo = os.path.join(nndm_data_dir, "tabicl_loo_residuals.csv")
    tabicl_loo_df.to_csv(out_tabicl_loo, index=False)
    print(f"Saved: {out_tabicl_loo}")

    # ---- LOO KpR --------------------------------------------------------------
    kpr_global_path = os.path.join(nndm_data_dir, "kpr_global_loo_features.csv")
    if not os.path.exists(kpr_global_path):
        raise FileNotFoundError(
            f"Run main_NNDM_preprocessing.R (Stage 1) first.\n"
            f"Missing: {kpr_global_path}"
        )

    kpr_global = pd.read_csv(kpr_global_path).set_index("obs_idx")
    kp_pred_all = kpr_global["krig_pred"].to_numpy()
    kp_var_all  = kpr_global["krig_var"].to_numpy()

    print(f"\n[LOO KpR-TabICL] N={N} fits...")
    y_pred_kpr_loo = np.empty(N, dtype=float)

    for i in range(N):
        mask = np.ones(N, dtype=bool); mask[i] = False

        X_tr = X_full.iloc[mask].copy().reset_index(drop=True)
        X_tr[f"{target}_feature"]  = kp_pred_all[mask]
        X_tr[f"{target}_krig_var"] = kp_var_all[mask]

        X_te = X_full.iloc[[i]].copy().reset_index(drop=True)
        X_te[f"{target}_feature"]  = kp_pred_all[i]
        X_te[f"{target}_krig_var"] = kp_var_all[i]

        X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)
        X_tr, X_te = nan_guard(X_tr, X_te)

        m = make_tabicl(N_ESTIMATORS_LOO)
        m.fit(X_tr, y_full[mask])
        y_pred_kpr_loo[i] = float(m.predict(X_te)[0])

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  KpR LOO {i+1}/{N}")

    resid_kpr_loo = y_full - y_pred_kpr_loo

    kpr_loo_df = coord_df.copy().reset_index(drop=True)
    kpr_loo_df.insert(0, "obs_idx", np.arange(N))
    kpr_loo_df[f"{target}_true"]      = y_full
    kpr_loo_df[f"{target}_pred_loo"]  = y_pred_kpr_loo
    kpr_loo_df[f"{target}_resid_loo"] = resid_kpr_loo

    out_kpr_loo = os.path.join(nndm_data_dir, "kpr_tabicl_loo_residuals.csv")
    kpr_loo_df.to_csv(out_kpr_loo, index=False)
    print(f"Saved: {out_kpr_loo}")

    print("\n[Stage 0 complete]")
    print("Next: run main_NNDM_preprocessing.R  (it will proceed to Stage 2 automatically)")

# =============================================================================
# RUN_MODE = "nndm"  —  Stage 2: NNDM evaluation
# =============================================================================
elif RUN_MODE == "nndm":

    out_vanilla = os.path.join(RESULTS, f"data_{date}", "NNDM", "Vanilla")
    out_rk      = os.path.join(RESULTS, f"data_{date}", "NNDM", "RK")
    os.makedirs(out_vanilla, exist_ok=True)
    os.makedirs(out_rk,      exist_ok=True)

    # ---- Vanilla NNDM evaluation (tabicl_nndm_folds) -------------------------
    tabicl_folds_path = os.path.join(nndm_data_dir, "tabicl_nndm_folds.csv")
    if not os.path.exists(tabicl_folds_path):
        raise FileNotFoundError(
            f"Run main_NNDM_preprocessing.R (Stage 2) first.\n"
            f"Missing: {tabicl_folds_path}"
        )
    tabicl_folds = pd.read_csv(tabicl_folds_path)

    print(f"\n[Vanilla NNDM TabICL] using tabicl_nndm_folds...")
    y_true_van  = np.empty(N, dtype=float)
    y_pred_van  = np.empty(N, dtype=float)
    yq_van      = []

    for fold_i in range(N):
        fold_rows = tabicl_folds[tabicl_folds["fold"] == fold_i]
        train_idx = fold_rows[fold_rows["role"] == "train"]["obs_idx"].to_numpy()
        test_idx  = int(fold_rows[fold_rows["role"] == "test"]["obs_idx"].iloc[0])

        if len(train_idx) == 0:
            raise ValueError(f"Vanilla fold {fold_i}: empty training set.")

        X_tr = X_full.iloc[train_idx].reset_index(drop=True)
        y_tr = y_full[train_idx]
        X_te = X_full.iloc[[test_idx]].reset_index(drop=True)
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)
        X_tr, X_te = nan_guard(X_tr, X_te)

        outer = make_tabicl(N_ESTIMATORS_EVAL)
        outer.fit(X_tr, y_tr)

        pred_test = float(outer.predict(X_te)[0])
        q_list = outer.predict(X_te, output_type="quantiles", alphas=quantiles)
        q_row  = np.asarray(q_list)[0]   # shape (n_quantiles,); q_list is (n_test, n_alphas)

        y_true_van[test_idx] = y_full[test_idx]
        y_pred_van[test_idx] = pred_test
        yq_van.append(q_row)

        if (fold_i + 1) % 50 == 0 or fold_i == 0:
            print(f"  Vanilla fold {fold_i+1}/{N} | n_train={len(train_idx)}")

    r2_van = r2_score(y_true_van, y_pred_van)
    print(f"\nVanilla R² (NNDM LOO) = {r2_van:.4f}")

    pd.DataFrame({f"{target}_true": y_true_van}).to_csv(
        os.path.join(out_vanilla, f"{target}_test.csv"), index=False)
    pd.DataFrame({f"{target}_pred": y_pred_van}).to_csv(
        os.path.join(out_vanilla, f"{target}_predictions.csv"), index=False)

    yq_mat = np.vstack(yq_van)   # shape (N, n_quantiles)
    pd.DataFrame(yq_mat, columns=[f"q{v}" for v in quantiles]).to_csv(
        os.path.join(out_vanilla, f"{target}_predictions_quantiles.csv"), index=False)

    print(f"Saved Vanilla outputs to: {out_vanilla}")

    # ---- RK data generation (rk_nndm_folds) ----------------------------------
    rk_folds_path = os.path.join(nndm_data_dir, "rk_nndm_folds.csv")
    if not os.path.exists(rk_folds_path):
        raise FileNotFoundError(
            f"Run main_NNDM_preprocessing.R (Stage 2) first.\n"
            f"Missing: {rk_folds_path}"
        )
    rk_folds = pd.read_csv(rk_folds_path)

    print(f"\n[RK data generation TabICL] using rk_nndm_folds...")

    for fold_i in range(N):
        fold_rows = rk_folds[rk_folds["fold"] == fold_i]
        train_idx = fold_rows[fold_rows["role"] == "train"]["obs_idx"].to_numpy()
        test_idx  = int(fold_rows[fold_rows["role"] == "test"]["obs_idx"].iloc[0])

        if len(train_idx) == 0:
            raise ValueError(f"RK fold {fold_i}: empty training set.")

        X_tr = X_full.iloc[train_idx].reset_index(drop=True)
        y_tr = y_full[train_idx]
        X_te = X_full.iloc[[test_idx]].reset_index(drop=True)
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)
        X_tr, X_te = nan_guard(X_tr, X_te)

        outer     = make_tabicl(N_ESTIMATORS_EVAL)
        outer.fit(X_tr, y_tr)
        pred_test = float(outer.predict(X_te)[0])

        if USE_LOO_RESIDUALS:
            pred_train = nested_loo_preds_tabicl(X_tr, y_tr)
        else:
            pred_train = outer.predict(X_tr).astype(float)
        resid = y_tr - pred_train

        tr_out = coord_df.iloc[train_idx].copy().reset_index(drop=True)
        tr_out[f"{target}_true"]      = y_tr
        tr_out[f"{target}_pred_LOO"]  = pred_train
        tr_out[f"{target}_resid_LOO"] = resid
        tr_out.to_csv(os.path.join(out_rk, f"{target}_train_residLOO_fold{fold_i}.csv"),
                      index=False)

        te_out = coord_df.iloc[[test_idx]].copy().reset_index(drop=True)
        te_out[f"{target}_true"] = y_full[test_idx]
        te_out[f"{target}_pred"] = pred_test
        te_out.to_csv(os.path.join(out_rk, f"{target}_test_true_pred_fold{fold_i}.csv"),
                      index=False)

        if (fold_i + 1) % 50 == 0 or fold_i == 0:
            print(f"  RK fold {fold_i+1}/{N} | n_train={len(train_idx)}")

    print(f"Saved per-fold RK inputs to: {out_rk}")
    print("\n[Stage 2 complete]")
    print("Next steps:")
    print("  1. main_NNDM_KpR_tabICL.py")
    print("  2. datasets_NNDM_RK_tabICL.R")
    print("  3. evaluate_NNDM_results_tabICL.py")

else:
    raise ValueError(f"Unknown RUN_MODE='{RUN_MODE}'. Use 'loo' or 'nndm'.")
