"""
main_NNDM_KpR_tabICL.py

KpR (Kriging prior Regression) TabICL under NNDM LOO CV for WeedMap step_2.

For each NNDM LOO fold:
  - Reads pre-computed kriging features from main_NNDM_preprocessing.R
  - Appends them to the base covariate matrix
  - Trains TabICL and predicts the single test point

Mirrors: main_NNDM_KpR.py (TabPFN V2.5 version)
Reads  : Dataset_for_python/data_{date}/NNDM/kpr_nndm_folds.csv
         Results_TabICL/data_{date}/NNDM/KpR_features.csv
Writes : Results_TabICL/data_{date}/NNDM/KpR/
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
ROOT      = r"/Users/takashi/LocalAnalysis/WeedMap"
DATA_ROOT = os.path.join(ROOT, "Dataset_for_python")
RESULTS   = os.path.join(ROOT, "Results_TabICL")
UAV_ROOT  = os.path.join(ROOT, "data")

date      = "20250602"
target    = "log1p_Chenopodium_Count"
quantiles = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
TOL_M     = 1.0
DEVICE       = "cpu"    # safe default on macOS
N_ESTIMATORS = 4        # ensemble size

def make_tabicl():
    return TabICLRegressor(n_estimators=N_ESTIMATORS, device=DEVICE)

# =============================================================================
# Column drop helpers
# =============================================================================
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


def build_X_base(df: pd.DataFrame) -> pd.DataFrame:
    """Base features: drop leakage + coords + ALL KpR columns."""
    log1p_cols  = [c for c in df.columns if c.startswith("log1p_")  and c.endswith("_Count")]
    logeps_cols = [c for c in df.columns if c.startswith("logeps_") and c.endswith("_Count")]
    kpr_cols    = [c for c in df.columns if c.endswith("_feature") or c.endswith("_krig_var")]
    drop = set(RAW_COUNT_COLS + COORD_COLS + GENERAL_COLS + log1p_cols + logeps_cols + kpr_cols)
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    return X.select_dtypes(include=[np.number])


# =============================================================================
# Main
# =============================================================================
os.chdir(ROOT)

# --- Load full dataset
full_df = load_result_csv(date)
roi_map = load_roi_features(date)
full_df = merge_roi_features(full_df, roi_map)

# --- Load KpR-specific NNDM fold assignments
folds_path = os.path.join(DATA_ROOT, f"data_{date}", "NNDM", "kpr_nndm_folds.csv")
if not os.path.exists(folds_path):
    raise FileNotFoundError(f"Run main_NNDM_preprocessing.R (Stage 2) first.\nMissing: {folds_path}")
folds_df = pd.read_csv(folds_path)

# --- Load KpR features (computed per fold in R)
kpr_path = os.path.join(RESULTS, f"data_{date}", "NNDM", "KpR_features.csv")
if not os.path.exists(kpr_path):
    raise FileNotFoundError(f"Run main_NNDM_preprocessing.R first.\nMissing: {kpr_path}")
kpr_all = pd.read_csv(kpr_path)
# Pre-index for fast lookup: dict[fold_i] -> DataFrame indexed by obs_idx
kpr_by_fold = {
    fold_i: grp.set_index("obs_idx")
    for fold_i, grp in kpr_all.groupby("fold")
}

X_base = build_X_base(full_df)
y_full = full_df[target].to_numpy()
N      = len(full_df)

print(f"N={N} | Target: {target} | Date: {date}")
if X_base.shape[1] == 0:
    raise ValueError("X_base has 0 columns after dropping. Check build_X_base().")

# --- Output directory
out_dir = os.path.join(RESULTS, f"data_{date}", "NNDM", "KpR")
os.makedirs(out_dir, exist_ok=True)

# --- NNDM LOO loop
y_true_all = np.empty(N, dtype=float)
y_pred_all = np.empty(N, dtype=float)
yq_all     = []

for fold_i in range(N):
    fold_rows = folds_df[folds_df["fold"] == fold_i]
    train_idx = fold_rows[fold_rows["role"] == "train"]["obs_idx"].to_numpy()
    test_idx  = int(fold_rows[fold_rows["role"] == "test"]["obs_idx"].iloc[0])

    if len(train_idx) == 0:
        raise ValueError(f"Fold {fold_i}: empty training set.")

    # --- Build X_train with KpR features
    kpr_fold  = kpr_by_fold.get(fold_i)
    if kpr_fold is None:
        raise KeyError(f"No KpR features found for fold {fold_i}. Check KpR_features.csv.")

    kpr_train = kpr_fold[kpr_fold["role"] == "train"]
    kpr_test  = kpr_fold[kpr_fold["role"] == "test"]

    # Align kpr_train rows to train_idx order
    try:
        kp_pred_train = kpr_train.loc[train_idx, "krig_pred"].to_numpy()
        kp_var_train  = kpr_train.loc[train_idx, "krig_var"].to_numpy()
    except KeyError as e:
        raise KeyError(f"Fold {fold_i}: missing KpR train entry for obs_idx {e}. "
                       f"Available: {kpr_train.index.tolist()[:10]}")

    try:
        kp_pred_test = float(kpr_test.loc[test_idx, "krig_pred"])
        kp_var_test  = float(kpr_test.loc[test_idx, "krig_var"])
    except KeyError:
        raise KeyError(f"Fold {fold_i}: missing KpR test entry for obs_idx {test_idx}.")

    X_train = X_base.iloc[train_idx].copy().reset_index(drop=True)
    X_train[f"{target}_feature"]  = kp_pred_train
    X_train[f"{target}_krig_var"] = kp_var_train

    X_test = X_base.iloc[[test_idx]].copy().reset_index(drop=True)
    X_test[f"{target}_feature"]   = kp_pred_test
    X_test[f"{target}_krig_var"]  = kp_var_test

    # Align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    # NaN guard
    if X_train.isna().any().any() or X_test.isna().any().any():
        med = X_train.median()
        X_train = X_train.fillna(med)
        X_test  = X_test.fillna(med)

    y_train = y_full[train_idx]

    model = make_tabicl()
    model.fit(X_train, y_train)

    y_true_all[test_idx] = y_full[test_idx]
    y_pred_all[test_idx] = float(model.predict(X_test)[0])

    q_list = model.predict(X_test, output_type="quantiles", alphas=quantiles)
    yq_all.append(np.asarray(q_list)[0])   # q_list is (n_test, n_alphas); take row 0

    if (fold_i + 1) % 50 == 0 or fold_i == 0:
        print(f"  fold {fold_i+1}/{N} done | test_idx={test_idx} | n_train={len(train_idx)}")

# --- Save outputs
r2 = r2_score(y_true_all, y_pred_all)
print(f"\nKpR R2 (NNDM LOO, TabICL) = {r2:.4f}")

pd.DataFrame({f"{target}_true": y_true_all}).to_csv(
    os.path.join(out_dir, f"{target}_test.csv"), index=False)
pd.DataFrame({f"{target}_pred": y_pred_all}).to_csv(
    os.path.join(out_dir, f"{target}_predictions.csv"), index=False)

yq_mat = np.vstack(yq_all)   # shape (N, n_quantiles)
pd.DataFrame(yq_mat, columns=[f"q{v}" for v in quantiles]).to_csv(
    os.path.join(out_dir, f"{target}_predictions_quantiles.csv"), index=False)

print(f"Saved KpR outputs to: {out_dir}")
