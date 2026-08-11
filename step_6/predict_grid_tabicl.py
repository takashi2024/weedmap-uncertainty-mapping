"""
predict_grid_tabicl.py
Train TabICL (Vanilla) and KpR on ALL observations per date,
then predict at all 1m grid cells.

Also saves in-sample predictions at training points (for RK residual kriging in predict_grid_RK.R).

Outputs (per date under grid_predictions/{date}/):
  Vanilla_grid.csv      — grid_id, x_25832, y_25832, pred, q0.1, q0.9
  Vanilla_train.csv     — ROI_ID, x_25832, y_25832, y_true, pred_train  (for RK)
  KpR_grid.csv          — grid_id, x_25832, y_25832, pred, q0.1, q0.9

RK is handled by predict_grid_RK.R (reads Vanilla_train.csv → kriging → RK_grid.csv).

Run with:  conda run -n tabpfn python3 step_6/predict_grid_tabicl.py
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from tabicl import TabICLRegressor

# =============================================================================
# Paths
# =============================================================================
ROOT     = "/Users/takashi/LocalAnalysis/WeedMap/ForGithub"
UAV_ROOT = os.path.join(ROOT, "data_20260805_updated")
OUT_ROOT = os.path.join(ROOT, "grid_predictions")

DATE_TAGS = {
    "20250414": "250414F3mRX",
    "20250424": "250424F3mRX",
    "20250430": "250430F3mRX",
    "20250506": "250506F3mRX",
    "20250513": "250513F3mRX",
    "20250520": "250520F3mRX",
    "20250526": "250526F3mRX",
    "20250603": "250603F3mRX",
}

TARGET = "log1p_Chenopodium_Count"
N_ESTIMATORS = 4
DEVICE = "cpu"
QUANTILES = [0.1, 0.9]

# =============================================================================
# Column helpers
# =============================================================================
META_COLS = {"ROI_ID", "Longitude", "Latitude", "Chenopodium_Count",
             "log1p_Chenopodium_Count", "logeps_Chenopodium_Count",
             "x_25832", "y_25832", "fold", "upload_id"}


def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS]


def load_training_data(tag):
    result = pd.read_csv(os.path.join(UAV_ROOT, tag, "result.csv"))
    feats  = pd.read_csv(os.path.join(UAV_ROOT, tag, "ROI_features_stacked.csv"))
    result["ROI_ID"] = result["ROI_ID"].astype(str).str.strip()
    feats["ROI_ID"]  = feats["ROI_ID"].astype(str).str.strip()
    return result.merge(feats, on="ROI_ID")


def load_grid_data(date):
    gf = pd.read_csv(os.path.join(OUT_ROOT, date, "grid_features.csv"))
    ok = pd.read_csv(os.path.join(OUT_ROOT, date, "OK_grid.csv"))
    gf = gf.merge(ok[["grid_id", "ok_pred", "ok_var"]], on="grid_id", how="left")
    return gf


# =============================================================================
# Main loop
# =============================================================================
for date, tag in DATE_TAGS.items():
    out_dir = os.path.join(OUT_ROOT, date)
    print(f"\n=== {date} ({tag}) ===")

    train_csv = os.path.join(UAV_ROOT, tag, "result.csv")
    ok_csv    = os.path.join(out_dir, "OK_grid.csv")
    if not os.path.exists(train_csv):
        print(f"  SKIP — missing {train_csv}"); continue
    if not os.path.exists(ok_csv):
        print(f"  SKIP — missing {ok_csv} (run predict_grid_OK.R first)"); continue

    train   = load_training_data(tag)
    grid_gf = load_grid_data(date)

    feat_cols = get_feature_cols(train)
    print(f"  Training N={len(train)}  Grid M={len(grid_gf)}  Features={len(feat_cols)}")

    X_train = train[feat_cols].values.astype("float32")
    y_train = train[TARGET].values.astype("float32")

    grid_complete = grid_gf.dropna(subset=feat_cols).copy()
    X_grid = grid_complete[feat_cols].values.astype("float32")
    valid_ids = grid_complete["grid_id"].values
    print(f"  Grid cells with complete features: {len(grid_complete)}/{len(grid_gf)}")

    # ── Vanilla TabICL ──────────────────────────────────────────────────────
    print("  Fitting Vanilla TabICL ...")
    model = TabICLRegressor(n_estimators=N_ESTIMATORS, device=DEVICE)
    model.fit(X_train, y_train)

    pred_grid  = model.predict(X_grid)
    quant_grid = model.predict(X_grid, output_type="quantiles", alphas=QUANTILES)

    van_grid_df = pd.DataFrame({
        "grid_id": valid_ids,
        "x_25832": grid_complete["x_25832"].values,
        "y_25832": grid_complete["y_25832"].values,
        "pred":    pred_grid,
        "q0.1":    quant_grid[:, 0],
        "q0.9":    quant_grid[:, 1],
    })
    van_grid_df.to_csv(os.path.join(out_dir, "Vanilla_grid.csv"), index=False)
    print(f"  Saved Vanilla_grid.csv  (pred: {pred_grid.min():.3f} – {pred_grid.max():.3f})")

    pred_train = model.predict(X_train)
    van_train_df = pd.DataFrame({
        "ROI_ID":     train["ROI_ID"].values,
        "x_25832":    train["x_25832"].values,
        "y_25832":    train["y_25832"].values,
        "y_true":     y_train,
        "pred_train": pred_train,
    })
    van_train_df.to_csv(os.path.join(out_dir, "Vanilla_train.csv"), index=False)
    print(f"  Saved Vanilla_train.csv")

    # ── KpR: LOO OK at training + OK at grid as extra features ─────────────
    loo_path = os.path.join(out_dir, "OK_train_loo.csv")
    if os.path.exists(loo_path):
        loo = pd.read_csv(loo_path)
        loo["ROI_ID"] = loo["ROI_ID"].astype(str).str.strip()
        train_kpr = train.merge(loo[["ROI_ID", "ok_loo_pred", "ok_loo_var"]],
                                on="ROI_ID", how="left")
        n_miss = train_kpr["ok_loo_pred"].isna().sum()
        if n_miss > 0:
            print(f"  Warning: {n_miss} training rows missing LOO OK — filling with 0")
            train_kpr[["ok_loo_pred", "ok_loo_var"]] = \
                train_kpr[["ok_loo_pred", "ok_loo_var"]].fillna(0)

        X_train_kpr = np.hstack([
            train_kpr[feat_cols].values.astype("float32"),
            train_kpr[["ok_loo_pred", "ok_loo_var"]].values.astype("float32")
        ])
        X_grid_kpr = np.hstack([
            X_grid,
            grid_complete[["ok_pred", "ok_var"]].values.astype("float32")
        ])

        print("  Fitting KpR TabICL ...")
        kpr = TabICLRegressor(n_estimators=N_ESTIMATORS, device=DEVICE)
        kpr.fit(X_train_kpr, y_train)

        pred_kpr  = kpr.predict(X_grid_kpr)
        quant_kpr = kpr.predict(X_grid_kpr, output_type="quantiles", alphas=QUANTILES)

        kpr_df = pd.DataFrame({
            "grid_id": valid_ids,
            "x_25832": grid_complete["x_25832"].values,
            "y_25832": grid_complete["y_25832"].values,
            "pred":    pred_kpr,
            "q0.1":    quant_kpr[:, 0],
            "q0.9":    quant_kpr[:, 1],
        })
        kpr_df.to_csv(os.path.join(out_dir, "KpR_grid.csv"), index=False)
        print(f"  Saved KpR_grid.csv  (pred: {pred_kpr.min():.3f} – {pred_kpr.max():.3f})")
    else:
        print(f"  SKIP KpR — {loo_path} not found (run predict_grid_OK.R first)")

print("\nDone. Now run predict_grid_RK.R to generate RK_grid.csv files.")
