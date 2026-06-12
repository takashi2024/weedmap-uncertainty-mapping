"""
main_prob_analysis_NNDM_RK.py

Probability analysis for Regression Kriging (RK) under NNDM LOO CV.
Reads  : Results/data_{date}/NNDM/RK/{target}_RK_predictions.csv
           columns: true, pred_rk, var_ok  (log scale)
         Results/data_{date}/NNDM/RK/{target}_RK_quantiles.csv  (optional)
           columns: q0.001 … q0.999

Probabilistic score: Normal distribution  P(Y > log1p(T)) from pred_rk + var_ok.
Writes : Results/NNDM_prob/NNDM_RK_{target}_runs_with_R2_T{T}.csv
         Results/NNDM_prob/NNDM_RK_{target}_taucurve_T{T}.csv
         Results/NNDM_prob/NNDM_RK_{target}_detpoint_T{T}.csv
         Results/NNDM_prob/NNDM_RK_{target}_topb_T{T}.csv
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, average_precision_score, roc_auc_score, brier_score_loss
from scipy.stats import norm

# =========================================================
# Settings
# =========================================================
ROOT         = r"/Users/takashi/LocalAnalysis/WeedMap"
RESULTS_ROOT = os.path.join(ROOT, "Results_TabICL")
OUT_DIR      = os.path.join(RESULTS_ROOT, "NNDM_prob")

T_COUNT       = 10
TARGET_FILTER = {"log1p_Chenopodium_Count"}
TAU_GRID      = np.round(np.linspace(0, 1, 101), 3)
B_GRID        = np.round(np.linspace(0.01, 1.00, 100), 3)

os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(ROOT)

# =========================================================
# Helpers (identical logic to original main_prob_analysis_RK.py)
# =========================================================

def parse_date_from_path(path: str):
    m = re.search(r"data_(\d{8})", path.replace("\\", "/"))
    return m.group(1) if m else "NA"


def metrics_from_confusion(tp, fp, tn, fn):
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    spec      = tn / (tn + fp + eps)
    acc       = (tp + tn) / (tp + fp + tn + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    bal_acc   = 0.5 * (recall + spec)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(spec),
        "accuracy": float(acc),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
    }


def confusion_from_mask(y_true_bin, spray_mask):
    y = np.asarray(y_true_bin, dtype=int)
    pred = np.asarray(spray_mask, dtype=bool).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def point_rule_det_at_T(y_true_bin, n_hat, T_count):
    spray = (np.asarray(n_hat, dtype=float) >= float(T_count))
    tp, fp, tn, fn = confusion_from_mask(y_true_bin, spray)
    met = metrics_from_confusion(tp, fp, tn, fn)
    return {
        "thr": float(T_count),
        "sprayed_frac": float(np.mean(spray)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        **met
    }


def topb_curve(y_true_bin, score, b_grid=B_GRID):
    y = np.asarray(y_true_bin, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s) & np.isfinite(y)
    y = y[ok]; s = s[ok]
    order = np.argsort(-s)
    y_sorted = y[order]
    n = len(y_sorted)
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    n_pos = int(np.sum(y_sorted == 1))
    n_neg = int(np.sum(y_sorted == 0))
    rows = []
    for b in b_grid:
        m = max(1, int(np.ceil(float(b) * n)))
        tp = int(tp_cum[m-1])
        fp = int(fp_cum[m-1])
        tn = n_neg - fp
        fn = n_pos - tp
        met = metrics_from_confusion(tp, fp, tn, fn)
        rows.append({"b": float(b), "sprayed_frac": float(m / n),
                     "tp": tp, "fp": fp, "tn": tn, "fn": fn, **met})
    return pd.DataFrame(rows)


def prob_exceed_normal_log(mu_log, var_log, t_count):
    """p = P(Y_log > log1p(t_count)) assuming Normal(mu_log, var_log)."""
    y_thresh = float(np.log1p(t_count))
    mu  = np.asarray(mu_log, dtype=float)
    var = np.asarray(var_log, dtype=float)
    var = np.where(var < 0, np.nan, var)
    sigma = np.sqrt(var)
    sigma = np.where(~np.isfinite(sigma) | (sigma <= 0), 1e-12, sigma)
    z = (y_thresh - mu) / sigma
    p = 1.0 - norm.cdf(z)
    return np.clip(p, 0.0, 1.0)


def confusion_from_threshold(y_true_bin, p, tau):
    y = np.asarray(y_true_bin, dtype=int)
    pred = (np.asarray(p, dtype=float) >= float(tau)).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


# =========================================================
# Scan NNDM RK files
# =========================================================
target_name = list(TARGET_FILTER)[0]

rk_pred_files = sorted(glob.glob(
    os.path.join(RESULTS_ROOT, "data_*", "NNDM", "RK",
                 f"{target_name}_RK_predictions.csv")
))
print(f"Found RK NNDM prediction files: {len(rk_pred_files)}")

all_runs, all_tau, all_det, all_topb = [], [], [], []

for fp in rk_pred_files:
    date    = parse_date_from_path(fp)
    run_dir = os.path.dirname(fp)

    df = pd.read_csv(fp)
    needed = {"true", "pred_rk", "var_ok"}
    if not needed.issubset(df.columns):
        print(f"  [SKIP] {date}: missing columns {needed - set(df.columns)}")
        continue

    y_true_log = df["true"].to_numpy(dtype=float)
    mu_log     = df["pred_rk"].to_numpy(dtype=float)
    var_log    = df["var_ok"].to_numpy(dtype=float)

    r2 = float(r2_score(y_true_log, mu_log))

    n_true = np.expm1(y_true_log)
    y_bin  = (n_true > float(T_COUNT)).astype(int)
    prevalence = float(y_bin.mean())

    p     = prob_exceed_normal_log(mu_log, var_log, t_count=T_COUNT)
    n_hat = np.expm1(mu_log)

    summ = {
        "prevalence": prevalence,
        "Brier":   float(brier_score_loss(y_bin, p)),
        "PR_AUC":  np.nan,
        "ROC_AUC": np.nan,
        "n":       int(len(y_bin)),
    }
    if y_bin.min() != y_bin.max():
        summ["PR_AUC"]  = float(average_precision_score(y_bin, p))
        summ["ROC_AUC"] = float(roc_auc_score(y_bin, p))

    # tau-curve
    rows = []
    for tau in TAU_GRID:
        spray = (p >= float(tau))
        tp, fp, tn, fn = confusion_from_mask(y_bin, spray)
        met = metrics_from_confusion(tp, fp, tn, fn)
        rows.append({"tau": float(tau), "sprayed_frac": float(np.mean(spray)),
                     "tp": tp, "fp": fp, "tn": tn, "fn": fn, **met})
    tau_df = pd.DataFrame(rows)
    tau_df["date"] = date; tau_df["method"] = "RK_NNDM"
    tau_df["target"] = target_name; tau_df["run_dir"] = run_dir
    tau_df["R2_log_reg"] = r2; tau_df["T_count"] = float(T_COUNT)
    tau_df["prevalence"] = prevalence
    all_tau.append(tau_df)

    # deterministic baseline
    det_point = point_rule_det_at_T(y_bin, n_hat, T_count=T_COUNT)
    det_point.update({"date": date, "method": "RK_NNDM", "target": target_name,
                      "run_dir": run_dir, "R2_log_reg": r2, "T_count": float(T_COUNT),
                      "prevalence": prevalence, "score_type": "det_nhat"})
    all_det.append(det_point)

    # top-b curves
    for bc, stype in [(topb_curve(y_bin, p),     "prob_p"),
                      (topb_curve(y_bin, n_hat),  "det_nhat")]:
        b = bc.copy()
        b["date"] = date; b["method"] = "RK_NNDM"; b["target"] = target_name
        b["run_dir"] = run_dir; b["R2_log_reg"] = r2; b["T_count"] = float(T_COUNT)
        b["prevalence"] = prevalence; b["score_type"] = stype
        all_topb.append(b)

    all_runs.append({
        "date": date, "method": "RK_NNDM", "target": target_name,
        "run_dir": run_dir, "R2_log_reg": r2, "T_count": float(T_COUNT),
        "var_col_used": "var_ok", **summ
    })
    print(f"  {date}: R2={r2:.4f}  prevalence={prevalence:.3f}  n={summ['n']}")

df_runs = pd.DataFrame(all_runs)
df_tau  = pd.concat(all_tau,  ignore_index=True) if all_tau  else pd.DataFrame()
df_det  = pd.DataFrame(all_det)
df_topb = pd.concat(all_topb, ignore_index=True) if all_topb else pd.DataFrame()

for df, suffix in [(df_runs, "runs_with_R2"), (df_tau, "taucurve"),
                   (df_det,  "detpoint"),      (df_topb, "topb")]:
    if df.empty:
        continue
    out = os.path.join(OUT_DIR, f"NNDM_RK_{target_name}_{suffix}_T{T_COUNT}.csv")
    df.sort_values([c for c in ["date", "tau", "b", "score_type"] if c in df.columns],
                   inplace=True, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"Saved: {out}  {df.shape}")
