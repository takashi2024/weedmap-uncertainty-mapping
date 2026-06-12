"""
main_prob_analysis_NNDM_OK.py

Probability analysis for Ordinary Kriging (OK) under NNDM LOO CV.
Reads  : Results/data_{date}/NNDM/OK_predictions.csv
           columns: obs_idx, ok_pred, ok_var  (log scale)
         Results/data_{date}/NNDM/Vanilla/{target}_test.csv
           first column: true values (shared LOO reference)
         Results/data_{date}/NNDM/{target}_OK_quantiles.csv  (optional)
           columns: q0.001 … q0.999

Probabilistic score: Normal distribution  P(Y > log1p(T)) from ok_pred + ok_var.
Writes : Results/NNDM_prob/NNDM_OK_{target}_runs_with_R2_T{T}.csv
         Results/NNDM_prob/NNDM_OK_{target}_taucurve_T{T}.csv
         Results/NNDM_prob/NNDM_OK_{target}_detpoint_T{T}.csv
         Results/NNDM_prob/NNDM_OK_{target}_topb_T{T}.csv
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
# Helpers (identical logic to original main_prob_analysis_OK.py)
# =========================================================

def parse_date_from_path(path: str):
    m = re.search(r"data_(\d{8})", path.replace("\\", "/"))
    return m.group(1) if m else None


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


def confusion_from_threshold(y_true_bin, p, tau):
    y = np.asarray(y_true_bin, dtype=int)
    pred = (np.asarray(p, dtype=float) >= float(tau)).astype(int)
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
    ok_mask = np.isfinite(s) & np.isfinite(y)
    y = y[ok_mask]; s = s[ok_mask]
    order = np.argsort(-s)
    y_sorted = y[order]
    n = len(y_sorted)
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    rows = []
    for b in b_grid:
        m = max(1, int(np.ceil(float(b) * n)))
        tp = int(tp_cum[m-1])
        fp = int(fp_cum[m-1])
        tn = int(np.sum(y_sorted == 0) - fp)
        fn = int(np.sum(y_sorted == 1) - tp)
        met = metrics_from_confusion(tp, fp, tn, fn)
        rows.append({"b": float(b), "sprayed_frac": float(m / n),
                     "tp": tp, "fp": fp, "tn": tn, "fn": fn, **met})
    return pd.DataFrame(rows)


def prob_exceed_normal_log(mu_log, var_log, t_count):
    y_thresh = float(np.log1p(t_count))
    mu  = np.asarray(mu_log, dtype=float)
    var = np.asarray(var_log, dtype=float)
    var = np.where(var < 0, np.nan, var)
    sigma = np.sqrt(var)
    sigma = np.where(~np.isfinite(sigma) | (sigma <= 0), 1e-12, sigma)
    z = (y_thresh - mu) / sigma
    p = 1.0 - norm.cdf(z)
    return np.clip(p, 0.0, 1.0)


# =========================================================
# Scan NNDM OK files
# =========================================================
target_name = list(TARGET_FILTER)[0]

ok_pred_files = sorted(glob.glob(
    os.path.join(RESULTS_ROOT, "data_*", "NNDM", "OK_predictions.csv")
))
print(f"Found OK NNDM prediction files: {len(ok_pred_files)}")

all_runs, all_tau, all_det, all_topb = [], [], [], []

for fp in ok_pred_files:
    date     = parse_date_from_path(fp)
    nndm_dir = os.path.dirname(fp)   # Results/data_{date}/NNDM/

    ok_df = pd.read_csv(fp)
    if "ok_pred" not in ok_df.columns:
        print(f"  [SKIP] {date}: 'ok_pred' column missing in OK_predictions.csv")
        continue
    ok_df = ok_df.sort_values("obs_idx").reset_index(drop=True)

    mu_log = ok_df["ok_pred"].to_numpy(dtype=float)

    # Variance: prefer ok_var column; fall back to nan (deterministic only)
    if "ok_var" in ok_df.columns:
        var_log = ok_df["ok_var"].to_numpy(dtype=float)
        has_var = True
    else:
        var_log = np.full_like(mu_log, np.nan)
        has_var = False

    # True values from shared Vanilla LOO test file
    true_fp = os.path.join(nndm_dir, "Vanilla", f"{target_name}_test.csv")
    if not os.path.exists(true_fp):
        print(f"  [SKIP] {date}: missing true-value file {true_fp}")
        continue
    y_true_log = pd.read_csv(true_fp).iloc[:, 0].to_numpy(dtype=float)

    if len(mu_log) != len(y_true_log):
        print(f"  [SKIP] {date}: length mismatch OK ({len(mu_log)}) vs true ({len(y_true_log)})")
        continue

    r2 = float(r2_score(y_true_log, mu_log))

    n_true = np.expm1(y_true_log)
    y_bin  = (n_true > float(T_COUNT)).astype(int)
    prevalence = float(y_bin.mean())
    n_hat  = np.expm1(mu_log)

    if has_var:
        p = prob_exceed_normal_log(mu_log, var_log, t_count=T_COUNT)
    else:
        # fall back: deterministic score rescaled to [0,1] — no true probabilistic output
        print(f"  [WARN] {date}: no variance column; using deterministic-only mode")
        p = np.clip(n_hat / (n_hat.max() + 1e-12), 0.0, 1.0)

    summ = {
        "prevalence": prevalence,
        "Brier":   float(brier_score_loss(y_bin, p)),
        "PR_AUC":  np.nan,
        "ROC_AUC": np.nan,
        "n":       int(len(y_bin)),
        "has_var": has_var,
    }
    if y_bin.min() != y_bin.max():
        summ["PR_AUC"]  = float(average_precision_score(y_bin, p))
        summ["ROC_AUC"] = float(roc_auc_score(y_bin, p))

    # tau-curve
    rows = []
    for tau in TAU_GRID:
        tp, fp, tn, fn = confusion_from_threshold(y_bin, p, tau)
        met = metrics_from_confusion(tp, fp, tn, fn)
        rows.append({"tau": float(tau), "sprayed_frac": float((p >= float(tau)).mean()),
                     "tp": tp, "fp": fp, "tn": tn, "fn": fn, **met})
    tau_df = pd.DataFrame(rows)
    tau_df["date"] = date; tau_df["method"] = "OK_NNDM"
    tau_df["target"] = target_name; tau_df["run_dir"] = nndm_dir
    tau_df["R2_log_reg"] = r2; tau_df["T_count"] = float(T_COUNT)
    tau_df["prevalence"] = prevalence
    all_tau.append(tau_df)

    # deterministic baseline
    det_point = point_rule_det_at_T(y_bin, n_hat, T_count=T_COUNT)
    det_point.update({"date": date, "method": "OK_NNDM", "target": target_name,
                      "run_dir": nndm_dir, "R2_log_reg": r2, "T_count": float(T_COUNT),
                      "prevalence": prevalence, "score_type": "det_nhat"})
    all_det.append(det_point)

    # top-b curves
    for bc, stype in [(topb_curve(y_bin, p),    "prob_p"),
                      (topb_curve(y_bin, n_hat), "det_nhat")]:
        b = bc.copy()
        b["date"] = date; b["method"] = "OK_NNDM"; b["target"] = target_name
        b["run_dir"] = nndm_dir; b["R2_log_reg"] = r2; b["T_count"] = float(T_COUNT)
        b["prevalence"] = prevalence; b["score_type"] = stype
        all_topb.append(b)

    all_runs.append({
        "date": date, "method": "OK_NNDM", "target": target_name,
        "run_dir": nndm_dir, "R2_log_reg": r2, "T_count": float(T_COUNT), **summ
    })
    print(f"  {date}: R2={r2:.4f}  prevalence={prevalence:.3f}  n={summ['n']}  var={'yes' if has_var else 'NO'}")

df_runs = pd.DataFrame(all_runs)
df_tau  = pd.concat(all_tau,  ignore_index=True) if all_tau  else pd.DataFrame()
df_det  = pd.DataFrame(all_det)
df_topb = pd.concat(all_topb, ignore_index=True) if all_topb else pd.DataFrame()

for df, suffix in [(df_runs, "runs_with_R2"), (df_tau, "taucurve"),
                   (df_det,  "detpoint"),      (df_topb, "topb")]:
    if df.empty:
        continue
    out = os.path.join(OUT_DIR, f"NNDM_OK_{target_name}_{suffix}_T{T_COUNT}.csv")
    df.sort_values([c for c in ["date", "tau", "b"] if c in df.columns],
                   inplace=True, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"Saved: {out}  {df.shape}")
