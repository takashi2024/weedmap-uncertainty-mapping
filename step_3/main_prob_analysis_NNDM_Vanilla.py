"""
main_prob_analysis_NNDM_Vanilla.py

Probability analysis for Vanilla TabPFN under NNDM LOO CV.
Reads  : Results/data_{date}/NNDM/Vanilla/{target}_test.csv
         Results/data_{date}/NNDM/Vanilla/{target}_predictions.csv
         Results/data_{date}/NNDM/Vanilla/{target}_predictions_quantiles.csv
Writes : Results/NNDM_prob/NNDM_Vanilla_TabPFN_{target}_runs_with_R2_T{T}.csv
         Results/NNDM_prob/NNDM_Vanilla_TabPFN_{target}_taucurve_T{T}.csv
         Results/NNDM_prob/NNDM_Vanilla_TabPFN_{target}_detpoint_T{T}.csv
         Results/NNDM_prob/NNDM_Vanilla_TabPFN_{target}_topb_T{T}.csv
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, average_precision_score, roc_auc_score, brier_score_loss

# =========================================================
# Settings
# =========================================================
ROOT         = r"/Users/takashi/LocalAnalysis/WeedMap/ForGithub"
RESULTS_ROOT = os.path.join(ROOT, "Results_TabICL")
OUT_DIR      = os.path.join(RESULTS_ROOT, "NNDM_prob")

T_COUNT       = 10
TARGET_FILTER = {"log1p_Chenopodium_Count"}
TAU_GRID      = np.round(np.linspace(0, 1, 101), 3)
B_GRID        = np.round(np.linspace(0.01, 1.00, 100), 3)

os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(ROOT)

# =========================================================
# Helpers (identical logic to original main_prob_analysis_Vanilla_TabPFN.py)
# =========================================================

def infer_q_levels_from_columns(qdf):
    q_levels = []
    for c in qdf.columns:
        if c.startswith("q"):
            try:
                q_levels.append(float(c[1:]))
            except ValueError:
                pass
    q_levels = sorted(set(q_levels))
    if len(q_levels) < 2:
        raise ValueError("Could not infer quantile levels from columns.")
    return q_levels


def confusion_from_mask(y_true_bin, spray_mask):
    y = np.asarray(y_true_bin, dtype=int)
    pred = np.asarray(spray_mask, dtype=bool).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


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
    m = metrics_from_confusion(tp, fp, tn, fn)
    return {
        "thr": float(T_count),
        "sprayed_frac": float(np.mean(spray)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        **m
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


def parse_date_from_path(path: str):
    m = re.search(r"data_(\d{8})", path.replace("\\", "/"))
    return m.group(1) if m else None


def cdf_interp_prob_exceed(yq, q_levels, y_thresh):
    y = np.asarray(yq, dtype=float)
    q = np.asarray(q_levels, dtype=float)
    order = np.argsort(y)
    y = y[order]; q = q[order]
    if y_thresh <= y[0]:
        F = q[0]
    elif y_thresh >= y[-1]:
        F = q[-1]
    else:
        F = np.interp(y_thresh, y, q)
    return float(1.0 - F)


def probs_from_quantile_df(qdf, q_levels, y_thresh):
    q_cols = [f"q{v}" for v in q_levels]
    missing = [c for c in q_cols if c not in qdf.columns]
    if missing:
        raise ValueError(f"Missing quantile columns: {missing[:10]}")
    YQ = qdf[q_cols].to_numpy(dtype=float)
    return np.array([cdf_interp_prob_exceed(YQ[i, :], q_levels, y_thresh)
                     for i in range(YQ.shape[0])], dtype=float)


def regression_r2(run_dir, target):
    test_fp = os.path.join(run_dir, f"{target}_test.csv")
    pred_fp = os.path.join(run_dir, f"{target}_predictions.csv")
    if not (os.path.exists(test_fp) and os.path.exists(pred_fp)):
        return None
    y_true = pd.read_csv(test_fp).iloc[:, 0].to_numpy(dtype=float)
    y_pred = pd.read_csv(pred_fp).iloc[:, 0].to_numpy(dtype=float)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return None
    return float(r2_score(y_true, y_pred))


def taucurve_from_quantiles(run_dir, target, t_count):
    test_fp = os.path.join(run_dir, f"{target}_test.csv")
    q_fp    = os.path.join(run_dir, f"{target}_predictions_quantiles.csv")
    pred_fp = os.path.join(run_dir, f"{target}_predictions.csv")
    if not (os.path.exists(test_fp) and os.path.exists(q_fp) and os.path.exists(pred_fp)):
        return None

    y_test_log = pd.read_csv(test_fp).iloc[:, 0].to_numpy(dtype=float)
    n_true = np.expm1(y_test_log)
    y_bin = (n_true > float(t_count)).astype(int)

    qdf = pd.read_csv(q_fp)
    q_levels = infer_q_levels_from_columns(qdf)
    p = probs_from_quantile_df(qdf, q_levels=q_levels, y_thresh=float(np.log1p(t_count)))

    y_pred_log = pd.read_csv(pred_fp).iloc[:, 0].to_numpy(dtype=float)
    n_hat = np.expm1(y_pred_log)

    rows = []
    for tau in TAU_GRID:
        tp, fp, tn, fn = confusion_from_threshold(y_bin, p, tau)
        met = metrics_from_confusion(tp, fp, tn, fn)
        rows.append({"tau": float(tau), "sprayed_frac": float((p >= float(tau)).mean()),
                     "tp": tp, "fp": fp, "tn": tn, "fn": fn, **met})
    tau_df = pd.DataFrame(rows)

    summ = {
        "prevalence": float(y_bin.mean()),
        "AP":      float(average_precision_score(y_bin, p)) if len(np.unique(y_bin)) > 1 else np.nan,
        "ROC_AUC": float(roc_auc_score(y_bin, p))           if len(np.unique(y_bin)) > 1 else np.nan,
        "Brier":   float(brier_score_loss(y_bin, p)),
        "n":       int(len(y_bin)),
    }

    det_point   = point_rule_det_at_T(y_bin, n_hat, T_count=t_count)
    bcurve_prob = topb_curve(y_bin, p)
    bcurve_det  = topb_curve(y_bin, n_hat)

    return tau_df, summ, det_point, bcurve_prob, bcurve_det


# =========================================================
# Scan NNDM Vanilla directories
# =========================================================
target_name = list(TARGET_FILTER)[0]

run_dirs = sorted({
    os.path.dirname(p)
    for p in glob.glob(os.path.join(RESULTS_ROOT, "data_*", "NNDM", "Vanilla",
                                    f"{target_name}_test.csv"))
})
print(f"Found Vanilla NNDM run dirs for {target_name}: {len(run_dirs)}")

all_runs, all_tau, all_detpts, all_bcurves = [], [], [], []

for run_dir in run_dirs:
    date = parse_date_from_path(run_dir)
    if date is None:
        continue

    r2 = regression_r2(run_dir, target_name)
    if r2 is None:
        print(f"  [SKIP] {date}: missing predictions for R2")
        continue

    ret = taucurve_from_quantiles(run_dir, target_name, t_count=T_COUNT)
    if ret is None:
        print(f"  [SKIP] {date}: missing files for tau-curve")
        continue

    tau_df, summ, det_point, bcurve_prob, bcurve_det = ret

    run_row = {"date": date, "method": "Vanilla_TabPFN_NNDM", "target": target_name,
               "run_dir": run_dir, "R2_log_reg": r2, "T_count": float(T_COUNT), **summ}

    tau_df = tau_df.copy()
    tau_df["date"] = date; tau_df["method"] = "Vanilla_TabPFN_NNDM"
    tau_df["target"] = target_name; tau_df["run_dir"] = run_dir
    tau_df["R2_log_reg"] = r2; tau_df["T_count"] = float(T_COUNT)
    tau_df["prevalence"] = run_row["prevalence"]

    dp = det_point.copy()
    dp.update({"date": date, "method": "Vanilla_TabPFN_NNDM", "target": target_name,
               "run_dir": run_dir, "R2_log_reg": r2, "T_count": float(T_COUNT),
               "prevalence": run_row["prevalence"]})
    all_detpts.append(dp)

    for bc, stype in [(bcurve_prob, "prob_p"), (bcurve_det, "det_nhat")]:
        b = bc.copy()
        b["date"] = date; b["method"] = "Vanilla_TabPFN_NNDM"; b["target"] = target_name
        b["run_dir"] = run_dir; b["R2_log_reg"] = r2; b["T_count"] = float(T_COUNT)
        b["prevalence"] = run_row["prevalence"]; b["score_type"] = stype
        all_bcurves.append(b)

    all_runs.append(run_row)
    all_tau.append(tau_df)
    print(f"  {date}: R2={r2:.4f}  prevalence={summ['prevalence']:.3f}  n={summ['n']}")

df_runs = pd.DataFrame(all_runs)
df_tau  = pd.concat(all_tau,     ignore_index=True) if all_tau     else pd.DataFrame()
df_det  = pd.DataFrame(all_detpts)
df_b    = pd.concat(all_bcurves, ignore_index=True) if all_bcurves else pd.DataFrame()

for df, suffix in [(df_runs, "runs_with_R2"), (df_tau, "taucurve"),
                   (df_det, "detpoint"),       (df_b,   "topb")]:
    if df.empty:
        continue
    out = os.path.join(OUT_DIR, f"NNDM_Vanilla_TabPFN_{target_name}_{suffix}_T{T_COUNT}.csv")
    df.sort_values([c for c in ["date", "tau", "b"] if c in df.columns],
                   inplace=True, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"Saved: {out}  {df.shape}")
