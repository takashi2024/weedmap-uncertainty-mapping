"""
evaluate_NNDM_results_tabICL.py

Compare R², RMSE, ME, and MAE across the four NNDM LOO CV models (TabICL version):
  OK  |  Vanilla TabICL  |  KpR TabICL  |  RK

Mirrors: evaluate_NNDM_results.py (TabPFN V2.5 version)
Reads  : Results_TabICL/data_{date}/NNDM/OK_predictions.csv
         Results_TabICL/data_{date}/NNDM/Vanilla/{target}_test.csv
         Results_TabICL/data_{date}/NNDM/Vanilla/{target}_predictions.csv
         Results_TabICL/data_{date}/NNDM/Vanilla/{target}_predictions_quantiles.csv
         Results_TabICL/data_{date}/NNDM/KpR/{target}_predictions.csv
         Results_TabICL/data_{date}/NNDM/KpR/{target}_predictions_quantiles.csv
         Results_TabICL/data_{date}/NNDM/RK/{target}_RK_predictions.csv
Writes : Results_TabICL/data_{date}/NNDM/model_comparison_{target}.csv
         Results_TabICL/data_{date}/NNDM/model_comparison_{target}.png  (if PLOT=True)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm
from sklearn.metrics import r2_score, mean_squared_error

# =============================================================================
# User settings
# =============================================================================
ROOT   = r"/Users/takashi/LocalAnalysis/WeedMap/ForGithub"
date   = "20250603"
target = "log1p_Chenopodium_Count"
PLOT   = True    # set False to skip the scatter plot
if len(sys.argv) > 1: date = sys.argv[1]

# =============================================================================
os.chdir(ROOT)
base = os.path.join("Results_TabICL", f"data_{date}", "NNDM")


def load_check(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing output file:\n  {path}\n"
            "Run the corresponding preprocessing / model script first."
        )
    return pd.read_csv(path)


# ---- True values (common reference) -----------------------------------------
van_test = load_check(os.path.join(base, "Vanilla", f"{target}_test.csv"))
y_true   = van_test[f"{target}_true"].to_numpy()
N        = len(y_true)
y_count  = np.expm1(y_true)

# ---- Load predictions --------------------------------------------------------
# OK
ok_df   = load_check(os.path.join(base, "OK_predictions.csv"))
ok_df   = ok_df.sort_values("obs_idx").reset_index(drop=True)
if len(ok_df) != N:
    raise ValueError(f"OK_predictions.csv has {len(ok_df)} rows but expected {N}.")
ok_pred = ok_df["ok_pred"].to_numpy()
ok_var  = ok_df["ok_var"].to_numpy()
ok_psi  = ok_df["ok_psi"].to_numpy()
ok_psi_safe = np.nan_to_num(ok_psi, nan=0.0)   # NA folds fall back to SK approx (psi=0)

# Vanilla
van_pred_df = load_check(os.path.join(base, "Vanilla", f"{target}_predictions.csv"))
van_pred    = van_pred_df[f"{target}_pred"].to_numpy()
van_q_df    = load_check(os.path.join(base, "Vanilla", f"{target}_predictions_quantiles.csv"))

# KpR
kpr_pred_df = load_check(os.path.join(base, "KpR", f"{target}_predictions.csv"))
kpr_pred    = kpr_pred_df[f"{target}_pred"].to_numpy()
kpr_q_df    = load_check(os.path.join(base, "KpR", f"{target}_predictions_quantiles.csv"))

# RK
rk_df    = load_check(os.path.join(base, "RK", f"{target}_RK_predictions.csv"))
rk_df    = rk_df.sort_values("obs_idx").reset_index(drop=True)
rk_pred  = rk_df["pred_rk"].to_numpy()
var_ok   = rk_df["var_ok"].to_numpy()

# ---- Sanity check lengths ---------------------------------------------------
for name, arr in [("Vanilla", van_pred), ("KpR", kpr_pred), ("RK", rk_pred)]:
    if len(arr) != N:
        raise ValueError(f"{name} predictions has {len(arr)} rows but expected {N}.")

# --- Mean back-transformation (lognormal correction) ---
#
# The model predictions are on the log1p scale: Y = log(1 + N).
# The naive back-transform expm1(Y_hat) returns the *median* of the
# conditional count distribution, not the mean.  For a log-normally
# distributed variable the conditional mean is:
#
#   E[N | x0] = exp(Y_hat + sigma^2 / 2) - 1
#
# where sigma^2 is the conditional variance on the log scale
# (Webster & Oliver 2007, Section 8.10, equation 8.37).
# Note: W&O use Y = ln(Z), giving E[Z] = exp(Y_hat + sigma^2/2).
# Here Y = ln(1 + N), so N = exp(Y) - 1 and
# E[N] = E[exp(Y)] - 1 = exp(Y_hat + sigma^2/2) - 1.
# The extra "- 1" corrects for the log1p offset.
#
# For Ordinary Kriging the exact formula (W&O eq. 8.38) subtracts the
# Lagrange multiplier psi:  E[N | x0] = exp(Y_hat + sigma^2/2 - psi) - 1.
# psi is extracted by manually solving the OK kriging system in R and
# stored in the ok_psi column of OK_predictions.csv.
# Folds where psi could not be computed (NA) fall back to psi=0 (SK approx).
#
# sigma^2 / back-transform source per model:
#   OK      : lognormal mean exp(Y + sigma^2/2 - psi) - 1
#             sigma^2 = ok_var (kriging variance), psi = Lagrange multiplier
#   RK      : lognormal mean exp(Y + sigma^2/2) - 1
#             sigma^2 = var_ok (residual kriging variance)
#   Vanilla : trapezoidal integration of back-transformed quantiles over [0, 1]
#   KpR     : same as Vanilla

# Quantile columns and probability levels available from TabICL output
_TABICL_Q_COLS = [
    "q0.05","q0.1","q0.15","q0.2","q0.25","q0.3","q0.35","q0.4","q0.45",
    "q0.5","q0.55","q0.6","q0.65","q0.7","q0.75","q0.8","q0.85","q0.9","q0.95"
]
_TABICL_PROBS = np.array([
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
])


def mean_count_from_quantiles(q_df: pd.DataFrame) -> np.ndarray:
    """
    Estimate E[count] via trapezoidal integration of the quantile function
    over the full [0, 1] probability range.

    Central region [0.05, 0.95]: direct trapezoidal integration over the 19
    available TabICL quantile levels.
    Lower tail  [0, 0.05]: Q_count(p=0) = 0  (counts are non-negative).
    Upper tail  [0.95, 1]: Q_log(p=1) linearly extrapolated from q0.90 and
    q0.95 on the log scale, then back-transformed.
    """
    q_log   = q_df[_TABICL_Q_COLS].to_numpy()             # (N, 19), log1p scale
    q_count = np.maximum(np.exp(q_log) - 1.0, 0.0)        # back-transform, clamp >= 0

    # Lower boundary: counts are non-negative
    lower = np.zeros((len(q_df), 1))

    # Upper boundary: linear extrapolation on log scale from q0.90 -> q0.95
    q_log_90 = q_df["q0.9"].to_numpy()
    q_log_95 = q_df["q0.95"].to_numpy()
    q_log_upper = q_log_95 + (q_log_95 - q_log_90)        # one step beyond q0.95
    upper = np.maximum(np.exp(q_log_upper) - 1.0, 0.0).reshape(-1, 1)

    # Full probability grid: [0, 0.05, 0.10, ..., 0.95, 1.0]  (21 points)
    all_probs   = np.concatenate([[0.0], _TABICL_PROBS, [1.0]])
    all_q_count = np.concatenate([lower, q_count, upper], axis=1)   # (N, 21)

    return np.maximum(np.trapezoid(all_q_count, x=all_probs, axis=1), 0.0)


def mean_back_transform(y_log: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """Lognormal mean back-transform: E[N] = exp(Y + sigma^2/2) - 1."""
    return np.maximum(np.exp(y_log + 0.5 * sigma2) - 1.0, 0.0)


ok_mean_count  = np.maximum(
    np.exp(ok_pred + 0.5 * np.maximum(ok_var, 0.0) - ok_psi_safe) - 1.0, 0.0
)
van_mean_count = mean_count_from_quantiles(van_q_df)
kpr_mean_count = mean_count_from_quantiles(kpr_q_df)
rk_mean_count  = mean_back_transform(rk_pred,  np.maximum(var_ok,  0.0))

# ---- Compute metrics --------------------------------------------------------
def metrics(y: np.ndarray, yhat: np.ndarray,
            y_cnt: np.ndarray, yhat_mean_cnt: np.ndarray,
            name: str) -> dict:
    r2         = r2_score(y, yhat)
    rmse_log   = np.sqrt(mean_squared_error(y, yhat))
    mae        = np.mean(np.abs(yhat - y))
    me_log     = np.mean(yhat - y)
    me_count   = np.mean(yhat_mean_cnt - y_cnt)
    rmse_count = np.sqrt(np.mean((yhat_mean_cnt - y_cnt) ** 2))
    return {
        "Model":      name,
        "R2":         round(r2,         4),
        "RMSE_log":   round(rmse_log,   4),
        "ME_log":     round(me_log,     4),
        "MAE":        round(mae,        4),
        "ME_count":   round(me_count,   4),
        "RMSE_count": round(rmse_count, 4),
    }

results = pd.DataFrame([
    metrics(y_true, ok_pred,  y_count, ok_mean_count,  "OK"),
    metrics(y_true, van_pred, y_count, van_mean_count, "Vanilla TabPFN"),   # keep same label as TabPFN25 for downstream compat
    metrics(y_true, kpr_pred, y_count, kpr_mean_count, "KpR TabPFN"),       # keep same label as TabPFN25 for downstream compat
    metrics(y_true, rk_pred,  y_count, rk_mean_count,  "RK"),
])

# ---- Print ------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  NNDM LOO CV (TabICL) — {target}")
print(f"  Date: {date}   N={N}")
print(f"{'='*60}")
print(results.to_string(index=False))
print(f"{'='*60}\n")

# ---- Save CSV ---------------------------------------------------------------
out_csv = os.path.join(base, f"model_comparison_{target}.csv")
results.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

# ---- Optional scatter plot --------------------------------------------------
if PLOT:
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for scripts
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()

        models = [
            ("OK",             ok_pred),
            ("Vanilla TabICL", van_pred),
            ("KpR TabICL",     kpr_pred),
            ("RK",             rk_pred),
        ]

        lims = (min(y_true.min(), ok_pred.min(), van_pred.min(),
                    kpr_pred.min(), rk_pred.min()) - 0.05,
                max(y_true.max(), ok_pred.max(), van_pred.max(),
                    kpr_pred.max(), rk_pred.max()) + 0.05)

        for ax, (name, yhat) in zip(axes, models):
            r2 = r2_score(y_true, yhat)
            ax.scatter(y_true, yhat, alpha=0.4, s=15, color="steelblue")
            ax.plot(lims, lims, "r--", linewidth=1)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
            ax.set_title(f"{name}  (R²={r2:.3f})")

        fig.suptitle(f"NNDM LOO CV (TabICL) — {target}\nDate: {date}  N={N}", fontsize=12)
        plt.tight_layout()

        out_png = os.path.join(base, f"model_comparison_{target}.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved plot: {out_png}")

    except ImportError:
        print("matplotlib not available — skipping plot.")
