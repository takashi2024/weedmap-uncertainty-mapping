"""
main_prob_analysis_NNDM_budget_overview.py

Extended budget-level summary for NNDM LOO CV probability analysis.

Reads  : Results/NNDM_prob/NNDM_{model}_topb_T10.csv   (score_type = prob_p / det_nhat)
Writes : Results/NNDM_prob/NNDM_budget_recall_table.csv
         Results/NNDM_prob/NNDM_budget_precision_table.csv
         Results/NNDM_prob/NNDM_budget_overview.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# =========================
# Settings
# =========================
ROOT = r"/Users/takashi/LocalAnalysis/WeedMap/ForGithub/Results_TabICL/NNDM_prob"

TARGET = "log1p_Chenopodium_Count"
T      = 10
TAU_MIN = 0.01

TOPB_FILES = {
    "KpR":     os.path.join(ROOT, f"NNDM_KpR_{TARGET}_topb_T{T}.csv"),
    "RK":      os.path.join(ROOT, f"NNDM_RK_{TARGET}_topb_T{T}.csv"),
    "Vanilla": os.path.join(ROOT, f"NNDM_Vanilla_TabPFN_{TARGET}_topb_T{T}.csv"),
    "OK":      os.path.join(ROOT, f"NNDM_OK_{TARGET}_topb_T{T}.csv"),
}

METHOD_ORDER = ["KpR", "RK", "Vanilla", "OK"]

BUDGET_GRID = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

BUDGET_VLINES = [0.10, 0.20, 0.50]   # reference lines on figure

COLORS = {
    "KpR":     "#cc6699",
    "RK":      "#e6b800",
    "Vanilla": "#009999",
    "OK":      "#222222",
}

# =========================
# Helpers
# =========================
def _interp_y_at_x(x, y, x0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return np.nan
    tmp = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)["y"].max()
    x = tmp["x"].to_numpy()
    y = tmp["y"].to_numpy()
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x0 < x.min() or x0 > x.max():
        return np.nan
    return float(np.interp(x0, x, y))


def load_topb(files: dict, score_type: str) -> pd.DataFrame:
    """Load and concatenate top-b CSVs, filtered to one score_type."""
    frames = []
    for method, fp in files.items():
        if not os.path.exists(fp):
            print(f"[WARN] Missing: {fp}")
            continue
        df = pd.read_csv(fp)
        df["algorithm"] = method
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[out["score_type"].astype(str) == score_type].copy()
    for col in ["b", "sprayed_frac", "recall", "precision", "prevalence"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["date"] = out["date"].astype(str)
    return out


def mean_at_budgets(df: pd.DataFrame, x_col: str, metric: str,
                    budget_grid: list, prev_gt0: bool = True) -> pd.DataFrame:
    """
    For each (date, algorithm), interpolate metric at each budget.
    Return mean across dates (optionally only prevalence > 0).
    """
    if prev_gt0 and "prevalence" in df.columns:
        df = df[df["prevalence"] > 0].copy()

    rows = []
    for (date, alg), g in df.groupby(["date", "algorithm"]):
        g = g.sort_values(x_col)
        for b in budget_grid:
            val = _interp_y_at_x(g[x_col], g[metric], b)
            rows.append({"date": date, "algorithm": alg, "budget": b, metric: val})

    long = pd.DataFrame(rows)
    mean_df = (long.groupby(["algorithm", "budget"])[metric]
                   .mean()
                   .reset_index())
    return mean_df


def make_wide_table(mean_prob: pd.DataFrame, mean_det: pd.DataFrame,
                    metric: str, method_order: list) -> pd.DataFrame:
    """Combine PROB and DET into wide format: rows=budget, cols=algorithm×rule."""
    p = mean_prob.rename(columns={metric: "val"}).copy()
    p["col"] = p["algorithm"] + "_PROB"
    d = mean_det.rename(columns={metric: "val"}).copy()
    d["col"] = d["algorithm"] + "_DET"

    combined = pd.concat([p, d], ignore_index=True)

    # desired column order: KpR_PROB, KpR_DET, RK_PROB, RK_DET, ...
    col_order = []
    for m in method_order:
        col_order += [f"{m}_PROB", f"{m}_DET"]

    wide = combined.pivot(index="budget", columns="col", values="val")
    # keep only columns that exist
    col_order = [c for c in col_order if c in wide.columns]
    wide = wide[col_order].reset_index()
    wide.insert(0, "budget_pct", (wide["budget"] * 100).astype(int).astype(str) + "%")
    return wide


# =========================
# Load data
# =========================
df_prob = load_topb(TOPB_FILES, score_type="prob_p")
df_det  = load_topb(TOPB_FILES, score_type="det_nhat")

# Mean curves for figure (continuous, all b values)
def mean_curve(df, x_col, metric, method_order, prev_gt0=True):
    if prev_gt0 and "prevalence" in df.columns:
        df = df[df["prevalence"] > 0].copy()
    return (df.groupby(["algorithm", x_col])[metric]
              .mean()
              .reset_index())

recall_prob_curve    = mean_curve(df_prob, "b", "recall",    METHOD_ORDER)
precision_prob_curve = mean_curve(df_prob, "b", "precision", METHOD_ORDER)
recall_det_curve     = mean_curve(df_det,  "b", "recall",    METHOD_ORDER)
precision_det_curve  = mean_curve(df_det,  "b", "precision", METHOD_ORDER)

# =========================
# Summary tables at BUDGET_GRID
# =========================
recall_prob_mean    = mean_at_budgets(df_prob, "b", "recall",    BUDGET_GRID)
recall_det_mean     = mean_at_budgets(df_det,  "b", "recall",    BUDGET_GRID)
precision_prob_mean = mean_at_budgets(df_prob, "b", "precision", BUDGET_GRID)
precision_det_mean  = mean_at_budgets(df_det,  "b", "precision", BUDGET_GRID)

recall_wide    = make_wide_table(recall_prob_mean,    recall_det_mean,    "recall",    METHOD_ORDER)
precision_wide = make_wide_table(precision_prob_mean, precision_det_mean, "precision", METHOD_ORDER)

# Print tables
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
print("\n=== RECALL (mean across 7 dates with prevalence > 0) ===")
print(recall_wide.to_string(index=False))
print("\n=== PRECISION (mean across 7 dates with prevalence > 0) ===")
print(precision_wide.to_string(index=False))

# Save tables
out_rec  = os.path.join(ROOT, "NNDM_budget_recall_table.csv")
out_prec = os.path.join(ROOT, "NNDM_budget_precision_table.csv")
recall_wide.to_csv(out_rec,  index=False)
precision_wide.to_csv(out_prec, index=False)
print(f"\nSaved: {out_rec}")
print(f"Saved: {out_prec}")

# =========================
# Figure: 2x2 (metric x rule)
# =========================
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)

panel_data = [
    (axes[0, 0], recall_prob_curve,    "recall",    "PROB (top-b by P(N>T))", True),
    (axes[0, 1], recall_det_curve,     "recall",    "DET  (top-b by n̂)",      False),
    (axes[1, 0], precision_prob_curve, "precision", "PROB (top-b by P(N>T))", True),
    (axes[1, 1], precision_det_curve,  "precision", "DET  (top-b by n̂)",      False),
]

for ax, curve_df, metric, rule_label, show_ylabel in panel_data:
    for method in METHOD_ORDER:
        sub = curve_df[curve_df["algorithm"] == method].sort_values("b")
        if sub.empty:
            continue
        ax.plot(sub["b"], sub[metric],
                color=COLORS[method], linewidth=2.0, label=method)

    for bv in BUDGET_VLINES:
        ax.axvline(bv, linestyle="--", linewidth=0.8, color="grey", alpha=0.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(rule_label, fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Sprayed fraction (budget)", fontsize=10)
    if show_ylabel:
        ax.set_ylabel(metric.capitalize(), fontsize=11)

# Row labels on the right
for row_idx, label in enumerate(["Recall", "Precision"]):
    axes[row_idx, 1].annotate(
        label, xy=(1.02, 0.5), xycoords="axes fraction",
        fontsize=13, fontweight="bold", va="center", rotation=-90
    )

# Shared legend below figure
legend_handles = [
    mlines.Line2D([], [], color=COLORS[m], linewidth=2, label=m)
    for m in METHOD_ORDER
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    fontsize=11,
    frameon=True,
    bbox_to_anchor=(0.5, 0.01)
)

fig.suptitle(
    f"NNDM LOO — Recall & Precision vs Sprayed fraction | T={T} | mean over 7 dates (prev>0)",
    fontsize=13, y=0.99
)

plt.tight_layout(rect=[0, 0.06, 0.97, 0.97])

out_fig = os.path.join(ROOT, "NNDM_budget_overview.png")
plt.savefig(out_fig, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_fig}")
