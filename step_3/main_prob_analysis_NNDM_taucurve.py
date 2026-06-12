"""
main_prob_analysis_NNDM_taucurve.py

Plot recall and precision as a function of the probability threshold τ,
faceted by survey date (one panel per date, same layout as the sprayed-fraction figures).

Reads  : Results_TabICL/NNDM_prob/NNDM_{model}_log1p_Chenopodium_Count_taucurve_T10.csv
Writes : Results_TabICL/NNDM_prob/NNDM_PROB_recall_vs_tau_budgets.png/.tiff
         Results_TabICL/NNDM_prob/NNDM_PROB_precision_vs_tau_budgets.png/.tiff
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# =============================================================================
# Settings
# =============================================================================
ROOT     = r"/Users/takashi/LocalAnalysis/WeedMap"
PROB_DIR = os.path.join(ROOT, "Results_TabICL", "NNDM_prob")
TARGET   = "log1p_Chenopodium_Count"
T        = 10
NCOLS    = 4   # panels per row (last slot in final row used for legend)

# Map from taucurve 'method' value -> display label
METHOD_MAP = {
    "Vanilla_TabPFN_NNDM": "TabICL_NNDM",
    "KpR_NNDM":            "TabICL-KpR_NNDM",
    "RK_NNDM":             "TabICL-RK_NNDM",
    "OK_NNDM":             "OK_NNDM",
}
METHOD_ORDER = ["TabICL-KpR_NNDM", "TabICL-RK_NNDM", "TabICL_NNDM", "OK_NNDM"]
LEGEND_LABELS = {
    "TabICL_NNDM":     "TabICL_NNDM",
    "TabICL-KpR_NNDM": "TabICL-KpR_NNDM",
    "TabICL-RK_NNDM":  "TabICL-RK_NNDM",
    "OK_NNDM":         "OK_NNDM",
}
COLORS = {
    "TabICL_NNDM":     "#009999",
    "TabICL-KpR_NNDM": "#0000ff",   # blue, matching reference figure
    "TabICL-RK_NNDM":  "#ff7f0e",   # orange
    "OK_NNDM":         "#d62728",   # red
}

# =============================================================================
# Load all taucurve files
# =============================================================================
pattern = os.path.join(PROB_DIR, f"NNDM_*_{TARGET}_taucurve_T{T}.csv")
files   = sorted(glob.glob(pattern))
if not files:
    raise FileNotFoundError(f"No taucurve files found matching:\n  {pattern}")

dfs = []
for fp in files:
    df = pd.read_csv(fp)
    df["label"] = df["method"].map(METHOD_MAP)
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

# Drop dates with no positives (prevalence = 0)
all_df = all_df[all_df["prevalence"] > 0].copy()
dates   = sorted(all_df["date"].unique())
n_dates = len(dates)
print(f"Dates with prevalence > 0: {dates}  (n={n_dates})")


# =============================================================================
# Helper: draw one faceted figure
# =============================================================================
def make_facet_figure(metric: str, ylabel: str, out_base: str):
    n_panels = n_dates
    n_rows   = math.ceil((n_panels + 1) / NCOLS)   # +1 for legend slot
    total_slots = n_rows * NCOLS

    fig, axes = plt.subplots(n_rows, NCOLS,
                             figsize=(NCOLS * 3.5, n_rows * 3.2),
                             sharey=True, sharex=True)
    axes_flat = axes.flatten()

    # vertical budget reference lines (matching reference figure style)
    budget_lines = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for i, date in enumerate(dates):
        ax = axes_flat[i]
        sub = all_df[all_df["date"] == date]
        prev = sub["prevalence"].iloc[0]

        for vline in budget_lines:
            ax.axvline(vline, color="steelblue", linestyle="--",
                       linewidth=0.7, alpha=0.5, zorder=1)

        for label in METHOD_ORDER:
            grp = sub[sub["label"] == label].sort_values("tau")
            if grp.empty:
                continue
            ax.plot(grp["tau"], grp[metric],
                    color=COLORS[label], linewidth=1.8, zorder=2)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{date} (prev={prev:.3f})", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(False)

    # Legend in the last slot
    legend_ax = axes_flat[n_dates]
    legend_ax.axis("off")
    handles = [
        mlines.Line2D([], [], color=COLORS[m], linewidth=2.0,
                      label=LEGEND_LABELS[m])
        for m in METHOD_ORDER
    ]
    legend_ax.legend(handles=handles, loc="center", fontsize=9,
                     frameon=True, title="Model", title_fontsize=9)

    # Hide any remaining empty slots
    for j in range(n_dates + 1, total_slots):
        axes_flat[j].axis("off")

    # Shared axis labels
    fig.text(0.5, 0.01, "Probability threshold τ", ha="center", fontsize=11)
    fig.text(0.01, 0.5, ylabel, va="center", rotation="vertical", fontsize=11)

    fig.suptitle(
        f"[NNDM LOO | PROB] Chenopodium (T={T}): {ylabel} vs τ  |  τ ≥ 0.01",
        fontsize=11, y=1.01
    )
    plt.tight_layout(rect=[0.03, 0.03, 1, 1])

    fig.savefig(out_base + ".png",  dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".tiff", dpi=300, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close()
    print(f"Saved: {out_base}.png")


# =============================================================================
# Produce recall and precision figures
# =============================================================================
make_facet_figure(
    "recall", "Recall",
    os.path.join(PROB_DIR, "NNDM_PROB_recall_vs_tau_budgets")
)
make_facet_figure(
    "precision", "Precision",
    os.path.join(PROB_DIR, "NNDM_PROB_precision_vs_tau_budgets")
)
