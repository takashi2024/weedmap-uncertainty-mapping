"""
main_3_visualize_MAE_RMSE_NNDM.py

Supplementary figures: ME (count scale) and RMSE (count scale) time-series
across survey dates.  Both metrics use the lognormal mean back-transform
exp(Y_hat + sigma^2/2) - 1 rather than the naive expm1(Y_hat) (median).
Mirrors main_3_visualize_all_performance_NNDM.py (same style as Figure 2).

Reads  : Results_TabICL/data_*/NNDM/model_comparison_*.csv
Writes : Results_TabICL/figures/NNDM_ME_count_timeseries_*.{png,tiff}
         Results_TabICL/figures/NNDM_RMSE_count_timeseries_*.{png,tiff}
         Results_TabICL/figures/NNDM_ME_RMSE_count_combined_*.{png,tiff}
"""

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

# =============================================================================
# Settings
# =============================================================================
ROOT         = r"/Users/takashi/LocalAnalysis/WeedMap"
RESULTS_ROOT = os.path.join(ROOT, "Results_TabICL")
DATE_MIN     = "20250414"
DATE_MAX     = "20250602"

MODEL_MAP = {
    "OK":             "OK",
    "Vanilla TabPFN": "TabICL",
    "KpR TabPFN":     "TabICL-KpR",
    "RK":             "TabICL-RK",
}
PLOT_MODEL_GROUPS = ["OK", "TabICL", "TabICL-KpR", "TabICL-RK"]

COLORS = {
    "OK":         "#4477AA",
    "TabICL-KpR": "#cc6699",
    "TabICL":     "#009999",
    "TabICL-RK":  "#e6b800",
}

# =============================================================================
# 1. Load model_comparison CSVs
# =============================================================================
pattern = os.path.join(RESULTS_ROOT, "data_*", "NNDM", "model_comparison_*.csv")
files   = sorted(glob.glob(pattern))
if not files:
    raise FileNotFoundError(f"No files found:\n{pattern}")

dfs = []
for fp in files:
    m = re.search(r"data_(\d{8})", fp.replace("\\", "/"))
    if not m:
        continue
    target = re.sub(r"^model_comparison_", "",
                    os.path.basename(fp)).replace(".csv", "")
    df = pd.read_csv(fp)
    df["date"]   = m.group(1)
    df["target"] = target
    dfs.append(df)

perf = pd.concat(dfs, ignore_index=True)
perf = perf.rename(columns={"Model": "model_group"})
perf["model_group"] = perf["model_group"].map(MODEL_MAP)
perf = perf.dropna(subset=["model_group"])
perf[["ME_count", "RMSE_count"]] = perf[["ME_count", "RMSE_count"]].apply(pd.to_numeric, errors="coerce")
perf = perf[(perf["date"] >= DATE_MIN) & (perf["date"] <= DATE_MAX)].copy()
perf["date_dt"] = pd.to_datetime(perf["date"], format="%Y%m%d")
perf = perf.sort_values(["target", "model_group", "date_dt"]).reset_index(drop=True)

out_dir = os.path.join(RESULTS_ROOT, "figures")
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# 2. Helper: draw one time-series panel
# =============================================================================
def draw_panel(ax, df_t, metric, ylabel):
    for mg in PLOT_MODEL_GROUPS:
        g = df_t[df_t["model_group"] == mg]
        if g.empty:
            continue
        ax.plot(g["date_dt"], g[metric],
                marker="o", label=mg,
                color=COLORS[mg], linewidth=1.8, markersize=6)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25, linewidth=0.6)

# =============================================================================
# 3. Per-target figures
# =============================================================================
targets = sorted(perf["target"].dropna().unique())

for tgt in targets:
    df_t = perf[perf["target"] == tgt].copy()

    # --- Individual ME (count scale) figure ---
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_panel(ax, df_t, "ME_count", "ME (count scale, plants m$^{-2}$)")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in (".png", ".tiff"):
        fig.savefig(os.path.join(out_dir, f"NNDM_ME_count_timeseries_{tgt}{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ME_count figure for {tgt}")

    # --- Individual RMSE (count scale) figure ---
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_panel(ax, df_t, "RMSE_count", "RMSE (count scale, plants m$^{-2}$)")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in (".png", ".tiff"):
        fig.savefig(os.path.join(out_dir, f"NNDM_RMSE_count_timeseries_{tgt}{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved RMSE_count figure for {tgt}")

    # --- Combined 2-row figure (ME top, RMSE bottom) ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    draw_panel(axes[0], df_t, "ME_count",   "ME (count scale, plants m$^{-2}$)")
    draw_panel(axes[1], df_t, "RMSE_count", "RMSE (count scale, plants m$^{-2}$)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set_xlabel("")
    fig.tight_layout()
    for ext in (".png", ".tiff"):
        fig.savefig(os.path.join(out_dir, f"NNDM_ME_RMSE_count_combined_{tgt}{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined ME+RMSE (count scale) figure for {tgt}")
