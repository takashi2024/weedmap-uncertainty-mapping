import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    "font.family":          "sans-serif",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
})

# =========================
# Settings
# =========================
ROOT         = r"/Users/takashi/LocalAnalysis/WeedMap"
RESULTS_ROOT = os.path.join(ROOT, "Results_TabICL")

METRIC   = "R2_log"   # NNDM LOO R² in log-scale (stored as "R2" in source files)
DATE_MIN = "20250414"
DATE_MAX = "20250602"

# Model name mapping: source CSV "Model" value -> canonical label
MODEL_MAP = {
    "OK":             "OK",
    "Vanilla TabPFN": "TabICL",
    "KpR TabPFN":     "TabICL-KpR",
    "RK":             "TabICL-RK",
}

PLOT_MODEL_GROUPS = ["OK", "TabICL", "TabICL-KpR", "TabICL-RK"]

# Color palette consistent with other NNDM figures
COLORS = {
    "OK":          "#4477AA",
    "TabICL-KpR":  "#cc6699",
    "TabICL":      "#009999",
    "TabICL-RK":   "#e6b800",
}

# =========================
# 1) Load model_comparison_{target}.csv from each data_YYYYMMDD/NNDM/ folder
# =========================
pattern = os.path.join(RESULTS_ROOT, "data_*", "NNDM", "model_comparison_*.csv")
files   = sorted(glob.glob(pattern))

if len(files) == 0:
    raise FileNotFoundError(f"No files found with pattern:\n{pattern}")

dfs = []
for fp in files:
    m = re.search(r"data_(\d{8})", fp.replace("\\", "/"))
    if not m:
        continue
    date = m.group(1)

    # Extract target from filename: model_comparison_{target}.csv
    target = re.sub(r"^model_comparison_", "", os.path.basename(fp)).replace(".csv", "")

    df = pd.read_csv(fp)
    df["date"]   = date
    df["target"] = target
    dfs.append(df)

perf = pd.concat(dfs, ignore_index=True)

# Rename columns to match expected format
perf = perf.rename(columns={"Model": "model_group", "R2": METRIC})
perf["model_group"] = perf["model_group"].map(MODEL_MAP)
perf = perf.dropna(subset=["model_group"])   # drop any unrecognised model rows

# Ensure metric is numeric
perf[METRIC] = pd.to_numeric(perf[METRIC], errors="coerce")

# Filter date range
perf = perf[(perf["date"] >= DATE_MIN) & (perf["date"] <= DATE_MAX)].copy()
if perf.empty:
    raise ValueError(f"No rows within date range {DATE_MIN}..{DATE_MAX}")

# Sort for plotting
perf["date_dt"] = pd.to_datetime(perf["date"], format="%Y%m%d")
perf = perf.sort_values(["target", "model_group", "date_dt"]).reset_index(drop=True)

# Save summary table
out_table = os.path.join(
    RESULTS_ROOT, "figures",
    f"NNDM_best_models_by_date_{DATE_MIN}_{DATE_MAX}_{METRIC}.csv"
)
os.makedirs(os.path.dirname(out_table), exist_ok=True)
perf.drop(columns=["date_dt"]).to_csv(out_table, index=False)
print("Saved summary table:", out_table)

# =========================
# 2) Plot: one figure per target, lines = model_group
# =========================
targets = sorted(perf["target"].dropna().unique().tolist())

for tgt in targets:
    df_t = perf[perf["target"] == tgt].copy()

    fig, ax = plt.subplots(figsize=(8, 4))

    for mg in PLOT_MODEL_GROUPS:
        g = df_t[df_t["model_group"] == mg].copy()
        if g.empty:
            continue
        ax.plot(
            g["date_dt"], g[METRIC],
            marker="o", label=mg,
            color=COLORS.get(mg),
            linewidth=1.8, markersize=6,
        )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel(r"$R^2$", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()

    fig_path = os.path.join(
        RESULTS_ROOT, "figures",
        f"NNDM_R2_timeseries_{tgt}_{METRIC}_{DATE_MIN}_{DATE_MAX}.png"
    )
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(fig_path.replace(".png", ".tiff"), dpi=300, bbox_inches="tight")
    print("Saved figure:", fig_path)

plt.show()
