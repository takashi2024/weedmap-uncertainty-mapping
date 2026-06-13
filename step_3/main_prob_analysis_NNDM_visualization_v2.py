# main_prob_analysis_NNDM_visualization_v2.py
#
# Visualization for NNDM LOO CV probability analysis results.
# Reads from Results/NNDM_prob/ (outputs of main_prob_analysis_NNDM_*.py scripts).
# Identical plotting logic to main_prob_analysis_visualization_together_v2.py.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Inputs
# =========================
ROOT = r"/Users/takashi/LocalAnalysis/WeedMap/ForGithub/Results_TabICL/NNDM_prob"

TARGET = "log1p_Chenopodium_Count"
T      = 10

# ---- probabilistic (tau-curve) inputs
TAU_BEST_FILES = {
    "OK_NNDM":             os.path.join(ROOT, f"NNDM_OK_{TARGET}_taucurve_T{T}.csv"),
    "Vanilla_TabPFN_NNDM": os.path.join(ROOT, f"NNDM_Vanilla_TabPFN_{TARGET}_taucurve_T{T}.csv"),
    "KpR_NNDM":            os.path.join(ROOT, f"NNDM_KpR_{TARGET}_taucurve_T{T}.csv"),
    "RK_NNDM":             os.path.join(ROOT, f"NNDM_RK_{TARGET}_taucurve_T{T}.csv"),
}

# ---- deterministic (top-b) inputs
TOPB_BEST_FILES = {
    "OK_NNDM":             os.path.join(ROOT, f"NNDM_OK_{TARGET}_topb_T{T}.csv"),
    "Vanilla_TabPFN_NNDM": os.path.join(ROOT, f"NNDM_Vanilla_TabPFN_{TARGET}_topb_T{T}.csv"),
    "KpR_NNDM":            os.path.join(ROOT, f"NNDM_KpR_{TARGET}_topb_T{T}.csv"),
    "RK_NNDM":             os.path.join(ROOT, f"NNDM_RK_{TARGET}_topb_T{T}.csv"),
}

METHOD_ORDER = ["KpR_NNDM", "RK_NNDM", "Vanilla_TabPFN_NNDM", "OK_NNDM"]
BUDGETS  = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
TAU_MIN  = 0.01

# ---- output figures (probabilistic)
OUT_RECALL_PROB       = os.path.join(ROOT, f"NNDM_PROB_recall_vs_sprayedfrac_tauGE{int(TAU_MIN*100)}_budgets.png")
OUT_PREC_PROB         = os.path.join(ROOT, f"NNDM_PROB_precision_vs_sprayedfrac_tauGE{int(TAU_MIN*100)}_budgets.png")
OUT_BOX_R_PROB        = os.path.join(ROOT, f"NNDM_PROB_box_recall_at10pct_tauGE{int(TAU_MIN*100)}.png")
OUT_BOX_P_PROB        = os.path.join(ROOT, f"NNDM_PROB_box_precision_at10pct_tauGE{int(TAU_MIN*100)}.png")
OUT_SPRAYEDFRAC_PROB  = os.path.join(ROOT, f"NNDM_PROB_sprayedfrac_vs_tau_budgets.png")

# ---- output figures (deterministic top-b)
OUT_RECALL_DET = os.path.join(ROOT, "NNDM_DET_recall_vs_sprayedfrac_topb_budgets.png")
OUT_PREC_DET   = os.path.join(ROOT, "NNDM_DET_precision_vs_sprayedfrac_topb_budgets.png")
OUT_BOX_R_DET  = os.path.join(ROOT, "NNDM_DET_box_recall_at10pct_topb.png")
OUT_BOX_P_DET  = os.path.join(ROOT, "NNDM_DET_box_precision_at10pct_topb.png")

# ---- output table for R LMM
OUT_LMM_CSV        = os.path.join(ROOT, f"NNDM_Chenopodium_T{T}_budget_metrics_for_R_LMM_tauGE{int(TAU_MIN*100)}.csv")
OUT_LMM_CSV_NOZERO = os.path.join(ROOT, f"NNDM_Chenopodium_T{T}_budget_metrics_for_R_LMM_tauGE{int(TAU_MIN*100)}_prevGT0.csv")

# =========================
# Helpers (identical to main_prob_analysis_visualization_together_v2.py)
# =========================
def _require_cols(df: pd.DataFrame, needed, name="df"):
    miss = set(needed) - set(df.columns)
    if miss:
        raise ValueError(f"{name} missing columns: {miss}. Found: {df.columns.tolist()}")

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

def load_all_taucurves(tau_files: dict, tau_min=0.5, strict=False) -> pd.DataFrame:
    frames = []
    missing = []
    for method, fp in tau_files.items():
        if not os.path.exists(fp):
            missing.append((method, fp))
            if strict:
                raise FileNotFoundError(f"Missing taucurve file for {method}: {fp}")
            continue
        df = pd.read_csv(fp)
        df["method"] = method
        frames.append(df)
    if missing:
        print("\n[WARN] Missing taucurve files (skipped):")
        for m, fp in missing:
            print(f"  - {m}: {fp}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    _require_cols(out, ["date", "tau", "sprayed_frac", "precision", "recall", "method"], "taucurves")
    out["date"]         = out["date"].astype(str)
    out["tau"]          = pd.to_numeric(out["tau"],          errors="coerce")
    out["sprayed_frac"] = pd.to_numeric(out["sprayed_frac"], errors="coerce")
    out["precision"]    = pd.to_numeric(out["precision"],    errors="coerce")
    out["recall"]       = pd.to_numeric(out["recall"],       errors="coerce")
    out = out[out["tau"] >= float(tau_min)].copy()
    out["sprayed_frac"] = out["sprayed_frac"].clip(0, 1)
    out["precision"]    = out["precision"].clip(0, 1)
    out["recall"]       = out["recall"].clip(0, 1)
    return out

def load_all_topb(topb_files: dict, want_score_type="det_nhat", strict=False) -> pd.DataFrame:
    frames = []
    missing = []
    for method, fp in topb_files.items():
        if not os.path.exists(fp):
            missing.append((method, fp))
            if strict:
                raise FileNotFoundError(f"Missing topb file for {method}: {fp}")
            continue
        df = pd.read_csv(fp)
        df["method"] = method
        frames.append(df)
    if missing:
        print("\n[WARN] Missing top-b files (skipped):")
        for m, fp in missing:
            print(f"  - {m}: {fp}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    _require_cols(out, ["date", "b", "sprayed_frac", "precision", "recall", "method"], "topb")
    out["date"]         = out["date"].astype(str)
    out["b"]            = pd.to_numeric(out["b"],            errors="coerce")
    out["sprayed_frac"] = pd.to_numeric(out["sprayed_frac"], errors="coerce")
    out["precision"]    = pd.to_numeric(out["precision"],    errors="coerce")
    out["recall"]       = pd.to_numeric(out["recall"],       errors="coerce")
    if "score_type" in out.columns and want_score_type is not None:
        out = out[out["score_type"].astype(str) == str(want_score_type)].copy()
    out["sprayed_frac"] = out["sprayed_frac"].clip(0, 1)
    out["precision"]    = out["precision"].clip(0, 1)
    out["recall"]       = out["recall"].clip(0, 1)
    return out

def plot_small_multiples(
    df: pd.DataFrame,
    ycol: str,
    title: str,
    outpath: str,
    budgets=(0.05, 0.10, 0.20),
    method_order=None,
    x_max=0.40,
    y_max=1.05,
    drop_zero_prevalence=True,
    legend_in_empty_panel=True
):
    method_order = method_order or sorted(df["method"].unique().tolist())

    if drop_zero_prevalence and "prevalence" in df.columns:
        prev_by_date = (
            df.groupby("date")["prevalence"]
              .apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().iloc[0]
                     if pd.to_numeric(s, errors="coerce").notna().any() else np.nan)
        )
        keep_dates = prev_by_date[prev_by_date > 0].index.tolist()
        df = df[df["date"].isin(keep_dates)].copy()

    dates = sorted(df["date"].unique().tolist())
    n = len(dates)

    extra_panel = 1 if legend_in_empty_panel else 0
    ncols = 4
    n_panels = n + extra_panel
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(4*ncols, 3.2*nrows),
        sharex=True, sharey=True
    )
    axes = np.array(axes).reshape(-1)

    for i, date in enumerate(dates):
        ax = axes[i]
        sub_d = df[df["date"] == date]

        prev_txt = ""
        if "prevalence" in sub_d.columns and sub_d["prevalence"].notna().any():
            prev = float(sub_d["prevalence"].dropna().iloc[0])
            prev_txt = f" (prev={prev:.3f})"
        ax.set_title(f"{date}{prev_txt}")

        for b in budgets:
            ax.axvline(b, linestyle="--", linewidth=1, alpha=0.30)

        for method in method_order:
            s = sub_d[sub_d["method"] == method].sort_values("sprayed_frac")
            if s.empty:
                continue
            ax.plot(s["sprayed_frac"], s[ycol], label=method, linewidth=2.0)

        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)

    if legend_in_empty_panel and len(axes) > n:
        leg_ax = axes[n]
        leg_ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        lab2hand = {lab: h for h, lab in zip(handles, labels)}
        ordered = [(lab2hand[m], m) for m in method_order if m in lab2hand]
        if ordered:
            leg_ax.legend(
                [h for h, _ in ordered],
                [m for _, m in ordered],
                loc="center",
                frameon=True,
                fontsize=11
            )
        start_off = n + 1
    else:
        start_off = n

    for j in range(start_off, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, y=0.98, fontsize=16)
    fig.text(0.5, 0.04, "Sprayed fraction", ha="center")
    fig.text(0.015, 0.5, ycol.capitalize(), va="center", rotation="vertical")

    plt.tight_layout(rect=[0.04, 0.06, 0.98, 0.94])
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", outpath)

def budget_summary_table(df: pd.DataFrame, ycol: str, budget: float) -> pd.DataFrame:
    rows = []
    for (date, method), g in df.groupby(["date", "method"]):
        g = g.sort_values("sprayed_frac")
        val = _interp_y_at_x(g["sprayed_frac"], g[ycol], budget)
        prev = np.nan
        if "prevalence" in g.columns and g["prevalence"].notna().any():
            prev = float(g["prevalence"].dropna().iloc[0])
        rows.append({"date": date, "method": method, "value": val, "prevalence": prev})
    out = pd.DataFrame(rows)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out

def plot_box_enhanced(df_sum: pd.DataFrame, title: str, outpath: str, method_order=None, y_min=0, y_max=1.05):
    method_order = method_order or sorted(df_sum["method"].unique().tolist())
    df_sum = df_sum.copy()
    df_sum = df_sum[df_sum["method"].isin(method_order)]
    df_sum["method"] = pd.Categorical(df_sum["method"], categories=method_order, ordered=True)

    groups = [df_sum.loc[df_sum["method"] == m, "value"].dropna().to_numpy() for m in method_order]

    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.boxplot(groups, labels=method_order)

    rng = np.random.default_rng(0)
    for i, (m, vals) in enumerate(zip(method_order, groups), start=1):
        if len(vals) == 0:
            continue
        x = i + rng.normal(0, 0.05, size=len(vals))
        ax.scatter(x, vals, s=26, alpha=0.75)
        mean_v = float(np.mean(vals))
        ax.scatter([i], [mean_v], s=65, marker="D", alpha=0.9)
        ax.text(i, y_max - 0.02, f"n={len(vals)}", ha="center", va="top", fontsize=9)

    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", outpath)

def export_budget_metrics_for_r_lmm(
    df_prob: pd.DataFrame,
    df_det: pd.DataFrame,
    budgets=(0.05, 0.10, 0.20),
    outpath=None,
    outpath_prev_gt0=None,
    method_order=None
):
    frames = []

    if df_prob is not None and not df_prob.empty:
        prob_rows = []
        for (date, method), g in df_prob.groupby(["date", "method"]):
            g = g.sort_values("sprayed_frac")
            prev = np.nan
            if "prevalence" in g.columns and g["prevalence"].notna().any():
                prev = float(pd.to_numeric(g["prevalence"], errors="coerce").dropna().iloc[0])
            tau_min_val = np.nan
            if "tau" in g.columns and g["tau"].notna().any():
                tau_min_val = float(pd.to_numeric(g["tau"], errors="coerce").min())
            for b in budgets:
                rec  = _interp_y_at_x(g["sprayed_frac"], g["recall"],    b)
                prec = _interp_y_at_x(g["sprayed_frac"], g["precision"], b)
                prob_rows.append({
                    "date": str(date), "algorithm": str(method),
                    "rule": "PROB", "budget": float(b),
                    "recall": rec, "precision": prec,
                    "prevalence": prev, "tau_min": tau_min_val, "score_type": np.nan
                })
        frames.append(pd.DataFrame(prob_rows))

    if df_det is not None and not df_det.empty:
        det_rows = []
        for (date, method), g in df_det.groupby(["date", "method"]):
            g = g.sort_values("sprayed_frac")
            prev = np.nan
            if "prevalence" in g.columns and g["prevalence"].notna().any():
                prev = float(pd.to_numeric(g["prevalence"], errors="coerce").dropna().iloc[0])
            score_type_val = np.nan
            if "score_type" in g.columns and g["score_type"].notna().any():
                score_type_val = str(g["score_type"].dropna().iloc[0])
            for b in budgets:
                rec  = _interp_y_at_x(g["sprayed_frac"], g["recall"],    b)
                prec = _interp_y_at_x(g["sprayed_frac"], g["precision"], b)
                det_rows.append({
                    "date": str(date), "algorithm": str(method),
                    "rule": "DET", "budget": float(b),
                    "recall": rec, "precision": prec,
                    "prevalence": prev, "tau_min": np.nan, "score_type": score_type_val
                })
        frames.append(pd.DataFrame(det_rows))

    if not frames:
        print("[INFO] No data available for LMM export.")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    for col in ["budget", "recall", "precision", "prevalence", "tau_min"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if method_order is not None:
        out = out[out["algorithm"].isin(method_order)].copy()
        out["algorithm"] = pd.Categorical(out["algorithm"], categories=method_order, ordered=True)
        out = out.sort_values(["date", "algorithm", "rule", "budget"])
        out["algorithm"] = out["algorithm"].astype(str)
    else:
        out = out.sort_values(["date", "algorithm", "rule", "budget"])

    if outpath is not None:
        out.to_csv(outpath, index=False)
        print("Saved:", outpath)

    if outpath_prev_gt0 is not None:
        out2 = out.copy()
        if "prevalence" in out2.columns:
            out2 = out2[(out2["prevalence"].isna()) | (out2["prevalence"] > 0)].copy()
        out2.to_csv(outpath_prev_gt0, index=False)
        print("Saved:", outpath_prev_gt0)

    return out


def plot_sprayed_frac_vs_tau(
    df: pd.DataFrame,
    title: str,
    outpath: str,
    tau_vlines=(0.3, 0.5, 0.7),
    method_order=None,
    drop_zero_prevalence=True,
    legend_in_empty_panel=True,
):
    """Faceted plot: sprayed fraction vs probability threshold τ, one panel per date."""
    method_order = method_order or sorted(df["method"].unique().tolist())

    if drop_zero_prevalence and "prevalence" in df.columns:
        prev_by_date = (
            df.groupby("date")["prevalence"]
              .apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().iloc[0]
                     if pd.to_numeric(s, errors="coerce").notna().any() else np.nan)
        )
        keep_dates = prev_by_date[prev_by_date > 0].index.tolist()
        df = df[df["date"].isin(keep_dates)].copy()

    dates = sorted(df["date"].unique().tolist())
    n = len(dates)

    extra_panel = 1 if legend_in_empty_panel else 0
    ncols = 4
    n_panels = n + extra_panel
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(4 * ncols, 3.2 * nrows),
        sharex=True, sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    for i, date in enumerate(dates):
        ax = axes[i]
        sub_d = df[df["date"] == date]

        prev_txt = ""
        if "prevalence" in sub_d.columns and sub_d["prevalence"].notna().any():
            prev = float(sub_d["prevalence"].dropna().iloc[0])
            prev_txt = f" (prev={prev:.3f})"
        ax.set_title(f"{date}{prev_txt}")

        for tv in tau_vlines:
            ax.axvline(tv, linestyle="--", linewidth=1, alpha=0.30)

        for method in method_order:
            s = sub_d[sub_d["method"] == method].sort_values("tau")
            if s.empty:
                continue
            ax.plot(s["tau"], s["sprayed_frac"], label=method, linewidth=2.0)

        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)

    if legend_in_empty_panel and len(axes) > n:
        leg_ax = axes[n]
        leg_ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        lab2hand = {lab: h for h, lab in zip(handles, labels)}
        ordered = [(lab2hand[m], m) for m in method_order if m in lab2hand]
        if ordered:
            leg_ax.legend(
                [h for h, _ in ordered],
                [m for _, m in ordered],
                loc="center", frameon=True, fontsize=11,
            )
        start_off = n + 1
    else:
        start_off = n

    for j in range(start_off, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, y=0.98, fontsize=16)
    fig.text(0.5, 0.04, "Probability threshold τ", ha="center")
    fig.text(0.015, 0.5, "Sprayed fraction", va="center", rotation="vertical")

    plt.tight_layout(rect=[0.04, 0.06, 0.98, 0.94])
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", outpath)


# =========================
# Run: PROBABILISTIC (tau curves)
# =========================
df_prob = load_all_taucurves(TAU_BEST_FILES, tau_min=TAU_MIN, strict=False)

if df_prob.empty:
    print("[INFO] No probabilistic taucurves found. Skipping PROB plots.")
else:
    method_order_prob = [m for m in METHOD_ORDER if m in df_prob["method"].unique().tolist()]

    plot_small_multiples(
        df_prob, ycol="recall",
        title=f"[NNDM LOO | PROB] Chenopodium (T={T}): Recall vs Sprayed fraction | tau ≥ {TAU_MIN:.2f}",
        outpath=OUT_RECALL_PROB,
        budgets=BUDGETS, method_order=method_order_prob, x_max=1.0, y_max=1.05
    )

    plot_small_multiples(
        df_prob, ycol="precision",
        title=f"[NNDM LOO | PROB] Chenopodium (T={T}): Precision vs Sprayed fraction | tau ≥ {TAU_MIN:.2f}",
        outpath=OUT_PREC_PROB,
        budgets=BUDGETS, method_order=method_order_prob, x_max=1.0, y_max=1.05
    )

    plot_sprayed_frac_vs_tau(
        df_prob,
        title=f"[NNDM LOO | PROB] Chenopodium (T={T}): Sprayed fraction vs τ",
        outpath=OUT_SPRAYEDFRAC_PROB,
        method_order=method_order_prob,
    )

    rec10_prob  = budget_summary_table(df_prob, ycol="recall",    budget=0.10)
    prec10_prob = budget_summary_table(df_prob, ycol="precision", budget=0.10)

    plot_box_enhanced(
        rec10_prob,
        title=f"[NNDM LOO | PROB] Recall @10% sprayed fraction (tau ≥ {TAU_MIN:.2f})",
        outpath=OUT_BOX_R_PROB, method_order=method_order_prob, y_min=0.2, y_max=0.5
    )

    plot_box_enhanced(
        prec10_prob,
        title=f"[NNDM LOO | PROB] Precision @10% sprayed fraction (tau ≥ {TAU_MIN:.2f})",
        outpath=OUT_BOX_P_PROB, method_order=method_order_prob, y_min=0.6, y_max=1.05
    )


# =========================
# Run: DETERMINISTIC (top-b curves)
# =========================
df_det = load_all_topb(TOPB_BEST_FILES, want_score_type="det_nhat", strict=False)

if df_det.empty:
    print("[INFO] No deterministic top-b curves found. Skipping DET plots.")
else:
    method_order_det = [m for m in METHOD_ORDER if m in df_det["method"].unique().tolist()]

    plot_small_multiples(
        df_det, ycol="recall",
        title=f"[NNDM LOO | DET] Chenopodium (T={T}): Recall vs Sprayed fraction | top-b by n_hat",
        outpath=OUT_RECALL_DET,
        budgets=BUDGETS, method_order=method_order_det, x_max=1.0, y_max=1.05
    )

    plot_small_multiples(
        df_det, ycol="precision",
        title=f"[NNDM LOO | DET] Chenopodium (T={T}): Precision vs Sprayed fraction | top-b by n_hat",
        outpath=OUT_PREC_DET,
        budgets=BUDGETS, method_order=method_order_det, x_max=1.0, y_max=1.05
    )

    rec10_det  = budget_summary_table(df_det, ycol="recall",    budget=0.10)
    prec10_det = budget_summary_table(df_det, ycol="precision", budget=0.10)

    plot_box_enhanced(
        rec10_det,
        title=f"[NNDM LOO | DET] Recall @10% sprayed fraction (top-b by n_hat)",
        outpath=OUT_BOX_R_DET, method_order=method_order_det, y_min=0.2, y_max=0.5
    )

    plot_box_enhanced(
        prec10_det,
        title=f"[NNDM LOO | DET] Precision @10% sprayed fraction (top-b by n_hat)",
        outpath=OUT_BOX_P_DET, method_order=method_order_det, y_min=0.6, y_max=1.05
    )


# =========================
# Export CSV for R LMM
# =========================
method_order_lmm = [m for m in METHOD_ORDER if m in set(
    ([] if df_prob.empty else df_prob["method"].unique().tolist()) +
    ([] if df_det.empty  else df_det["method"].unique().tolist())
)]

df_lmm = export_budget_metrics_for_r_lmm(
    df_prob=df_prob,
    df_det=df_det,
    budgets=BUDGETS,
    outpath=OUT_LMM_CSV,
    outpath_prev_gt0=OUT_LMM_CSV_NOZERO,
    method_order=method_order_lmm
)
