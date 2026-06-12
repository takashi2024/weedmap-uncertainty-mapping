# WeedMap — Uncertainty-aware Site-specific Weed Density Mapping

This repository contains the analysis code for the manuscript comparing four spatial prediction methods — Ordinary Kriging (OK), TabICL, TabICL-KpR, and TabICL-RK — for mapping *Chenopodium* weed density from UAV imagery using NNDM leave-one-out cross-validation. All code is for the TabICL pipeline; no TabPFN dependency is required.

---

## Requirements

### R (≥ 4.3)
```r
install.packages(c("NNDM", "automap", "gstat", "sf", "sp",
                   "dplyr", "tidyr", "ggplot2", "patchwork", "scales",
                   "lme4", "lmerTest", "emmeans", "multcomp", "tidyverse"))
```
> `NNDM` may need to be installed from GitHub: `remotes::install_github("HannesOberreiter/NNDM")`

### Python
Two conda environments are used:

| Environment | Purpose | Key packages |
|-------------|---------|--------------|
| `tabpfn` | Model execution + probabilistic analysis | `tabicl`, `numpy`, `pandas`, `scikit-learn` |
| `py39` | Standalone visualisation scripts | `pandas`, `matplotlib`, `numpy` |

Create the `tabpfn` environment:
```bash
conda create -n tabpfn python=3.10
conda activate tabpfn
pip install tabicl numpy pandas scikit-learn
```

---

## Data

All input data are in `data/`:

| File | Description |
|------|-------------|
| `data/<date>/result.csv` | Observation points per date: *Chenopodium* counts (raw, log1p, logeps), coordinates (WGS84 + UTM 32N), fold assignments |
| `data/<date>/ROI_features_stacked.csv` | UAV texture and spectral index features per ROI per date |

Date folder naming convention: `YYMMDD + F3mRX` (e.g., `250414F3mRX` = 14 April 2025).

---

## Repository layout

```
WeedMap/
├── data/                                # Input data
│   └── <date>/                          # result.csv + ROI_features_stacked.csv (8 dates)
├── step_1/                              # Preprocessing (NNDM fold structure)
│   └── main_NNDM_preprocessing.R        # Global variogram, LOO kriging, OK evaluation, KpR features
├── step_2/                              # Model execution (per-date predictions)
│   ├── main_NNDM_tabICL_for_RK.py      # TabICL LOO residuals + NNDM predictions
│   ├── main_NNDM_KpR_tabICL.py         # TabICL-KpR predictions
│   ├── datasets_NNDM_RK_tabICL.R       # TabICL-RK fold aggregation
│   └── evaluate_NNDM_results_tabICL.py # Per-date model comparison CSV
├── step_3/                              # Probabilistic analysis (quantile outputs, τ-curves)
│   ├── main_prob_analysis_NNDM_Vanilla.py
│   ├── main_prob_analysis_NNDM_KpR.py
│   ├── main_prob_analysis_NNDM_RK.py
│   ├── main_prob_analysis_NNDM_OK.py
│   ├── main_prob_analysis_NNDM_budget_overview.py
│   ├── main_prob_analysis_NNDM_taucurve.py
│   └── main_prob_analysis_NNDM_visualization_v2.py
├── step_4/                              # Performance evaluation + QCP uncertainty analysis
│   ├── main_4_visualize_R2_NNDM.py     # Fig. 2 — R² time series
│   ├── main_4_visualize_MAE_RMSE_NNDM.py  # Fig. 3 — MAE/RMSE/ME
│   ├── main_4_boxplot_NNDM_R2.R        # R² boxplots
│   ├── main_4_QCP_NNDM.R               # Fig. 4 — QCP + PI width
│   ├── main_4_QCP_NNDM_TabICL.R        # QCP (TabICL-specific output)
│   ├── main_4_QCP_NNDM_sensitivity.R   # QCP sensitivity analysis
│   └── main_4_PR_curve_by_date.R       # Fig. S.14 — per-date PR curves
├── step_5/                              # LMM statistical inference
│   ├── main_5_LMM_NNDM_R2.R
│   ├── main_5_LMM_NNDM_recall_precision.R
│   ├── main_5_LMM_NNDM_tau_recall_precision.R
│   └── main_5_LMM_NNDM_tau_recall_precision_sprayedfrac.R
├── figures/                             # Manuscript and supplementary figures
│   ├── main_Fig1_spatial_distribution.R       # Fig. 1 — observation maps
│   ├── main_supp_combined_maps.R              # Fig. S.9–12 — prediction + uncertainty maps
│   ├── main_supp_prediction_maps.R
│   ├── main_supp_uncertainty_maps.R
│   └── main_supp_variogram_plots.R            # Fig. S.1–2 — variograms
├── run_pipeline_TabICL.sh                     # Full pipeline automation
└── README.md
```

All outputs go to `Results_TabICL/` (created automatically).

---

## Configuration

Before running, set `ROOT` at the top of each script (or in `run_pipeline_TabICL.sh`) to the absolute path of this repository on your system:

```r
ROOT <- "/path/to/WeedMap"
```

---

## Running the pipeline

### Automated (recommended)

```bash
bash run_pipeline_TabICL.sh
```

This runs all phases in order:

| Phase | Folder | Description |
|-------|--------|-------------|
| 1 | `step_1/` | Preprocessing per date: NNDM fold structure, OK kriging, KpR features |
| 2 | `step_2/` | Per-date model runs: TabICL, TabICL-KpR, TabICL-RK, evaluation |
| 3 | `step_3/` | Probabilistic analysis: quantile outputs, τ-curves |
| 4 | `step_4/` | Performance evaluation: R², MAE/RMSE, QCP calibration, PR curves |
| 5 | `step_5/` | LMM statistical inference |
| — | `figures/` | Manuscript and supplementary figure generation |

### Manual step-by-step

Per date (repeat for each of the 8 survey dates):

```bash
# Phase 1 — preprocessing (step_1/)
Rscript step_1/main_NNDM_preprocessing.R <date_id>   # date_id: 1–8 maps to the 8 survey dates (Stage 1)
# (set RUN_MODE = "loo" in step_2/main_NNDM_tabICL_for_RK.py first)
conda activate tabpfn && python step_2/main_NNDM_tabICL_for_RK.py
Rscript step_1/main_NNDM_preprocessing.R <date_id>   # Stage 2

# Phase 2 — model runs (step_2/)
# (set RUN_MODE = "nndm" in main_NNDM_tabICL_for_RK.py)
python step_2/main_NNDM_tabICL_for_RK.py
python step_2/main_NNDM_KpR_tabICL.py
Rscript step_2/datasets_NNDM_RK_tabICL.R
python step_2/evaluate_NNDM_results_tabICL.py
```

After all dates:

```bash
# Phase 3 — probabilistic analysis (step_3/)
python step_3/main_prob_analysis_NNDM_Vanilla.py
python step_3/main_prob_analysis_NNDM_KpR.py
python step_3/main_prob_analysis_NNDM_RK.py
python step_3/main_prob_analysis_NNDM_OK.py
python step_3/main_prob_analysis_NNDM_budget_overview.py
python step_3/main_prob_analysis_NNDM_visualization_v2.py

# Phase 4 — performance evaluation + QCP (step_4/)
python step_4/main_4_visualize_R2_NNDM.py
python step_4/main_4_visualize_MAE_RMSE_NNDM.py
Rscript step_4/main_4_boxplot_NNDM_R2.R
Rscript step_4/main_4_QCP_NNDM.R
Rscript step_4/main_4_PR_curve_by_date.R

# Phase 5 — LMM statistical analysis (step_5/)
Rscript step_5/main_5_LMM_NNDM_R2.R
Rscript step_5/main_5_LMM_NNDM_recall_precision.R

# Figures
Rscript figures/main_supp_combined_maps.R
Rscript figures/main_supp_variogram_plots.R
Rscript figures/main_Fig1_spatial_distribution.R
```

---

## Output structure

```
Results_TabICL/
├── data_YYYYMMDD/NNDM/          # Per-date predictions
│   ├── OK_predictions.csv
│   ├── <target>_OK_quantiles.csv
│   ├── KpR_features.csv
│   ├── Vanilla/                 # TabICL outputs
│   ├── KpR/                     # TabICL-KpR outputs
│   └── RK/                      # TabICL-RK outputs
├── NNDM_prob/                   # Probabilistic outputs (quantile + τ-curve CSVs)
└── figures/                     # All manuscript figures and tables
    └── supp/                    # Supplementary figures
```

---

## Key definitions

| Term | Meaning |
|------|---------|
| **NNDM LOO CV** | Nearest-Neighbour Distance Matching Leave-One-Out Cross-Validation |
| **KpR** | Kriging of TabICL prediction residuals using local spatial neighbourhood features |
| **RK** | Regression Kriging: TabICL predictions + kriged residual correction (Hengl et al., 2007) |
| **δ_QCP** | Quantile Coverage Probability deviation; 0 = perfect calibration |
| **τ** | Probability threshold for binary spray/no-spray decision |

---

## Colour palette

| Method | Colour |
|--------|--------|
| OK | `#4477AA` (blue) |
| TabICL | `#009999` (teal) |
| TabICL-KpR | `#cc6699` (pink) |
| TabICL-RK | `#e6b800` (gold) |
