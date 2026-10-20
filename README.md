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
| `py39` | Grid feature extraction + visualisation | `geopandas`, `rasterio`, `scikit-image`, `pandas`, `matplotlib`, `numpy` |

Create the environments:
```bash
conda create -n tabpfn python=3.10
conda activate tabpfn
pip install tabicl numpy pandas scikit-learn

conda create -n py39 python=3.9
conda activate py39
pip install geopandas rasterio scikit-image pandas matplotlib numpy
```

---

## Data

All input data are in `data/`:

| File | Description |
|------|-------------|
| `data/<date>/result.csv` | Observation points per date: *Chenopodium* counts (raw, log1p, logeps), coordinates (WGS84 + UTM 32N), fold assignments |
| `data/<date>/ROI_features_stacked.csv` | UAV texture and spectral index features per ROI per date |

Date folder naming convention: `YYMMDD + F3mRX` (e.g., `250414F3mRX` = 14 April 2025).

Spatial reference data for the 1m grid:

| File | Description |
|------|-------------|
| `spatial/field_1m_grid.gpkg` | Regular 1m × 1m grid cells within the field boundary (EPSG:32632) |
| `grid_predictions/<date>/grid_features.csv` | Spectral/texture features extracted at all grid cells |
| `grid_predictions/<date>/OK_grid.csv` | OK predictions at grid cells (log1p scale) |
| `grid_predictions/<date>/OK_train_loo.csv` | LOO OK predictions at training locations (for KpR) |
| `grid_predictions/<date>/Vanilla_grid.csv` | TabICL predictions + full quantile forecast (q0.05–q0.95) at grid cells |
| `grid_predictions/<date>/KpR_grid.csv` | TabICL-KpR predictions + full quantile forecast (q0.05–q0.95) at grid cells |
| `grid_predictions/<date>/RK_grid.csv` | TabICL-RK predictions at grid cells |

> `field_1m_grid.gpkg` is built from `Field.gpkg` in the UAV analysis folder (outside this repository). Re-generate with `step_6/build_field_grid.py` if needed; the pre-built file is included so this step can be skipped.

---

## Repository layout

```
WeedMap/
├── data/                                # Input data
│   └── <date>/                          # result.csv + ROI_features_stacked.csv (8 dates)
├── spatial/                             # Spatial reference files
│   └── field_1m_grid.gpkg               # 1m × 1m grid cells within field boundary (EPSG:32632)
├── grid_predictions/                    # Wall-to-wall 1m grid prediction outputs
│   └── <date>/                          # grid_features.csv, OK_grid.csv, Vanilla_grid.csv, etc.
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
│   ├── main_4_visualize_MAE_RMSE_NNDM.py  # Fig. S4 — ME/RMSE time series
│   ├── main_4_boxplot_NNDM_R2.R        # R² boxplots (disabled in pipeline)
│   ├── main_4_QCP_NNDM_TabICL.R        # Fig. 3 — QCP calibration plot
│   ├── main_4_QCP_NNDM_sensitivity.R   # QCP sensitivity (log vs count scale)
│   └── main_4_PR_curve_by_date.R       # Fig. S15 — Per-date precision-recall curves
├── step_5/                              # LMM statistical inference (Figure 4)
│   └── main_5_LMM_NNDM_tau_recall_precision_sprayedfrac.R  # Fig. 4 — recall, precision + sprayed fraction vs τ
├── step_6/                              # Wall-to-wall 1m grid spatial predictions
│   ├── build_field_grid.py              # Build grid GeoPackage from Field.gpkg boundary
│   ├── extract_grid_features.py        # Extract spectral/texture features at grid cells
│   ├── predict_grid_OK.R               # OK kriging at all grid cells
│   ├── predict_grid_tabicl.py          # TabICL + KpR predictions at grid cells
│   └── predict_grid_RK.R               # RK residual kriging at grid cells
├── figures/                             # Manuscript and supplementary figures
│   ├── main_Fig1_spatial_distribution.R       # Fig. 1 — observation maps
│   ├── main_supp_combined_maps.R              # Fig. S13–S14 — per-date uncertainty maps (NNDM LOO CV)
│   ├── main_supp_prediction_maps.R            # Per-date NNDM LOO CV prediction maps
│   ├── figures_grid_maps.R                    # Fig. S5–S14 — 1m grid prediction + uncertainty maps
│   └── main_supp_variogram_plots.R            # Fig. S2–S3 — variograms (OK, RK residuals)
├── archive/                             # Archived scripts no longer used in main pipeline
│   ├── step_5/                          # Earlier LMM scripts (R² and recall/precision only)
│   └── Results_TabICL/                  # Archived LMM outputs
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
| 4 | `step_4/` | Performance evaluation: R², ME/RMSE, QCP calibration |
| 5 | `step_5/` | LMM statistical inference |
| 6 | `step_6/` | Wall-to-wall 1m grid predictions (OK, TabICL, KpR, RK) |
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
Rscript step_5/main_5_LMM_NNDM_tau_recall_precision_sprayedfrac.R

# Phase 6 — 1m grid spatial predictions (step_6/)
# (skip build_field_grid.py if spatial/field_1m_grid.gpkg already exists)
conda activate py39 && python step_6/build_field_grid.py
python step_6/extract_grid_features.py
Rscript step_6/predict_grid_OK.R
conda activate tabpfn && python step_6/predict_grid_tabicl.py
Rscript step_6/predict_grid_RK.R

# Figures
Rscript figures/main_Fig1_spatial_distribution.R
Rscript figures/main_supp_combined_maps.R
Rscript figures/figures_grid_maps.R
Rscript figures/main_supp_variogram_plots.R
```

---

## Output structure

```
Results_TabICL/
├── data_YYYYMMDD/NNDM/          # Per-date NNDM CV predictions
│   ├── OK_predictions.csv
│   ├── <target>_OK_quantiles.csv
│   ├── KpR_features.csv
│   ├── Vanilla/                 # TabICL outputs
│   ├── KpR/                     # TabICL-KpR outputs
│   └── RK/                      # TabICL-RK outputs
├── NNDM_prob/                   # Probabilistic outputs (quantile + τ-curve CSVs)
├── NNDM_LMM_outputs/            # LMM results (EMM tables, ANOVA)
└── figures/                     # All manuscript figures
    └── supp/                    # Supplementary figures (variograms, pred/unc maps)

grid_predictions/
└── YYYYMMDD/                    # Per-date 1m grid outputs
    ├── grid_features.csv        # Spectral/texture features at all grid cells
    ├── OK_grid.csv              # OK predictions + kriging variance
    ├── OK_train_loo.csv         # LOO OK predictions at training locations
    ├── Vanilla_grid.csv         # TabICL predictions + full quantile forecast (q0.05–q0.95)
    ├── KpR_grid.csv             # TabICL-KpR predictions + full quantile forecast (q0.05–q0.95)
    └── RK_grid.csv              # TabICL-RK predictions
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
