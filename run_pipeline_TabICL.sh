#!/usr/bin/env bash
# =============================================================================
# run_pipeline_TabICL.sh
# Full NNDM pipeline for TabICL — self-contained, no TabPFN dependency.
#
# Prerequisites:
#   - conda environment "tabpfn" with tabicl, numpy, pandas, scikit-learn
#   - R ≥ 4.3 with packages: NNDM, automap, gstat, sf, sp, dplyr, tidyr
#
# Set ROOT to the repository root before running.
# All outputs go to Results_TabICL/.
# =============================================================================

set -euo pipefail

ROOT="/Users/takashi/LocalAnalysis/WeedMap/ForGithub"   # <-- set to your local path
CONDA_ENV="tabpfn"

PREPROCESS_R="$ROOT/step_1/main_NNDM_preprocessing.R"
TABICL_RK="$ROOT/step_2/main_NNDM_tabICL_for_RK.py"
KPR="$ROOT/step_2/main_NNDM_KpR_tabICL.py"
RK_R="$ROOT/step_2/datasets_NNDM_RK_tabICL.R"
EVAL="$ROOT/step_2/evaluate_NNDM_results_tabICL.py"

DATES=(20250414 20250424 20250430 20250506 20250513 20250520 20250526 20250602)

LOG="$ROOT/Results_TabICL/pipeline_run.log"
mkdir -p "$ROOT/Results_TabICL"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------------------
# Helper: patch the date variable in a script (in-place, macOS-safe)
# ---------------------------------------------------------------------------
set_date_py() {
  local file="$1" date="$2"
  perl -i -pe "s|^date\s*=\s*\"[0-9]+\"|date      = \"$date\"|" "$file"
}
set_date_r() {
  local file="$1" date="$2"
  perl -i -pe "s|^date\s*<-\s*\"[0-9]+\"|date   <- \"$date\"|" "$file"
}
set_runmode() {
  local file="$1" mode="$2"
  perl -i -pe "s|^RUN_MODE\s*=\s*\"[a-z]+\"\s*# loo or nndm|RUN_MODE  = \"$mode\" # loo or nndm|" "$file"
}

# ---------------------------------------------------------------------------
# Phase 1: per-date preprocessing
#   Stage 1 (R) → LOO Python → Stage 2 (R)
# ---------------------------------------------------------------------------
log "===== Phase 1: preprocessing ====="

for i in "${!DATES[@]}"; do
  DATE="${DATES[$i]}"
  DATE_ID=$((i + 1))   # R script expects 1-based index
  log "----- Date: $DATE (date_id=$DATE_ID) -----"

  DST_NNDM="$ROOT/Results_TabICL/data_$DATE/NNDM"
  LOO_FILE="$ROOT/Dataset_for_python/data_$DATE/NNDM/tabicl_loo_residuals.csv"

  # Skip if preprocessing Stage 2 outputs already exist
  if [ -f "$DST_NNDM/OK_predictions.csv" ] && [ -f "$DST_NNDM/KpR_features.csv" ]; then
    log "  Preprocessing already done for $DATE, skipping"
    continue
  fi

  # Stage 1: global variogram + LOO kriging (always re-run if Stage 2 not done)
  log "  Stage 1: LOO kriging (R)"
  Rscript "$PREPROCESS_R" "$DATE_ID" >> "$LOG" 2>&1
  log "  Stage 1 done"

  # Python LOO residuals (needed by Stage 2)
  if [ ! -f "$LOO_FILE" ]; then
    log "  Python LOO: TabICL LOO residuals (date=$DATE)"
    set_date_py "$TABICL_RK" "$DATE"
    set_runmode "$TABICL_RK" "loo"
    conda run -n "$CONDA_ENV" python "$TABICL_RK" >> "$LOG" 2>&1
    log "  Python LOO done"
  else
    log "  LOO residuals already exist for $DATE, skipping Python LOO"
  fi

  # Stage 2: fold CSVs + OK_predictions + KpR_features
  log "  Stage 2: fold CSVs + OK eval + KpR features (R)"
  Rscript "$PREPROCESS_R" "$DATE_ID" >> "$LOG" 2>&1
  log "  Stage 2 done"
done

log "===== Phase 1 complete ====="

# ---------------------------------------------------------------------------
# Phase 2: per-date model runs (Vanilla, KpR, RK, evaluate)
# ---------------------------------------------------------------------------
log "===== Phase 2: per-date model runs ====="

for DATE in "${DATES[@]}"; do
  log "----- Date: $DATE -----"

  # Skip if Vanilla output already exists
  if [ -d "$ROOT/Results_TabICL/data_$DATE/NNDM/Vanilla" ]; then
    log "  Vanilla already done for $DATE, skipping steps 1–2"
  else
    # Step 1: NNDM evaluation (Vanilla + RK fold data)
    log "  Step 1: TabICL NNDM (Vanilla + RK, date=$DATE)"
    set_date_py "$TABICL_RK" "$DATE"
    set_runmode "$TABICL_RK" "nndm"
    conda run -n "$CONDA_ENV" python "$TABICL_RK" >> "$LOG" 2>&1
    log "  Step 1 done"
  fi

  # Step 2: KpR
  log "  Step 2: KpR TabICL (date=$DATE)"
  set_date_py "$KPR" "$DATE"
  conda run -n "$CONDA_ENV" python "$KPR" >> "$LOG" 2>&1
  log "  Step 2 done"

  # Step 3: RK fold aggregation (R)
  log "  Step 3: RK fold aggregation (date=$DATE)"
  set_date_r "$RK_R" "$DATE"
  Rscript "$RK_R" >> "$LOG" 2>&1
  log "  Step 3 done"

  # Step 4: Evaluate — model_comparison CSV
  log "  Step 4: evaluate results (date=$DATE)"
  perl -i -pe "s|^date\s*=\s*\"[0-9]+\"|date   = \"$DATE\"|" "$EVAL"
  conda run -n "$CONDA_ENV" python "$EVAL" >> "$LOG" 2>&1
  log "  Step 4 done"
done

log "===== Phase 2 complete ====="

# ---------------------------------------------------------------------------
# Phase 3: probabilistic analysis (once, over all dates)
# ---------------------------------------------------------------------------
log "===== Phase 3: probabilistic analysis ====="

cd "$ROOT"

PROB_DIR="step_3"

log "  Prob analysis — Vanilla"
conda run -n "$CONDA_ENV" python "$PROB_DIR/main_prob_analysis_NNDM_Vanilla.py" >> "$LOG" 2>&1

log "  Prob analysis — KpR"
conda run -n "$CONDA_ENV" python "$PROB_DIR/main_prob_analysis_NNDM_KpR.py" >> "$LOG" 2>&1

log "  Prob analysis — RK"
conda run -n "$CONDA_ENV" python "$PROB_DIR/main_prob_analysis_NNDM_RK.py" >> "$LOG" 2>&1

log "  Prob analysis — OK"
conda run -n "$CONDA_ENV" python "$PROB_DIR/main_prob_analysis_NNDM_OK.py" >> "$LOG" 2>&1

log "  Budget overview"
conda run -n "$CONDA_ENV" python "$PROB_DIR/main_prob_analysis_NNDM_budget_overview.py" >> "$LOG" 2>&1

log "  NNDM visualization (generates LMM CSV)"
conda run -n "$CONDA_ENV" python "$PROB_DIR/main_prob_analysis_NNDM_visualization_v2.py" >> "$LOG" 2>&1

log "===== Phase 3 complete ====="

# ---------------------------------------------------------------------------
# Phase 4: visualisation and statistics
# ---------------------------------------------------------------------------
log "===== Phase 4: performance evaluation + QCP (step_4/) ====="

log "  R² time-series figure"
conda run -n "$CONDA_ENV" python step_4/main_4_visualize_R2_NNDM.py >> "$LOG" 2>&1

log "  MAE/RMSE/ME figure"
conda run -n "$CONDA_ENV" python step_4/main_4_visualize_MAE_RMSE_NNDM.py >> "$LOG" 2>&1

log "  R² boxplot"
Rscript step_4/main_4_boxplot_NNDM_R2.R >> "$LOG" 2>&1

log "  QCP uncertainty calibration"
Rscript step_4/main_4_QCP_NNDM.R >> "$LOG" 2>&1

log "  Precision-recall curves"
Rscript step_4/main_4_PR_curve_by_date.R >> "$LOG" 2>&1

log "===== Phase 4 complete ====="

log "===== Phase 5: LMM statistical analysis (step_5/) ====="

log "  LMM R²"
Rscript step_5/main_5_LMM_NNDM_R2.R >> "$LOG" 2>&1

log "  LMM recall/precision"
Rscript step_5/main_5_LMM_NNDM_recall_precision.R >> "$LOG" 2>&1

log "  LMM τ × recall/precision"
Rscript step_5/main_5_LMM_NNDM_tau_recall_precision.R >> "$LOG" 2>&1

log "  LMM τ × sprayed fraction"
Rscript step_5/main_5_LMM_NNDM_tau_recall_precision_sprayedfrac.R >> "$LOG" 2>&1

log "===== Phase 5 complete ====="

log "===== Figures (figures/) ====="

log "  Fig 1 — spatial distribution"
Rscript figures/main_Fig1_spatial_distribution.R >> "$LOG" 2>&1

log "  Supplementary maps (prediction + uncertainty)"
Rscript figures/main_supp_combined_maps.R >> "$LOG" 2>&1

log "  Supplementary variograms"
Rscript figures/main_supp_variogram_plots.R >> "$LOG" 2>&1

log "===== Pipeline complete. Results in Results_TabICL/ ====="
