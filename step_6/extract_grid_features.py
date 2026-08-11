"""
extract_grid_features.py
Extract 40 spectral/texture features at each 1m grid cell for all 8 survey dates.
Reads pre-processed rasters from UAV analysis folder (resampled + VIs already done).
Output: grid_predictions/{date}/grid_features.csv  (M × 41 columns: grid_id + 40 features)

Run with:  conda run -n py39 python3 step_6/extract_grid_features.py
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from skimage.feature import graycomatrix, graycoprops

# =============================================================================
# Paths
# =============================================================================
ROOT      = "/Users/takashi/LocalAnalysis/WeedMap/ForGithub"
UAV_ANA   = "/Users/takashi/LocalAnalysis/WeedMap/UAV analysis"   # outside repo — rasters only
GRID_GPKG = os.path.join(ROOT, "spatial", "field_1m_grid.gpkg")
OUT_ROOT  = os.path.join(ROOT, "grid_predictions")

# date string → imagery campaign folder
# Jun 03 YOLO data uses the Jun 02 UAV imagery folder
DATE_TO_IMG = {
    "20250414": "250414F3mRX",
    "20250424": "250424F3mRX",
    "20250430": "250430F3mRX",
    "20250506": "250506F3mRX",
    "20250513": "250513F3mRX",
    "20250520": "250520F3mRX",
    "20250526": "250526F3mRX",
    "20250603": "250602F3mRX",
}

# Layer order must match ROI_features_stacked.csv column order
LAYER_ORDER = ["nir", "red", "red_edge", "green", "NDVI", "NDRE", "CIgreen", "CIrededge"]

# =============================================================================
# Haralick texture (verbatim from main_texture_analysis_v4.ipynb)
# =============================================================================
def haralick_from_values(values, levels=32):
    values = values[np.isfinite(values)]
    if len(values) < 20:
        return np.nan, np.nan, np.nan

    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return 0.0, 0.0, 1.0

    q = np.floor((values - vmin) / (vmax - vmin) * (levels - 1)).astype("uint8")
    glcm = graycomatrix(
        q.reshape(-1, 1),
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )
    entropy     = -np.sum(glcm * np.log2(glcm + 1e-12))
    contrast    = graycoprops(glcm, "contrast").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    return float(entropy), float(contrast), float(homogeneity)


# =============================================================================
# Load grid once
# =============================================================================
print("Loading grid ...")
gdf = gpd.read_file(GRID_GPKG)
gdf = gdf.set_crs("EPSG:32632", allow_override=True)
n_cells = len(gdf)
print(f"  {n_cells} grid cells")

# =============================================================================
# Process each date
# =============================================================================
for date, img_tag in DATE_TO_IMG.items():
    proc_root = os.path.join(UAV_ANA, img_tag,
                             "Processed data resampled and texture analysis")
    raster_paths = {
        **{b: os.path.join(proc_root, "resampled", f"{b}.tif")
           for b in ["nir", "red", "red_edge", "green"]},
        **{v: os.path.join(proc_root, "vegetation_indices", f"{v}.tif")
           for v in ["NDVI", "NDRE", "CIgreen", "CIrededge"]},
    }

    print(f"\n=== {date} (imagery: {img_tag}) ===")
    missing = [l for l, p in raster_paths.items() if not os.path.exists(p)]
    if missing:
        print(f"  SKIP — missing rasters: {missing}")
        continue

    records = []

    for layer_name in LAYER_ORDER:
        rpath = raster_paths[layer_name]
        print(f"  {layer_name} ...", end="", flush=True)
        n_ok = 0

        with rasterio.open(rpath) as src:
            nodata = src.nodata
            grid_reproj = gdf.to_crs(src.crs)

            for idx, row in grid_reproj.iterrows():
                try:
                    out, _ = mask(src, [row.geometry], crop=True)
                except Exception:
                    continue

                data = out[0].astype("float32")
                if nodata is not None:
                    data = data[data != nodata]
                data = data[np.isfinite(data)]

                if data.size == 0:
                    continue

                ent, con, hom = haralick_from_values(data)
                records.append({
                    "grid_id":     gdf.loc[idx, "grid_id"],
                    "layer":       layer_name,
                    "mean":        float(np.nanmean(data)),
                    "std":         float(np.nanstd(data)),
                    "entropy":     ent,
                    "contrast":    con,
                    "homogeneity": hom,
                })
                n_ok += 1

        print(f" {n_ok}/{n_cells} cells extracted")

    if not records:
        print("  No records — skipping")
        continue

    df = pd.DataFrame(records)
    df_wide = df.pivot_table(
        index="grid_id",
        columns="layer",
        values=["mean", "std", "entropy", "contrast", "homogeneity"],
        aggfunc="first",
    )
    df_wide.columns = [f"{layer}_{stat}" for stat, layer in df_wide.columns]
    df_wide = df_wide.reset_index()

    stat_order = ["mean", "std", "entropy", "contrast", "homogeneity"]
    ordered_cols = ["grid_id"] + [
        f"{layer}_{stat}"
        for layer in LAYER_ORDER
        for stat in stat_order
        if f"{layer}_{stat}" in df_wide.columns
    ]
    df_wide = df_wide[ordered_cols]

    coords = gdf[["grid_id", "x_25832", "y_25832"]]
    df_wide = coords.merge(df_wide, on="grid_id", how="left")

    out_dir = os.path.join(OUT_ROOT, date)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "grid_features.csv")
    df_wide.to_csv(out_path, index=False)

    n_complete = df_wide.dropna().shape[0]
    print(f"  Saved {len(df_wide)} rows ({n_complete} complete) → {out_path}")

print("\nDone.")
