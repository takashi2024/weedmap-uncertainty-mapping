"""
build_field_grid.py
Create a regular 1m × 1m grid of cells within the field boundary defined by Field.gpkg.
Output: spatial/field_1m_grid.gpkg

Run with:  conda run -n py39 python3 step_6/build_field_grid.py
"""

import os
import geopandas as gpd
import numpy as np
from shapely.geometry import box

ROOT       = "/Users/takashi/LocalAnalysis/WeedMap/ForGithub"
UAV_ANA    = "/Users/takashi/LocalAnalysis/WeedMap/UAV analysis"   # outside repo — rasters only
FIELD_GPKG = os.path.join(UAV_ANA, "Field.gpkg")
OUT_GPKG   = os.path.join(ROOT, "spatial", "field_1m_grid.gpkg")

print("Loading Field.gpkg ...")
field = gpd.read_file(FIELD_GPKG)
print(f"  CRS: {field.crs}")
print(f"  Features: {len(field)}")
print(f"  Bounds: {field.total_bounds}")

# Dissolve all 145 plot polygons into one field polygon
field_union = field.geometry.unary_union
x_min, y_min, x_max, y_max = field_union.bounds
print(f"  Union bounds: {x_min:.1f}–{x_max:.1f} E, {y_min:.1f}–{y_max:.1f} N")

# Generate 1m × 1m grid centroids at 0.5m offsets so centroids fall at integer + 0.5
xs = np.arange(np.floor(x_min) + 0.5, np.ceil(x_max), 1.0)
ys = np.arange(np.floor(y_min) + 0.5, np.ceil(y_max), 1.0)
print(f"  Candidate cells: {len(xs)} × {len(ys)} = {len(xs)*len(ys)}")

cells = []
for x in xs:
    for y in ys:
        cell = box(x - 0.5, y - 0.5, x + 0.5, y + 0.5)
        if field_union.intersects(cell):
            cells.append({
                "grid_id": len(cells) + 1,
                "x_25832": x,
                "y_25832": y,
                "geometry": cell
            })

os.makedirs(os.path.dirname(OUT_GPKG), exist_ok=True)
gdf = gpd.GeoDataFrame(cells, crs=field.crs)
gdf.to_file(OUT_GPKG, driver="GPKG", layer="grid_1m")
print(f"\nSaved {len(gdf)} grid cells to {OUT_GPKG}")
