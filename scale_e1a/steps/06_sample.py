#!/usr/bin/env python3
"""E1a stage 06: sample all feature rasters at ATL08 footprints."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import pyproj

ROOT = Path(__file__).resolve().parent.parent
FP = ROOT / "samples" / "atl08_footprints.csv"
OUT = ROOT / "samples" / "pilot_e1a_full.csv"

if OUT.exists() and OUT.stat().st_size > 100_000:
    print(f"✓ {OUT.name} already exists")
    sys.exit(0)

df = pd.read_csv(FP)
print(f"Loaded {len(df)} footprints")

raster_dirs = {
    "fabdem": [ROOT / "dem"],
    "terrain": [ROOT / "factors"],
    "hydro": [ROOT / "hydro"],
    "satellite": [ROOT / "satellite"],
}
rasters = {}
for cat, dirs in raster_dirs.items():
    for d in dirs:
        for tif in sorted(d.glob("*.tif")):
            rasters[tif.stem] = (cat, tif)
print(f"Found {len(rasters)} feature rasters")

lon = df["lon"].values
lat = df["lat"].values
pts_geo = list(zip(lon, lat))

# Pre-transform to UTM 19S once
to_utm = pyproj.Transformer.from_crs(4326, 32719, always_xy=True)
x_utm, y_utm = to_utm.transform(lon, lat)
pts_utm = list(zip(x_utm, y_utm))

for name, (cat, path) in rasters.items():
    with rasterio.open(path) as src:
        crs_epsg = src.crs.to_epsg() if src.crs else None
        pts = pts_utm if crs_epsg == 32719 else pts_geo
        vals = np.array(list(src.sample(pts, indexes=1, masked=False))).ravel()
        if src.nodata is not None:
            vals = np.where(vals == src.nodata, np.nan, vals)
    df[name] = vals
    valid = int(np.isfinite(vals).sum())
    print(f"  {name:<25s}  ({cat})  CRS={crs_epsg}  valid={valid}/{len(vals)}")

# Compute residual
df["residual_corrected"] = df["h_te_orthometric"] - df["fabdem"]
df.to_csv(OUT, index=False)
print(f"\n→ {OUT}")
print(f"  Rows: {len(df)}")
print(f"  Columns: {len(df.columns)}")
print(f"\nResidual (corrected) stats:")
r = df.residual_corrected.dropna()
print(f"  Mean:   {r.mean():+.3f} m")
print(f"  Median: {r.median():+.3f} m")
print(f"  Std:    {r.std():.3f} m")
print(f"  P5–P95: [{r.quantile(0.05):+.2f}, {r.quantile(0.95):+.2f}]")
