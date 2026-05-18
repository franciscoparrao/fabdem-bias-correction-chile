#!/usr/bin/env python3
"""Stage 06: sample feature rasters at ATL08 footprints → tile_samples.csv."""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import pyproj

ROOT = Path(os.environ.get("PIPELINE_WORKDIR", str(Path(__file__).resolve().parent.parent)))
FP = ROOT / "samples" / "atl08_footprints.csv"
OUT = ROOT / "samples" / "tile_samples.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
print(f"[06_sample] WORKDIR={ROOT}")

if OUT.exists() and OUT.stat().st_size > 100_000:
    print(f"✓ {OUT.name} already exists")
    sys.exit(0)

df = pd.read_csv(FP)
print(f"Loaded {len(df)} footprints")

rasters = {}
for cat, sub in [("dem", "dem"), ("terrain", "factors"), ("hydro", "hydro"), ("sat", "satellite")]:
    d = ROOT / sub
    for tif in sorted(d.glob("*.tif")):
        rasters[tif.stem] = (cat, tif)
print(f"Found {len(rasters)} feature rasters")

lon = df["lon"].values; lat = df["lat"].values
pts_geo = list(zip(lon, lat))
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
    print(f"  {name:<25s}  {cat}  valid={valid}/{len(vals)}")

df["residual_corrected"] = df["h_te_orthometric"] - df["fabdem"]
df.to_csv(OUT, index=False)
print(f"→ {OUT}  ({len(df)} × {len(df.columns)})")
