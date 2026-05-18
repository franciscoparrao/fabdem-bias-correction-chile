#!/usr/bin/env python3
"""Sample DEM + terrain + hydrology features at ATL08 footprints.

Input:
  ../icesat2/atl08_footprints_filtered.csv  (lon, lat, h_te, h_te_unc, ...)
  ../dem/fabdem.tif
  ../factors/*.tif      (17 terrain factors)
  ../hydro/*.tif        (8 hydrology factors)

Output:
  pilot_samples.csv  (one row per footprint, columns = all features + target)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.sample import sample_gen

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem/pilot_teno")
FP = ROOT / "icesat2" / "atl08_footprints_filtered.csv"

# Build feature stack
feature_rasters = {"fabdem": ROOT / "dem" / "fabdem.tif"}
for d in [ROOT / "factors", ROOT / "hydro"]:
    for tif in sorted(d.glob("*.tif")):
        feature_rasters[tif.stem] = tif

print(f"Features: {len(feature_rasters)}")
for name in feature_rasters:
    print(f"  - {name}")

# Read footprints
df = pd.read_csv(FP)
pts = list(zip(df.lon.values, df.lat.values))
print(f"\nFootprints: {len(df)}")

# Sample each raster
for name, path in feature_rasters.items():
    with rasterio.open(path) as src:
        vals = np.array(list(src.sample(pts, indexes=1, masked=False))).ravel()
        # Replace NoData with NaN
        if src.nodata is not None:
            vals = np.where(vals == src.nodata, np.nan, vals)
    df[name] = vals

# Drop rows where any critical feature is NaN
critical = ["fabdem", "slope", "hand"]
before = len(df)
df_clean = df.dropna(subset=[c for c in critical if c in df.columns])
after = len(df_clean)
print(f"\nFootprints after dropping critical-NaN: {after} (was {before})")

# Stats
out = ROOT / "samples" / "pilot_samples.csv"
df_clean.to_csv(out, index=False)
print(f"\n→ {out}")
print(f"\n=== SUMMARY ===")
print(f"  Footprints:      {len(df_clean)}")
print(f"  Features:        {len(feature_rasters)}")
print(f"  Cols total:      {len(df_clean.columns)}")
print(f"  CSV size:        {out.stat().st_size / 1024:.1f} KB")

# Feature correlation with target (h_te)
print(f"\n=== Top 10 features by |corr| with h_te (target) ===")
target = df_clean["h_te"]
corrs = {}
for col in feature_rasters.keys():
    if col in df_clean.columns:
        s = df_clean[col].dropna()
        t = target.loc[s.index]
        if len(s) > 10 and s.std() > 0:
            corrs[col] = np.corrcoef(s, t)[0, 1]
for col, c in sorted(corrs.items(), key=lambda x: -abs(x[1]))[:10]:
    print(f"  {col:<25s} r = {c:+.4f}")
