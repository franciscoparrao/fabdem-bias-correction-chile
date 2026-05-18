#!/usr/bin/env python3
"""Add Sentinel-1/2 features to the EGM2008-corrected sample dataset.

Inputs:
  pilot_samples_egm2008.csv  (888 footprints × 27 terrain features + h_te_ortho)
  ../satellite/*.tif          (8 satellite features: ndvi,ndwi,ndmi,bsi,ndbi, vv,vh,ratio)

Output:
  pilot_samples_full.csv     (all features + EGM2008-corrected residual)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import pyproj

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem/pilot_teno")
df = pd.read_csv(ROOT / "samples" / "pilot_samples_egm2008.csv")
print(f"Loaded {len(df)} footprints with {len(df.columns)} columns")

sat_dir = ROOT / "satellite"
sat_features = sorted(sat_dir.glob("*.tif"))
print(f"Adding {len(sat_features)} satellite features:")

lon = df["lon"].values
lat = df["lat"].values
pts_geo = list(zip(lon, lat))

# Pre-transform once for UTM rasters
to_utm = pyproj.Transformer.from_crs(4326, 32719, always_xy=True)
x_utm, y_utm = to_utm.transform(lon, lat)
pts_utm = list(zip(x_utm, y_utm))

for tif in sat_features:
    name = tif.stem
    with rasterio.open(tif) as src:
        crs_epsg = src.crs.to_epsg() if src.crs else None
        pts = pts_utm if crs_epsg == 32719 else pts_geo
        vals = np.array(list(src.sample(pts, indexes=1, masked=False))).ravel()
        if src.nodata is not None:
            vals = np.where(vals == src.nodata, np.nan, vals)
    df[name] = vals
    valid = np.sum(np.isfinite(vals))
    print(f"  {name:<20s}  CRS={crs_epsg}  valid={valid}/{len(vals)}")

# Drop rows where any satellite feature is NaN (could happen at edges)
sat_cols = [t.stem for t in sat_features]
before = len(df)
df_clean = df.dropna(subset=sat_cols)
after = len(df_clean)
print(f"\nFootprints after dropping satellite-NaN: {after} (was {before})")

out = ROOT / "samples" / "pilot_samples_full.csv"
df_clean.to_csv(out, index=False)
print(f"\n→ {out}")
print(f"Final: {len(df_clean)} rows × {len(df_clean.columns)} columns")

# Correlations with corrected residual (only new + interesting old features)
print(f"\n=== Correlations with EGM2008-corrected residual ===")
target = df_clean.residual_corrected
all_feats = (
    sat_cols
    + ["valley_depth", "tri", "vrm", "openness_negative", "slope",
       "twi", "hand", "geomorphons", "mrvbf", "dev"]
)
corrs = {}
for c in all_feats:
    if c not in df_clean.columns:
        continue
    s = df_clean[c].dropna()
    if s.std() > 0 and len(s) > 10:
        t = target.loc[s.index]
        corrs[c] = np.corrcoef(s, t)[0, 1]

# Group output: SAT vs TERRAIN
print("\n  --- New satellite features ---")
for c in sat_cols:
    if c in corrs:
        print(f"  {c:<22s} r = {corrs[c]:+.4f}")

print("\n  --- Strongest terrain features (ref) ---")
for c, r in sorted(
    [(k, v) for k, v in corrs.items() if k not in sat_cols],
    key=lambda x: -abs(x[1]),
)[:10]:
    print(f"  {c:<22s} r = {r:+.4f}")
