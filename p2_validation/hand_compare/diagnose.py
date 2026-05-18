#!/usr/bin/env python3
"""Diagnose why HAND-based comparison shows no improvement.

Hypotheses to test:
  H1: Both DEMs produce similar drainage → similar HAND → no differentiation
  H2: Correction is small relative to local terrain gradient
  H3: Within CIGIDEN flood zone, corrected DEM is actually closer to water surface
"""
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
P2 = ROOT / "p2_validation"
INFER = ROOT / "scale_p1" / "inference"
BBOX = [-72.0, -35.26, -71.69, -34.88]
W, S, E, N = BBOX


def read_window(path, bbox):
    with rasterio.open(path) as src:
        w = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=w).astype(np.float32)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return arr, src.window_transform(w), w.height, w.width


# Load all 5 rasters in Licantén window
print("=== Loading rasters within Licantén bbox ===")
fabdem_raw, t, H, W_w = read_window(INFER / "mm_fabdem_raw.tif", BBOX)
fabdem_cor, _, _, _ = read_window(INFER / "mm_corrected.tif", BBOX)
residual, _, _, _ = read_window(INFER / "mm_residual.tif", BBOX)
hand_raw, _, _, _ = read_window(P2 / "hand_compare" / "mm_hand_raw.tif", BBOX)
hand_cor, _, _, _ = read_window(P2 / "hand_compare" / "hydro_corrected" / "mm_hand_corrected.tif", BBOX)

# Rasterize CIGIDEN
print("Rasterizing CIGIDEN...")
gdf = gpd.read_file(P2 / "cigiden" / "inundacion_licanten.shp").to_crs(4326)
gdf = gdf.clip(box(W, S, E, N))
cig = rasterize(
    ((g, 1) for g in gdf.geometry if g and not g.is_empty),
    out_shape=(H, W_w), transform=t, fill=0, dtype=np.uint8,
).astype(bool)

valid = (np.isfinite(fabdem_raw) & np.isfinite(fabdem_cor) &
         np.isfinite(hand_raw) & np.isfinite(hand_cor))
inside = cig & valid
outside = ~cig & valid

print(f"\nLicantén bbox: {H}×{W_w} px ({H*W_w:,})")
print(f"  Valid:         {valid.sum():,}")
print(f"  CIGIDEN flood: {inside.sum():,} px ({100*inside.sum()/valid.sum():.2f}%)")
print(f"  Non-flooded:   {outside.sum():,} px")


# ===== H1: Are HAND distributions different in flooded zone? =====
print("\n" + "="*68)
print("H1: HAND distribution within vs outside CIGIDEN flood")
print("="*68)
for label, arr in [("HAND_raw  inside ", hand_raw[inside]),
                    ("HAND_raw  outside", hand_raw[outside]),
                    ("HAND_corr inside ", hand_cor[inside]),
                    ("HAND_corr outside", hand_cor[outside])]:
    arr = arr[np.isfinite(arr)]
    print(f"  {label}: n={len(arr):>7d}  median={np.median(arr):>6.2f}  "
          f"p25={np.percentile(arr,25):>5.2f}  p75={np.percentile(arr,75):>5.2f}  "
          f"max={arr.max():.1f}")

# Separation: are flood pixels much lower in HAND than non-flood pixels?
sep_raw = np.median(hand_raw[outside & np.isfinite(hand_raw)]) - np.median(hand_raw[inside & np.isfinite(hand_raw)])
sep_cor = np.median(hand_cor[outside & np.isfinite(hand_cor)]) - np.median(hand_cor[inside & np.isfinite(hand_cor)])
print(f"\n  Separation (median outside − inside):")
print(f"    HAND_raw : {sep_raw:.2f} m")
print(f"    HAND_corr: {sep_cor:.2f} m")
print(f"    Larger separation = DEM does better at distinguishing flood vs land")


# ===== H2: Distribution of correction (residual) inside vs outside flood =====
print("\n" + "="*68)
print("H2: Correction magnitude inside vs outside CIGIDEN flood")
print("="*68)
res_inside = residual[inside]
res_outside = residual[outside]
res_inside = res_inside[np.isfinite(res_inside)]
res_outside = res_outside[np.isfinite(res_outside)]
print(f"  Inside flood  : mean={res_inside.mean():+.3f}  median={np.median(res_inside):+.3f}  std={res_inside.std():.3f}")
print(f"  Outside flood : mean={res_outside.mean():+.3f}  median={np.median(res_outside):+.3f}  std={res_outside.std():.3f}")
print(f"  Δ mean: {res_inside.mean() - res_outside.mean():+.3f} m  "
      f"({'flood zones get more correction' if res_inside.mean() < res_outside.mean() else 'flood zones less correction'})")


# ===== H3: Elevation distribution within flood polygons (flatness check) =====
# Inside a single flood polygon, water surface is ~flat. Tighter DEM
# distribution = DEM matches flat water better.
print("\n" + "="*68)
print("H3: DEM flatness within flood polygons (water surface should be flat)")
print("="*68)
poly_stats = []
for poly in gdf.geometry:
    if poly is None or poly.is_empty:
        continue
    poly_mask = rasterize([(poly, 1)], out_shape=(H, W_w), transform=t,
                           fill=0, dtype=np.uint8).astype(bool)
    pm = poly_mask & valid
    if pm.sum() < 30:  # skip tiny polygons
        continue
    z_raw = fabdem_raw[pm]
    z_cor = fabdem_cor[pm]
    poly_stats.append({"n": int(pm.sum()),
                        "std_raw": float(np.std(z_raw)),
                        "std_cor": float(np.std(z_cor)),
                        "iqr_raw": float(np.percentile(z_raw, 75) - np.percentile(z_raw, 25)),
                        "iqr_cor": float(np.percentile(z_cor, 75) - np.percentile(z_cor, 25))})

import pandas as pd
ps = pd.DataFrame(poly_stats)
print(f"  Analyzed {len(ps)} polygons with ≥30 valid px")
print(f"  Median IQR raw:  {ps.iqr_raw.median():.3f} m")
print(f"  Median IQR corr: {ps.iqr_cor.median():.3f} m  "
      f"({'tighter→better' if ps.iqr_cor.median() < ps.iqr_raw.median() else 'wider'})")
print(f"  Median STD raw:  {ps.std_raw.median():.3f} m")
print(f"  Median STD corr: {ps.std_cor.median():.3f} m")
n_better = (ps.iqr_cor < ps.iqr_raw).sum()
print(f"  Polygons where corrected is tighter (lower IQR): {n_better}/{len(ps)} "
      f"({100*n_better/len(ps):.1f}%)")
ps.to_csv(P2 / "hand_compare" / "polygon_flatness.csv", index=False)

# ===== H4: At ATL08 footprints WITHIN Licantén bbox =====
print("\n" + "="*68)
print("H4: Improvement specifically at ATL08 footprints within Licantén bbox")
print("="*68)
SAMP = ROOT / "scale_p1" / "samples_unified"
df = pd.read_csv(SAMP / "samples_mm_full.csv")
m = (df.lon >= W) & (df.lon <= E) & (df.lat >= S) & (df.lat <= N)
sub = df[m].copy()
print(f"  Footprints in bbox: {len(sub)}")
if len(sub) > 0:
    rmse_raw = float(np.sqrt(np.mean((sub.h_te_orthometric - sub.fabdem)**2)))
    # Sample corrected at these points
    with rasterio.open(INFER / "mm_corrected.tif") as src:
        pts = list(zip(sub.lon.values, sub.lat.values))
        h_cor = np.array(list(src.sample(pts, indexes=1, masked=False))).ravel()
    sub = sub[np.isfinite(h_cor)]
    h_cor = h_cor[np.isfinite(h_cor)]
    rmse_cor = float(np.sqrt(np.mean((sub.h_te_orthometric - h_cor)**2)))
    print(f"  RMSE raw  in Licantén bbox: {rmse_raw:.3f} m")
    print(f"  RMSE corr in Licantén bbox: {rmse_cor:.3f} m   Δ {100*(rmse_cor-rmse_raw)/rmse_raw:+.1f}%")
