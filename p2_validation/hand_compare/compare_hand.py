#!/usr/bin/env python3
"""HAND-based flood extent comparison: FABDEM raw vs FABDEM-ML vs CIGIDEN reference.

Rasterizes CIGIDEN Licantén flood polygons onto our 30m EPSG:4326 grid,
then computes flood masks from HAND at multiple thresholds and compares.

Limitations to declare:
  - HAND threshold = uniform "height above drainage" — NOT a hydrodynamic model
  - Doesn't account for discharge, channel slope, friction, time evolution
  - Intended as preliminary comparison BEFORE LISFLOOD-FP investment
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
import geopandas as gpd
from pathlib import Path

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem/p2_validation")
SHP = ROOT / "cigiden" / "inundacion_licanten.shp"
HAND_RAW = ROOT / "hand_compare" / "mm_hand_raw.tif"
HAND_COR = ROOT / "hand_compare" / "hydro_corrected" / "mm_hand_corrected.tif"

# Restrict comparison to Licantén bbox intersected with our DEM coverage
# CIGIDEN bounds: [-72.21, -35.26, -71.69, -34.88], DEM W edge = -72
# Use the intersection
BBOX = [-72.0, -35.26, -71.69, -34.88]
W, S, E, N = BBOX

print("=== Loading CIGIDEN polygons ===")
cigiden = gpd.read_file(SHP).to_crs(4326)
print(f"  {len(cigiden)} polygons in EPSG:4326")
# Clip to BBOX
from shapely.geometry import box
bbox_geom = box(W, S, E, N)
cigiden_clip = cigiden.clip(bbox_geom)
print(f"  After clip to bbox: {len(cigiden_clip)} polygons")
area_km2 = cigiden_clip.to_crs(32719).geometry.area.sum() / 1e6
print(f"  Total flood area: {area_km2:.3f} km²")


def get_window_arr(path, bbox):
    """Read raster within bbox, return (array, transform)."""
    with rasterio.open(path) as src:
        w = from_bounds(*bbox, transform=src.transform)
        w = w.round_offsets().round_lengths()
        arr = src.read(1, window=w).astype(np.float32)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        # Also replace inf
        arr = np.where(np.isfinite(arr), arr, np.nan)
        transform = src.window_transform(w)
        return arr, transform, w.height, w.width


print("\n=== Reading HAND raw + corrected within bbox ===")
hand_raw, trans_r, H_r, W_r = get_window_arr(HAND_RAW, BBOX)
hand_cor, trans_c, H_c, W_c = get_window_arr(HAND_COR, BBOX)
assert (H_r, W_r) == (H_c, W_c), f"Mismatch raw {H_r}x{W_r} vs corr {H_c}x{W_c}"
print(f"  Window: {H_r} × {W_r} pixels")
print(f"  HAND raw  : valid={np.isfinite(hand_raw).sum():,}  min={np.nanmin(hand_raw):.2f}  max={np.nanmax(hand_raw):.2f}")
print(f"  HAND corr : valid={np.isfinite(hand_cor).sum():,}  min={np.nanmin(hand_cor):.2f}  max={np.nanmax(hand_cor):.2f}")

print("\n=== Rasterizing CIGIDEN onto same grid ===")
shapes = ((geom, 1) for geom in cigiden_clip.geometry if geom and not geom.is_empty)
cigiden_mask = rasterize(
    shapes,
    out_shape=(H_r, W_r),
    transform=trans_r,
    fill=0, dtype=np.uint8,
)
cigiden_bool = cigiden_mask.astype(bool)
n_cigiden = cigiden_bool.sum()
print(f"  CIGIDEN flooded pixels: {n_cigiden:,} ({100*n_cigiden/(H_r*W_r):.2f}% of bbox)")
print(f"  Equivalent area: {n_cigiden * 30 * 30 / 1e6:.3f} km² (approx, geographic CRS)")

# Mask to valid HAND area
valid = np.isfinite(hand_raw) & np.isfinite(hand_cor)
print(f"  Valid for comparison: {valid.sum():,} pixels")


def metrics(pred_bool, gt_bool, valid_mask):
    pred = pred_bool & valid_mask
    gt = gt_bool & valid_mask
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt & valid_mask).sum()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n_pred = pred.sum()
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "iou": float(iou), "recall": float(recall),
            "precision": float(precision), "f1": float(f1),
            "n_pred_flooded": int(n_pred)}

print("\n=== IoU comparison: HAND threshold sweep ===")
print(f"  CIGIDEN reference: {n_cigiden:,} flooded pixels in bbox")
print()
print(f"  {'thr':>4s} | {'src':>4s} | {'pred_flood':>10s} | {'IoU':>6s} | {'recall':>6s} | {'prec':>6s} | {'F1':>6s}")
print("  " + "-" * 64)

rows = []
for thr in [1, 2, 3, 5, 8, 12, 18, 25]:
    m_raw = (hand_raw < thr) & np.isfinite(hand_raw)
    m_cor = (hand_cor < thr) & np.isfinite(hand_cor)
    mr = metrics(m_raw, cigiden_bool, valid)
    mc = metrics(m_cor, cigiden_bool, valid)
    print(f"  {thr:>4d} | {'raw':>4s} | {mr['n_pred_flooded']:>10d} | "
          f"{mr['iou']:>.4f} | {mr['recall']:>.4f} | {mr['precision']:>.4f} | {mr['f1']:>.4f}")
    print(f"  {thr:>4d} | {'corr':>4s} | {mc['n_pred_flooded']:>10d} | "
          f"{mc['iou']:>.4f} | {mc['recall']:>.4f} | {mc['precision']:>.4f} | {mc['f1']:>.4f}")
    print()
    rows.append({"thr": thr, "src": "raw", **mr})
    rows.append({"thr": thr, "src": "corr", **mc})

results = pd.DataFrame(rows)
results.to_csv(ROOT / "hand_compare" / "iou_results.csv", index=False)
print(f"→ {ROOT/'hand_compare'/'iou_results.csv'}")

# Best IoU per source
print(f"\n=== Best IoU per source ===")
for src in ["raw", "corr"]:
    sub = results[results.src == src]
    best = sub.loc[sub.iou.idxmax()]
    print(f"  {src:>4s}: thr={best.thr:.0f}m  IoU={best.iou:.4f}  recall={best.recall:.4f}  precision={best.precision:.4f}")

# Improvement at best raw threshold
best_thr = results[results.src == "raw"].loc[results[results.src == "raw"].iou.idxmax(), "thr"]
row_raw = results[(results.src == "raw") & (results.thr == best_thr)].iloc[0]
row_cor = results[(results.src == "corr") & (results.thr == best_thr)].iloc[0]
print(f"\n=== At thr={best_thr:.0f}m (best for raw) ===")
print(f"  RAW : IoU={row_raw.iou:.4f}")
print(f"  CORR: IoU={row_cor.iou:.4f}")
delta = (row_cor.iou - row_raw.iou) / row_raw.iou * 100 if row_raw.iou else 0
print(f"  Δ IoU: {delta:+.1f}%")
