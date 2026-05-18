#!/usr/bin/env python3
"""Raster-wide inference for one tile.

Inputs:
  - dem/fabdem.tif
  - factors/*.tif (terrain)
  - hydro/*.tif (hydrology)
  - satellite/*.tif (S1/S2, UTM 32719 @ 60m)
Outputs:
  - inference/<tile>_residual.tif  (predicted residual, m)
  - inference/<tile>_corrected.tif (FABDEM + residual)

Aligns all features to FABDEM grid (EPSG:4326 @ 30m) using reproject.

Usage: python3 infer_tile.py <tile_workdir>
"""
import sys, json, time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import xgboost as xgb

SCALE_P1 = Path(__file__).resolve().parent.parent
SAMP = SCALE_P1 / "samples_unified"
OUT = SCALE_P1 / "inference"
OUT.mkdir(parents=True, exist_ok=True)

# Load feature order + booster
metrics = json.loads((SAMP / "mm_metrics.json").read_text())
FEATURES = metrics["features"]
print(f"n_features = {len(FEATURES)}")

booster = xgb.Booster()
booster.load_model(str(SAMP / "xgb_mm_booster.json"))


def find_raster(tile_dir, feat_name):
    """Locate raster file for a given feature name across subdirs."""
    for sub in ("factors", "hydro", "satellite", "dem"):
        cands = list((tile_dir / sub).glob(f"{feat_name}.tif"))
        if cands:
            return cands[0]
    return None


def read_aligned(path, dst_profile, dst_transform, dst_crs, dst_shape):
    """Read a raster aligned to a destination grid.

    Strategy:
      1. Same CRS + transform + shape → read raw.
      2. Same shape (likely SurtGis output that shares FABDEM grid even if
         CRS metadata is broken / EngineeringCRS) → read raw and trust grid.
      3. Different grid → reproject.
    """
    with rasterio.open(path) as src:
        src_nodata = src.nodata
        # Case 1: identical
        if src.crs == dst_crs and src.transform == dst_transform and src.shape == dst_shape:
            data = src.read(1).astype(np.float32)
            if src_nodata is not None:
                data = np.where(data == src_nodata, np.nan, data)
            return data
        # Case 2: same shape but different/missing CRS metadata (SurtGis quirk)
        # Trust the grid since it was computed from FABDEM
        crs_epsg = None
        try:
            crs_epsg = src.crs.to_epsg() if src.crs else None
        except Exception:
            pass
        if src.shape == dst_shape and (crs_epsg is None or crs_epsg == dst_crs.to_epsg()):
            data = src.read(1).astype(np.float32)
            if src_nodata is not None:
                data = np.where(data == src_nodata, np.nan, data)
            return data
        # Case 3: actually different grid → reproject
        arr = np.full(dst_shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return arr


def infer_tile(tile_dir):
    tile_name = tile_dir.name
    print(f"\n=== {tile_name} ===")
    t0 = time.time()
    dem_path = tile_dir / "dem" / "fabdem.tif"
    with rasterio.open(dem_path) as fd:
        profile = fd.profile.copy()
        dst_transform = fd.transform
        dst_crs = fd.crs
        H, W = fd.shape
        fabdem_arr = fd.read(1).astype(np.float32)
        if fd.nodata is not None:
            fabdem_arr = np.where(fabdem_arr == fd.nodata, np.nan, fabdem_arr)
    print(f"  FABDEM grid: {H}×{W}  CRS={dst_crs}")

    # Stack features
    n_pixels = H * W
    print(f"  Building feature matrix ({n_pixels:,} px × {len(FEATURES)} feat)...")
    feat_mat = np.empty((n_pixels, len(FEATURES)), dtype=np.float32)
    for i, name in enumerate(FEATURES):
        path = find_raster(tile_dir, name)
        if path is None:
            print(f"    ⚠ missing {name}, filling with NaN")
            feat_mat[:, i] = np.nan
            continue
        arr = read_aligned(path, profile, dst_transform, dst_crs, (H, W))
        if name == "stream_network":
            arr = np.nan_to_num(arr, nan=0.0)  # match training fillna(0)
        feat_mat[:, i] = arr.ravel()

    # Predict residual
    print(f"  Predicting residual...")
    dmat = xgb.DMatrix(feat_mat, missing=np.nan)
    pred = booster.predict(dmat).reshape(H, W)
    # Mask: where FABDEM is invalid, prediction is meaningless
    pred[~np.isfinite(fabdem_arr)] = np.nan

    # Apply correction
    corrected = fabdem_arr + pred

    # Output COG-style profile (tiled, deflate)
    out_profile = profile.copy()
    out_profile.update(
        dtype="float32", nodata=np.nan,
        tiled=True, blockxsize=512, blockysize=512,
        compress="deflate", predictor=2,
    )
    res_path = OUT / f"{tile_name}_residual.tif"
    cor_path = OUT / f"{tile_name}_corrected.tif"

    with rasterio.open(res_path, "w", **out_profile) as ds:
        ds.write(pred.astype(np.float32), 1)
        ds.build_overviews([2, 4, 8, 16, 32], Resampling.average)
        ds.update_tags(ns="rio_overview", resampling="average")
    with rasterio.open(cor_path, "w", **out_profile) as ds:
        ds.write(corrected.astype(np.float32), 1)
        ds.build_overviews([2, 4, 8, 16, 32], Resampling.average)
        ds.update_tags(ns="rio_overview", resampling="average")

    elapsed = time.time() - t0
    valid = np.isfinite(pred).sum()
    print(f"  → {res_path.name}  ({res_path.stat().st_size/1024/1024:.1f} MB)")
    print(f"  → {cor_path.name}  ({cor_path.stat().st_size/1024/1024:.1f} MB)")
    print(f"  Stats — pred: min={np.nanmin(pred):+.3f}  max={np.nanmax(pred):+.3f}  "
          f"mean={np.nanmean(pred):+.3f}  std={np.nanstd(pred):.3f}")
    print(f"  Valid px: {valid:,}/{n_pixels:,} ({100*valid/n_pixels:.1f}%)")
    print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: infer_tile.py <tile_workdir>")
        sys.exit(1)
    infer_tile(Path(sys.argv[1]))
