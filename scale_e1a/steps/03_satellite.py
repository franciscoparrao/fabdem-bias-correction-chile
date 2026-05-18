#!/usr/bin/env python3
"""E1a stage 03: Sentinel-1/2 stack via planetary-computer + stackstac.

Same approach as pilot but adapted to 1° tile bbox.
Per-MGRS-tile top-N to ensure full coverage.
"""
import sys
from pathlib import Path
import numpy as np
import planetary_computer as pc
import pystac_client
import stackstac
import rioxarray  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "satellite"
BBOX = [-72.0, -36.0, -71.0, -35.0]
TIME = "2022-12-01/2023-02-28"  # austral summer
N_KEEP = 3
RESOLUTION = 60  # 60 m to cap RAM on 1° tile; sampleo en footprints es tolerante

# Skip if already done
if all((OUT / f"{n}.tif").exists() and (OUT / f"{n}.tif").stat().st_size > 100_000
       for n in ["ndvi", "ndwi", "ndmi", "bsi", "ndbi",
                  "s1_vv_db", "s1_vh_db", "s1_vv_vh_ratio"]):
    print("✓ satellite stack already done")
    sys.exit(0)

CATALOG = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

def save(da, name):
    da = da.rio.write_crs("EPSG:32719")
    p = OUT / f"{name}.tif"
    da.rio.to_raster(p, compress="deflate", dtype="float32")
    finite = int(np.isfinite(da.values).sum())
    total = int(da.values.size)
    print(f"    → {name}.tif  ({finite}/{total} finite [{100*finite/total:.1f}%])")

# ---------------- Sentinel-2 ----------------
print("=== Sentinel-2 L2A ===")
items_s2 = list(CATALOG.search(
    collections=["sentinel-2-l2a"], bbox=BBOX, datetime=TIME,
    query={"eo:cloud_cover": {"lt": 20}},
).items())
print(f"  {len(items_s2)} candidate scenes (cloud<20%)")
by_tile = {}
for it in items_s2:
    tile = it.properties.get("s2:mgrs_tile", "?")
    by_tile.setdefault(tile, []).append(it)
selected = []
for tile, group in by_tile.items():
    group.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    selected.extend(group[:N_KEEP])
print(f"  {len(selected)} scenes selected (top {N_KEEP} per MGRS tile, {len(by_tile)} tiles)")
for it in selected:
    print(f"    {it.properties.get('s2:mgrs_tile')} cc={it.properties.get('eo:cloud_cover'):.1f}%  {it.id[:40]}")

da = stackstac.stack(
    selected,
    assets=["B02", "B03", "B04", "B08", "B11", "B12"],
    bounds_latlon=BBOX,
    resolution=RESOLUTION,
    epsg=32719,
    chunksize=1024,
).astype("float32") / 10000.0

print("  computing S2 median...")
med = da.median(dim="time", skipna=True).compute()
B02, B03, B04, B08, B11, B12 = [med.sel(band=b).drop_vars("band", errors="ignore")
                                 for b in ["B02", "B03", "B04", "B08", "B11", "B12"]]
ndvi = (B08 - B04) / (B08 + B04 + 1e-9)
ndwi = (B03 - B08) / (B03 + B08 + 1e-9)
ndmi = (B08 - B11) / (B08 + B11 + 1e-9)
bsi = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02) + 1e-9)
ndbi = (B11 - B08) / (B11 + B08 + 1e-9)

print("  saving S2 indices...")
for name, d in [("ndvi", ndvi), ("ndwi", ndwi), ("ndmi", ndmi),
                 ("bsi", bsi), ("ndbi", ndbi)]:
    save(d, name)

# Free memory
del da, med, B02, B03, B04, B08, B11, B12, ndvi, ndwi, ndmi, bsi, ndbi

# ---------------- Sentinel-1 RTC ----------------
print("\n=== Sentinel-1 RTC ===")
items_s1 = list(CATALOG.search(
    collections=["sentinel-1-rtc"], bbox=BBOX, datetime=TIME,
).items())
print(f"  {len(items_s1)} candidate scenes")
# Limit to keep RAM bounded
items_s1 = items_s1[:12]
print(f"  using {len(items_s1)} scenes")

da_s1 = stackstac.stack(
    items_s1, assets=["vv", "vh"],
    bounds_latlon=BBOX, resolution=RESOLUTION, epsg=32719, chunksize=1024,
).astype("float32")
print("  computing S1 median...")
s1_med = da_s1.median(dim="time", skipna=True).compute()
vv = s1_med.sel(band="vv").drop_vars("band", errors="ignore")
vh = s1_med.sel(band="vh").drop_vars("band", errors="ignore")
vv_db = 10 * np.log10(np.maximum(vv, 1e-6))
vh_db = 10 * np.log10(np.maximum(vh, 1e-6))
ratio = vv_db - vh_db

print("  saving S1 features...")
for name, d in [("s1_vv_db", vv_db), ("s1_vh_db", vh_db), ("s1_vv_vh_ratio", ratio)]:
    save(d, name)
print("Done.")
