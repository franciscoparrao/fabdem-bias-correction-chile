#!/usr/bin/env python3
"""E1a stage 01: download FABDEM tile S36W072 via GEE."""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BBOX = [-72.0, -36.0, -71.0, -35.0]
OUT = ROOT / "dem" / "fabdem.tif"

if OUT.exists() and OUT.stat().st_size > 1_000_000:
    print(f"✓ {OUT.name} already exists ({OUT.stat().st_size/1024/1024:.1f} MB)")
    sys.exit(0)

import ee, geemap
ee.Initialize()
bbox = ee.Geometry.Rectangle(BBOX, proj="EPSG:4326", geodesic=False)
fabdem = (ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
          .filterBounds(bbox).mosaic().clip(bbox).rename("elevation"))
print(f"Downloading FABDEM for bbox {BBOX} @ 30m...")
geemap.download_ee_image(image=fabdem, filename=str(OUT),
                          region=bbox, crs="EPSG:4326", scale=30)
print(f"→ {OUT}  ({OUT.stat().st_size/1024/1024:.1f} MB)")
