#!/usr/bin/env python3
"""P1 stage 01: download FABDEM tile via GEE.
Reads BBOX from $PIPELINE_BBOX, workdir from $PIPELINE_WORKDIR.
"""
import os, sys
from pathlib import Path

ROOT = Path(os.environ.get("PIPELINE_WORKDIR", str(Path(__file__).resolve().parent.parent)))
BBOX = [float(x) for x in os.environ.get("PIPELINE_BBOX", "-72,-36,-71,-35").split(",")]
OUT = ROOT / "dem" / "fabdem.tif"
OUT.parent.mkdir(parents=True, exist_ok=True)
print(f"[01_fabdem] WORKDIR={ROOT}  BBOX={BBOX}")

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
