#!/usr/bin/env python3
"""Stage 04: search & download ATL08 granules for the tile."""
import os, sys
from pathlib import Path

ROOT = Path(os.environ.get("PIPELINE_WORKDIR", str(Path(__file__).resolve().parent.parent)))
ICE = ROOT / "icesat2"
ICE.mkdir(parents=True, exist_ok=True)
BBOX = [float(x) for x in os.environ.get("PIPELINE_BBOX", "-72,-36,-71,-35").split(",")]
N_MAX = 30
TIME_START = "2023-01-01"
TIME_END = "2024-12-31"
print(f"[04_atl08] WORKDIR={ROOT}  BBOX={BBOX}")

existing = list(ICE.glob("ATL08_*.h5"))
if len(existing) >= N_MAX:
    print(f"✓ already have {len(existing)} ATL08 granules")
    sys.exit(0)

import earthaccess
print("Authenticating with NASA Earthdata...")
earthaccess.login(strategy="netrc")

print(f"Searching ATL08 over {BBOX} ({TIME_START} to {TIME_END})...")
results = earthaccess.search_data(
    short_name="ATL08", version="007",
    bounding_box=tuple(BBOX),
    temporal=(TIME_START, TIME_END),
    count=N_MAX,
)
print(f"Found {len(results)} granules; downloading...")
files = earthaccess.download(results, local_path=str(ICE))
print(f"Downloaded {len(files)} files to {ICE}")
