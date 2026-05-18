#!/usr/bin/env python3
"""E1a stage 04: search & download ATL08 granules for the tile.

Strategy:
  - Search recent granules (2023-01 to 2024-12) crossing the bbox
  - Download top-N most recent (cap to keep disk usage bounded)
  - Use curl + netrc (no MCP overhead)
"""
import subprocess
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
ICE = ROOT / "icesat2"
BBOX = [-72.0, -36.0, -71.0, -35.0]
N_MAX = 30  # cap to keep disk usage bounded (~2-3 GB)
TIME_START = "2023-01-01"
TIME_END = "2024-12-31"

# Skip if we already have enough granules
existing = list(ICE.glob("ATL08_*.h5"))
if len(existing) >= N_MAX:
    print(f"✓ already have {len(existing)} ATL08 granules")
    sys.exit(0)

# Search via earthaccess (preferred over MCP for direct Python use)
try:
    import earthaccess
except ImportError:
    print("⚠ earthaccess not installed, using fallback curl-only approach")
    print("ABORT: please install earthaccess: pip install --user --break-system-packages earthaccess")
    sys.exit(1)

print("Authenticating with NASA Earthdata...")
earthaccess.login(strategy="netrc")

print(f"Searching ATL08 over {BBOX} ({TIME_START} to {TIME_END})...")
results = earthaccess.search_data(
    short_name="ATL08",
    version="007",
    bounding_box=tuple(BBOX),
    temporal=(TIME_START, TIME_END),
    count=N_MAX,
)
print(f"Found {len(results)} granules; downloading...")
files = earthaccess.download(results, local_path=str(ICE))
print(f"Downloaded {len(files)} files to {ICE}")
for f in files[:5]:
    print(f"  - {Path(f).name}")
if len(files) > 5:
    print(f"  ... +{len(files)-5} more")
