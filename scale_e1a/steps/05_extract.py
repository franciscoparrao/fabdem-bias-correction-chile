#!/usr/bin/env python3
"""E1a stage 05: extract ATL08 footprints across all granules, filter quality,
apply EGM2008 correction.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

os.environ.setdefault("PROJ_NETWORK", "ON")
import pyproj
pyproj.network.set_network_enabled(True)

ROOT = Path(__file__).resolve().parent.parent
ICE = ROOT / "icesat2"
OUT = ROOT / "samples" / "atl08_footprints.csv"
OUT.parent.mkdir(exist_ok=True)
BBOX = [-72.0, -36.0, -71.0, -35.0]
W, S, E, N = BBOX
BEAMS = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]

if OUT.exists() and OUT.stat().st_size > 10_000:
    print(f"✓ {OUT.name} already exists")
    sys.exit(0)

granules = sorted(ICE.glob("ATL08_*.h5"))
print(f"Processing {len(granules)} granules...")

records = []
for gi, gpath in enumerate(granules):
    try:
        with h5py.File(gpath, "r") as f:
            for beam in BEAMS:
                if beam not in f:
                    continue
                ls = f[beam]["land_segments"]
                lon = ls["longitude"][:]
                lat = ls["latitude"][:]
                m = (lon >= W) & (lon <= E) & (lat >= S) & (lat <= N)
                if not m.any():
                    continue
                idx = np.where(m)[0]
                df = pd.DataFrame({
                    "granule": gpath.stem,
                    "beam": beam,
                    "lon": lon[idx], "lat": lat[idx],
                    "h_te": ls["terrain/h_te_best_fit"][:][idx],
                    "h_te_unc": ls["terrain/h_te_uncertainty"][:][idx],
                    "night": ls["night_flag"][:][idx],
                    "terrain_flg": ls["terrain_flg"][:][idx],
                    "n_te_phot": ls["terrain/n_te_photons"][:][idx],
                    "snow": ls["segment_snowcover"][:][idx],
                    "water": ls["segment_watermask"][:][idx],
                    "cloud": ls["cloud_flag_atm"][:][idx],
                })
                records.append(df)
    except Exception as e:
        print(f"  ⚠ {gpath.name}: {e}")
        continue
    if (gi + 1) % 5 == 0:
        print(f"  {gi+1}/{len(granules)} granules processed, "
              f"{sum(len(r) for r in records)} points so far")

if not records:
    print("✗ no footprints extracted")
    sys.exit(1)

raw = pd.concat(records, ignore_index=True)
print(f"\nRaw points in bbox: {len(raw)}")

# Quality filters
mask = (
    np.isfinite(raw.h_te) & (raw.h_te > -1000) & (raw.h_te < 9000) &
    (raw.terrain_flg == 1) &
    (raw.h_te_unc < 10.0) &
    (raw.n_te_phot >= 30) &
    (raw.snow == 1) &
    (raw.water == 0) &
    (raw.cloud <= 1)
)
filt = raw[mask].reset_index(drop=True)
print(f"After quality filters: {len(filt)} ({100*len(filt)/len(raw):.1f}%)")

# EGM2008 correction
print("Applying EGM2008 correction (PROJ network)...")
t = pyproj.Transformer.from_crs(4979, 9518, always_xy=True)
_, _, h_ortho = t.transform(filt.lon.values, filt.lat.values, filt.h_te.values)
filt["h_te_orthometric"] = h_ortho
filt["geoid_N"] = filt.h_te - filt.h_te_orthometric

print(f"  Geoid N: min={filt.geoid_N.min():.2f}, max={filt.geoid_N.max():.2f}, "
      f"mean={filt.geoid_N.mean():.2f}, range={filt.geoid_N.max()-filt.geoid_N.min():.3f} m")

filt.to_csv(OUT, index=False)
print(f"\n→ {OUT} ({len(filt)} rows × {len(filt.columns)} cols)")
print(f"Per-granule yield (top 10):")
print(filt.groupby("granule").size().sort_values(ascending=False).head(10).to_string())
