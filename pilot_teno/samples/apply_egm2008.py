#!/usr/bin/env python3
"""Apply EGM2008 geoid correction to ATL08 elevations.

ATL08 v7 reports h_te_best_fit as WGS84 ellipsoidal height.
FABDEM uses EGM2008 orthometric heights.

Pipeline:
  h_te_orthometric = h_te_ellipsoidal - N(lat, lon)
where N is geoid undulation from EGM2008.

Compare residuals before/after correction.
"""
import os
os.environ["PROJ_NETWORK"] = "ON"
import numpy as np
import pandas as pd
import pyproj

pyproj.network.set_network_enabled(True)

# WGS84 3D ellipsoidal (4979) → WGS84 + EGM2008 orthometric (9518)
transformer = pyproj.Transformer.from_crs(4979, 9518, always_xy=True)

df = pd.read_csv("pilot_samples.csv")
print(f"Loaded {len(df)} footprints")

# Per-footprint geoid correction
lon = df["lon"].values
lat = df["lat"].values
h_ell = df["h_te"].values

_, _, h_ortho = transformer.transform(lon, lat, h_ell)
df["h_te_orthometric"] = h_ortho
df["geoid_N"] = h_ell - h_ortho

print(f"\n=== EGM2008 geoid undulation N (m) over bbox ===")
print(f"  Min:  {df.geoid_N.min():.3f}")
print(f"  Max:  {df.geoid_N.max():.3f}")
print(f"  Mean: {df.geoid_N.mean():.3f}")
print(f"  Std:  {df.geoid_N.std():.3f}")
print(f"  Range across bbox: {df.geoid_N.max() - df.geoid_N.min():.3f} m")

# Residual before/after
df["residual_raw"] = df["h_te"] - df["fabdem"]
df["residual_corrected"] = df["h_te_orthometric"] - df["fabdem"]

print(f"\n=== RESIDUAL: raw vs EGM2008-corrected ===")
for label, col in [("Raw (ellipsoidal)", "residual_raw"),
                    ("Corrected (orthometric)", "residual_corrected")]:
    r = df[col]
    print(f"\n  {label}:")
    print(f"    Mean:    {r.mean():+8.3f} m")
    print(f"    Median:  {r.median():+8.3f} m")
    print(f"    Std:     {r.std():8.3f} m")
    print(f"    P5–P95:  [{r.quantile(0.05):+.2f}, {r.quantile(0.95):+.2f}]")
    print(f"    |res|<1m: {(r.abs()<1).sum():>4d} ({100*(r.abs()<1).sum()/len(r):.1f}%)")
    print(f"    |res|<3m: {(r.abs()<3).sum():>4d} ({100*(r.abs()<3).sum()/len(r):.1f}%)")
    print(f"    |res|>5m: {(r.abs()>5).sum():>4d} ({100*(r.abs()>5).sum()/len(r):.1f}%)")

# Feature correlations with CORRECTED residual
print(f"\n=== Feature correlations with EGM2008-corrected residual ===")
exclude = {"lon","lat","h_te","h_te_unc","te_qual","night","terrain_flg","n_te_phot",
           "snow","water","cloud","beam","fabdem","filled","h_te_orthometric",
           "geoid_N","residual_raw","residual_corrected"}
feats = [c for c in df.columns if c not in exclude]
corrs = {}
for c in feats:
    s = df[c].dropna()
    t = df.residual_corrected.loc[s.index]
    if s.std() > 0 and len(s) > 10:
        corrs[c] = np.corrcoef(s, t)[0, 1]
for c, r in sorted(corrs.items(), key=lambda x: -abs(x[1]))[:15]:
    print(f"  {c:<25s} r = {r:+.4f}")

df.to_csv("pilot_samples_egm2008.csv", index=False)
print(f"\n→ pilot_samples_egm2008.csv")
