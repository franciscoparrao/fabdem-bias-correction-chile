#!/usr/bin/env python3
"""Stratify Vietnam + Atacama OOD predictions to identify sub-regimes where
the Mediterranean-trained model retains or loses transferability.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
P4 = ROOT / "scale_p4" / "samples_unified"

df = pd.read_csv(P4 / "samples_p4_with_mm_predictions.csv")
print(f"Loaded {len(df)} rows")

# Errors
if "err_raw" not in df.columns:
    df["err_raw"] = df["fabdem"] - df["h_te_orthometric"]
    df["err_corr"] = (df["fabdem"] + df["pred_residual_mm_model"]) - df["h_te_orthometric"]


def metrics(sub):
    if len(sub) < 30:
        return None
    rmse_r = float(np.sqrt(np.mean(sub.err_raw**2)))
    rmse_c = float(np.sqrt(np.mean(sub.err_corr**2)))
    bias_r = float(sub.err_raw.mean())
    bias_c = float(sub.err_corr.mean())
    delta = 100.0 * (rmse_c - rmse_r) / rmse_r
    improved = float((np.abs(sub.err_corr) < np.abs(sub.err_raw)).mean())
    return {
        "n": len(sub), "rmse_raw": rmse_r, "rmse_corr": rmse_c,
        "bias_raw": bias_r, "bias_corr": bias_c,
        "delta_pct": delta, "frac_improved": improved,
    }


def stratify(d, label):
    print(f"\n{'='*78}")
    print(f"STRATIFIED: {label}  (n={len(d)})")
    print('='*78)

    # Elevation bands
    print(f"\n  By elevation (m):")
    elev_bins = [-100, 50, 200, 500, 1000, 2000, 9000]
    elev_labels = ["<50", "50-200", "200-500", "500-1k", "1k-2k", ">2k"]
    d["elev_band"] = pd.cut(d.h_te_orthometric, bins=elev_bins, labels=elev_labels)
    for b in elev_labels:
        sub = d[d.elev_band == b]
        m = metrics(sub)
        if m:
            print(f"    {b:>10s}  n={m['n']:>5d}  raw={m['rmse_raw']:5.2f}  "
                  f"corr={m['rmse_corr']:5.2f}  Δ={m['delta_pct']:+6.1f}%  "
                  f"impr={m['frac_improved']*100:>4.1f}%  "
                  f"bias_raw={m['bias_raw']:+.2f}→{m['bias_corr']:+.2f}")

    # Slope bands
    print(f"\n  By slope (degrees):")
    slope_bins = [-1, 1, 5, 15, 30, 90]
    slope_labels = ["<1°", "1-5°", "5-15°", "15-30°", ">30°"]
    d["slope_band"] = pd.cut(d.slope, bins=slope_bins, labels=slope_labels)
    for b in slope_labels:
        sub = d[d.slope_band == b]
        m = metrics(sub)
        if m:
            print(f"    {b:>10s}  n={m['n']:>5d}  raw={m['rmse_raw']:5.2f}  "
                  f"corr={m['rmse_corr']:5.2f}  Δ={m['delta_pct']:+6.1f}%  "
                  f"impr={m['frac_improved']*100:>4.1f}%")

    # NDVI bands
    print(f"\n  By NDVI:")
    ndvi_bins = [-2, 0.1, 0.3, 0.5, 0.7, 1.0]
    ndvi_labels = ["<0.1 bare", "0.1-0.3", "0.3-0.5", "0.5-0.7", ">0.7 dense"]
    d["ndvi_band"] = pd.cut(d.ndvi, bins=ndvi_bins, labels=ndvi_labels)
    for b in ndvi_labels:
        sub = d[d.ndvi_band == b]
        m = metrics(sub)
        if m:
            print(f"    {b:>12s}  n={m['n']:>5d}  raw={m['rmse_raw']:5.2f}  "
                  f"corr={m['rmse_corr']:5.2f}  Δ={m['delta_pct']:+6.1f}%  "
                  f"impr={m['frac_improved']*100:>4.1f}%")


for region, sub in df.groupby("regime"):
    stratify(sub.copy(), region)

# Save per-stratum table
rows = []
for region, sub in df.groupby("regime"):
    # elevation
    for col, bins, labels in [
        ("h_te_orthometric", [-100, 50, 200, 500, 1000, 2000, 9000],
            ["<50", "50-200", "200-500", "500-1k", "1k-2k", ">2k"]),
        ("slope", [-1, 1, 5, 15, 30, 90],
            ["<1deg", "1-5deg", "5-15deg", "15-30deg", ">30deg"]),
        ("ndvi", [-2, 0.1, 0.3, 0.5, 0.7, 1.0],
            ["bare", "ndvi_0.1-0.3", "ndvi_0.3-0.5", "ndvi_0.5-0.7", "ndvi_dense"]),
    ]:
        sub["band"] = pd.cut(sub[col], bins=bins, labels=labels)
        for b in labels:
            ss = sub[sub.band == b]
            m = metrics(ss)
            if m:
                m["dimension"] = col
                m["stratum"] = b
                m["region"] = region
                rows.append(m)

pd.DataFrame(rows).to_csv(P4 / "strat_p4.csv", index=False)
print(f"\n→ {P4 / 'strat_p4.csv'}")
