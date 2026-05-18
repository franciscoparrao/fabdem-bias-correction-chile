#!/usr/bin/env python3
"""L3a: Stratified elevation-centric validation across geomorphometric regimes.

Uses OOF spatial-CV predictions from mm_predictions.csv (honest, paper-grade)
+ feature stack to stratify performance.

Strata reported:
  - Elevation band: <500, 500-1000, 1000-2000, 2000-3000, >3000 m
  - Slope class:    flat (<5°), gentle (5-15°), moderate (15-25°), steep (>25°)
  - HAND band:      <2m, 2-10m, 10-50m, >50m  (proxy for drainage proximity)
  - Geomorphons:    10 classes (Jasiewicz & Stepinski 2013)
  - Tile / Sub-region

For each stratum: n, RMSE_raw, RMSE_corr, MAE_raw, MAE_corr, improvement %.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
SAMP = ROOT / "scale_p1" / "samples_unified"
OUT = ROOT / "p2_validation" / "stratified"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SAMP / "mm_predictions.csv")
print(f"Loaded {len(df)} predictions")
# OOF residual prediction is in df.pred_residual; FABDEM + pred = fabdem_corrected
# True = h_te_orthometric

df["err_raw"]  = df["fabdem"] - df["h_te_orthometric"]
df["err_corr"] = df["fabdem_corrected"] - df["h_te_orthometric"]
df["abs_err_raw"]  = df.err_raw.abs()
df["abs_err_corr"] = df.err_corr.abs()

def stratum_stats(group):
    return pd.Series({
        "n": len(group),
        "rmse_raw":  float(np.sqrt(np.mean(group.err_raw**2))),
        "rmse_corr": float(np.sqrt(np.mean(group.err_corr**2))),
        "mae_raw":   float(group.abs_err_raw.mean()),
        "mae_corr":  float(group.abs_err_corr.mean()),
        "bias_raw":  float(group.err_raw.mean()),
        "bias_corr": float(group.err_corr.mean()),
    })


# ===== Stratum 1: Elevation band =====
print("\n" + "="*72)
print("STRATUM 1: Elevation band (h_te_orthometric)")
print("="*72)
df["elev_band"] = pd.cut(
    df.h_te_orthometric,
    bins=[-100, 500, 1000, 2000, 3000, 9000],
    labels=["<500m", "500–1000m", "1000–2000m", "2000–3000m", ">3000m"],
)
g1 = df.groupby("elev_band", observed=True).apply(stratum_stats, include_groups=False)
g1["improve_pct"] = 100 * (g1.rmse_raw - g1.rmse_corr) / g1.rmse_raw
print(g1.round(3).to_string())
g1.to_csv(OUT / "by_elevation.csv")


# ===== Stratum 2: Slope class =====
print("\n" + "="*72)
print("STRATUM 2: Slope class (in degrees)")
print("="*72)
df["slope_class"] = pd.cut(
    df.slope,
    bins=[-1, 5, 15, 25, 90],
    labels=["flat (<5°)", "gentle (5–15°)", "moderate (15–25°)", "steep (>25°)"],
)
g2 = df.groupby("slope_class", observed=True).apply(stratum_stats, include_groups=False)
g2["improve_pct"] = 100 * (g2.rmse_raw - g2.rmse_corr) / g2.rmse_raw
print(g2.round(3).to_string())
g2.to_csv(OUT / "by_slope.csv")


# ===== Stratum 3: HAND band (drainage proximity proxy) =====
print("\n" + "="*72)
print("STRATUM 3: HAND band (height above nearest drainage)")
print("="*72)
df["hand_band"] = pd.cut(
    df.hand,
    bins=[-1, 2, 10, 50, 200, 9999],
    labels=["<2m (near stream)", "2–10m (floodplain)", "10–50m (lower hills)",
            "50–200m (hillside)", ">200m (montane)"],
)
g3 = df.groupby("hand_band", observed=True).apply(stratum_stats, include_groups=False)
g3["improve_pct"] = 100 * (g3.rmse_raw - g3.rmse_corr) / g3.rmse_raw
print(g3.round(3).to_string())
g3.to_csv(OUT / "by_hand.csv")


# ===== Stratum 4: Geomorphons (10 classes) =====
print("\n" + "="*72)
print("STRATUM 4: Geomorphon class")
print("="*72)
GEOM_NAMES = {1:"flat", 2:"peak", 3:"ridge", 4:"shoulder", 5:"spur",
               6:"slope", 7:"hollow", 8:"footslope", 9:"valley", 10:"pit"}
df["geom_class"] = df.geomorphons.map(GEOM_NAMES)
g4 = df.groupby("geom_class", observed=True).apply(stratum_stats, include_groups=False)
g4["improve_pct"] = 100 * (g4.rmse_raw - g4.rmse_corr) / g4.rmse_raw
g4 = g4.sort_values("n", ascending=False)
print(g4.round(3).to_string())
g4.to_csv(OUT / "by_geomorphon.csv")


# ===== Stratum 5: Per-tile (already done in finalize, replicated here for completeness) =====
print("\n" + "="*72)
print("STRATUM 5: Per tile (spatial CV recap)")
print("="*72)
g5 = df.groupby("tile", observed=True).apply(stratum_stats, include_groups=False)
g5["improve_pct"] = 100 * (g5.rmse_raw - g5.rmse_corr) / g5.rmse_raw
g5 = g5.sort_values("improve_pct", ascending=False)
print(g5.round(3).to_string())
g5.to_csv(OUT / "by_tile.csv")


# ===== Stratum 6: NDVI bands (vegetation cover proxy) =====
print("\n" + "="*72)
print("STRATUM 6: NDVI band (vegetation cover)")
print("="*72)
df["ndvi_band"] = pd.cut(
    df.ndvi,
    bins=[-1, 0.0, 0.2, 0.4, 0.6, 1.0],
    labels=["bare (<0)", "sparse (0–0.2)", "low veg (0.2–0.4)",
            "moderate (0.4–0.6)", "dense (>0.6)"],
)
g6 = df.groupby("ndvi_band", observed=True).apply(stratum_stats, include_groups=False)
g6["improve_pct"] = 100 * (g6.rmse_raw - g6.rmse_corr) / g6.rmse_raw
print(g6.round(3).to_string())
g6.to_csv(OUT / "by_ndvi.csv")


# ===== Joint: HAND × slope (where the corrections matter most) =====
print("\n" + "="*72)
print("CROSS-STRATUM: HAND band × Slope class")
print("="*72)
joint = df.groupby(["hand_band", "slope_class"], observed=True).apply(
    stratum_stats, include_groups=False
)
joint["improve_pct"] = 100 * (joint.rmse_raw - joint.rmse_corr) / joint.rmse_raw
print(joint.round(3).to_string())
joint.to_csv(OUT / "by_hand_x_slope.csv")


# ===== Summary table for paper =====
print("\n" + "="*72)
print("SUMMARY TABLE: top improvement strata across all dimensions")
print("="*72)
summary_rows = []
for name, g in [("Elevation", g1), ("Slope", g2), ("HAND", g3),
                 ("Geomorphon", g4), ("Tile", g5), ("NDVI", g6)]:
    for idx, row in g.iterrows():
        summary_rows.append({
            "dimension": name, "stratum": str(idx),
            "n": int(row.n), "rmse_raw": float(row.rmse_raw),
            "rmse_corr": float(row.rmse_corr),
            "improve_pct": float(row.improve_pct),
        })
summary = pd.DataFrame(summary_rows).sort_values("improve_pct", ascending=False)
top10 = summary.head(15)
worst10 = summary.tail(10)
print("\nTop 15 best-improving strata:")
print(top10.round(2).to_string(index=False))
print("\nBottom 10 worst-improving strata:")
print(worst10.round(2).to_string(index=False))
summary.to_csv(OUT / "all_strata_summary.csv", index=False)

# Final aggregate metrics
print("\n" + "="*72)
print("AGGREGATE")
print("="*72)
n = len(df)
rmse_raw  = float(np.sqrt(np.mean(df.err_raw**2)))
rmse_corr = float(np.sqrt(np.mean(df.err_corr**2)))
print(f"  N: {n:,}")
print(f"  RMSE raw  : {rmse_raw:.3f} m")
print(f"  RMSE corr : {rmse_corr:.3f} m   Δ {100*(rmse_corr-rmse_raw)/rmse_raw:+.1f}%")
print(f"  → Detailed strata CSVs saved in {OUT}")
