#!/usr/bin/env python3
"""G3 + G4: Stratified validation + SHAP comparison for OOD generalization test.

G3: stratify humid_temperate footprints by NDVI/slope/HAND/elevation/geomorphon/tile
    using M-M model predictions (no retrain).
G4: compute SHAP on humid_temperate sample; compare with original M-M SHAP
    (computed on Mediterranean training data) to see if feature importance
    changes when the model crosses regimes.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
P3A = ROOT / "scale_p3a"
P1 = ROOT / "scale_p1"
OUT_DIR = P3A / "ood_analysis"
OUT_DIR.mkdir(exist_ok=True)

# Load samples with M-M predictions
df = pd.read_csv(P3A / "samples_unified" / "samples_p3a_with_mm_predictions.csv")
print(f"Loaded {len(df)} rows × {len(df.columns)} cols")
print(f"By regime: {df['regime'].value_counts().to_dict()}")

# Compute errors
df["err_raw"] = df["fabdem"] - df["h_te_orthometric"]
df["err_corr"] = (df["fabdem"] + df["pred_residual_mm_model"]) - df["h_te_orthometric"]

# ===== G3: Stratified validation on humid_temperate =====
ht = df[df.regime == "humid_temperate"].copy()
md = df[df.regime == "mediterranean"].copy()
print(f"\n{'='*72}")
print(f"G3: STRATIFIED VALIDATION — humid_temperate (OOD) vs mediterranean (in-sample)")
print(f"{'='*72}")

def stats(group):
    return pd.Series({
        "n": len(group),
        "rmse_raw":  float(np.sqrt(np.mean(group.err_raw**2))),
        "rmse_corr": float(np.sqrt(np.mean(group.err_corr**2))),
        "mae_raw":   float(np.mean(np.abs(group.err_raw))),
        "mae_corr":  float(np.mean(np.abs(group.err_corr))),
        "bias_raw":  float(group.err_raw.mean()),
        "bias_corr": float(group.err_corr.mean()),
    })

def stratify(d, label):
    out = {}
    # Elevation
    d = d.assign(elev_band=pd.cut(d.h_te_orthometric,
        bins=[-100, 200, 500, 1000, 2000, 9000],
        labels=["<200m", "200–500m", "500–1000m", "1000–2000m", ">2000m"]))
    g = d.groupby("elev_band", observed=True).apply(stats, include_groups=False)
    g["improve_pct"] = 100 * (g.rmse_raw - g.rmse_corr) / g.rmse_raw
    out["elevation"] = g
    # NDVI band
    d = d.assign(ndvi_band=pd.cut(d.ndvi,
        bins=[-1, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["sparse (<0.2)", "low (0.2–0.4)", "med (0.4–0.6)", "dense (0.6–0.8)", "very dense (>0.8)"]))
    g = d.groupby("ndvi_band", observed=True).apply(stats, include_groups=False)
    g["improve_pct"] = 100 * (g.rmse_raw - g.rmse_corr) / g.rmse_raw
    out["ndvi"] = g
    # HAND band
    d = d.assign(hand_band=pd.cut(d.hand,
        bins=[-1, 2, 10, 50, 200, 9999],
        labels=["<2m", "2–10m", "10–50m", "50–200m", ">200m"]))
    g = d.groupby("hand_band", observed=True).apply(stats, include_groups=False)
    g["improve_pct"] = 100 * (g.rmse_raw - g.rmse_corr) / g.rmse_raw
    out["hand"] = g
    # Slope
    d = d.assign(slope_band=pd.cut(d.slope,
        bins=[-1, 5, 15, 25, 90],
        labels=["flat <5°", "gentle 5–15°", "mod 15–25°", "steep >25°"]))
    g = d.groupby("slope_band", observed=True).apply(stats, include_groups=False)
    g["improve_pct"] = 100 * (g.rmse_raw - g.rmse_corr) / g.rmse_raw
    out["slope"] = g
    # Geomorphon
    GEOM = {1:"flat",2:"peak",3:"ridge",4:"shoulder",5:"spur",
            6:"slope",7:"hollow",8:"footslope",9:"valley",10:"pit"}
    d = d.assign(geom=d.geomorphons.map(GEOM))
    g = d.groupby("geom", observed=True).apply(stats, include_groups=False)
    g["improve_pct"] = 100 * (g.rmse_raw - g.rmse_corr) / g.rmse_raw
    out["geomorphon"] = g.sort_values("n", ascending=False)
    return out


print(f"\n--- HUMID TEMPERATE (out-of-distribution, M-M model) ---")
ht_strata = stratify(ht, "humid_temperate")
for name, g in ht_strata.items():
    print(f"\n  by {name}:")
    print(g.round(3).to_string())
    g.to_csv(OUT_DIR / f"ht_by_{name}.csv")

print(f"\n--- MEDITERRANEAN (in-sample reference) ---")
md_strata = stratify(md, "mediterranean")
for name, g in md_strata.items():
    g.to_csv(OUT_DIR / f"md_by_{name}.csv")

# Side-by-side comparison: improvement % by NDVI between regimes
print(f"\n{'='*72}")
print(f"COMPARISON: Improvement % by NDVI band, by regime")
print(f"{'='*72}")
ndvi_compare = pd.DataFrame({
    "humid_temperate": ht_strata["ndvi"].improve_pct,
    "mediterranean": md_strata["ndvi"].improve_pct,
    "n_ht": ht_strata["ndvi"].n,
    "n_md": md_strata["ndvi"].n,
})
print(ndvi_compare.round(2).to_string())
ndvi_compare.to_csv(OUT_DIR / "compare_ndvi_by_regime.csv")

print(f"\n--- by HAND ---")
hand_compare = pd.DataFrame({
    "humid_temperate": ht_strata["hand"].improve_pct,
    "mediterranean": md_strata["hand"].improve_pct,
    "n_ht": ht_strata["hand"].n,
    "n_md": md_strata["hand"].n,
})
print(hand_compare.round(2).to_string())
hand_compare.to_csv(OUT_DIR / "compare_hand_by_regime.csv")


# ===== G4: SHAP comparison =====
print(f"\n{'='*72}")
print(f"G4: SHAP COMPARISON — M-M model applied to humid_temperate vs Mediterranean")
print(f"{'='*72}")

# Load M-M model + features
booster = xgb.Booster()
booster.load_model(str(P1 / "samples_unified" / "xgb_mm_booster.json"))
FEATURES = json.loads((P1 / "samples_unified" / "mm_metrics.json").read_text())["features"]
print(f"Features ({len(FEATURES)}): order matches training")

# Existing M-M SHAP
mm_shap = pd.read_csv(P1 / "samples_unified" / "mm_shap.csv")
mm_shap = mm_shap.set_index("feature").rename(columns={"mean_abs_shap": "shap_mediterranean"})

# Compute SHAP on humid_temperate (sample 5000 for speed)
ht_features = ht[FEATURES].fillna(np.nan).values
sample_n = min(5000, len(ht_features))
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(ht_features), sample_n, replace=False)
X_ht = ht_features[sample_idx]

# Use sklearn wrapper for SHAP TreeExplainer
print(f"\nComputing SHAP on humid_temperate sample (n={sample_n})...")
explainer = shap.TreeExplainer(booster)
sv = explainer.shap_values(X_ht)
ht_shap = pd.DataFrame({"feature": FEATURES, "shap_humid_temperate": np.abs(sv).mean(axis=0)}).set_index("feature")

# Combine
shap_cmp = mm_shap.join(ht_shap)
shap_cmp["delta_abs"] = shap_cmp["shap_humid_temperate"] - shap_cmp["shap_mediterranean"]
shap_cmp["ratio"] = shap_cmp["shap_humid_temperate"] / shap_cmp["shap_mediterranean"]
shap_cmp["rank_md"] = shap_cmp["shap_mediterranean"].rank(ascending=False).astype(int)
shap_cmp["rank_ht"] = shap_cmp["shap_humid_temperate"].rank(ascending=False).astype(int)
shap_cmp["rank_shift"] = shap_cmp["rank_md"] - shap_cmp["rank_ht"]
shap_cmp = shap_cmp.sort_values("shap_humid_temperate", ascending=False)

print(f"\n  {'feature':<25s} {'SHAP_md':>9s} {'SHAP_ht':>9s} {'ratio':>7s} {'rank_md':>8s} {'rank_ht':>8s} {'shift':>6s}")
print("  " + "-"*78)
for f, row in shap_cmp.iterrows():
    print(f"  {f:<25s} {row.shap_mediterranean:>9.4f} {row.shap_humid_temperate:>9.4f} "
          f"{row.ratio:>7.2f} {row.rank_md:>8d} {row.rank_ht:>8d} {row.rank_shift:>+6d}")

shap_cmp.to_csv(OUT_DIR / "shap_regime_comparison.csv")
print(f"\n→ {OUT_DIR/'shap_regime_comparison.csv'}")

# Headline takeaways
print(f"\n{'='*72}")
print(f"HEADLINE FINDINGS")
print(f"{'='*72}")

# Top SHAP gainers / losers
top_gainers = shap_cmp.nlargest(5, "delta_abs")
top_losers = shap_cmp.nsmallest(5, "delta_abs")
print(f"\nTop 5 features that GAIN importance in humid_temperate (vs Mediterranean):")
for f, row in top_gainers.iterrows():
    print(f"  {f:<25s} Δ = {row.delta_abs:+.4f}  (rank {row.rank_md} → {row.rank_ht})")

print(f"\nTop 5 features that LOSE importance:")
for f, row in top_losers.iterrows():
    print(f"  {f:<25s} Δ = {row.delta_abs:+.4f}  (rank {row.rank_md} → {row.rank_ht})")

# Best/worst strata in humid_temperate
print(f"\nBest improvement stratum in humid_temperate by NDVI:")
best = ht_strata["ndvi"].loc[ht_strata["ndvi"].improve_pct.idxmax()]
print(f"  NDVI={ht_strata['ndvi'].improve_pct.idxmax()}: n={int(best.n)}, "
      f"RMSE {best.rmse_raw:.3f}→{best.rmse_corr:.3f} ({best.improve_pct:+.1f}%)")

print(f"\nBest improvement stratum in humid_temperate by HAND:")
best = ht_strata["hand"].loc[ht_strata["hand"].improve_pct.idxmax()]
print(f"  HAND={ht_strata['hand'].improve_pct.idxmax()}: n={int(best.n)}, "
      f"RMSE {best.rmse_raw:.3f}→{best.rmse_corr:.3f} ({best.improve_pct:+.1f}%)")
