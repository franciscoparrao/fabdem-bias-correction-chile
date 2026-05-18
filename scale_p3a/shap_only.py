#!/usr/bin/env python3
"""SHAP comparison only (recover from format bug)."""
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

df = pd.read_csv(P3A / "samples_unified" / "samples_p3a_with_mm_predictions.csv")
ht = df[df.regime == "humid_temperate"].copy()
print(f"humid_temperate: {len(ht)} footprints")

booster = xgb.Booster()
booster.load_model(str(P1 / "samples_unified" / "xgb_mm_booster.json"))
FEATURES = json.loads((P1 / "samples_unified" / "mm_metrics.json").read_text())["features"]

# Fill stream_network NaN to match training
if "stream_network" in ht.columns:
    ht["stream_network"] = ht["stream_network"].fillna(0)
X_ht = ht[FEATURES].values

# Sample 5000 for SHAP
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_ht), min(5000, len(X_ht)), replace=False)
X_sample = X_ht[sample_idx]

print(f"Computing SHAP on {len(X_sample)} humid_temperate samples...")
explainer = shap.TreeExplainer(booster)
sv = explainer.shap_values(X_sample)
ht_shap = pd.DataFrame({
    "feature": FEATURES,
    "shap_humid_temperate": np.abs(sv).mean(axis=0),
}).set_index("feature")

# Load existing MD SHAP
mm_shap = pd.read_csv(P1 / "samples_unified" / "mm_shap.csv").set_index("feature")
mm_shap = mm_shap.rename(columns={"mean_abs_shap": "shap_mediterranean"})

cmp = mm_shap.join(ht_shap)
cmp["delta_abs"] = cmp["shap_humid_temperate"] - cmp["shap_mediterranean"]
cmp["ratio"] = cmp["shap_humid_temperate"] / cmp["shap_mediterranean"].replace(0, np.nan)
cmp["rank_md"] = cmp["shap_mediterranean"].rank(ascending=False, method="min")
cmp["rank_ht"] = cmp["shap_humid_temperate"].rank(ascending=False, method="min")
cmp["rank_shift"] = (cmp["rank_md"] - cmp["rank_ht"]).fillna(0)
# Cast safely
for c in ["rank_md", "rank_ht", "rank_shift"]:
    cmp[c] = cmp[c].astype(int)
cmp = cmp.sort_values("shap_humid_temperate", ascending=False)

cmp.to_csv(OUT_DIR / "shap_regime_comparison.csv")

print(f"\n{'feature':<25s} {'SHAP_md':>9s} {'SHAP_ht':>9s} {'ratio':>7s} {'rk_md':>6s} {'rk_ht':>6s} {'shift':>6s}")
print("-" * 78)
for f, r in cmp.iterrows():
    print(f"{f:<25s} {r.shap_mediterranean:>9.4f} {r.shap_humid_temperate:>9.4f} "
          f"{r.ratio:>7.2f} {r.rank_md:>6d} {r.rank_ht:>6d} {r.rank_shift:>+6d}")

print(f"\n=== Top 5 features that GAIN importance in humid_temperate ===")
for f, r in cmp.nlargest(5, "delta_abs").iterrows():
    print(f"  {f:<25s} Δ={r.delta_abs:+.4f}  rank {r.rank_md} → {r.rank_ht} (shift {r.rank_shift:+d})")

print(f"\n=== Top 5 features that LOSE importance ===")
for f, r in cmp.nsmallest(5, "delta_abs").iterrows():
    print(f"  {f:<25s} Δ={r.delta_abs:+.4f}  rank {r.rank_md} → {r.rank_ht} (shift {r.rank_shift:+d})")
