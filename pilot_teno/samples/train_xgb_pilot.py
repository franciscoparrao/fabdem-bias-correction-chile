#!/usr/bin/env python3
"""First XGBoost on pilot Teno dataset.

Target:   residual_corrected = h_te_orthometric - fabdem  [m]
Features: 27 terrain/hydro + 8 satellite = 35

Honest validation:
  - Spatial block CV (5×5 km blocks, K=5) — proper test
  - Random K=5 CV for comparison (illustrates autocorrelation leakage)

Outputs:
  pilot_xgb_metrics.json  — fold-level + aggregate metrics
  pilot_xgb_shap.csv      — feature importance ranking
  pilot_xgb_predictions.csv — out-of-fold predictions
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem/pilot_teno")
SAMP = ROOT / "samples"
df = pd.read_csv(SAMP / "pilot_samples_full.csv")
print(f"Loaded {len(df)} footprints × {len(df.columns)} cols")

TARGET = "residual_corrected"
META = {"lon", "lat", "h_te", "h_te_unc", "te_qual", "night", "terrain_flg",
        "n_te_phot", "snow", "water", "cloud", "beam",
        "h_te_orthometric", "geoid_N", "residual_raw", "residual_corrected",
        "fabdem", "filled"}
FEATURES = [c for c in df.columns if c not in META and df[c].dtype != "O"]
print(f"Features ({len(FEATURES)}):")
for f in FEATURES:
    print(f"  - {f}")

# stream_network NaN means "no stream" → 0 (semantic)
df["stream_network"] = df["stream_network"].fillna(0)

# Print NaN summary
nan_cnt = df[FEATURES].isna().sum()
nan_cnt = nan_cnt[nan_cnt > 0]
if len(nan_cnt) > 0:
    print(f"\nFeatures with remaining NaN (XGBoost handles natively):")
    for c, n in nan_cnt.items():
        print(f"  {c}: {n} NaN ({100*n/len(df):.1f}%)")

X = df[FEATURES].values
y = df[TARGET].values
lon, lat = df.lon.values, df.lat.values

# Only drop rows with NaN target (features may have NaN — XGBoost handles)
keep = np.isfinite(y)
X, y, lon, lat = X[keep], y[keep], lon[keep], lat[keep]
print(f"\nAfter dropping NaN target: {keep.sum()} rows")

# Spatial blocks: 5×5 km grid in UTM 19S
from pyproj import Transformer
t = Transformer.from_crs(4326, 32719, always_xy=True)
x_utm, y_utm = t.transform(lon, lat)
BLOCK_M = 5000.0
block_x = (x_utm // BLOCK_M).astype(int)
block_y = (y_utm // BLOCK_M).astype(int)
blocks = block_x * 1000 + block_y
uniq_blocks = np.unique(blocks)
print(f"\nSpatial blocks ({BLOCK_M:.0f} m): {len(uniq_blocks)} unique")
for b in uniq_blocks:
    print(f"  block {b}: {(blocks==b).sum()} points")


def fit_eval(name, splits, X, y):
    preds = np.full_like(y, np.nan, dtype=np.float64)
    fold_metrics = []
    for fold, (tr, te) in enumerate(splits):
        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_lambda=1.0, tree_method="hist", random_state=42,
            n_jobs=4, early_stopping_rounds=30,
        )
        model.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
        p = model.predict(X[te])
        preds[te] = p
        fold_metrics.append({
            "fold": fold,
            "n_train": int(tr.size), "n_test": int(te.size),
            "rmse": float(np.sqrt(mean_squared_error(y[te], p))),
            "mae":  float(mean_absolute_error(y[te], p)),
            "r2":   float(r2_score(y[te], p)),
            "best_iter": int(model.best_iteration) if hasattr(model, "best_iteration") else None,
        })
    rmse_oof = float(np.sqrt(mean_squared_error(y, preds)))
    mae_oof  = float(mean_absolute_error(y, preds))
    r2_oof   = float(r2_score(y, preds))
    print(f"\n=== {name} ===")
    for fm in fold_metrics:
        print(f"  fold{fm['fold']}: n_test={fm['n_test']}  "
              f"RMSE={fm['rmse']:.3f}  MAE={fm['mae']:.3f}  R²={fm['r2']:+.3f}  "
              f"best_iter={fm['best_iter']}")
    print(f"  OOF aggregate: RMSE={rmse_oof:.3f}  MAE={mae_oof:.3f}  R²={r2_oof:+.3f}")
    return {"name": name, "folds": fold_metrics,
            "oof": {"rmse": rmse_oof, "mae": mae_oof, "r2": r2_oof}}, preds


# Baseline metrics: if we used FABDEM raw (predicted residual = 0)
rmse_base = float(np.sqrt(mean_squared_error(y, np.zeros_like(y))))
mae_base  = float(mean_absolute_error(y, np.zeros_like(y)))
print(f"\n=== Baseline (FABDEM raw, predicted residual = 0) ===")
print(f"  RMSE = {rmse_base:.3f} m")
print(f"  MAE  = {mae_base:.3f} m")
print(f"  std target = {y.std():.3f} m, mean target = {y.mean():+.3f} m")


# Spatial block CV
gkf = GroupKFold(n_splits=5)
splits_spatial = list(gkf.split(X, y, groups=blocks))
res_spatial, pred_spatial = fit_eval("Spatial Block CV (5×5 km, K=5)", splits_spatial, X, y)

# Random CV (for leakage comparison)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
splits_random = list(kf.split(X))
res_random, pred_random = fit_eval("Random CV (K=5) [optimistic]", splits_random, X, y)


# Improvement vs baseline
def improvement(name, pred):
    new_rmse = float(np.sqrt(mean_squared_error(y, pred)))
    new_mae  = float(mean_absolute_error(y, pred))
    print(f"  {name:<32s}  RMSE {rmse_base:.3f} → {new_rmse:.3f} "
          f"({100*(rmse_base-new_rmse)/rmse_base:+.1f}%);  "
          f"MAE {mae_base:.3f} → {new_mae:.3f}")
    return new_rmse, new_mae

print(f"\n=== Improvement: FABDEM raw → FABDEM+XGBoost predicted residual ===")
improvement("Spatial block CV", pred_spatial)
improvement("Random CV (optimistic)", pred_random)


# Train on all data for SHAP
print(f"\n=== Final model on all data + SHAP ===")
final = xgb.XGBRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    tree_method="hist", random_state=42, n_jobs=4,
)
final.fit(X, y)

explainer = shap.TreeExplainer(final)
shap_values = explainer.shap_values(X)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": mean_abs_shap})
shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
print(shap_df.to_string(index=False))

shap_df.to_csv(SAMP / "pilot_xgb_shap.csv", index=False)

# Save
with open(SAMP / "pilot_xgb_metrics.json", "w") as f:
    json.dump({
        "baseline_fabdem_raw": {"rmse": rmse_base, "mae": mae_base,
                                  "target_mean": float(y.mean()),
                                  "target_std": float(y.std())},
        "spatial_block_cv": res_spatial,
        "random_cv": res_random,
        "n_samples": int(keep.sum()),
        "n_features": len(FEATURES),
        "n_blocks": int(len(uniq_blocks)),
    }, f, indent=2)

pred_df = df.loc[keep].copy()
pred_df["pred_residual_spatial"] = pred_spatial
pred_df["pred_residual_random"] = pred_random
pred_df["fabdem_corrected"] = pred_df["fabdem"] + pred_df["pred_residual_spatial"]
pred_df.to_csv(SAMP / "pilot_xgb_predictions.csv", index=False)
print(f"\n→ Saved: pilot_xgb_metrics.json, pilot_xgb_shap.csv, pilot_xgb_predictions.csv")
