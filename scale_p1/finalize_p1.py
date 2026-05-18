#!/usr/bin/env python3
"""Recovery script: redo only the training stage with known-best params,
save model via booster (sklearn save_model has compat issue), write metrics JSON,
generate SHAP.

Uses the best_params already discovered by the prior Optuna run (logged).
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import pyproj

warnings.filterwarnings("ignore")
ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem/scale_p1")
SAMP = ROOT / "samples_unified"
CSV = SAMP / "samples_mm_full.csv"

# Best params from prior Optuna 100 trials
BEST = {
    "n_estimators": 1000,
    "max_depth": 9,
    "learning_rate": 0.011289111763110846,
    "subsample": 0.9533188971118374,
    "colsample_bytree": 0.6354986604850102,
    "min_child_weight": 14,
    "reg_lambda": 7.40970392425014,
    "reg_alpha": 1.5969596058836053,
    "gamma": 2.2017805740056464,
}

df = pd.read_csv(CSV)
if "stream_network" in df.columns:
    df["stream_network"] = df["stream_network"].fillna(0)

TARGET = "residual_corrected"
META = {"granule", "beam", "lon", "lat", "h_te", "h_te_unc", "te_qual",
        "night", "terrain_flg", "n_te_phot", "snow", "water", "cloud",
        "h_te_orthometric", "geoid_N", "residual_raw", "residual_corrected",
        "fabdem", "filled", "tile"}
FEATURES = [c for c in df.columns if c not in META and df[c].dtype != "O"]
print(f"Rows: {len(df)}  Features: {len(FEATURES)}")
print(f"Per-tile: {df['tile'].value_counts().to_dict()}")

X = df[FEATURES].values
y = df[TARGET].values
lon, lat = df.lon.values, df.lat.values
keep = np.isfinite(y)
X, y, lon, lat = X[keep], y[keep], lon[keep], lat[keep]

t = pyproj.Transformer.from_crs(4326, 32719, always_xy=True)
x_utm, y_utm = t.transform(lon, lat)
BLOCK_M = 10000.0
blocks = (x_utm // BLOCK_M).astype(int) * 10000 + (y_utm // BLOCK_M).astype(int)
uniq = np.unique(blocks)
print(f"\nSpatial blocks (10 km): {len(uniq)}")

splits_sp = list(GroupKFold(n_splits=5).split(X, y, groups=blocks))
splits_rn = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X))

rmse_base = float(np.sqrt(mean_squared_error(y, np.zeros_like(y))))
mae_base = float(mean_absolute_error(y, np.zeros_like(y)))
print(f"Baseline: RMSE={rmse_base:.3f}  MAE={mae_base:.3f}")


def cv_predict(splits):
    preds = np.full_like(y, np.nan, dtype=np.float64)
    for tr, te in splits:
        m = xgb.XGBRegressor(**BEST, tree_method="hist", random_state=42,
                              n_jobs=4, early_stopping_rounds=30)
        m.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
        preds[te] = m.predict(X[te])
    return preds


print("\n=== Spatial CV (5-fold, 10 km blocks) ===")
pred_sp = cv_predict(splits_sp)
rmse_sp = float(np.sqrt(mean_squared_error(y, pred_sp)))
mae_sp = float(mean_absolute_error(y, pred_sp))
r2_sp = float(r2_score(y, pred_sp))
print(f"  RMSE={rmse_sp:.3f}  MAE={mae_sp:.3f}  R²={r2_sp:+.3f}  Δ={100*(rmse_sp-rmse_base)/rmse_base:+.1f}%")

print("\n=== Random CV (5-fold) ===")
pred_rn = cv_predict(splits_rn)
rmse_rn = float(np.sqrt(mean_squared_error(y, pred_rn)))
mae_rn = float(mean_absolute_error(y, pred_rn))
r2_rn = float(r2_score(y, pred_rn))
print(f"  RMSE={rmse_rn:.3f}  MAE={mae_rn:.3f}  R²={r2_rn:+.3f}  Δ={100*(rmse_rn-rmse_base)/rmse_base:+.1f}%")

# Per-fold (spatial)
print("\n=== Per-fold (spatial) ===")
fold_sum = []
for i, (_, te) in enumerate(splits_sp):
    rm = float(np.sqrt(mean_squared_error(y[te], pred_sp[te])))
    fm = float(mean_absolute_error(y[te], pred_sp[te]))
    r2 = float(r2_score(y[te], pred_sp[te]))
    fold_sum.append({"fold": i, "n_test": int(te.size), "rmse": rm, "mae": fm, "r2": r2})
    print(f"  fold{i}: n={te.size:>5d}  RMSE={rm:.3f}  MAE={fm:.3f}  R²={r2:+.3f}")

# Final model on all data — save BOOSTER (not sklearn) to avoid compat bug
print("\n=== Final fit on all data ===")
final = xgb.XGBRegressor(**BEST, tree_method="hist", random_state=42, n_jobs=4)
final.fit(X, y, verbose=False)

# Save via booster: cross-platform compatible
final.get_booster().save_model(str(SAMP / "xgb_mm_booster.json"))
print(f"  → {SAMP/'xgb_mm_booster.json'}")

# Also save with joblib for full sklearn re-instantiation
import joblib
joblib.dump(final, SAMP / "xgb_mm_sklearn.joblib")
print(f"  → {SAMP/'xgb_mm_sklearn.joblib'}")

# SHAP on a sample
print("\n=== SHAP (sample 8k) ===")
explainer = shap.TreeExplainer(final)
sample_idx = np.random.default_rng(42).choice(len(X), min(8000, len(X)), replace=False)
sv = explainer.shap_values(X[sample_idx])
shap_imp = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": np.abs(sv).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
print(shap_imp.head(20).to_string(index=False))
shap_imp.to_csv(SAMP / "mm_shap.csv", index=False)
print(f"\n→ {SAMP/'mm_shap.csv'}")

# Per-tile breakdown of error
print("\n=== Per-tile error breakdown (spatial CV) ===")
df_pred = df.loc[keep].copy()
df_pred["pred_residual"] = pred_sp
df_pred["fabdem_corrected"] = df_pred["fabdem"] + pred_sp
df_pred["abs_err_raw"] = df_pred[TARGET].abs()
df_pred["abs_err_ml"]  = (df_pred[TARGET] - pred_sp).abs()
per_tile = df_pred.groupby("tile").agg(
    n=("lon", "count"),
    rmse_raw=("residual_corrected", lambda s: float(np.sqrt((s**2).mean()))),
    rmse_ml=("abs_err_ml", lambda s: float(np.sqrt((s**2).mean()))),
    mae_raw=("abs_err_raw", "mean"),
    mae_ml=("abs_err_ml", "mean"),
)
per_tile["improve_pct"] = 100 * (per_tile.rmse_raw - per_tile.rmse_ml) / per_tile.rmse_raw
print(per_tile.round(3).to_string())

df_pred.to_csv(SAMP / "mm_predictions.csv", index=False)
per_tile.to_csv(SAMP / "mm_per_tile.csv")

metrics = {
    "n_samples": int(keep.sum()),
    "n_features": len(FEATURES),
    "features": FEATURES,
    "n_blocks": int(len(uniq)),
    "block_size_m": BLOCK_M,
    "per_tile_counts": df.loc[keep, "tile"].value_counts().to_dict(),
    "baseline_rmse": rmse_base, "baseline_mae": mae_base,
    "spatial_cv": {"rmse": rmse_sp, "mae": mae_sp, "r2": r2_sp, "folds": fold_sum},
    "random_cv":  {"rmse": rmse_rn, "mae": mae_rn, "r2": r2_rn},
    "best_params": BEST,
    "per_tile_error": per_tile.to_dict(orient="index"),
}
(SAMP / "mm_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
print(f"\n→ {SAMP/'mm_metrics.json'}")
print("\n✅ Finalize complete")
