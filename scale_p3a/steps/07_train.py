#!/usr/bin/env python3
"""E1a stage 07: train XGBoost with spatial CV + Optuna on full tile dataset."""
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import shap
import pyproj

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
SAMP = ROOT / "samples"
df = pd.read_csv(SAMP / "pilot_e1a_full.csv")

# Fix stream_network NaN → 0
if "stream_network" in df.columns:
    df["stream_network"] = df["stream_network"].fillna(0)

TARGET = "residual_corrected"
META = {"granule", "beam", "lon", "lat", "h_te", "h_te_unc", "te_qual",
        "night", "terrain_flg", "n_te_phot", "snow", "water", "cloud",
        "h_te_orthometric", "geoid_N", "residual_corrected",
        "fabdem", "filled"}
FEATURES = [c for c in df.columns if c not in META and df[c].dtype != "O"]
print(f"Rows: {len(df)}  Features: {len(FEATURES)}")

X = df[FEATURES].values
y = df[TARGET].values
lon, lat = df.lon.values, df.lat.values
keep = np.isfinite(y)
X, y, lon, lat = X[keep], y[keep], lon[keep], lat[keep]
print(f"Rows with finite target: {keep.sum()}")

# Spatial blocks 10 km
t = pyproj.Transformer.from_crs(4326, 32719, always_xy=True)
x_utm, y_utm = t.transform(lon, lat)
BLOCK_M = 10000.0
block_x = (x_utm // BLOCK_M).astype(int)
block_y = (y_utm // BLOCK_M).astype(int)
blocks = block_x * 1000 + block_y
uniq = np.unique(blocks)
print(f"\nSpatial blocks ({BLOCK_M:.0f} m): {len(uniq)}")
for b in uniq:
    print(f"  block {b}: {(blocks==b).sum()} pts")

n_splits = min(5, len(uniq))
gkf = GroupKFold(n_splits=n_splits)
splits = list(gkf.split(X, y, groups=blocks))

# Baseline
rmse_base = float(np.sqrt(mean_squared_error(y, np.zeros_like(y))))
mae_base = float(mean_absolute_error(y, np.zeros_like(y)))
print(f"\nBaseline FABDEM raw: RMSE={rmse_base:.3f}  MAE={mae_base:.3f}")
print(f"  target mean={y.mean():+.3f}  std={y.std():.3f}")


def cv_rmse(params, splits):
    preds = np.full_like(y, np.nan, dtype=np.float64)
    for tr, te in splits:
        m = xgb.XGBRegressor(
            **params, tree_method="hist", random_state=42, n_jobs=4,
            early_stopping_rounds=30,
        )
        m.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
        preds[te] = m.predict(X[te])
    return float(np.sqrt(mean_squared_error(y, preds))), preds


def objective(trial):
    params = {
        "n_estimators": 800,
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
    }
    rmse, _ = cv_rmse(params, splits)
    return rmse

print(f"\n=== Optuna tuning (50 trials, spatial K={n_splits}) ===")
study = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=False)
print(f"Best RMSE (spatial CV): {study.best_value:.4f}")
print(f"Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

best = {**study.best_params, "n_estimators": 800}

# Final spatial CV
rmse_sp, pred_sp = cv_rmse(best, splits)
mae_sp = float(mean_absolute_error(y, pred_sp))
r2_sp = float(r2_score(y, pred_sp))
print(f"\nSpatial CV OOF: RMSE={rmse_sp:.3f}  MAE={mae_sp:.3f}  R²={r2_sp:+.3f}")

# Random CV for leakage comparison
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rs = list(kf.split(X))
rmse_rn, pred_rn = cv_rmse(best, rs)
mae_rn = float(mean_absolute_error(y, pred_rn))
r2_rn = float(r2_score(y, pred_rn))
print(f"Random CV OOF:  RMSE={rmse_rn:.3f}  MAE={mae_rn:.3f}  R²={r2_rn:+.3f}")

print(f"\n=== Improvement vs FABDEM raw ===")
print(f"  Spatial: {rmse_base:.3f} → {rmse_sp:.3f}  ({100*(rmse_base-rmse_sp)/rmse_base:+.1f}%)")
print(f"  Random:  {rmse_base:.3f} → {rmse_rn:.3f}  ({100*(rmse_base-rmse_rn)/rmse_base:+.1f}%)")

# Per-fold (spatial)
print(f"\n=== Per-fold (spatial 10km) ===")
fold_sum = []
for i, (_, te) in enumerate(splits):
    rm = float(np.sqrt(mean_squared_error(y[te], pred_sp[te])))
    fm = float(mean_absolute_error(y[te], pred_sp[te]))
    r2 = float(r2_score(y[te], pred_sp[te]))
    fold_sum.append({"fold": i, "n_test": int(te.size), "rmse": rm, "mae": fm, "r2": r2})
    print(f"  fold{i}: n={te.size:>5d}  RMSE={rm:.3f}  MAE={fm:.3f}  R²={r2:+.3f}")

# SHAP on full data with best params
print(f"\n=== SHAP top 15 ===")
final = xgb.XGBRegressor(**best, tree_method="hist", random_state=42, n_jobs=4)
final.fit(X, y, verbose=False)
explainer = shap.TreeExplainer(final)
# Use sample for speed on big datasets
sample_n = min(5000, len(X))
sample_idx = np.random.default_rng(42).choice(len(X), sample_n, replace=False)
sv = explainer.shap_values(X[sample_idx])
shap_imp = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": np.abs(sv).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
print(shap_imp.head(15).to_string(index=False))
shap_imp.to_csv(SAMP / "e1a_shap.csv", index=False)

with open(SAMP / "e1a_metrics.json", "w") as f:
    json.dump({
        "n_samples": int(keep.sum()),
        "n_features": len(FEATURES),
        "n_blocks": int(len(uniq)),
        "block_size_m": BLOCK_M,
        "n_optuna_trials": 50,
        "baseline_rmse": rmse_base, "baseline_mae": mae_base,
        "spatial_cv": {"rmse": rmse_sp, "mae": mae_sp, "r2": r2_sp, "folds": fold_sum},
        "random_cv": {"rmse": rmse_rn, "mae": mae_rn, "r2": r2_rn},
        "best_params": best,
    }, f, indent=2)

# Save predictions
df_pred = df.loc[keep].copy()
df_pred["pred_residual"] = pred_sp
df_pred["fabdem_corrected"] = df_pred["fabdem"] + pred_sp
df_pred.to_csv(SAMP / "e1a_predictions.csv", index=False)
print(f"\n→ samples/e1a_metrics.json, e1a_shap.csv, e1a_predictions.csv")
