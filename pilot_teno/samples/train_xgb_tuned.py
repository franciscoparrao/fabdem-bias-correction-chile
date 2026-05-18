#!/usr/bin/env python3
"""XGBoost + Optuna tuning + 10×10 km spatial blocks (D1).

Improvements over D0:
  - 10 km blocks: larger than ATL08 inter-track distance (~3 km) → less leakage
  - Optuna with 30 trials, objective = mean OOF RMSE under spatial K-fold
  - SHAP on best model

Caveats (declared):
  - Tuning uses the same spatial CV as final evaluation → mildly optimistic.
    Proper nested CV would be ideal but expensive at 888 rows.
"""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import shap

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem/pilot_teno")
SAMP = ROOT / "samples"
df = pd.read_csv(SAMP / "pilot_samples_full.csv")
df["stream_network"] = df["stream_network"].fillna(0)

TARGET = "residual_corrected"
META = {"lon", "lat", "h_te", "h_te_unc", "te_qual", "night", "terrain_flg",
        "n_te_phot", "snow", "water", "cloud", "beam",
        "h_te_orthometric", "geoid_N", "residual_raw", "residual_corrected",
        "fabdem", "filled"}
FEATURES = [c for c in df.columns if c not in META and df[c].dtype != "O"]

X = df[FEATURES].values
y = df[TARGET].values
lon, lat = df.lon.values, df.lat.values
keep = np.isfinite(y)
X, y, lon, lat = X[keep], y[keep], lon[keep], lat[keep]
print(f"Rows: {keep.sum()}  Features: {len(FEATURES)}")

# 10 km blocks
from pyproj import Transformer
t = Transformer.from_crs(4326, 32719, always_xy=True)
x_utm, y_utm = t.transform(lon, lat)
BLOCK_M = 10000.0
block_x = (x_utm // BLOCK_M).astype(int)
block_y = (y_utm // BLOCK_M).astype(int)
blocks = block_x * 1000 + block_y
uniq = np.unique(blocks)
print(f"\nBlocks (10×10 km): {len(uniq)}")
for b in uniq:
    print(f"  block {b}: {(blocks==b).sum()} pts")

n_splits = min(5, len(uniq))
print(f"Spatial CV K={n_splits}\n")

# Baseline
rmse_base = float(np.sqrt(mean_squared_error(y, np.zeros_like(y))))
print(f"Baseline FABDEM raw: RMSE={rmse_base:.3f}, MAE={float(mean_absolute_error(y,np.zeros_like(y))):.3f}")

# ----- Optuna objective: mean OOF RMSE under spatial K-fold -----
gkf = GroupKFold(n_splits=n_splits)
spatial_splits = list(gkf.split(X, y, groups=blocks))

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
        "n_estimators": 600,
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
    }
    rmse, _ = cv_rmse(params, spatial_splits)
    return rmse

print("=== Optuna tuning (30 trials, spatial K-fold objective) ===")
study = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=False)

print(f"\nBest RMSE (spatial CV mean): {study.best_value:.4f} m")
print(f"Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

best = {**study.best_params, "n_estimators": 600}

# Final evaluation: same spatial CV with best params + random CV for leakage demo
print("\n=== Spatial Block CV (10×10 km) with tuned params ===")
rmse_sp, pred_sp = cv_rmse(best, spatial_splits)
mae_sp = float(mean_absolute_error(y, pred_sp))
r2_sp = float(r2_score(y, pred_sp))
print(f"  OOF: RMSE={rmse_sp:.3f}  MAE={mae_sp:.3f}  R²={r2_sp:+.3f}")

print("\n=== Random CV (K=5) [optimistic] with tuned params ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
random_splits = list(kf.split(X))
rmse_rn, pred_rn = cv_rmse(best, random_splits)
mae_rn = float(mean_absolute_error(y, pred_rn))
r2_rn = float(r2_score(y, pred_rn))
print(f"  OOF: RMSE={rmse_rn:.3f}  MAE={mae_rn:.3f}  R²={r2_rn:+.3f}")

print(f"\n=== Summary table ===")
print(f"  Method                                   RMSE   MAE   R²    Δ vs baseline")
print(f"  Baseline (FABDEM raw)                    {rmse_base:.3f}  "
      f"{float(mean_absolute_error(y,np.zeros_like(y))):.3f}  —      —")
print(f"  XGB default + 5km spatial CV (D0)        1.589  0.764  +0.269  -22.0%")
print(f"  XGB tuned + 10km spatial CV (D1)         {rmse_sp:.3f}  {mae_sp:.3f}  "
      f"{r2_sp:+.3f}  {100*(rmse_sp-rmse_base)/rmse_base:+.1f}%")
print(f"  XGB tuned + random CV (optimistic)       {rmse_rn:.3f}  {mae_rn:.3f}  "
      f"{r2_rn:+.3f}  {100*(rmse_rn-rmse_base)/rmse_base:+.1f}%")

# Fold metrics with best params
print(f"\n=== Per-fold (spatial 10km, tuned) ===")
preds = pred_sp
fold_summary = []
for i, (_, te) in enumerate(spatial_splits):
    rm = float(np.sqrt(mean_squared_error(y[te], preds[te])))
    fm = float(mean_absolute_error(y[te], preds[te]))
    r2 = float(r2_score(y[te], preds[te]))
    fold_summary.append({"fold": i, "n_test": int(te.size),
                         "rmse": rm, "mae": fm, "r2": r2})
    print(f"  fold{i}: n={te.size:>4d}  RMSE={rm:.3f}  MAE={fm:.3f}  R²={r2:+.3f}")

# SHAP on tuned model fit on all data
print(f"\n=== SHAP (top 15) on tuned full-data model ===")
final = xgb.XGBRegressor(**best, tree_method="hist", random_state=42, n_jobs=4)
final.fit(X, y, verbose=False)
explainer = shap.TreeExplainer(final)
sv = explainer.shap_values(X)
shap_imp = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": np.abs(sv).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
print(shap_imp.head(15).to_string(index=False))

shap_imp.to_csv(SAMP / "pilot_xgb_tuned_shap.csv", index=False)

# Save metrics + best params + predictions
with open(SAMP / "pilot_xgb_tuned_metrics.json", "w") as f:
    json.dump({
        "baseline_rmse": rmse_base,
        "best_params": best,
        "spatial_cv_10km": {"rmse": rmse_sp, "mae": mae_sp, "r2": r2_sp,
                              "folds": fold_summary},
        "random_cv": {"rmse": rmse_rn, "mae": mae_rn, "r2": r2_rn},
        "n_samples": int(keep.sum()),
        "n_features": len(FEATURES),
        "n_blocks": int(len(uniq)),
        "block_size_m": BLOCK_M,
        "optuna_n_trials": 30,
    }, f, indent=2)

df_pred = df.loc[keep].copy()
df_pred["pred_residual_tuned"] = pred_sp
df_pred["fabdem_corrected_tuned"] = df_pred["fabdem"] + pred_sp
df_pred.to_csv(SAMP / "pilot_xgb_tuned_predictions.csv", index=False)
print(f"\n→ pilot_xgb_tuned_metrics.json, pilot_xgb_tuned_shap.csv, pilot_xgb_tuned_predictions.csv")
