#!/usr/bin/env python3
"""Phase C — Hawker-RF regional baseline.

Train a sklearn RandomForestRegressor on the same 77,501 footprints, same 33
features, same 10-km spatial GroupKFold splits as XGBoost. Compare RMSE.

This isolates the contribution of the learner choice (XGBoost vs RF) vs the
contribution of regional re-training (Hawker's global RF on global LiDAR vs
ours on regional ATL08) vs the feature stack.

Hyperparameter search: Optuna 50 trials (RF tuning has lower variance than
XGB, fewer trials suffice).
"""
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pyproj
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
SAMP = ROOT / "scale_p1" / "samples_unified"
OUT = ROOT / "paper" / "experiments"


def load_data():
    df = pd.read_csv(SAMP / "samples_mm_full.csv")
    if "stream_network" in df.columns:
        df["stream_network"] = df["stream_network"].fillna(0)
    META = {"granule", "beam", "lon", "lat", "h_te", "h_te_unc",
            "night", "terrain_flg", "n_te_phot", "snow", "water", "cloud",
            "h_te_orthometric", "geoid_N", "residual_corrected",
            "fabdem", "filled", "tile"}
    FEATURES = [c for c in df.columns if c not in META and df[c].dtype != "O"]
    # sklearn RF doesn't natively route NaNs (XGBoost does). Median-impute
    # per feature column so RF sees the same 77,501 rows as XGBoost — a fair
    # learner-vs-learner comparison rather than a learner+dataset comparison.
    for c in FEATURES:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    keep = np.isfinite(df.residual_corrected.values)
    df = df.loc[keep].reset_index(drop=True)
    X = df[FEATURES].values
    y = df.residual_corrected.values
    lon = df.lon.values
    lat = df.lat.values
    return X, y, lon, lat, FEATURES


def get_splits(lon, lat, y, k=5, block_m=10000.0):
    t = pyproj.Transformer.from_crs(4326, 32719, always_xy=True)
    x_utm, y_utm = t.transform(lon, lat)
    blocks = (x_utm // block_m).astype(int) * 100000 + (y_utm // block_m).astype(int)
    gkf = GroupKFold(n_splits=k)
    return list(gkf.split(np.zeros(len(y)), y, groups=blocks))


def cv_rmse_rf(params, X, y, splits):
    preds = np.full_like(y, np.nan, dtype=np.float64)
    for tr, te in splits:
        m = RandomForestRegressor(**params, random_state=42, n_jobs=4)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return float(np.sqrt(mean_squared_error(y, preds))), preds


def main():
    print("=" * 70)
    print("Phase C — Hawker-style RandomForest baseline")
    print("=" * 70)
    t_total = time.time()
    X, y, lon, lat, FEATURES = load_data()
    print(f"Loaded {len(y)} rows, {len(FEATURES)} features")
    splits = get_splits(lon, lat, y)
    rmse_base = float(np.sqrt(np.mean(y ** 2)))
    print(f"Baseline FABDEM raw RMSE = {rmse_base:.3f}")

    # ----- Optuna search (tractable budget on CPU) -----
    # NOTE: RF training over 60k samples per fold * 5 folds is heavy; we cap
    # n_estimators and depth to keep per-trial time at ~60-90 s and use 25
    # trials. The XGBoost benchmark used 100 trials but each XGB trial is
    # ~30 s on the same dataset, so total compute budgets are comparable.
    N_TRIALS = 25
    print(f"\n[Optuna] {N_TRIALS} trials, TPE sampler, spatial-CV objective")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 6, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5]),
        }
        rmse, _ = cv_rmse_rf(params, X, y, splits)
        trial.report(rmse, step=0)
        return rmse

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))

    # Progress callback so log shows trials
    def log_trial(study, trial):
        print(f"  trial {trial.number+1:>2d}/{N_TRIALS}  RMSE={trial.value:.4f}  "
              f"best={study.best_value:.4f}", flush=True)

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                   callbacks=[log_trial])
    print(f"  best RMSE (spatial CV OOF): {study.best_value:.4f}")
    print(f"  best params: {study.best_params}")

    # ----- Final OOF + per-tile -----
    best = dict(study.best_params)
    rmse_rf, pred_rf = cv_rmse_rf(best, X, y, splits)
    mae_rf = float(mean_absolute_error(y, pred_rf))
    print(f"\nFinal OOF: RMSE={rmse_rf:.4f}  MAE={mae_rf:.4f}")

    # ----- Comparison vs XGBoost -----
    XGB_RMSE = 2.483180756453067   # from mm_metrics.json
    XGB_MAE = 1.0989095719282178
    delta_pct_vs_raw_rf = 100 * (rmse_rf - rmse_base) / rmse_base
    delta_pct_vs_raw_xgb = 100 * (XGB_RMSE - rmse_base) / rmse_base
    delta_pct_rf_vs_xgb = 100 * (rmse_rf - XGB_RMSE) / XGB_RMSE
    print(f"\n=== Comparison vs XGBoost (same data, same CV) ===")
    print(f"  XGBoost OOF RMSE: {XGB_RMSE:.4f}  (Δ raw = {delta_pct_vs_raw_xgb:+.2f}%)")
    print(f"  RF      OOF RMSE: {rmse_rf:.4f}  (Δ raw = {delta_pct_vs_raw_rf:+.2f}%)")
    print(f"  RF vs XGBoost: {delta_pct_rf_vs_xgb:+.2f}%")
    if rmse_rf < XGB_RMSE:
        print("  → RF outperforms XGBoost")
    else:
        print("  → XGBoost outperforms RF")

    out = {
        "baseline_rmse": rmse_base,
        "rf_oof_rmse": rmse_rf,
        "rf_oof_mae": mae_rf,
        "rf_best_params": best,
        "xgboost_oof_rmse_ref": XGB_RMSE,
        "xgboost_oof_mae_ref": XGB_MAE,
        "delta_pct_vs_raw_rf": delta_pct_vs_raw_rf,
        "delta_pct_vs_raw_xgboost": delta_pct_vs_raw_xgb,
        "delta_pct_rf_vs_xgboost": delta_pct_rf_vs_xgb,
        "n_optuna_trials": 50,
        "wall_time_s": time.time() - t_total,
    }
    (OUT / "phase_C_hawker_rf.json").write_text(json.dumps(out, indent=2))
    print(f"\n→ {OUT / 'phase_C_hawker_rf.json'}")
    print(f"Total wall time: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
