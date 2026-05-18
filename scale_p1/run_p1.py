#!/usr/bin/env python3
"""P1 orchestrator: process 5 new tiles + reuse S36W072 from E1a, unify, train.

Tiles for Mataquito + Maule complete:
  S35W072 — bbox [-72, -35, -71, -34]   (Curicó, north Mataquito)
  S35W071 — bbox [-71, -35, -70, -34]   (Andes north Mataquito)
  S36W072 — already done as E1a — REUSED
  S36W071 — bbox [-71, -36, -70, -35]   (Andes central Maule)
  S37W072 — bbox [-72, -37, -71, -36]   (south Maule + Linares)
  S37W071 — bbox [-71, -37, -70, -36]   (Andes south Maule)
"""
import os, sys, time, json, subprocess
from pathlib import Path

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
SCALE_P1 = ROOT / "scale_p1"
SCALE_E1A = ROOT / "scale_e1a"

# (tile_name, bbox W,S,E,N)
NEW_TILES = [
    ("S35W072", "-72,-35,-71,-34"),
    ("S35W071", "-71,-35,-70,-34"),
    ("S36W071", "-71,-36,-70,-35"),
    ("S37W072", "-72,-37,-71,-36"),
    ("S37W071", "-71,-37,-70,-36"),
]


def run_tile(tile_name, bbox):
    print(f"\n{'='*70}\nTILE {tile_name}\n{'='*70}", flush=True)
    rc = subprocess.call(
        ["python3", str(SCALE_P1 / "run_tile.py"), tile_name, bbox],
    )
    return rc == 0


def unify_samples():
    """Concatenate all per-tile samples + reuse E1a."""
    import pandas as pd

    out = SCALE_P1 / "samples_unified" / "samples_mm_full.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"✓ {out.name} already exists")
        return out

    dfs = []
    # E1a (S36W072)
    e1a_csv = SCALE_E1A / "samples" / "pilot_e1a_full.csv"
    if e1a_csv.exists():
        df = pd.read_csv(e1a_csv)
        df["tile"] = "S36W072"
        dfs.append(df)
        print(f"  + S36W072 (E1a): {len(df)} pts")

    # P1 tiles
    for tile_name, _ in NEW_TILES:
        csv = SCALE_P1 / "tiles" / tile_name / "samples" / "tile_samples.csv"
        if not csv.exists():
            print(f"  ⚠ missing {csv}, skip")
            continue
        df = pd.read_csv(csv)
        df["tile"] = tile_name
        dfs.append(df)
        print(f"  + {tile_name}: {len(df)} pts")

    if not dfs:
        print("✗ no per-tile samples found")
        return None

    # Align columns (some tiles may have different feature set if a raster failed)
    common_cols = set(dfs[0].columns)
    for d in dfs[1:]:
        common_cols &= set(d.columns)
    common_cols = sorted(common_cols)
    print(f"  common features across tiles: {len(common_cols)}")

    unified = pd.concat([d[common_cols] for d in dfs], ignore_index=True)
    print(f"  unified rows: {len(unified)}  cols: {len(unified.columns)}")
    unified.to_csv(out, index=False)
    print(f"→ {out}  ({out.stat().st_size/1024/1024:.1f} MB)")
    return out


def train_xgb(unified_csv):
    """Optuna 100 trials + spatial CV on unified dataset."""
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import optuna
    import shap
    import pyproj

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out_dir = SCALE_P1 / "samples_unified"

    df = pd.read_csv(unified_csv)
    if "stream_network" in df.columns:
        df["stream_network"] = df["stream_network"].fillna(0)

    TARGET = "residual_corrected"
    META = {"granule", "beam", "lon", "lat", "h_te", "h_te_unc", "te_qual",
            "night", "terrain_flg", "n_te_phot", "snow", "water", "cloud",
            "h_te_orthometric", "geoid_N", "residual_raw", "residual_corrected",
            "fabdem", "filled", "tile"}
    FEATURES = [c for c in df.columns if c not in META and df[c].dtype != "O"]
    print(f"\nRows: {len(df)}  Features: {len(FEATURES)}")
    print(f"Per-tile counts: {df['tile'].value_counts().to_dict()}")

    X = df[FEATURES].values
    y = df[TARGET].values
    lon, lat = df.lon.values, df.lat.values
    keep = np.isfinite(y)
    X, y, lon, lat = X[keep], y[keep], lon[keep], lat[keep]
    print(f"With finite target: {keep.sum()}")

    t = pyproj.Transformer.from_crs(4326, 32719, always_xy=True)
    x_utm, y_utm = t.transform(lon, lat)
    BLOCK_M = 10000.0
    blocks = (x_utm // BLOCK_M).astype(int) * 10000 + (y_utm // BLOCK_M).astype(int)
    uniq = np.unique(blocks)
    print(f"\nSpatial blocks ({BLOCK_M:.0f} m): {len(uniq)}")

    n_splits = min(5, len(uniq))
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups=blocks))

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
            "n_estimators": 1000,
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        }
        return cv_rmse(params, splits)[0]

    print("\n=== Optuna 100 trials (spatial K=5) ===")
    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=100, show_progress_bar=False)
    print(f"Best RMSE: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best = {**study.best_params, "n_estimators": 1000}
    rmse_sp, pred_sp = cv_rmse(best, splits)
    mae_sp = float(mean_absolute_error(y, pred_sp))
    r2_sp = float(r2_score(y, pred_sp))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_rn, pred_rn = cv_rmse(best, list(kf.split(X)))
    mae_rn = float(mean_absolute_error(y, pred_rn))
    r2_rn = float(r2_score(y, pred_rn))

    print(f"\n=== Final ===")
    print(f"  Baseline:     RMSE={rmse_base:.3f}")
    print(f"  Spatial CV:   RMSE={rmse_sp:.3f}  MAE={mae_sp:.3f}  R²={r2_sp:+.3f}  Δ={100*(rmse_sp-rmse_base)/rmse_base:+.1f}%")
    print(f"  Random  CV:   RMSE={rmse_rn:.3f}  MAE={mae_rn:.3f}  R²={r2_rn:+.3f}  Δ={100*(rmse_rn-rmse_base)/rmse_base:+.1f}%")

    final = xgb.XGBRegressor(**best, tree_method="hist", random_state=42, n_jobs=4)
    final.fit(X, y, verbose=False)
    final.save_model(str(out_dir / "xgb_mm.json"))
    print(f"\n→ model saved: {out_dir/'xgb_mm.json'}")

    explainer = shap.TreeExplainer(final)
    sample_idx = np.random.default_rng(42).choice(len(X), min(8000, len(X)), replace=False)
    sv = explainer.shap_values(X[sample_idx])
    shap_imp = pd.DataFrame({
        "feature": FEATURES,
        "mean_abs_shap": np.abs(sv).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    print(f"\n=== SHAP top 15 ===")
    print(shap_imp.head(15).to_string(index=False))
    shap_imp.to_csv(out_dir / "mm_shap.csv", index=False)

    fold_sum = []
    for i, (_, te) in enumerate(splits):
        fold_sum.append({
            "fold": i, "n_test": int(te.size),
            "rmse": float(np.sqrt(mean_squared_error(y[te], pred_sp[te]))),
            "mae": float(mean_absolute_error(y[te], pred_sp[te])),
            "r2": float(r2_score(y[te], pred_sp[te])),
        })

    metrics = {
        "n_samples": int(keep.sum()),
        "n_features": len(FEATURES),
        "n_blocks": int(len(uniq)),
        "block_size_m": BLOCK_M,
        "n_optuna_trials": 100,
        "per_tile_counts": df.loc[keep, "tile"].value_counts().to_dict(),
        "baseline_rmse": rmse_base, "baseline_mae": mae_base,
        "spatial_cv": {"rmse": rmse_sp, "mae": mae_sp, "r2": r2_sp, "folds": fold_sum},
        "random_cv": {"rmse": rmse_rn, "mae": mae_rn, "r2": r2_rn},
        "best_params": best,
    }
    (out_dir / "mm_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n→ metrics: {out_dir/'mm_metrics.json'}")


def main():
    print("="*70)
    print("P1 — Maule + Mataquito complete (6 tiles)")
    print("="*70)
    t0 = time.time()

    # Phase 1: process new tiles
    failed = []
    for tile_name, bbox in NEW_TILES:
        ok = run_tile(tile_name, bbox)
        if not ok:
            failed.append(tile_name)

    if failed:
        print(f"\n⚠ Failed tiles: {failed}")

    # Phase 2: unify
    print(f"\n{'='*70}\nUNIFY samples\n{'='*70}")
    unified = unify_samples()
    if not unified:
        print("Cannot proceed to training without unified samples")
        return 1

    # Phase 3: train
    print(f"\n{'='*70}\nTRAIN XGBoost on unified dataset\n{'='*70}")
    train_xgb(unified)

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"✅ P1 complete in {total/60:.1f} min")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
