#!/usr/bin/env python3
"""Feature ablation + block-size sensitivity for Mediterranean dataset.

Two experiments using the locked-in best_params from Optuna (no re-tuning):
  (A) Feature ablation: 5 conditions removing feature groups
  (B) Block-size sensitivity: 5, 10, 20 km spatial-block CV

Output: ablation_blocksize_results.json
"""
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pyproj

warnings.filterwarnings("ignore")

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
SAMP = ROOT / "scale_p1" / "samples_unified"
OUT_JSON = ROOT / "paper" / "experiments" / "ablation_blocksize_results.json"

# Same locked-in best params from Optuna (mm_metrics.json)
BEST_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 9,
    "learning_rate": 0.011289111763110846,
    "subsample": 0.9533188971118374,
    "colsample_bytree": 0.6354986604850102,
    "min_child_weight": 14,
    "reg_lambda": 7.40970392425014,
    "reg_alpha": 1.5969596058836053,
    "gamma": 2.2017805740056464,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 4,
}

# Feature groups
GROUPS = {
    "terrain_1st": ["slope", "aspect", "hillshade", "eastness", "northness"],
    "terrain_2nd": ["curvature", "tpi", "tri", "vrm", "dev", "convergence"],
    "shape_3d": ["openness_negative", "openness_positive", "svf"],
    "landform": ["mrvbf", "mrrtf", "geomorphons"],
    "hydro": ["hand", "twi", "flow_accumulation", "flow_accumulation_mfd",
              "flow_direction_d8", "flow_direction_dinf", "stream_network", "valley_depth"],
    "s2_optical": ["ndvi", "ndwi", "ndmi", "bsi", "ndbi"],
    "s1_sar": ["s1_vh_db", "s1_vv_db", "s1_vv_vh_ratio"],
}
ALL_FEATURES = [f for g in GROUPS.values() for f in g]
assert len(ALL_FEATURES) == 33, f"Expected 33, got {len(ALL_FEATURES)}"

# Ablation conditions: drop one or two groups at a time
ABLATIONS = {
    "all_33": [],
    "drop_optical": ["s2_optical"],          # 33 - 5 = 28
    "drop_sar": ["s1_sar"],                  # 33 - 3 = 30
    "drop_satellite": ["s2_optical", "s1_sar"],   # 33 - 8 = 25
    "drop_hydro": ["hydro"],                 # 33 - 8 = 25
    "drop_3d_shape": ["shape_3d"],           # 33 - 3 = 30
    "drop_landform": ["landform"],           # 33 - 3 = 30
    "terrain_only": ["hydro", "s2_optical", "s1_sar", "landform"],  # 14
}

# Block sizes (meters)
BLOCK_SIZES = [5000.0, 10000.0, 20000.0]


def load_data():
    df = pd.read_csv(SAMP / "samples_mm_full.csv")
    if "stream_network" in df.columns:
        df["stream_network"] = df["stream_network"].fillna(0)
    target = "residual_corrected"
    keep = np.isfinite(df[target].values)
    df = df.loc[keep].reset_index(drop=True)
    print(f"Loaded {len(df)} rows with finite target")
    return df


def get_blocks(lon, lat, block_m):
    t = pyproj.Transformer.from_crs(4326, 32719, always_xy=True)
    x_utm, y_utm = t.transform(lon, lat)
    block_x = (x_utm // block_m).astype(int)
    block_y = (y_utm // block_m).astype(int)
    return block_x * 100000 + block_y


def cv_rmse(X, y, splits):
    """Spatial CV with early stopping on test fold."""
    preds = np.full_like(y, np.nan, dtype=np.float64)
    for tr, te in splits:
        m = xgb.XGBRegressor(**BEST_PARAMS, early_stopping_rounds=30)
        m.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
        preds[te] = m.predict(X[te])
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    return rmse, mae, preds


def main():
    df = load_data()
    y = df["residual_corrected"].values
    lon = df["lon"].values
    lat = df["lat"].values
    rmse_base = float(np.sqrt(mean_squared_error(y, np.zeros_like(y))))
    mae_base = float(mean_absolute_error(y, np.zeros_like(y)))
    print(f"FABDEM raw baseline: RMSE={rmse_base:.3f}  MAE={mae_base:.3f}\n")

    results = {
        "n_samples": len(df),
        "baseline_rmse": rmse_base,
        "baseline_mae": mae_base,
        "best_params": {k: v for k, v in BEST_PARAMS.items()
                        if k not in ("tree_method", "random_state", "n_jobs")},
        "ablation": {},
        "block_size": {},
    }

    # ===================== A) FEATURE ABLATION (block=10km fixed) ======
    print("=" * 60)
    print("A) FEATURE ABLATION  (block=10 km fixed, K=5)")
    print("=" * 60)
    blocks_10km = get_blocks(lon, lat, 10000.0)
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.zeros(len(df)), y, groups=blocks_10km))

    for name, drop_groups in ABLATIONS.items():
        t0 = time.time()
        dropped = [f for g in drop_groups for f in GROUPS[g]]
        features = [f for f in ALL_FEATURES if f not in dropped]
        X = df[features].values
        rmse, mae, _ = cv_rmse(X, y, splits)
        dt = time.time() - t0
        delta_pct = 100.0 * (rmse_base - rmse) / rmse_base
        results["ablation"][name] = {
            "n_features": len(features),
            "dropped_groups": drop_groups,
            "dropped_features": dropped,
            "rmse_oof": rmse,
            "mae_oof": mae,
            "delta_pct_vs_raw": delta_pct,
            "elapsed_s": dt,
        }
        print(f"  {name:18s} | n_feat={len(features):>3d} | RMSE={rmse:.4f} | "
              f"MAE={mae:.4f} | Δ={delta_pct:+.2f}% | {dt:.1f}s")

    # ===================== B) BLOCK-SIZE SENSITIVITY (all features) ====
    print("\n" + "=" * 60)
    print("B) BLOCK-SIZE SENSITIVITY  (all 33 features, K=5)")
    print("=" * 60)
    X = df[ALL_FEATURES].values
    for bm in BLOCK_SIZES:
        t0 = time.time()
        blocks = get_blocks(lon, lat, bm)
        n_blocks = len(np.unique(blocks))
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(np.zeros(len(df)), y, groups=blocks))
        rmse, mae, _ = cv_rmse(X, y, splits)
        dt = time.time() - t0
        delta_pct = 100.0 * (rmse_base - rmse) / rmse_base
        key = f"{int(bm/1000)}km"
        results["block_size"][key] = {
            "block_size_m": bm,
            "n_blocks": int(n_blocks),
            "rmse_oof": rmse,
            "mae_oof": mae,
            "delta_pct_vs_raw": delta_pct,
            "elapsed_s": dt,
        }
        print(f"  {key:>5s} | n_blocks={n_blocks:>5d} | RMSE={rmse:.4f} | "
              f"MAE={mae:.4f} | Δ={delta_pct:+.2f}% | {dt:.1f}s")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n→ {OUT_JSON}")


if __name__ == "__main__":
    main()
