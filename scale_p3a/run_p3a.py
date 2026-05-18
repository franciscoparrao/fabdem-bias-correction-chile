#!/usr/bin/env python3
"""P3a orchestrator: extend to humid temperate forest regime (37-39°S).

Adds 4 new tiles south of Maule (Bío-Bío + Araucanía), then unifies with
E1a + P1 samples (10 tiles total) and retrains.

This tests whether the M-M-trained pattern (NDVI denso → −41% RMSE) generalizes
to a fundamentally different climate regime.
"""
import os, sys, time, json, subprocess
from pathlib import Path

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
SCALE_P3A = ROOT / "scale_p3a"
SCALE_P1 = ROOT / "scale_p1"
SCALE_E1A = ROOT / "scale_e1a"

# 4 new tiles for humid temperate forest regime (Bío-Bío + Araucanía)
NEW_TILES = [
    ("S38W072", "-72,-38,-71,-37"),  # Bío-Bío valley/Andes
    ("S38W073", "-73,-38,-72,-37"),  # Bío-Bío coast/valley
    ("S39W072", "-72,-39,-71,-38"),  # Araucanía valley/Andes
    ("S39W073", "-73,-39,-72,-38"),  # Araucanía coast (wettest)
]


def run_tile(tile_name, bbox):
    print(f"\n{'='*70}\nTILE {tile_name}  bbox={bbox}\n{'='*70}", flush=True)
    rc = subprocess.call(
        ["python3", str(SCALE_P3A / "run_tile.py"), tile_name, bbox],
    )
    return rc == 0


def unify_samples():
    """Concatenate E1a + P1 (5 new) + P3a (4 new) = 10 tiles."""
    import pandas as pd
    out = SCALE_P3A / "samples_unified" / "samples_p3a_full.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    dfs = []
    # E1a
    e1a_csv = SCALE_E1A / "samples" / "pilot_e1a_full.csv"
    if e1a_csv.exists():
        df = pd.read_csv(e1a_csv); df["tile"] = "S36W072"; df["regime"] = "mediterranean"
        dfs.append(df); print(f"  + S36W072 (E1a/M): {len(df)} pts")
    # P1 (Mediterranean)
    for t in ["S35W072", "S35W071", "S36W071", "S37W072", "S37W071"]:
        csv = SCALE_P1 / "tiles" / t / "samples" / "tile_samples.csv"
        if csv.exists():
            df = pd.read_csv(csv); df["tile"] = t; df["regime"] = "mediterranean"
            dfs.append(df); print(f"  + {t} (P1/M): {len(df)} pts")
    # P3a (humid temperate)
    for t, _ in NEW_TILES:
        csv = SCALE_P3A / "tiles" / t / "samples" / "tile_samples.csv"
        if csv.exists():
            df = pd.read_csv(csv); df["tile"] = t; df["regime"] = "humid_temperate"
            dfs.append(df); print(f"  + {t} (P3a/HT): {len(df)} pts")

    if not dfs:
        print("✗ no samples found")
        return None

    common = set(dfs[0].columns)
    for d in dfs[1:]:
        common &= set(d.columns)
    common = sorted(common)
    unified = pd.concat([d[common] for d in dfs], ignore_index=True)
    print(f"  Unified: {len(unified)} rows × {len(unified.columns)} cols")
    print(f"  By regime: {unified['regime'].value_counts().to_dict()}")
    unified.to_csv(out, index=False)
    print(f"→ {out}  ({out.stat().st_size/1024/1024:.1f} MB)")
    return out


def evaluate_generalization(unified_csv):
    """Critical test: how does the M-M-only model perform on P3a tiles
    WITHOUT re-training? This is the key generalization claim.
    """
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    out_dir = SCALE_P3A / "samples_unified"
    print(f"\n{'='*70}\nGENERALIZATION TEST: M-M model on P3a tiles\n{'='*70}")

    df = pd.read_csv(unified_csv)
    if "stream_network" in df.columns:
        df["stream_network"] = df["stream_network"].fillna(0)

    META = {"granule", "beam", "lon", "lat", "h_te", "h_te_unc", "te_qual",
            "night", "terrain_flg", "n_te_phot", "snow", "water", "cloud",
            "h_te_orthometric", "geoid_N", "residual_raw", "residual_corrected",
            "fabdem", "filled", "tile", "regime"}
    FEATURES_FILE = SCALE_P1 / "samples_unified" / "mm_metrics.json"
    FEATURES = json.loads(FEATURES_FILE.read_text())["features"]

    # Load M-M model (trained on Mediterranean only)
    booster = xgb.Booster()
    booster.load_model(str(SCALE_P1 / "samples_unified" / "xgb_mm_booster.json"))

    # Predict residual on ALL data; segregate by regime
    X = df[FEATURES].values
    dmat = xgb.DMatrix(X, missing=np.nan)
    pred = booster.predict(dmat)

    df["pred_residual_mm_model"] = pred
    df["err_raw"] = df["fabdem"] - df["h_te_orthometric"]
    df["err_corr_mm_model"] = (df["fabdem"] + pred) - df["h_te_orthometric"]

    # Per regime
    print(f"\n  {'regime':<20s} {'n':>8s} {'rmse_raw':>10s} {'rmse_corr':>11s} {'Δ%':>7s}")
    print("  " + "-"*60)
    for regime, sub in df.groupby("regime"):
        if len(sub) == 0:
            continue
        rmse_r = float(np.sqrt(np.mean(sub.err_raw**2)))
        rmse_c = float(np.sqrt(np.mean(sub.err_corr_mm_model**2)))
        d = 100 * (rmse_c - rmse_r) / rmse_r
        print(f"  {regime:<20s} {len(sub):>8d} {rmse_r:>10.3f} {rmse_c:>11.3f} {d:>+7.2f}")

    # Per tile
    print(f"\n  Per-tile (M-M model applied):")
    print(f"  {'tile':<10s} {'regime':<18s} {'n':>7s} {'rmse_raw':>10s} {'rmse_corr':>11s} {'Δ%':>7s}")
    print("  " + "-"*65)
    for (tile, regime), sub in df.groupby(["tile", "regime"]):
        rmse_r = float(np.sqrt(np.mean(sub.err_raw**2)))
        rmse_c = float(np.sqrt(np.mean(sub.err_corr_mm_model**2)))
        d = 100 * (rmse_c - rmse_r) / rmse_r
        print(f"  {tile:<10s} {regime:<18s} {len(sub):>7d} {rmse_r:>10.3f} {rmse_c:>11.3f} {d:>+7.2f}")

    df.to_csv(out_dir / "samples_p3a_with_mm_predictions.csv", index=False)


def main():
    print("="*70)
    print("P3a — Humid temperate regime extension (4 tiles, 37-39°S)")
    print("="*70)
    t0 = time.time()

    failed = []
    for tile_name, bbox in NEW_TILES:
        ok = run_tile(tile_name, bbox)
        if not ok:
            failed.append(tile_name)
    if failed:
        print(f"\n⚠ Failed tiles: {failed}")

    print(f"\n{'='*70}\nUNIFY samples (E1a + P1 + P3a = 10 tiles)\n{'='*70}")
    unified = unify_samples()
    if unified:
        evaluate_generalization(unified)

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"✅ P3a complete in {total/60:.1f} min")
    print(f"{'='*70}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
