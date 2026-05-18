#!/usr/bin/env python3
"""P4 orchestrator: out-of-distribution test on Vietnam Mekong + Atacama.

Goal: extend the OOD generalisation claim of the paper from within-Chile
cross-regime (Mediterranean → humid temperate) to *across continents and
climate extremes*. Two new tiles selected for maximum climatic contrast:

  - N10E105  Vietnam, Mekong Delta (10–11°N, 105–106°E)
              Tropical wet, dense agriculture/wetland, flat to gentle relief.
              Same study area as Hawker et al. (2024) Vietnam flood paper.
              UTM zone 48 North → EPSG:32648.

  - S24W069  Chile, Atacama (23–24°S, 69–68°W)
              Hyperarid (<10 mm/yr), zero vegetation, Andean foothills +
              Salar de Atacama playa. Maximum opposite of humid temperate.
              UTM zone 19 South → EPSG:32719 (same as M-M).

Applies the Mediterranean-trained XGBoost model (xgb_mm_booster.json) to both
tiles WITHOUT retraining and reports RMSE/MAE/sign-test/Diebold-Mariano per
tile and per region.
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
SCALE_P4 = ROOT / "scale_p4"
SCALE_P1 = ROOT / "scale_p1"

# Tile_name, bbox "W,S,E,N", UTM EPSG, S2 time window, region label
NEW_TILES = [
    ("N10E105", "105,10,106,11",  32648, "2022-12-01/2023-04-30", "tropical_wet"),
    ("S24W069", "-69,-24,-68,-23", 32719, "2022-12-01/2023-02-28", "hyperarid"),
]


def run_tile(tile_name, bbox, utm_epsg, time_window):
    print(f"\n{'='*72}", flush=True)
    print(f"TILE {tile_name}  bbox={bbox}  UTM=EPSG:{utm_epsg}  TIME={time_window}",
          flush=True)
    print('='*72, flush=True)
    env = os.environ.copy()
    env["PIPELINE_UTM_EPSG"] = str(utm_epsg)
    env["PIPELINE_TIME_WINDOW"] = time_window
    rc = subprocess.call(
        ["python3", str(SCALE_P4 / "run_tile.py"), tile_name, bbox],
        env=env,
    )
    return rc == 0


def unify_p4_samples():
    """Concatenate Vietnam + Atacama samples for OOD analysis."""
    import pandas as pd
    out_dir = SCALE_P4 / "samples_unified"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "samples_p4_full.csv"

    dfs = []
    for tile_name, _, _, _, region in NEW_TILES:
        csv = SCALE_P4 / "tiles" / tile_name / "samples" / "tile_samples.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["tile"] = tile_name
            df["regime"] = region
            dfs.append(df)
            print(f"  + {tile_name} ({region}): {len(df)} pts")
        else:
            print(f"  ✗ {tile_name}: tile_samples.csv missing")

    if not dfs:
        print("✗ No P4 samples produced")
        return None

    common = sorted(set.intersection(*(set(d.columns) for d in dfs)))
    unified = pd.concat([d[common] for d in dfs], ignore_index=True)
    unified.to_csv(out, index=False)
    print(f"→ {out}  ({len(unified)} rows × {len(common)} cols)")
    return out


def evaluate_ood(unified_csv):
    """Apply M-M model to P4 samples without retraining."""
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    out_dir = SCALE_P4 / "samples_unified"
    print(f"\n{'='*72}")
    print("OOD EVALUATION: Mediterranean-trained model on Vietnam + Atacama")
    print('='*72)

    df = pd.read_csv(unified_csv)
    if "stream_network" in df.columns:
        df["stream_network"] = df["stream_network"].fillna(0)

    features_meta = json.loads((SCALE_P1 / "samples_unified" /
                                "mm_metrics.json").read_text())
    FEATURES = features_meta["features"]
    print(f"Loaded {len(df)} rows, {len(FEATURES)} features")

    booster = xgb.Booster()
    booster.load_model(str(SCALE_P1 / "samples_unified" / "xgb_mm_booster.json"))
    X = df[FEATURES].values
    dmat = xgb.DMatrix(X, missing=np.nan)
    pred = booster.predict(dmat)

    df["pred_residual_mm_model"] = pred
    df["err_raw"] = df["fabdem"] - df["h_te_orthometric"]
    df["err_corr"] = (df["fabdem"] + pred) - df["h_te_orthometric"]

    # Per region
    print(f"\n  {'region':<18s} {'n':>8s} {'rmse_raw':>10s} {'rmse_corr':>11s} "
          f"{'mae_raw':>10s} {'mae_corr':>11s} {'bias_raw':>10s} {'bias_corr':>11s} {'Δ%':>7s}")
    print("  " + "-"*96)
    summary = {}
    for regime, sub in df.groupby("regime"):
        rmse_r = float(np.sqrt(np.mean(sub.err_raw**2)))
        rmse_c = float(np.sqrt(np.mean(sub.err_corr**2)))
        mae_r = float(np.mean(np.abs(sub.err_raw)))
        mae_c = float(np.mean(np.abs(sub.err_corr)))
        bias_r = float(sub.err_raw.mean())
        bias_c = float(sub.err_corr.mean())
        d = 100 * (rmse_c - rmse_r) / rmse_r
        # Sign test: fraction of footprints improved
        improved = int((np.abs(sub.err_corr) < np.abs(sub.err_raw)).sum())
        frac_improved = improved / len(sub)
        # Diebold–Mariano (simple)
        d_sq = sub.err_raw**2 - sub.err_corr**2
        dm_t = float(d_sq.mean() / (d_sq.std() / np.sqrt(len(sub))))
        summary[regime] = {
            "n": int(len(sub)),
            "rmse_raw": rmse_r, "rmse_corr": rmse_c, "delta_pct": d,
            "mae_raw": mae_r, "mae_corr": mae_c,
            "bias_raw": bias_r, "bias_corr": bias_c,
            "frac_improved": frac_improved, "improved": improved,
            "dm_t_stat": dm_t,
        }
        print(f"  {regime:<18s} {len(sub):>8d} {rmse_r:>10.3f} {rmse_c:>11.3f} "
              f"{mae_r:>10.3f} {mae_c:>11.3f} {bias_r:>+10.3f} {bias_c:>+11.3f} {d:>+7.2f}")

    # Per tile
    print(f"\n  Per-tile:")
    print(f"  {'tile':<10s} {'regime':<18s} {'n':>7s} {'rmse_raw':>10s} "
          f"{'rmse_corr':>11s} {'Δ%':>7s} {'improved%':>10s}")
    print("  " + "-"*72)
    for (tile, regime), sub in df.groupby(["tile", "regime"]):
        rmse_r = float(np.sqrt(np.mean(sub.err_raw**2)))
        rmse_c = float(np.sqrt(np.mean(sub.err_corr**2)))
        d = 100 * (rmse_c - rmse_r) / rmse_r
        frac = float((np.abs(sub.err_corr) < np.abs(sub.err_raw)).mean())
        print(f"  {tile:<10s} {regime:<18s} {len(sub):>7d} {rmse_r:>10.3f} "
              f"{rmse_c:>11.3f} {d:>+7.2f} {frac*100:>9.1f}%")

    df.to_csv(out_dir / "samples_p4_with_mm_predictions.csv", index=False)
    (out_dir / "ood_p4_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ samples_p4_with_mm_predictions.csv")
    print(f"→ ood_p4_summary.json")


def main():
    print("="*72)
    print("P4 — OOD multi-continent test (Vietnam Mekong + Atacama)")
    print("="*72)
    t0 = time.time()
    failed = []
    for tile_name, bbox, utm, time_window, _ in NEW_TILES:
        ok = run_tile(tile_name, bbox, utm, time_window)
        if not ok:
            failed.append(tile_name)

    if failed:
        print(f"\n⚠ Failed tiles: {failed}")

    unified = unify_p4_samples()
    if unified:
        evaluate_ood(unified)

    print(f"\n{'='*72}")
    print(f"✅ P4 complete in {(time.time()-t0)/60:.1f} min")
    print(f"{'='*72}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
