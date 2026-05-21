#!/usr/bin/env python3
"""P5 orchestrator: Andes tropicales (Cusco, Perú) — 5th climate regime.

Designed as a CONFIRMATORY test of the bounded-transferability hypothesis
identified in Section 6 of the paper. Prediction: favourable transfer
(negative Δ% RMSE), because Cusco satisfies both pre-conditions identified
empirically in §6.4:
  (i) substantial relief variation (500–6000 m elevation range, Andean
      tectonic substrate same as M-M training tiles),
  (ii) positive FABDEM canopy bias (humid montane Andean forest plus
       puna / paramo at elevation, NDVI moderate-to-high).

If the Mediterranean-trained model transfers favourably here, it falsifies the
alternative explanation that the within-Chile transfer relies on idiosyncratic
patterns peculiar to Chilean Mediterranean training data.

Tile: S13W072 (Cusco–Vilcanota region, Peruvian Andes)
  bbox lon, lat: -73 to -72, -14 to -13
  UTM zone: 18 South → EPSG:32718
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
SCALE_P5 = ROOT / "scale_p5"
SCALE_P1 = ROOT / "scale_p1"

# (tile_name, bbox "W,S,E,N", UTM EPSG, S2 time window, region label)
NEW_TILES = [
    ("S13W072", "-73,-14,-72,-13", 32718, "2022-06-01/2023-08-31", "tropical_montane"),
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
        ["python3", str(SCALE_P5 / "run_tile.py"), tile_name, bbox],
        env=env,
    )
    return rc == 0


def evaluate_ood():
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    out_dir = SCALE_P5 / "samples_unified"
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for tile_name, _, _, _, region in NEW_TILES:
        csv = SCALE_P5 / "tiles" / tile_name / "samples" / "tile_samples.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["tile"] = tile_name
            df["regime"] = region
            dfs.append(df)
    if not dfs:
        print("✗ no samples")
        return
    df = pd.concat(dfs, ignore_index=True)
    if "stream_network" in df.columns:
        df["stream_network"] = df["stream_network"].fillna(0)

    features_meta = json.loads((SCALE_P1 / "samples_unified" /
                                "mm_metrics.json").read_text())
    FEATURES = features_meta["features"]
    print(f"\nLoaded {len(df)} rows, {len(FEATURES)} features")

    booster = xgb.Booster()
    booster.load_model(str(SCALE_P1 / "samples_unified" / "xgb_mm_booster.json"))
    X = df[FEATURES].values
    dmat = xgb.DMatrix(X, missing=np.nan)
    pred = booster.predict(dmat)
    df["pred_residual_mm_model"] = pred
    df["err_raw"] = df.fabdem - df.h_te_orthometric
    df["err_corr"] = df.fabdem + pred - df.h_te_orthometric

    print(f"\n{'='*72}")
    print("OOD EVALUATION: Mediterranean-trained model on Cusco (tropical montane Andes)")
    print('='*72)
    print(f"  {'region':<20s} {'n':>8s} {'rmse_raw':>10s} {'rmse_corr':>11s} "
          f"{'mae_raw':>10s} {'mae_corr':>11s} {'bias_raw':>10s} {'bias_corr':>11s} {'Δ%':>7s}")
    print("  " + "-"*102)
    summary = {}
    for regime, sub in df.groupby("regime"):
        rr = float(np.sqrt(np.mean(sub.err_raw ** 2)))
        rc = float(np.sqrt(np.mean(sub.err_corr ** 2)))
        mr = float(np.mean(np.abs(sub.err_raw)))
        mc = float(np.mean(np.abs(sub.err_corr)))
        br, bc = float(sub.err_raw.mean()), float(sub.err_corr.mean())
        d = 100 * (rc - rr) / rr
        improved = int((np.abs(sub.err_corr) < np.abs(sub.err_raw)).sum())
        d_sq = sub.err_raw ** 2 - sub.err_corr ** 2
        dm_t = float(d_sq.mean() / (d_sq.std() / np.sqrt(len(sub))))
        summary[regime] = {
            "n": int(len(sub)), "rmse_raw": rr, "rmse_corr": rc,
            "delta_pct": d, "mae_raw": mr, "mae_corr": mc,
            "bias_raw": br, "bias_corr": bc,
            "improved": improved, "frac_improved": improved / len(sub),
            "dm_t_stat": dm_t,
        }
        print(f"  {regime:<20s} {len(sub):>8d} {rr:>10.3f} {rc:>11.3f} "
              f"{mr:>10.3f} {mc:>11.3f} {br:>+10.3f} {bc:>+11.3f} {d:>+7.2f}")

    df.to_csv(out_dir / "samples_p5_with_mm_predictions.csv", index=False)
    (out_dir / "ood_p5_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n→ samples_p5_with_mm_predictions.csv")
    print(f"→ ood_p5_summary.json")

    # Prediction outcome
    cusco_delta = summary.get("tropical_montane", {}).get("delta_pct", None)
    if cusco_delta is None:
        return
    print(f"\n{'='*72}")
    print("HYPOTHESIS TEST OUTCOME")
    print('='*72)
    if cusco_delta < 0:
        print(f"  ✓ Cusco Δ% = {cusco_delta:+.2f}% — FAVOURABLE transfer.")
        print(f"    Consistent with the two pre-condition hypothesis (relief + canopy bias).")
    else:
        print(f"  ✗ Cusco Δ% = {cusco_delta:+.2f}% — UNFAVOURABLE transfer.")
        print(f"    Falsifies the two pre-condition hypothesis as currently stated.")


def main():
    print("="*72)
    print("P5 — Cusco (tropical montane Andes) confirmatory OOD test")
    print("="*72)
    t0 = time.time()
    failed = []
    for tile_name, bbox, utm, time_window, _ in NEW_TILES:
        ok = run_tile(tile_name, bbox, utm, time_window)
        if not ok:
            failed.append(tile_name)
    if failed:
        print(f"\n⚠ Failed tiles: {failed}")
    evaluate_ood()
    print(f"\n{'='*72}\n✅ P5 complete in {(time.time()-t0)/60:.1f} min\n{'='*72}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
