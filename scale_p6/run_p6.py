#!/usr/bin/env python3
"""P6 orchestrator: tres tiles tropicales montanos adicionales.

Diseñados para densificar el factorial relief × canopy bias detrás del split
de Section 5.5 / 6.3. Predicción uniforme: favourable transfer (Δ% < 0) si la
hipótesis de las dos pre-condiciones (Andean-style relief + positive FABDEM
canopy bias) se sostiene. Si alguno degrada, refutamos parcialmente.

  - N04W074  Colombia, Boyacá / Cordillera Oriental
              bbox -74,4,-73,5  UTM 18N → EPSG:32618
              dry season window 2022-12 → 2023-03
  - S01W079  Ecuador, Pichincha-Cotopaxi sur
              bbox -79,-1,-78,0  UTM 17S → EPSG:32717
              dry season window 2023-06 → 2023-09
  - S16W068  Bolivia, La Paz Yungas / Cordillera Real
              bbox -68,-16,-67,-15  UTM 19S → EPSG:32719
              austral dry window 2023-05 → 2023-10

Each tile reuses scale_p4 steps via symlink (run_tile.py + steps/).
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
SCALE_P6 = ROOT / "scale_p6"
SCALE_P4 = ROOT / "scale_p4"   # symlink target: tiles land there
SCALE_P1 = ROOT / "scale_p1"

# (tile_name, bbox W,S,E,N, UTM EPSG, S2 time, region label)
NEW_TILES = [
    ("N04W074", "-74,4,-73,5",     32618, "2022-12-01/2023-03-31", "tropical_montane_co"),
    ("S01W079", "-79,-1,-78,0",    32717, "2023-06-01/2023-09-30", "tropical_montane_ec"),
    ("S16W068", "-68,-16,-67,-15", 32719, "2023-05-01/2023-10-31", "tropical_montane_bo"),
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
        ["python3", str(SCALE_P6 / "run_tile.py"), tile_name, bbox],
        env=env,
    )
    return rc == 0


def evaluate_ood():
    """Apply M-M model (no retrain) to the new 3 tropical-montane tiles and
    consolidate with the existing Cusco S13W072 sample.
    """
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    out_dir = SCALE_P6 / "samples_unified"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The tiles land under scale_p4/tiles/ because run_tile.py's __file__
    # resolves to its symlink target (scale_p4) — same trick we already
    # exploit in scale_p5. So we collect from scale_p4/tiles/.
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
        print("✗ no samples produced")
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
    print("OOD EVALUATION: M-M model on 3 tropical-montane tiles (Col, Ec, Bo)")
    print('='*72)
    print(f"  {'regime/tile':<22s} {'n':>8s} {'rmse_raw':>10s} {'rmse_corr':>11s} "
          f"{'mae_raw':>10s} {'mae_corr':>11s} {'bias_raw':>10s} {'bias_corr':>11s} {'Δ%':>7s}")
    print("  " + "-"*108)
    summary = {}
    for tile, sub in df.groupby("tile"):
        rr = float(np.sqrt(np.mean(sub.err_raw ** 2)))
        rc = float(np.sqrt(np.mean(sub.err_corr ** 2)))
        mr = float(np.mean(np.abs(sub.err_raw)))
        mc = float(np.mean(np.abs(sub.err_corr)))
        br, bc = float(sub.err_raw.mean()), float(sub.err_corr.mean())
        d = 100 * (rc - rr) / rr
        improved = int((np.abs(sub.err_corr) < np.abs(sub.err_raw)).sum())
        d_sq = sub.err_raw ** 2 - sub.err_corr ** 2
        dm_t = float(d_sq.mean() / (d_sq.std() / np.sqrt(len(sub))))
        summary[tile] = {
            "regime": sub.regime.iloc[0],
            "n": int(len(sub)), "rmse_raw": rr, "rmse_corr": rc,
            "delta_pct": d, "mae_raw": mr, "mae_corr": mc,
            "bias_raw": br, "bias_corr": bc,
            "improved": improved, "frac_improved": improved / len(sub),
            "dm_t_stat": dm_t,
        }
        print(f"  {tile:<22s} {len(sub):>8d} {rr:>10.3f} {rc:>11.3f} "
              f"{mr:>10.3f} {mc:>11.3f} {br:>+10.3f} {bc:>+11.3f} {d:>+7.2f}")

    df.to_csv(out_dir / "samples_p6_with_mm_predictions.csv", index=False)
    (out_dir / "ood_p6_summary.json").write_text(json.dumps(summary, indent=2))

    # Hypothesis outcome per tile
    print(f"\n{'='*72}\nHYPOTHESIS OUTCOMES (favourable Δ<0; degradation Δ>0)\n{'='*72}")
    for tile, s in summary.items():
        d = s["delta_pct"]
        verdict = "✓ favourable" if d < 0 else "✗ degradation"
        print(f"  {tile}  {verdict}  Δ%={d:+.2f}  n={s['n']}  DM={s['dm_t_stat']:.2f}")


def main():
    print("="*72)
    print("P6 — three tropical-montane tiles densifying the relief × canopy probe")
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
    print(f"\n{'='*72}\n✅ P6 complete in {(time.time()-t0)/60:.1f} min\n{'='*72}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
