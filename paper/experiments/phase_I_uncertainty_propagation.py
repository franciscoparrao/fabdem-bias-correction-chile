#!/usr/bin/env python3
"""Phase I (medium) — propagate ATL08 h_te_uncertainty into the observed RMSE.

The ATL08 ATL terrain height `h_te` carries a 1-sigma uncertainty
`h_te_uncertainty` (median ~2.8 m, RMS ~4.3 m over Mediterranean Chile filtered
sample). Under independence, the observed residual variance decomposes as:

    Var(residual_observed) = Var(residual_intrinsic) + σ_ATL08²

where σ_ATL08² is the RMS of h_te_uncertainty across the sample. We compute
the RMS σ_ATL08 per tile and per regime; we then check whether subtracting
σ_ATL08² from RMSE_observed² produces a real (positive) quantity. If yes,
RMSE_intrinsic = sqrt(RMSE_observed² - σ_ATL08²) is the model-attributable
component. If σ_ATL08² > RMSE_observed², the residual variance is dominated
by ATL08 noise — the model has approached the noise floor of the ground
truth and the observed RMSE is itself an upper bound on the intrinsic error.

We also report inverse-variance-weighted RMSE as an alternative honest metric
(weights ∝ 1/h_te_uncertainty² downweight noisy footprints).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
SAMP = ROOT / "scale_p1" / "samples_unified"
P3A_PRED = ROOT / "scale_p3a" / "samples_unified" / "samples_p3a_with_mm_predictions.csv"
P4_PRED = ROOT / "scale_p4" / "samples_unified" / "samples_p4_with_mm_predictions.csv"
P5_PRED = ROOT / "scale_p5" / "samples_unified" / "samples_p5_with_mm_predictions.csv"
P6_PRED = ROOT / "scale_p6" / "samples_unified" / "samples_p6_with_mm_predictions.csv"
OUT = ROOT / "paper" / "experiments" / "phase_I_uncertainty_propagation.json"


def metrics_with_uncertainty(err_raw, err_corr, unc):
    """RMSE, MAE, σ_ATL08 RMS, intrinsic RMSE (if real) and inverse-var-weighted RMSE.

    err_raw : array of FABDEM_raw - h_te_orthometric
    err_corr : array of FABDEM_corrected - h_te_orthometric
    unc : array of h_te_uncertainty (1-σ per footprint, m)
    """
    n = len(err_raw)
    rmse_raw = float(np.sqrt(np.mean(err_raw ** 2)))
    rmse_corr = float(np.sqrt(np.mean(err_corr ** 2)))
    mae_raw = float(np.mean(np.abs(err_raw)))
    mae_corr = float(np.mean(np.abs(err_corr)))
    sigma_atl08_rms = float(np.sqrt(np.mean(unc ** 2)))
    sigma_atl08_med = float(np.median(unc))

    def intrinsic(rmse_obs, sigma):
        diff = rmse_obs ** 2 - sigma ** 2
        if diff > 0:
            return float(np.sqrt(diff)), True
        return float(np.sqrt(abs(diff))), False  # |·|, flag False

    raw_int, raw_real = intrinsic(rmse_raw, sigma_atl08_rms)
    corr_int, corr_real = intrinsic(rmse_corr, sigma_atl08_rms)

    # Inverse-variance weighted RMSE: w_i = 1/unc_i^2 (per-footprint precision)
    # WRMSE^2 = Σ w_i * err_i^2 / Σ w_i
    w = 1.0 / np.maximum(unc ** 2, 1e-6)
    wrmse_raw = float(np.sqrt(np.sum(w * err_raw ** 2) / np.sum(w)))
    wrmse_corr = float(np.sqrt(np.sum(w * err_corr ** 2) / np.sum(w)))

    return {
        "n": int(n),
        "rmse_observed_raw": rmse_raw,
        "rmse_observed_corr": rmse_corr,
        "mae_observed_raw": mae_raw,
        "mae_observed_corr": mae_corr,
        "sigma_atl08_rms": sigma_atl08_rms,
        "sigma_atl08_median": sigma_atl08_med,
        "rmse_intrinsic_raw": raw_int,
        "rmse_intrinsic_raw_is_real": bool(raw_real),
        "rmse_intrinsic_corr": corr_int,
        "rmse_intrinsic_corr_is_real": bool(corr_real),
        "wrmse_raw": wrmse_raw,
        "wrmse_corr": wrmse_corr,
        "rmse_obs_minus_wrmse_corr_pct": float(100 * (rmse_corr - wrmse_corr) / rmse_corr),
    }


def load_combined():
    """Combine OOF Mediterranean (mm_predictions.csv) with all OOD samples."""
    # Mediterranean OOF
    df_md = pd.read_csv(SAMP / "mm_predictions.csv")
    df_md["err_raw"] = df_md.fabdem - df_md.h_te_orthometric
    df_md["err_corr"] = df_md.fabdem_corrected - df_md.h_te_orthometric
    df_md["regime"] = "mediterranean"
    df_md["unc"] = df_md.h_te_unc

    # Humid temperate (from p3a, regime tag preserved in the file)
    df_p3a = pd.read_csv(P3A_PRED)
    ht = df_p3a[df_p3a.regime == "humid_temperate"].copy()
    ht["err_raw"] = ht.fabdem - ht.h_te_orthometric
    ht["err_corr"] = ht.fabdem + ht.pred_residual_mm_model - ht.h_te_orthometric
    ht["regime"] = "humid_temperate"
    ht["unc"] = ht.h_te_unc

    # P4 (Vietnam tropical_wet + Atacama hyperarid)
    df_p4 = pd.read_csv(P4_PRED)
    df_p4["unc"] = df_p4.h_te_unc

    # P5 (Cusco tropical_montane)
    df_p5 = pd.read_csv(P5_PRED)
    df_p5["regime"] = "tropical_montane"
    df_p5["tile"] = "S13W072"
    df_p5["unc"] = df_p5.h_te_unc

    # P6 (Colombia/Ecuador/Bolivia tropical_montane). Different regime strings
    df_p6 = pd.read_csv(P6_PRED)
    df_p6["regime"] = "tropical_montane"
    df_p6["unc"] = df_p6.h_te_unc

    cols = ["tile", "regime", "err_raw", "err_corr", "unc"]
    return pd.concat(
        [df_md[cols], ht[cols], df_p4[cols], df_p5[cols], df_p6[cols]],
        ignore_index=True,
    )


def main():
    df = load_combined()
    print(f"Total footprints across all OOD regimes: {len(df):,}")

    out = {"per_regime": {}, "per_tile": {}}

    # Per regime
    print("\n" + "=" * 80)
    print("PER REGIME — uncertainty-aware RMSE")
    print("=" * 80)
    print(f"{'regime':<22} {'n':>8} {'rmse_obs':>9} {'σ_ATL08':>9} {'intrinsic?':>11} {'wRMSE':>9}")
    print("-" * 80)
    for regime, sub in df.groupby("regime"):
        m = metrics_with_uncertainty(sub.err_raw.values, sub.err_corr.values, sub.unc.values)
        out["per_regime"][regime] = m
        intr_str = f"{m['rmse_intrinsic_corr']:.3f}" if m["rmse_intrinsic_corr_is_real"] else "—"
        print(f"{regime:<22} {m['n']:>8d} {m['rmse_observed_corr']:>9.3f} "
              f"{m['sigma_atl08_rms']:>9.3f} {intr_str:>11} {m['wrmse_corr']:>9.3f}")

    # Per tile
    print("\n" + "=" * 80)
    print("PER TILE — uncertainty-aware RMSE")
    print("=" * 80)
    print(f"{'tile':<10} {'n':>7} {'rmse_obs_corr':>13} {'σ_ATL08':>9} {'intrinsic_corr':>14} {'wRMSE_corr':>11}")
    print("-" * 80)
    for tile, sub in df.groupby("tile"):
        m = metrics_with_uncertainty(sub.err_raw.values, sub.err_corr.values, sub.unc.values)
        out["per_tile"][tile] = m
        intr_str = f"{m['rmse_intrinsic_corr']:.3f}" if m["rmse_intrinsic_corr_is_real"] else "noise-floor"
        print(f"{tile:<10} {m['n']:>7d} {m['rmse_observed_corr']:>13.3f} "
              f"{m['sigma_atl08_rms']:>9.3f} {intr_str:>14} {m['wrmse_corr']:>11.3f}")

    # Headline: how often does the model hit the ATL08 noise floor?
    noise_floor_tiles = [t for t, m in out["per_tile"].items()
                         if not m["rmse_intrinsic_corr_is_real"]]
    print(f"\nTiles where corrected DEM hits ATL08 noise floor "
          f"(σ_ATL08² > RMSE_observed²): {len(noise_floor_tiles)}/{len(out['per_tile'])}")
    print(f"  {noise_floor_tiles}")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
