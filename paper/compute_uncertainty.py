#!/usr/bin/env python3
"""Bootstrap CIs and Diebold-Mariano tests for paper claims (D3 fix).

Uses the appropriate prediction source per regime:
  - Mediterranean: spatial-CV out-of-fold predictions (mm_predictions.csv)
    → honest performance number
  - Humid temperate: M-M model applied without retraining
    (samples_p3a_with_mm_predictions.csv) → true OOD evaluation
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
P1 = ROOT / "scale_p1"
P3A = ROOT / "scale_p3a"

rng = np.random.default_rng(42)
N_BOOTSTRAP = 2000


def bootstrap_rmse(errors, n_boot=N_BOOTSTRAP):
    e = errors[np.isfinite(errors)]
    n = len(e)
    rmses = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        rmses[i] = np.sqrt(np.mean(e[idx] ** 2))
    return float(np.sqrt(np.mean(e**2))), float(np.percentile(rmses, 2.5)), float(np.percentile(rmses, 97.5))


def diebold_mariano(err_raw, err_corr):
    mask = np.isfinite(err_raw) & np.isfinite(err_corr)
    d = err_raw[mask]**2 - err_corr[mask]**2
    n = len(d)
    d_mean = d.mean()
    d_var = d.var(ddof=1)
    dm_stat = d_mean / np.sqrt(d_var / n)
    p = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p), int(n)


def sign_test(err_raw, err_corr):
    mask = np.isfinite(err_raw) & np.isfinite(err_corr)
    a = np.abs(err_raw[mask]); b = np.abs(err_corr[mask])
    n_better = int((a > b).sum()); n_total = int(len(a))
    res = stats.binomtest(n_better, n_total, p=0.5, alternative="greater")
    return n_better, n_total, 100 * n_better / n_total, float(res.pvalue)


# ============================================================================
# Mediterranean: OOF spatial CV predictions (honest)
# ============================================================================
print("=" * 72)
print("MEDITERRANEAN — Spatial-block CV OOF (honest performance)")
print("=" * 72)
mm = pd.read_csv(P1 / "samples_unified" / "mm_predictions.csv")
# In this file: residual_corrected = h_te_orthometric - fabdem (truth − raw)
# pred_residual = spatial CV OOF prediction
# We define error as model − truth, consistent with literature:
#   err_raw  = fabdem - h_te_orthometric
#   err_corr_oof = (fabdem + pred_residual) - h_te_orthometric
mm["err_raw"] = mm["fabdem"] - mm["h_te_orthometric"]
mm["err_corr_oof"] = (mm["fabdem"] + mm["pred_residual"]) - mm["h_te_orthometric"]
print(f"  n = {len(mm):,}")
rmse_r, lo_r, hi_r = bootstrap_rmse(mm.err_raw.values)
rmse_c, lo_c, hi_c = bootstrap_rmse(mm.err_corr_oof.values)
print(f"  RMSE raw      : {rmse_r:.3f} m  [95% CI {lo_r:.3f}, {hi_r:.3f}]")
print(f"  RMSE corr (OOF): {rmse_c:.3f} m  [95% CI {lo_c:.3f}, {hi_c:.3f}]")
print(f"  Improvement   : {rmse_r - rmse_c:.3f} m absolute  ({100*(rmse_r-rmse_c)/rmse_r:+.1f}%)")
dm, p_dm, n_dm = diebold_mariano(mm.err_raw.values, mm.err_corr_oof.values)
print(f"  Diebold-Mariano: DM = {dm:.2f}  p = {p_dm:.2e}")
n_better, n_total, pct, p_sign = sign_test(mm.err_raw.values, mm.err_corr_oof.values)
print(f"  Sign test: {n_better:,}/{n_total:,} ({pct:.1f}%) corrections improve; p = {p_sign:.2e}")

med_stats = {
    "source": "Mediterranean, spatial-CV OOF",
    "n": len(mm),
    "rmse_raw": {"point": rmse_r, "ci_lo": lo_r, "ci_hi": hi_r},
    "rmse_corr": {"point": rmse_c, "ci_lo": lo_c, "ci_hi": hi_c},
    "improvement_abs_m": rmse_r - rmse_c,
    "improvement_pct": 100 * (rmse_r - rmse_c) / rmse_r,
    "diebold_mariano": {"statistic": dm, "p_value": p_dm, "n": n_dm},
    "sign_test": {"n_better": n_better, "n_total": n_total, "pct_better": pct, "p_value": p_sign},
}

# ============================================================================
# Humid temperate: M-M model applied without retraining (true OOD)
# ============================================================================
print(f"\n" + "=" * 72)
print("HUMID TEMPERATE — M-M model applied without retraining (OOD)")
print("=" * 72)
ht_all = pd.read_csv(P3A / "samples_unified" / "samples_p3a_with_mm_predictions.csv")
ht = ht_all[ht_all.regime == "humid_temperate"].copy()
ht["err_raw"] = ht["fabdem"] - ht["h_te_orthometric"]
ht["err_corr"] = (ht["fabdem"] + ht["pred_residual_mm_model"]) - ht["h_te_orthometric"]
print(f"  n = {len(ht):,}")
rmse_r, lo_r, hi_r = bootstrap_rmse(ht.err_raw.values)
rmse_c, lo_c, hi_c = bootstrap_rmse(ht.err_corr.values)
print(f"  RMSE raw      : {rmse_r:.3f} m  [95% CI {lo_r:.3f}, {hi_r:.3f}]")
print(f"  RMSE corr (OOD): {rmse_c:.3f} m  [95% CI {lo_c:.3f}, {hi_c:.3f}]")
print(f"  Improvement   : {rmse_r - rmse_c:.3f} m absolute  ({100*(rmse_r-rmse_c)/rmse_r:+.1f}%)")
dm, p_dm, n_dm = diebold_mariano(ht.err_raw.values, ht.err_corr.values)
print(f"  Diebold-Mariano: DM = {dm:.2f}  p = {p_dm:.2e}")
n_better, n_total, pct, p_sign = sign_test(ht.err_raw.values, ht.err_corr.values)
print(f"  Sign test: {n_better:,}/{n_total:,} ({pct:.1f}%) corrections improve; p = {p_sign:.2e}")

ht_stats = {
    "source": "Humid temperate, M-M model applied (true OOD)",
    "n": len(ht),
    "rmse_raw": {"point": rmse_r, "ci_lo": lo_r, "ci_hi": hi_r},
    "rmse_corr": {"point": rmse_c, "ci_lo": lo_c, "ci_hi": hi_c},
    "improvement_abs_m": rmse_r - rmse_c,
    "improvement_pct": 100 * (rmse_r - rmse_c) / rmse_r,
    "diebold_mariano": {"statistic": dm, "p_value": p_dm, "n": n_dm},
    "sign_test": {"n_better": n_better, "n_total": n_total, "pct_better": pct, "p_value": p_sign},
}

# Save
out = ROOT / "paper" / "uncertainty_stats.json"
out.write_text(json.dumps({"mediterranean_oof": med_stats, "humid_temperate_ood": ht_stats},
                            indent=2))
print(f"\n→ {out}")
print("\n" + "=" * 72)
print("SUMMARY (numbers ready for paper)")
print("=" * 72)
print(f"Mediterranean (spatial-CV OOF, n={med_stats['n']:,}):")
print(f"  RMSE: {med_stats['rmse_raw']['point']:.3f} [{med_stats['rmse_raw']['ci_lo']:.2f}–{med_stats['rmse_raw']['ci_hi']:.2f}] "
      f"→ {med_stats['rmse_corr']['point']:.3f} [{med_stats['rmse_corr']['ci_lo']:.2f}–{med_stats['rmse_corr']['ci_hi']:.2f}] m, "
      f"−{med_stats['improvement_pct']:.1f}%")
print(f"  DM p < 0.001, sign test {med_stats['sign_test']['pct_better']:.1f}% better, p < 0.001")
print(f"Humid temperate (OOD, n={ht_stats['n']:,}):")
print(f"  RMSE: {ht_stats['rmse_raw']['point']:.3f} [{ht_stats['rmse_raw']['ci_lo']:.2f}–{ht_stats['rmse_raw']['ci_hi']:.2f}] "
      f"→ {ht_stats['rmse_corr']['point']:.3f} [{ht_stats['rmse_corr']['ci_lo']:.2f}–{ht_stats['rmse_corr']['ci_hi']:.2f}] m, "
      f"−{ht_stats['improvement_pct']:.1f}%")
print(f"  DM p < 0.001, sign test {ht_stats['sign_test']['pct_better']:.1f}% better, p < 0.001")
