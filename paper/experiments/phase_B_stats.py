#!/usr/bin/env python3
"""Phase B — statistical rigour fixes from paper-review dry-run.

(1) Block bootstrap: resample the 570 10-km spatial blocks (not the 77,501
    footprints) to compute honest CIs on RMSE and the DM differential.
(2) Empirical variogram: compute on per-footprint OOF residuals to justify
    the 10-km block size choice quantitatively.
(3) Seed jitter: retrain XGBoost with 5 different random seeds to quantify
    model-training uncertainty (orthogonal to footprint-resampling uncertainty).

Outputs:
  - phase_B_results.json
  - phase_B_variogram.{pdf,png}

This run is heavy: seed jitter alone is ~5 retrains * ~10 min ≈ 50 min.
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
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
SAMP = ROOT / "scale_p1" / "samples_unified"
OUT = ROOT / "paper" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

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
    "n_jobs": 4,
}


def load_oof():
    """Load existing OOF predictions (the headline result)."""
    df = pd.read_csv(SAMP / "mm_predictions.csv")
    df["err_raw"] = df.fabdem - df.h_te_orthometric
    df["err_corr"] = df.fabdem_corrected - df.h_te_orthometric
    # Add 10-km block label
    t = pyproj.Transformer.from_crs(4326, 32719, always_xy=True)
    x_utm, y_utm = t.transform(df.lon.values, df.lat.values)
    df["block_id"] = (x_utm // 10000).astype(int) * 100000 + (y_utm // 10000).astype(int)
    df["x_utm"] = x_utm
    df["y_utm"] = y_utm
    print(f"Loaded {len(df)} OOF predictions, {df.block_id.nunique()} unique 10-km blocks")
    return df


# ============================================================
# (1) BLOCK BOOTSTRAP
# ============================================================
def block_bootstrap(df, n_boot=2000, seed=42):
    """Resample blocks (with replacement), recompute RMSE_raw and RMSE_corr.

    Returns CIs honest to the autocorrelation structure.
    """
    print(f"\n[BLOCK BOOTSTRAP] n_boot={n_boot}, sampling 570 blocks with replacement")
    blocks = df.block_id.values
    err_raw = df.err_raw.values
    err_corr = df.err_corr.values

    unique_blocks = np.unique(blocks)
    n_blocks = len(unique_blocks)
    block_to_idx = {b: np.where(blocks == b)[0] for b in unique_blocks}

    rng = np.random.default_rng(seed)
    rmse_raw_b = np.zeros(n_boot)
    rmse_corr_b = np.zeros(n_boot)
    dm_b = np.zeros(n_boot)

    t0 = time.time()
    for i in range(n_boot):
        sampled = rng.choice(unique_blocks, size=n_blocks, replace=True)
        idx_list = [block_to_idx[b] for b in sampled]
        idx = np.concatenate(idx_list)
        er, ec = err_raw[idx], err_corr[idx]
        rmse_raw_b[i] = np.sqrt(np.mean(er ** 2))
        rmse_corr_b[i] = np.sqrt(np.mean(ec ** 2))
        d_sq = er ** 2 - ec ** 2
        dm_b[i] = d_sq.mean() / (d_sq.std() / np.sqrt(len(idx)))
    print(f"  elapsed {time.time()-t0:.1f}s")

    out = {
        "n_boot": n_boot, "n_blocks": int(n_blocks),
        "rmse_raw": {
            "median": float(np.median(rmse_raw_b)),
            "ci95_lo": float(np.percentile(rmse_raw_b, 2.5)),
            "ci95_hi": float(np.percentile(rmse_raw_b, 97.5)),
            "mean": float(rmse_raw_b.mean()),
        },
        "rmse_corr": {
            "median": float(np.median(rmse_corr_b)),
            "ci95_lo": float(np.percentile(rmse_corr_b, 2.5)),
            "ci95_hi": float(np.percentile(rmse_corr_b, 97.5)),
            "mean": float(rmse_corr_b.mean()),
        },
        "rmse_reduction_pct": {
            "median": float(np.median(100 * (rmse_corr_b - rmse_raw_b) / rmse_raw_b)),
            "ci95_lo": float(np.percentile(100 * (rmse_corr_b - rmse_raw_b) / rmse_raw_b, 2.5)),
            "ci95_hi": float(np.percentile(100 * (rmse_corr_b - rmse_raw_b) / rmse_raw_b, 97.5)),
        },
        "dm_stat": {
            "median": float(np.median(dm_b)),
            "ci95_lo": float(np.percentile(dm_b, 2.5)),
            "ci95_hi": float(np.percentile(dm_b, 97.5)),
            "frac_positive": float((dm_b > 0).mean()),
        },
    }
    print(f"  RMSE_raw  block-bootstrap CI: [{out['rmse_raw']['ci95_lo']:.3f}, "
          f"{out['rmse_raw']['ci95_hi']:.3f}]  (footprint-boot was [2.95, 3.18])")
    print(f"  RMSE_corr block-bootstrap CI: [{out['rmse_corr']['ci95_lo']:.3f}, "
          f"{out['rmse_corr']['ci95_hi']:.3f}]  (footprint-boot was [2.37, 2.62])")
    print(f"  Δ%      block-bootstrap CI: [{out['rmse_reduction_pct']['ci95_lo']:+.2f}%, "
          f"{out['rmse_reduction_pct']['ci95_hi']:+.2f}%]  (point estimate -18.7%)")
    print(f"  DM stat block-bootstrap CI: [{out['dm_stat']['ci95_lo']:.1f}, "
          f"{out['dm_stat']['ci95_hi']:.1f}]  (positive in {out['dm_stat']['frac_positive']*100:.1f}% of resamples)")
    return out


# ============================================================
# (2) EMPIRICAL VARIOGRAM
# ============================================================
def empirical_variogram(df, n_pairs=500000, n_bins=30, max_dist_m=50000, seed=42):
    """Compute empirical variogram on OOF squared residuals.

    Sample pairs randomly to keep memory bounded.
    Returns (bin_centres_m, semivar, n_per_bin).
    """
    print(f"\n[VARIOGRAM] sampling {n_pairs:,} random pairs of OOF residuals")
    x = df.x_utm.values
    y = df.y_utm.values
    z = df.err_corr.values
    n = len(df)

    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]

    dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
    sqdiff = 0.5 * (z[i] - z[j]) ** 2

    keep = dist <= max_dist_m
    dist, sqdiff = dist[keep], sqdiff[keep]
    print(f"  retained {len(dist):,} pairs within {max_dist_m/1000:.0f} km")

    bin_edges = np.linspace(0, max_dist_m, n_bins + 1)
    bin_idx = np.digitize(dist, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    semivar = np.zeros(n_bins)
    n_per_bin = np.zeros(n_bins, dtype=int)
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for b in range(n_bins):
        sel = bin_idx == b
        n_per_bin[b] = int(sel.sum())
        if n_per_bin[b] > 0:
            semivar[b] = float(sqdiff[sel].mean())

    print(f"  variogram computed across {n_bins} bins, lag 0-{max_dist_m/1000:.0f} km")
    return {
        "bin_centres_m": centres.tolist(),
        "semivariance": semivar.tolist(),
        "n_per_bin": n_per_bin.tolist(),
        "max_dist_m": max_dist_m,
        "n_pairs_used": int(len(dist)),
    }


def plot_variogram(variogram, out_path):
    centres = np.array(variogram["bin_centres_m"]) / 1000  # to km
    semivar = np.array(variogram["semivariance"])
    n_bin = np.array(variogram["n_per_bin"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(centres, semivar, "o-", color="#1F77B4", markersize=6,
            linewidth=1.5, label="Empirical $\\gamma(h)$")
    ax.axvline(10, color="orange", linestyle="--", alpha=0.7,
               label="10 km block size (primary)")
    ax.axvline(5, color="gray", linestyle=":", alpha=0.5, label="5 km")
    ax.axvline(20, color="gray", linestyle=":", alpha=0.5, label="20 km")
    ax.set_xlabel("Lag distance $h$ (km)")
    ax.set_ylabel("Semivariance $\\gamma(h)$ of OOF residuals (m$^2$)")
    ax.set_title("Empirical variogram on Mediterranean OOF residuals\n"
                 f"({variogram['n_pairs_used']:,} sampled pairs, "
                 f"lag 0–{variogram['max_dist_m']/1000:.0f} km)")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(centres) * 1.02)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = out_path.with_suffix(f".{ext}")
        fig.savefig(p, format=ext, dpi=300, bbox_inches="tight")
        print(f"  → {p}")
    plt.close(fig)


# ============================================================
# (3) SEED JITTER
# ============================================================
def seed_jitter(seeds=(7, 13, 42, 100, 2026)):
    """Re-train XGBoost with multiple seeds, report distribution of OOF RMSE.

    Note: spatial-CV splits are computed from coordinates (deterministic),
    so the source of variation is XGBoost internal sampling (subsample,
    colsample) and tree initialisation, which use the `random_state` seed.
    """
    print(f"\n[SEED JITTER] training XGBoost with {len(seeds)} different seeds")
    df = pd.read_csv(SAMP / "samples_mm_full.csv")
    if "stream_network" in df.columns:
        df["stream_network"] = df["stream_network"].fillna(0)
    META = {"granule", "beam", "lon", "lat", "h_te", "h_te_unc",
            "night", "terrain_flg", "n_te_phot", "snow", "water", "cloud",
            "h_te_orthometric", "geoid_N", "residual_corrected",
            "fabdem", "filled", "tile"}
    FEATURES = [c for c in df.columns if c not in META and df[c].dtype != "O"]
    y = df.residual_corrected.values
    X = df[FEATURES].values
    keep = np.isfinite(y)
    X, y = X[keep], y[keep]
    lon = df.lon.values[keep]
    lat = df.lat.values[keep]
    t = pyproj.Transformer.from_crs(4326, 32719, always_xy=True)
    x_utm, y_utm = t.transform(lon, lat)
    blocks = (x_utm // 10000).astype(int) * 100000 + (y_utm // 10000).astype(int)
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups=blocks))

    rmse_base = float(np.sqrt(np.mean(y ** 2)))
    results = {"baseline_rmse": rmse_base, "seeds": {}}
    for seed in seeds:
        t0 = time.time()
        preds = np.full_like(y, np.nan, dtype=np.float64)
        for tr, te in splits:
            m = xgb.XGBRegressor(**{**BEST_PARAMS, "random_state": seed},
                                 early_stopping_rounds=30)
            m.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False)
            preds[te] = m.predict(X[te])
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = float(mean_absolute_error(y, preds))
        dt = time.time() - t0
        results["seeds"][seed] = {"rmse": rmse, "mae": mae, "elapsed_s": dt}
        print(f"  seed={seed:5d}  RMSE={rmse:.4f}  MAE={mae:.4f}  ({dt:.1f}s)")

    rmses = [r["rmse"] for r in results["seeds"].values()]
    results["summary"] = {
        "n_seeds": len(seeds),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses, ddof=1)),
        "rmse_min": float(np.min(rmses)),
        "rmse_max": float(np.max(rmses)),
    }
    print(f"  Across {len(seeds)} seeds: RMSE = {results['summary']['rmse_mean']:.4f} "
          f"± {results['summary']['rmse_std']:.4f} "
          f"(range {results['summary']['rmse_min']:.4f} – {results['summary']['rmse_max']:.4f})")
    return results


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("Phase B — Statistical rigour fixes")
    print("=" * 70)

    df = load_oof()

    # (1) Block bootstrap (~1 min)
    bb = block_bootstrap(df, n_boot=2000)

    # (2) Variogram (~30 s)
    vg = empirical_variogram(df, n_pairs=500_000, n_bins=30, max_dist_m=50_000)
    plot_variogram(vg, OUT / "phase_B_variogram")

    # (3) Seed jitter (~50 min)
    sj = seed_jitter(seeds=(7, 13, 42, 100, 2026))

    results = {
        "block_bootstrap": bb,
        "variogram": vg,
        "seed_jitter": sj,
        "wall_time_s": time.time() - t0,
    }
    out_path = OUT / "phase_B_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n→ {out_path}")
    print(f"Total wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
