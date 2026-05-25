#!/usr/bin/env python3
"""Fix F3, F5, F8 — apply issues found in H5 review."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))
from style_v2 import setup_style, COLORS_REGIME, COLOR_RAW
setup_style()

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
P1 = ROOT / "scale_p1"
P3A = ROOT / "scale_p3a"
INFER = P1 / "inference"
FIG = ROOT / "paper" / "figures"

COLOR_MD = COLORS_REGIME["mediterranean"]
COLOR_HT = COLORS_REGIME["humid_temperate"]
CITIES = [
    ("Curicó",     -71.24, -34.98),
    ("Talca",      -71.66, -35.43),
    ("Linares",    -71.60, -35.85),
    ("Chillán",    -72.10, -36.61),
]


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", format=ext)
    plt.close(fig)


# ============================================================================
# F3 FIX: sort mediterranean first (so annotations are correct)
# ============================================================================
def fix_f3():
    print("F3 fix: regime order + annotation placement")
    df = pd.read_csv(P3A / "samples_unified" / "samples_p3a_with_mm_predictions.csv")
    df["err_raw"] = df["fabdem"] - df["h_te_orthometric"]
    df["err_corr"] = (df["fabdem"] + df["pred_residual_mm_model"]) - df["h_te_orthometric"]
    agg = df.groupby(["tile", "regime"]).agg(
        n=("lon", "count"),
        rmse_raw=("err_raw", lambda s: float(np.sqrt(np.mean(s**2)))),
        rmse_corr=("err_corr", lambda s: float(np.sqrt(np.mean(s**2)))),
    ).reset_index()
    agg["improve"] = 100 * (agg.rmse_raw - agg.rmse_corr) / agg.rmse_raw
    # Sort: mediterranean FIRST, then humid_temperate, then by tile name
    agg["regime_rank"] = agg["regime"].map({"mediterranean": 0, "humid_temperate": 1})
    agg = agg.sort_values(["regime_rank", "tile"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(agg))
    w = 0.35
    colors_corr = [COLOR_MD if r == "mediterranean" else COLOR_HT for r in agg.regime]
    ax.bar(x - w/2, agg.rmse_raw, w, label="FABDEM raw", color=COLOR_RAW, edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, agg.rmse_corr, w, label="FABDEM + ML correction", color=colors_corr,
           edgecolor="black", linewidth=0.4)
    for i, row in agg.iterrows():
        ax.text(i, max(row.rmse_raw, row.rmse_corr) + 0.15,
                f"{row.improve:+.0f}%", ha="center", fontsize=8,
                color="darkgreen" if row.improve > 0 else "darkred", fontweight="bold")
        ax.text(i, -0.45, f"n={row.n:,}", ha="center", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(agg.tile, rotation=30, ha="right")
    ax.set_ylabel("RMSE (m)")
    # Title removed — caption in LaTeX
    ax.legend(loc="upper right")
    ax.set_ylim(-0.8, agg.rmse_raw.max() * 1.22)

    # Regime annotations CORRECTLY placed (now MD on left, HT on right after re-sort)
    md_xs = [i for i, r in enumerate(agg.regime) if r == "mediterranean"]
    ht_xs = [i for i, r in enumerate(agg.regime) if r == "humid_temperate"]
    if md_xs:
        ax.axvspan(min(md_xs)-0.5, max(md_xs)+0.5, alpha=0.06, color=COLOR_MD)
        ax.text((min(md_xs)+max(md_xs))/2, ax.get_ylim()[1]*0.93,
                "Mediterranean (training, in-sample)", ha="center",
                color=COLOR_MD, fontweight="bold", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_MD, alpha=0.95))
    if ht_xs:
        ax.axvspan(min(ht_xs)-0.5, max(ht_xs)+0.5, alpha=0.06, color=COLOR_HT)
        ax.text((min(ht_xs)+max(ht_xs))/2, ax.get_ylim()[1]*0.93,
                "Humid temperate (OOD, no retrain)", ha="center",
                color=COLOR_HT, fontweight="bold", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_HT, alpha=0.95))
    savefig(fig, "F3_per_tile_rmse")


# ============================================================================
# F5 FIX: better label placement to avoid overlap
# ============================================================================
def fix_f5():
    print("F5 fix: label offsets to reduce overlap")
    df = pd.read_csv(P3A / "samples_unified" / "samples_p3a_with_mm_predictions.csv")
    df["err_raw"] = df["fabdem"] - df["h_te_orthometric"]
    df["err_corr"] = (df["fabdem"] + df["pred_residual_mm_model"]) - df["h_te_orthometric"]
    agg = df.groupby(["tile", "regime"]).agg(
        n=("lon", "count"),
        rmse_raw=("err_raw", lambda s: float(np.sqrt(np.mean(s**2)))),
        rmse_corr=("err_corr", lambda s: float(np.sqrt(np.mean(s**2)))),
    ).reset_index()
    agg["improve"] = 100 * (agg.rmse_raw - agg.rmse_corr) / agg.rmse_raw

    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    # Custom per-tile label offsets to avoid overlap
    OFFSETS = {
        # tile : (dx_pts, dy_pts)
        "S35W071": (-15, 9),
        "S35W072": (10, 5),
        "S36W071": (10, -12),
        "S36W072": (10, 9),
        "S37W071": (-15, -12),
        "S37W072": (10, 9),
        "S38W072": (10, -12),
        "S38W073": (-50, 9),
        "S39W072": (10, 5),
        "S39W073": (10, -12),
    }

    for regime, marker, color, label in [
        ("mediterranean", "o", COLOR_MD, "Mediterranean (training, in-sample)"),
        ("humid_temperate", "s", COLOR_HT, "Humid temperate (OOD, no retrain)"),
    ]:
        sub = agg[agg.regime == regime]
        ax.scatter(sub.rmse_raw, sub.improve, s=np.sqrt(sub.n)*8, c=color, marker=marker,
                   label=label, alpha=0.8, edgecolor="black", linewidth=0.6)
        for _, r in sub.iterrows():
            dx, dy = OFFSETS.get(r.tile, (8, 4))
            ax.annotate(r.tile, (r.rmse_raw, r.improve), fontsize=8,
                        xytext=(dx, dy), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    ax.set_xlabel("FABDEM raw RMSE (m)")
    ax.set_ylabel("RMSE improvement from ML correction (%)")
    # Title removed — caption in LaTeX
    ax.legend(loc="lower right")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlim(1.8, 7.6)
    ax.set_ylim(-5, 68)
    # Annotation box top-left, clear of points
    ax.text(0.02, 0.97,
            "Single M-M-trained model evaluated on BOTH regimes\n"
            "without retraining — humid temperate is true OOD",
            transform=ax.transAxes, fontsize=9, va="top", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray"))
    savefig(fig, "F5_ood_generalization")


# ============================================================================
# F8 FIX: only show tiles inside residual extent + cleaner inset
# ============================================================================
def fix_f8():
    print("F8 fix: filter tiles to residual extent, fix inset")
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    src_path = INFER / "mm_residual.tif"
    with rasterio.open(src_path) as src:
        factor = 6
        h = src.height // factor
        w = src.width // factor
        arr = src.read(1, out_shape=(h, w), resampling=Resampling.average)
        bounds = src.bounds
    arr = np.where(np.isfinite(arr) & (arr > -100) & (arr < 100), arr, np.nan)

    # Only the 6 tiles inside residual extent (M-M)
    TILES_IN_EXTENT = [
        ("S35W072", (-72, -35, -71, -34)),
        ("S35W071", (-71, -35, -70, -34)),
        ("S36W072", (-72, -36, -71, -35)),
        ("S36W071", (-71, -36, -70, -35)),
        ("S37W072", (-72, -37, -71, -36)),
        ("S37W071", (-71, -37, -70, -36)),
    ]

    fig = plt.figure(figsize=(8.5, 11))
    gs = GridSpec(2, 1, height_ratios=[2.3, 1], hspace=0.22)
    ax_main = fig.add_subplot(gs[0])
    ax_inset = fig.add_subplot(gs[1])

    vmin, vmax = -8, 8
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    im = ax_main.imshow(arr, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                        extent=extent, interpolation="nearest", origin="upper")

    for tile, (lo, la, lo2, la2) in TILES_IN_EXTENT:
        ax_main.add_patch(Rectangle((lo, la), lo2-lo, la2-la,
                                     fill=False, edgecolor="black", linewidth=0.5,
                                     linestyle="--", alpha=0.6))
        ax_main.text(lo+0.5, la+0.5, tile, ha="center", va="center",
                     fontsize=7, color="black", alpha=0.75,
                     bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.4))

    for city, lon, lat in CITIES:
        if bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top:
            ax_main.plot(lon, lat, "o", color="black", markersize=4, markerfacecolor="yellow", zorder=5)
            ax_main.annotate(city, (lon, lat), xytext=(5, -2), textcoords="offset points",
                              fontsize=7,
                              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    ax_main.set_xlabel("Longitude (°)")
    ax_main.set_ylabel("Latitude (°)")
    # Panel label (a) bold top-left; title removed (caption in LaTeX)
    ax_main.text(0.02, 0.98, "(a)", transform=ax_main.transAxes,
                 fontweight="bold", fontsize=10, va="top",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax_main.set_xlim(bounds.left, bounds.right)
    ax_main.set_ylim(bounds.bottom, bounds.top)

    cbar = fig.colorbar(im, ax=ax_main, orientation="vertical", fraction=0.04, pad=0.02)
    cbar.set_label("Predicted residual (m)\n(negative = FABDEM overestimates)", fontsize=9)

    # Inset
    LIC_BBOX = (-72.0, -35.25, -71.69, -34.88)
    with rasterio.open(src_path) as src:
        wd = from_bounds(*LIC_BBOX, transform=src.transform).round_offsets().round_lengths()
        sub_arr = src.read(1, window=wd)
        sub_bounds = src.window_bounds(wd)
    sub_arr = np.where(np.isfinite(sub_arr) & (sub_arr > -100) & (sub_arr < 100), sub_arr, np.nan)
    im2 = ax_inset.imshow(sub_arr, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                          extent=(sub_bounds[0], sub_bounds[2], sub_bounds[1], sub_bounds[3]),
                          interpolation="nearest", origin="upper")
    ax_inset.plot(-72.07, -34.985, "*", color="gold", markersize=20,
                  markeredgecolor="black", markeredgewidth=1, zorder=6)
    ax_inset.annotate("Licantén", (-72.07, -34.985), xytext=(10, 8),
                      textcoords="offset points", fontsize=10, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.95))
    ax_inset.set_xlabel("Longitude (°)")
    ax_inset.set_ylabel("Latitude (°)")
    # Panel label (b); description goes to caption
    ax_inset.text(0.02, 0.98, "(b)", transform=ax_inset.transAxes,
                  fontweight="bold", fontsize=10, va="top",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    cbar2 = fig.colorbar(im2, ax=ax_inset, orientation="vertical", fraction=0.04, pad=0.02)
    cbar2.set_label("residual (m)", fontsize=8)

    ax_main.add_patch(Rectangle((LIC_BBOX[0], LIC_BBOX[1]),
                                  LIC_BBOX[2]-LIC_BBOX[0], LIC_BBOX[3]-LIC_BBOX[1],
                                  fill=False, edgecolor="gold", linewidth=1.8))
    savefig(fig, "F8_residual_map")


if __name__ == "__main__":
    fix_f3()
    fix_f5()
    fix_f8()
    print("✓ Fixes applied to F3, F5, F8")
