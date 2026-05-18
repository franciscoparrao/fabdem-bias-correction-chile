#!/usr/bin/env python3
"""Generate F1, F2, F8 — completing the 8 flagship figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
P1 = ROOT / "scale_p1"
P3A = ROOT / "scale_p3a"
INFER = P1 / "inference"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COLOR_MD = "#D7642E"
COLOR_HT = "#2E8B8B"
COLOR_RAW = "#888888"
COLOR_CORR = "#1F77B4"

# Tile inventory (name, bbox W,S,E,N, regime)
TILES = [
    ("S35W072", (-72, -35, -71, -34), "mediterranean"),
    ("S35W071", (-71, -35, -70, -34), "mediterranean"),
    ("S36W072", (-72, -36, -71, -35), "mediterranean"),
    ("S36W071", (-71, -36, -70, -35), "mediterranean"),
    ("S37W072", (-72, -37, -71, -36), "mediterranean"),
    ("S37W071", (-71, -37, -70, -36), "mediterranean"),
    ("S38W072", (-72, -38, -71, -37), "humid_temperate"),
    ("S38W073", (-73, -38, -72, -37), "humid_temperate"),
    ("S39W072", (-72, -39, -71, -38), "humid_temperate"),
    ("S39W073", (-73, -39, -72, -38), "humid_temperate"),
]

# Major rivers (approximate centerline coords for labeling)
RIVERS = [
    ("Río Mataquito",  -71.5,  -35.05),
    ("Río Maule",      -71.6,  -35.45),
    ("Río Itata",      -72.4,  -36.5),
    ("Río Biobío",     -72.5,  -36.85),
    ("Río Imperial",   -72.8,  -38.85),
]
# Major cities (lat/lon)
CITIES = [
    ("Curicó",     -71.24, -34.98),
    ("Talca",      -71.66, -35.43),
    ("Linares",    -71.60, -35.85),
    ("Chillán",    -72.10, -36.61),
    ("Concepción", -73.05, -36.83),
    ("Los Ángeles",-72.35, -37.47),
    ("Temuco",     -72.59, -38.74),
]


def savefig(fig, name):
    for ext in ("pdf", "png"):
        p = FIG / f"{name}.{ext}"
        fig.savefig(p, format=ext)
        print(f"  → {p}")
    plt.close(fig)


# ============================================================================
# F1: Study area map
# ============================================================================
def fig_study_area():
    print("F1: study area map")
    import geopandas as gpd
    print("  loading Chile boundary...")
    world = gpd.read_file("https://datahub.io/core/geo-countries/r/countries.geojson")
    chile = world[world.name == "Chile"]
    s_am = world[world.continent == "South America"] if "continent" in world.columns else world

    fig = plt.figure(figsize=(10, 9))
    gs = GridSpec(1, 2, width_ratios=[3, 1.6], wspace=0.03)
    ax = fig.add_subplot(gs[0])
    ax_inset = fig.add_subplot(gs[1])

    # Main panel: study area zoom
    bbox_study = (-74, -39.5, -69.5, -33.5)
    chile.plot(ax=ax, color="white", edgecolor="black", linewidth=0.8)
    ax.set_xlim(bbox_study[0], bbox_study[2])
    ax.set_ylim(bbox_study[1], bbox_study[3])
    ax.set_aspect("equal")

    # Tiles
    for tile, (w, s, e, n), regime in TILES:
        color = COLOR_MD if regime == "mediterranean" else COLOR_HT
        rect = Rectangle((w, s), e-w, n-s,
                          fill=True, facecolor=color, alpha=0.35,
                          edgecolor=color, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(w + 0.5, s + 0.5, tile, ha="center", va="center",
                fontsize=8, fontweight="bold",
                color="black",
                path_effects=[__import__("matplotlib.patheffects").patheffects.withStroke(linewidth=2, foreground="white")])

    # Rivers
    for name, lon, lat in RIVERS:
        ax.text(lon, lat, name, fontsize=8, color="navy", style="italic",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # Cities
    for city, lon, lat in CITIES:
        ax.plot(lon, lat, "o", color="black", markersize=5, markerfacecolor="white", zorder=5)
        ax.annotate(city, (lon, lat), xytext=(6, -2), textcoords="offset points", fontsize=8)

    # Regime annotations
    ax.text(-69.7, -35.5, "Mediterranean\nregime\n(training)",
            ha="right", va="center", color=COLOR_MD, fontsize=11,
            fontweight="bold", linespacing=1.0,
            bbox=dict(boxstyle="round", fc="white", ec=COLOR_MD, alpha=0.95))
    ax.text(-69.7, -38, "Humid temperate\nregime\n(OOD test)",
            ha="right", va="center", color=COLOR_HT, fontsize=11,
            fontweight="bold", linespacing=1.0,
            bbox=dict(boxstyle="round", fc="white", ec=COLOR_HT, alpha=0.95))

    # Grid and labels
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Study area: 10 FABDEM tiles, 2 climate regimes (central-south Chile)",
                  fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle=":")

    # North arrow + scale (rough)
    ax.annotate("N", xy=(-69.8, -33.8), xytext=(-69.8, -34.2),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                 ha="center", fontsize=10, fontweight="bold")
    # Scale: 1° lat ≈ 111 km. Show 100 km bar at lat -39
    bar_lat = -39.3
    km_per_deg_lat = 111
    bar_lon0 = -73.5
    bar_lon1 = bar_lon0 + 100 / km_per_deg_lat
    ax.plot([bar_lon0, bar_lon1], [bar_lat, bar_lat], "k-", lw=2)
    ax.text((bar_lon0+bar_lon1)/2, bar_lat-0.12, "100 km", ha="center", fontsize=8)

    # Inset: South America
    s_am.plot(ax=ax_inset, color="lightgrey", edgecolor="black", linewidth=0.5)
    chile.plot(ax=ax_inset, color="white", edgecolor="black", linewidth=0.5)
    # Highlight box
    aoi = Rectangle((bbox_study[0], bbox_study[1]),
                     bbox_study[2]-bbox_study[0], bbox_study[3]-bbox_study[1],
                     fill=False, edgecolor="red", linewidth=2)
    ax_inset.add_patch(aoi)
    ax_inset.set_xlim(-83, -33)
    ax_inset.set_ylim(-56, 14)
    ax_inset.set_aspect("equal")
    ax_inset.set_xticks([]); ax_inset.set_yticks([])
    ax_inset.set_title("South America", fontsize=9)

    savefig(fig, "F1_study_area")


# ============================================================================
# F2: Pipeline flowchart
# ============================================================================
def fig_pipeline_flowchart():
    print("F2: pipeline flowchart")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x, y, w, h, text, color="white", edge="black", fontsize=9, fontweight="normal"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                            facecolor=color, edgecolor=edge, linewidth=1.3)
        ax.add_patch(b)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, wrap=True)

    def arrow(x1, y1, x2, y2, color="dimgray"):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle="->", mutation_scale=18,
                              color=color, linewidth=1.4)
        ax.add_patch(a)

    # Data sources (left column)
    box(0.2, 7.7, 2.2, 0.9, "FABDEM v1.2\n(GEE asset)", color="#FFE0B2", fontsize=9, fontweight="bold")
    box(0.2, 6.5, 2.2, 0.9, "ICESat-2 ATL08 v7\n(earthaccess)", color="#FFE0B2", fontsize=9, fontweight="bold")
    box(0.2, 5.3, 2.2, 0.9, "Sentinel-1/2\n(Planetary Computer)", color="#FFE0B2", fontsize=9, fontweight="bold")
    ax.text(1.3, 8.85, "Data sources", ha="center", fontsize=10, fontweight="bold", color="darkgoldenrod")

    # Preprocessing
    box(3.5, 7.4, 2.6, 1.3,
        "Terrain features\n• SurtGis (Rust)\n• 17 terrain + 8 hydro", color="#B3E5FC", fontsize=8)
    box(3.5, 5.7, 2.6, 1.3,
        "Footprint extraction\n• Quality filters\n• EGM2008 correction", color="#B3E5FC", fontsize=8)
    box(3.5, 4.0, 2.6, 1.3,
        "Satellite indices\n• NDVI/NDWI/NDMI/BSI\n• Sentinel-1 VV/VH dB", color="#B3E5FC", fontsize=8)
    ax.text(4.8, 8.95, "Per-tile pipeline", ha="center", fontsize=10, fontweight="bold", color="steelblue")

    # Feature stack
    box(7.2, 5.6, 2.4, 1.6,
        "Feature stack\n33 features\n× 135,350 footprints", color="#C8E6C9", fontsize=9, fontweight="bold")

    # ML training
    box(10.4, 6.1, 3.3, 1.3,
        "XGBoost regression\n• Target: residual\n  (h_te_ortho − fabdem)\n• Optuna 100 trials TPE",
        color="#FFCDD2", fontsize=8)
    box(10.4, 4.3, 3.3, 1.3,
        "Spatial-block CV\n• 10-km blocks (n=570)\n• GroupKFold K=5",
        color="#FFCDD2", fontsize=8)
    ax.text(12.0, 7.75, "ML training", ha="center", fontsize=10, fontweight="bold", color="darkred")

    # Outputs
    box(10.4, 2.4, 3.3, 1.3,
        "Outputs\n• Corrected DEM COG\n• SHAP importance\n• Per-tile metrics",
        color="#E1BEE7", fontsize=8, fontweight="bold")

    # OOD test
    box(7.2, 2.4, 2.4, 1.3,
        "OOD test\nApply to humid\ntemperate w/o retrain", color="#FFF59D", fontsize=8, fontweight="bold")

    # Arrows: sources → preprocessing
    arrow(2.4, 8.15, 3.5, 8.05)
    arrow(2.4, 6.95, 3.5, 6.95)
    arrow(2.4, 5.75, 3.5, 4.65)

    # Preprocessing → feature stack
    arrow(6.1, 8.05, 7.2, 6.8)
    arrow(6.1, 6.35, 7.2, 6.6)
    arrow(6.1, 4.65, 7.2, 6.0)

    # Feature stack → training
    arrow(9.6, 6.7, 10.4, 6.7)
    arrow(9.6, 6.0, 10.4, 5.0)

    # Training → outputs
    arrow(12.0, 4.3, 12.0, 3.7)

    # Outputs → OOD test
    arrow(10.4, 2.9, 9.6, 3.0)

    fig.suptitle("FABDEM-ML pipeline: data → features → spatial-CV training → outputs + OOD test",
                  fontsize=11, fontweight="bold")
    savefig(fig, "F2_pipeline_flowchart")


# ============================================================================
# F8: Residual map (mm_residual.tif) + Licantén inset
# ============================================================================
def fig_residual_map():
    print("F8: residual map")
    import rasterio
    from rasterio.enums import Resampling

    src_path = INFER / "mm_residual.tif"
    with rasterio.open(src_path) as src:
        # Downsample for display (factor 6 → ~1237×1854)
        factor = 6
        h = src.height // factor
        w = src.width // factor
        arr = src.read(1, out_shape=(h, w), resampling=Resampling.average)
        bounds = src.bounds
    arr = np.where(np.isfinite(arr) & (arr > -100) & (arr < 100), arr, np.nan)
    print(f"  display array: {arr.shape}, range [{np.nanmin(arr):.2f}, {np.nanmax(arr):.2f}]")

    fig = plt.figure(figsize=(8.5, 11))
    gs = GridSpec(2, 1, height_ratios=[2.3, 1], hspace=0.18)
    ax_main = fig.add_subplot(gs[0])
    ax_inset = fig.add_subplot(gs[1])

    vmin, vmax = -8, 8  # cap colormap
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    im = ax_main.imshow(arr, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                          extent=extent, interpolation="nearest", origin="upper")

    # Tile boundaries
    for tile, (lo, la, lo2, la2), regime in TILES:
        rect = Rectangle((lo, la), lo2-lo, la2-la,
                          fill=False, edgecolor="black", linewidth=0.5, linestyle="--", alpha=0.6)
        ax_main.add_patch(rect)
        ax_main.text(lo+0.5, la+0.5, tile, ha="center", va="center",
                       fontsize=7, color="black", alpha=0.8)

    # Cities
    for city, lon, lat in CITIES:
        ax_main.plot(lon, lat, "o", color="black", markersize=3, markerfacecolor="yellow", zorder=5)
        ax_main.annotate(city, (lon, lat), xytext=(5, -2), textcoords="offset points", fontsize=7,
                          bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    ax_main.set_xlabel("Longitude (°)")
    ax_main.set_ylabel("Latitude (°)")
    ax_main.set_title("Spatial pattern of predicted residual (correction layer)\n"
                       "mm_residual.tif: ML-predicted FABDEM bias [m]",
                       fontsize=10, fontweight="bold")
    ax_main.set_xlim(bounds.left, bounds.right)
    ax_main.set_ylim(bounds.bottom, bounds.top)

    cbar = fig.colorbar(im, ax=ax_main, orientation="vertical", fraction=0.04, pad=0.02)
    cbar.set_label("Predicted residual (m)\n(positive = correct downward)", fontsize=9)

    # Inset: Licantén zoom (where CIGIDEN reference exists)
    LIC_BBOX = (-72.0, -35.25, -71.69, -34.88)
    # Re-read at higher resolution within this window
    with rasterio.open(src_path) as src:
        from rasterio.windows import from_bounds
        wd = from_bounds(*LIC_BBOX, transform=src.transform).round_offsets().round_lengths()
        sub_arr = src.read(1, window=wd)
        sub_bounds = src.window_bounds(wd)
    sub_arr = np.where(np.isfinite(sub_arr) & (sub_arr > -100) & (sub_arr < 100), sub_arr, np.nan)
    im2 = ax_inset.imshow(sub_arr, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                            extent=(sub_bounds[0], sub_bounds[2], sub_bounds[1], sub_bounds[3]),
                            interpolation="nearest", origin="upper")
    # Mark Licantén
    ax_inset.plot(-72.07, -34.985, "*", color="gold", markersize=18,
                    markeredgecolor="black", markeredgewidth=1, zorder=6)
    ax_inset.annotate("Licantén", (-72.07, -34.985), xytext=(8, 6),
                        textcoords="offset points", fontsize=10, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.95))
    ax_inset.set_xlabel("Longitude (°)")
    ax_inset.set_ylabel("Latitude (°)")
    ax_inset.set_title("Inset: Licantén area (Mataquito mouth, jun 2023 flood reference)",
                         fontsize=9, fontweight="bold")
    cbar2 = fig.colorbar(im2, ax=ax_inset, orientation="vertical", fraction=0.04, pad=0.02)
    cbar2.set_label("residual (m)", fontsize=8)

    # Box in main ax showing inset region
    inset_rect = Rectangle((LIC_BBOX[0], LIC_BBOX[1]),
                             LIC_BBOX[2]-LIC_BBOX[0], LIC_BBOX[3]-LIC_BBOX[1],
                             fill=False, edgecolor="gold", linewidth=1.8)
    ax_main.add_patch(inset_rect)

    savefig(fig, "F8_residual_map")


if __name__ == "__main__":
    print(f"Generating F1, F2, F8 into {FIG}")
    fig_pipeline_flowchart()  # cheapest, do first
    fig_study_area()
    fig_residual_map()
    print(f"\n✅ Done. Total figures in {FIG}:")
    for f in sorted(FIG.glob("*.pdf")):
        print(f"  - {f.name}")
