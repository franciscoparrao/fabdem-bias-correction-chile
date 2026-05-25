#!/usr/bin/env python3
"""Generate flagship figures for the paper.

Produces PDF + PNG for each figure at publication-grade quality.
Single command: run this script to (re-)generate all figures.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Publication style — load shared rcParams from style_v2 (Helvetica/Arial,
# spines bottom+left only, ticks `in`, frameon=False, pdf.fonttype=42, etc.)
sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))
from style_v2 import setup_style, COLORS_REGIME, COLOR_RAW, COLOR_CORR
setup_style()

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
P1 = ROOT / "scale_p1"
P3A = ROOT / "scale_p3a"
P2V = ROOT / "p2_validation"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COLOR_MD = COLORS_REGIME["mediterranean"]
COLOR_HT = COLORS_REGIME["humid_temperate"]


def savefig(fig, name):
    for ext in ("pdf", "png"):
        p = FIG / f"{name}.{ext}"
        fig.savefig(p, format=ext)
        print(f"  → {p}")
    plt.close(fig)


# ============================================================================
# F3: Per-tile RMSE bar chart
# ============================================================================
def fig_per_tile():
    print("F3: per-tile RMSE bar chart")
    # Load M-M model applied to all tiles
    df = pd.read_csv(P3A / "samples_unified" / "samples_p3a_with_mm_predictions.csv")
    df["err_raw"] = df["fabdem"] - df["h_te_orthometric"]
    df["err_corr"] = (df["fabdem"] + df["pred_residual_mm_model"]) - df["h_te_orthometric"]
    agg = df.groupby(["tile", "regime"]).agg(
        n=("lon", "count"),
        rmse_raw=("err_raw", lambda s: float(np.sqrt(np.mean(s**2)))),
        rmse_corr=("err_corr", lambda s: float(np.sqrt(np.mean(s**2)))),
    ).reset_index()
    agg["improve"] = 100 * (agg.rmse_raw - agg.rmse_corr) / agg.rmse_raw
    agg = agg.sort_values(["regime", "tile"])

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(agg))
    w = 0.35

    colors_corr = [COLOR_MD if r == "mediterranean" else COLOR_HT for r in agg.regime]
    ax.bar(x - w/2, agg.rmse_raw, w, label="FABDEM raw", color=COLOR_RAW, edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, agg.rmse_corr, w, label="FABDEM + ML correction", color=colors_corr, edgecolor="black", linewidth=0.4)
    for i, row in agg.reset_index(drop=True).iterrows():
        ax.text(i, max(row.rmse_raw, row.rmse_corr) + 0.15,
                f"{row.improve:+.0f}%", ha="center", fontsize=8,
                color="darkgreen" if row.improve > 0 else "darkred", fontweight="bold")
        ax.text(i, -0.4, f"n={row.n:,}", ha="center", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(agg.tile, rotation=30, ha="right")
    ax.set_ylabel("RMSE (m)")
    # Title removed — caption in LaTeX
    ax.legend(loc="upper left")
    ax.set_ylim(0, agg.rmse_raw.max() * 1.2)
    # Regime annotation
    md_xs = [i for i, r in enumerate(agg.regime) if r == "mediterranean"]
    ht_xs = [i for i, r in enumerate(agg.regime) if r == "humid_temperate"]
    if md_xs and ht_xs:
        ax.axvspan(min(md_xs)-0.5, max(md_xs)+0.5, alpha=0.06, color=COLOR_MD, ymin=0, ymax=1)
        ax.axvspan(min(ht_xs)-0.5, max(ht_xs)+0.5, alpha=0.06, color=COLOR_HT, ymin=0, ymax=1)
        ax.text((min(md_xs)+max(md_xs))/2, ax.get_ylim()[1]*0.93,
                "Mediterranean (training)", ha="center", color=COLOR_MD, fontweight="bold", fontsize=10)
        ax.text((min(ht_xs)+max(ht_xs))/2, ax.get_ylim()[1]*0.93,
                "Humid temperate (OOD)", ha="center", color=COLOR_HT, fontweight="bold", fontsize=10)
    savefig(fig, "F3_per_tile_rmse")


# ============================================================================
# F4: Stratification 4-panel (NDVI / HAND / slope / elevation × regime)
# ============================================================================
def fig_stratification():
    print("F4: stratification 4-panel")
    # Use OOD analysis comparison files
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    # NDVI panel (already have side-by-side)
    df_ndvi = pd.read_csv(P3A / "ood_analysis" / "compare_ndvi_by_regime.csv", index_col=0)
    df_hand = pd.read_csv(P3A / "ood_analysis" / "compare_hand_by_regime.csv", index_col=0)
    # For elev + slope: build from individual md/ht
    def load_by(name):
        m = pd.read_csv(P3A / "ood_analysis" / f"md_by_{name}.csv", index_col=0)
        h = pd.read_csv(P3A / "ood_analysis" / f"ht_by_{name}.csv", index_col=0)
        return pd.DataFrame({"mediterranean": m["improve_pct"], "humid_temperate": h["improve_pct"],
                              "n_md": m["n"], "n_ht": h["n"]})
    df_elev = load_by("elevation")
    df_slope = load_by("slope")

    panels = [
        ("NDVI band", df_ndvi, axes[0,0]),
        ("HAND band (m above drainage)", df_hand, axes[0,1]),
        ("Elevation band", df_elev, axes[1,0]),
        ("Slope class", df_slope, axes[1,1]),
    ]
    for title, df, ax in panels:
        n = len(df)
        x = np.arange(n)
        w = 0.4
        ax.bar(x - w/2, df.mediterranean, w, label="Mediterranean", color=COLOR_MD, edgecolor="black", linewidth=0.4)
        ax.bar(x + w/2, df.humid_temperate, w, label="Humid temperate (OOD)", color=COLOR_HT, edgecolor="black", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("RMSE improvement (%)")
        # Panel label (top-left bold sans-serif, distinct from caption)
        ax.text(0.02, 0.96, title, transform=ax.transAxes,
                fontweight="bold", fontsize=9, va="top")
        if title.startswith("NDVI"):
            ax.legend(loc="upper left", fontsize=7)
        # Annotate sample sizes
        for i, (idx, row) in enumerate(df.iterrows()):
            try:
                n_text = f"{int(row.n_md):,} / {int(row.n_ht):,}"
            except KeyError:
                n_text = ""
            ax.text(i, ax.get_ylim()[0] + 1, n_text, ha="center", fontsize=6, color="gray")
    # Suptitle removed — caption in LaTeX
    fig.tight_layout()
    savefig(fig, "F4_stratification_4panel")


# ============================================================================
# F5: OOD generalization scatter
# ============================================================================
def fig_ood_scatter():
    print("F5: OOD generalization scatter")
    df = pd.read_csv(P3A / "samples_unified" / "samples_p3a_with_mm_predictions.csv")
    df["err_raw"] = df["fabdem"] - df["h_te_orthometric"]
    df["err_corr"] = (df["fabdem"] + df["pred_residual_mm_model"]) - df["h_te_orthometric"]
    agg = df.groupby(["tile", "regime"]).agg(
        n=("lon", "count"),
        rmse_raw=("err_raw", lambda s: float(np.sqrt(np.mean(s**2)))),
        rmse_corr=("err_corr", lambda s: float(np.sqrt(np.mean(s**2)))),
    ).reset_index()
    agg["improve"] = 100 * (agg.rmse_raw - agg.rmse_corr) / agg.rmse_raw

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for regime, marker, color, label in [
        ("mediterranean", "o", COLOR_MD, "Mediterranean (training, in-sample)"),
        ("humid_temperate", "s", COLOR_HT, "Humid temperate (OOD, no retrain)"),
    ]:
        sub = agg[agg.regime == regime]
        ax.scatter(sub.rmse_raw, sub.improve, s=np.sqrt(sub.n)*8, c=color, marker=marker,
                    label=label, alpha=0.8, edgecolor="black", linewidth=0.6)
        for _, r in sub.iterrows():
            ax.annotate(r.tile, (r.rmse_raw, r.improve), fontsize=8,
                         xytext=(7, 4), textcoords="offset points")
    ax.set_xlabel("FABDEM raw RMSE (m)")
    ax.set_ylabel("RMSE improvement from ML correction (%)")
    ax.set_title("Per-tile improvement vs baseline RMSE\n(marker area ∝ √n footprints)")
    ax.legend(loc="lower right")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.text(0.02, 0.96, "M-M-trained model applied to BOTH regimes without retraining",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray"))
    savefig(fig, "F5_ood_generalization")


# ============================================================================
# F6: SHAP regime comparison
# ============================================================================
def fig_shap_comparison():
    print("F6: SHAP comparison")
    df = pd.read_csv(P3A / "ood_analysis" / "shap_regime_comparison.csv")
    df = df.sort_values("shap_humid_temperate", ascending=True).tail(15)  # top 15 from bottom for hbar

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(df))
    w = 0.4
    ax.barh(y - w/2, df.shap_mediterranean, w, label="Mediterranean (training)", color=COLOR_MD, edgecolor="black", linewidth=0.4)
    ax.barh(y + w/2, df.shap_humid_temperate, w, label="Humid temperate (OOD)", color=COLOR_HT, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(df.feature)
    ax.set_xlabel(r"Mean $|\mathrm{SHAP}|$ value (m)")
    # Title removed — caption in LaTeX
    ax.legend(loc="lower right", fontsize=8)
    # Annotate rank changes
    for i, (_, r) in enumerate(df.iterrows()):
        shift = int(r.rank_shift)
        if abs(shift) >= 2:
            color = "green" if shift > 0 else "red"
            ax.text(max(r.shap_mediterranean, r.shap_humid_temperate) + 0.005,
                    i, f"rank {int(r.rank_md)}→{int(r.rank_ht)}",
                    fontsize=8, va="center", color=color, fontweight="bold")
    savefig(fig, "F6_shap_regime_comparison")


# ============================================================================
# F7: Bias histogram before/after, per regime
# ============================================================================
def fig_bias_histogram():
    print("F7: bias histogram")
    df = pd.read_csv(P3A / "samples_unified" / "samples_p3a_with_mm_predictions.csv")
    df["err_raw"] = df["fabdem"] - df["h_te_orthometric"]
    df["err_corr"] = (df["fabdem"] + df["pred_residual_mm_model"]) - df["h_te_orthometric"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    bins = np.linspace(-12, 12, 80)
    for ax, regime, color in [
        (axes[0], "mediterranean", COLOR_MD),
        (axes[1], "humid_temperate", COLOR_HT),
    ]:
        sub = df[df.regime == regime]
        n = len(sub)
        m_raw = sub.err_raw.mean()
        s_raw = sub.err_raw.std()
        m_corr = sub.err_corr.mean()
        s_corr = sub.err_corr.std()
        ax.hist(sub.err_raw, bins=bins, alpha=0.5, label=f"Raw (μ={m_raw:+.2f}, σ={s_raw:.2f})",
                color=COLOR_RAW, edgecolor="black", linewidth=0.3, density=True)
        ax.hist(sub.err_corr, bins=bins, alpha=0.6, label=f"Corrected (μ={m_corr:+.2f}, σ={s_corr:.2f})",
                color=color, edgecolor="black", linewidth=0.3, density=True)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.axvline(m_raw, color="dimgray", linewidth=1, linestyle="--", alpha=0.7)
        ax.axvline(m_corr, color=color, linewidth=1, linestyle="--", alpha=0.9)
        panel_label = ("(a) Mediterranean" if regime == "mediterranean"
                       else "(b) Humid temperate (OOD)") + f" — n={n:,}"
        ax.text(0.02, 0.96, panel_label, transform=ax.transAxes,
                fontweight="bold", fontsize=9, va="top")
        ax.set_xlabel("FABDEM error (FABDEM − ICESat-2 orthometric) (m)")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_xlim(-12, 12)
    axes[0].set_ylabel("Density")
    # Suptitle removed — caption in LaTeX
    fig.tight_layout()
    savefig(fig, "F7_bias_histogram")


# ============================================================================
# Run all
# ============================================================================
if __name__ == "__main__":
    print(f"Generating figures into {FIG}")
    fig_per_tile()
    fig_stratification()
    fig_ood_scatter()
    fig_shap_comparison()
    fig_bias_histogram()
    print(f"\n✅ {len(list(FIG.glob('*.pdf')))} PDF + {len(list(FIG.glob('*.png')))} PNG figures in {FIG}")
