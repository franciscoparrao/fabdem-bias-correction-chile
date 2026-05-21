#!/usr/bin/env python3
"""Regenerate F5 with the four-regime OOD evidence.

Replaces the original M-M (in-sample, training) + HT (OOD Chile) scatter with
a four-regime scatter that adds the non-Chilean OOD tiles (Vietnam Mekong
Delta, Atacama). Mediterranean tiles now use spatial-CV OOF predictions for
honest comparison (no in-sample leakage in the visual either).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    "axes.grid": True,
    "grid.alpha": 0.3,
})

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
FIG = ROOT / "paper" / "figures"
ISPRS_FIG = ROOT / "paper" / "isprs" / "figures"

COLORS = {
    "mediterranean":   "#D7642E",  # warm orange — training
    "humid_temperate": "#2E8B8B",  # teal — OOD Chile (favourable)
    "tropical_wet":    "#7BA05B",  # sage green — OOD Vietnam (boundary, marginal degrade)
    "hyperarid":       "#C9956B",  # tan/sand — OOD Atacama (boundary, catastrophic)
    "tropical_montane": "#7B3F7B", # purple — OOD Cusco (confirmatory favourable)
}
MARKERS = {
    "mediterranean":   "o",
    "humid_temperate": "s",
    "tropical_wet":    "D",
    "hyperarid":       "^",
    "tropical_montane": "P",
}
LABELS = {
    "mediterranean":   "Mediterranean (training, spatial-CV OOF)",
    "humid_temperate": "Humid temperate Chile (OOD favourable)",
    "tropical_wet":    "Tropical wet — Vietnam Mekong (OOD boundary)",
    "hyperarid":       "Hyperarid — Atacama (OOD boundary)",
    "tropical_montane": "Tropical montane Andes (Co/Ec/Pe/Bo, OOD confirmatory)",
}


def per_tile_metrics(df, regime):
    """Return DataFrame per tile. Sign convention matches the manuscript:
    delta_pct = (RMSE_corr - RMSE_raw) / RMSE_raw * 100
    Negative values indicate RMSE reduction (improvement).
    """
    out = []
    for tile, sub in df.groupby("tile"):
        rmse_r = float(np.sqrt(np.mean(sub.err_raw ** 2)))
        rmse_c = float(np.sqrt(np.mean(sub.err_corr ** 2)))
        out.append({
            "tile": tile, "regime": regime, "n": len(sub),
            "rmse_raw": rmse_r, "rmse_corr": rmse_c,
            "delta_pct": 100.0 * (rmse_c - rmse_r) / rmse_r,
        })
    return pd.DataFrame(out)


# --- Mediterranean: spatial-CV OOF predictions ---
df_md = pd.read_csv(ROOT / "scale_p1" / "samples_unified" / "mm_predictions.csv")
df_md["err_raw"] = df_md.fabdem - df_md.h_te_orthometric
df_md["err_corr"] = df_md.fabdem_corrected - df_md.h_te_orthometric
md = per_tile_metrics(df_md, "mediterranean")

# --- Humid temperate: M-M model applied direct (no retrain) ---
df_p3a = pd.read_csv(ROOT / "scale_p3a" / "samples_unified" /
                     "samples_p3a_with_mm_predictions.csv")
ht_df = df_p3a[df_p3a.regime == "humid_temperate"].copy()
ht_df["err_raw"] = ht_df.fabdem - ht_df.h_te_orthometric
ht_df["err_corr"] = ht_df.fabdem + ht_df.pred_residual_mm_model - ht_df.h_te_orthometric
ht = per_tile_metrics(ht_df, "humid_temperate")

# --- Tropical wet + Hyperarid: M-M model applied direct ---
df_p4 = pd.read_csv(ROOT / "scale_p4" / "samples_unified" /
                    "samples_p4_with_mm_predictions.csv")
# err_raw and err_corr already exist in p4 file
tw_df = df_p4[df_p4.regime == "tropical_wet"].copy()
ha_df = df_p4[df_p4.regime == "hyperarid"].copy()
tw = per_tile_metrics(tw_df, "tropical_wet")
ha = per_tile_metrics(ha_df, "hyperarid")

# --- Tropical montane Andes (4-site panel) ---
# Cusco (Peru) from scale_p5; Colombia/Ecuador/Bolivia from scale_p6 (samples
# materialise in scale_p4/tiles/ because run_tile.py is symlinked there).
df_p5 = pd.read_csv(ROOT / "scale_p5" / "samples_unified" /
                    "samples_p5_with_mm_predictions.csv")
df_p5["tile"] = "S13W072"
df_p6 = pd.read_csv(ROOT / "scale_p6" / "samples_unified" /
                    "samples_p6_with_mm_predictions.csv")
tma_df = pd.concat([df_p5, df_p6], ignore_index=True)
tma = per_tile_metrics(tma_df, "tropical_montane")

agg = pd.concat([md, ht, tw, ha, tma], ignore_index=True)
print(agg.to_string(index=False))

# Save aggregated for reuse
agg.to_csv(FIG / "F5_per_tile_4regimes.csv", index=False)

# ============================================================================
# Plot
# ============================================================================
fig, ax = plt.subplots(figsize=(9.2, 5.8))

for regime in ["mediterranean", "humid_temperate", "tropical_wet", "hyperarid", "tropical_montane"]:
    sub = agg[agg.regime == regime]
    ax.scatter(
        sub.rmse_raw, sub.delta_pct,
        s=np.sqrt(sub.n) * 8,
        c=COLORS[regime], marker=MARKERS[regime],
        label=LABELS[regime], alpha=0.85,
        edgecolor="black", linewidth=0.6,
    )

# Per-tile labels with anti-overlap offsets
OFFSETS = {
    "S35W071": (10, 4),     "S35W072": (10, 4),
    "S36W071": (10, -14),   "S36W072": (10, 4),
    "S37W071": (-55, -14),  "S37W072": (10, 4),
    "S38W072": (10, 4),     "S38W073": (10, 4),
    "S39W072": (-55, 4),    "S39W073": (10, -12),
    "N10E105": (-65, 4),    "S24W069": (10, 4),
    "S13W072": (-65, 4),    "N04W074": (10, -14),
    "S01W079": (-65, 4),    "S16W068": (10, 4),
}
for _, r in agg.iterrows():
    dx, dy = OFFSETS.get(r.tile, (8, 4))
    ax.annotate(
        r.tile, (r.rmse_raw, r.delta_pct),
        fontsize=8, xytext=(dx, dy), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
    )

ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.7)

# Shade the improvement region (negative) and the degradation region (positive)
ax.axhspan(-65, 0, color="#E4F4E4", alpha=0.4, zorder=-1)   # improvement
ax.axhspan(0, 60, color="#FFE4E4", alpha=0.5, zorder=-1)    # degradation

ax.set_xlabel("FABDEM raw RMSE (m)")
ax.set_ylabel(r"$\Delta$RMSE $= (\mathrm{RMSE}_{\mathrm{corr}} - \mathrm{RMSE}_{\mathrm{raw}}) / \mathrm{RMSE}_{\mathrm{raw}} \times 100$ (\%)" "\n← improvement        degradation →")
ax.set_title("Per-tile correction performance across five climate regimes\n"
             "(tropical montane Andes panel = 4 tiles spanning Colombia 4°N to Bolivia 16°S;\n"
             "marker area ∝ √n footprints; single M-M model, no retraining)")
ax.legend(loc="upper right", framealpha=0.95)
ax.set_xlim(0.3, 13)
ax.set_ylim(-65, 60)

# Annotation: the bounded transferability message (bottom-left now, since
# improvement region is below the y=0 line)
ax.text(
    0.02, 0.04,
    "Favourable transfer requires (i) Andean-style relief\n"
    "AND (ii) positive FABDEM canopy bias to coexist.\n"
    "All 4 tropical montane Andean tiles (Co, Ec, Pe, Bo) satisfy both\n"
    "→ favourable transfer confirmed across 53° of latitude.\n"
    "Vietnam Delta lacks (i); Atacama lacks (ii).",
    transform=ax.transAxes, fontsize=9, va="bottom", linespacing=1.3,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
              edgecolor="gray", alpha=0.95),
)

fig.tight_layout()

for ext in ("pdf", "png"):
    p = FIG / f"F5_ood_generalization.{ext}"
    fig.savefig(p, format=ext)
    print(f"  → {p}")
    # Copy to ISPRS dir
    p2 = ISPRS_FIG / f"F5_ood_generalization.{ext}"
    fig.savefig(p2, format=ext)
    print(f"  → {p2}")

plt.close(fig)
print("✓ F5 multi-regime regenerated")
