"""Publication-grade matplotlib style for the FABDEM-ML paper.

Eliminates DejaVu Sans default, enforces Helvetica/Arial, removes top+right
spines, sets ticks `in`, disables legend frame by default, and sets DPI for
PDF vector output.

Import at the start of any figure-generation script:
    from style_v2 import setup_style, COLORS_REGIME
    setup_style()
"""
import matplotlib.pyplot as plt
import matplotlib as mpl


def setup_style():
    """Apply publication-grade rcParams. Call once at module load."""
    mpl.rcParams.update({
        # ---- Tipografía: Helvetica/Arial (NOT DejaVu Sans default)
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": 9,

        # ---- Jerarquía de tamaños (panels > axis > ticks > annotations)
        "axes.titlesize": 10,   # most titles will be removed, kept as fallback
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        # ---- Spines: only bottom + left
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,

        # ---- Ticks: dirección `in`, sizes correctos
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,

        # ---- Grid: sutil
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,

        # ---- Legend: sin frame por defecto
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.borderpad": 0.4,

        # ---- Output: 300 DPI minimum, tight bbox
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # ---- Misc
        "pdf.fonttype": 42,   # TrueType (Type 42) — embeds, not Type 3 (bitmap)
        "ps.fonttype": 42,
    })


# Cross-figure consistent colors by regime (used in F1, F3, F4, F5, F6, F7)
COLORS_REGIME = {
    "mediterranean":   "#D7642E",  # warm orange — training
    "humid_temperate": "#2E8B8B",  # teal — Chile OOD favourable
    "tropical_wet":    "#7BA05B",  # sage green — Vietnam OOD boundary
    "hyperarid":       "#C9956B",  # tan/sand — Atacama OOD boundary
    "tropical_montane": "#7B3F7B", # purple — Cusco/Col/Ec/Bo OOD confirmatory
}

# Raw vs corrected (used in F3, F4, F7)
COLOR_RAW = "#888888"
COLOR_CORR = "#1F77B4"
