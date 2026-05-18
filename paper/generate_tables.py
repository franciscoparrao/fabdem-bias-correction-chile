#!/usr/bin/env python3
"""Generate LaTeX tables T1-T5 from existing CSVs and metadata.

Outputs:
  paper/tables/T1_data_sources.tex   + .md preview
  paper/tables/T2_hyperparameters.tex + .md
  paper/tables/T3_per_tile.tex        + .md
  paper/tables/T4_stratification.tex  + .md
  paper/tables/T5_literature.tex      + .md

Uses booktabs style (toprule / midrule / bottomrule).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
P1 = ROOT / "scale_p1"
P3A = ROOT / "scale_p3a"
STRAT = ROOT / "p2_validation" / "stratified"
OOD = P3A / "ood_analysis"
TBL = ROOT / "paper" / "tables"
TBL.mkdir(parents=True, exist_ok=True)


def write_pair(name, tex, md):
    (TBL / f"{name}.tex").write_text(tex)
    (TBL / f"{name}.md").write_text(md)
    print(f"  → {name}.tex + .md")


# ============================================================================
# T1: Data sources summary
# ============================================================================
def t1_data_sources():
    print("T1: data sources")
    rows = [
        # (source, type, native_res, coverage, format, license, access)
        ("FABDEM v1.2", "Bare-earth DEM", "1 arcsec ($\\sim$30 m)",
         "$\\pm$80$^{\\circ}$ lat", "GeoTIFF", "CC BY-NC-4.0",
         "GEE \\texttt{projects/sat-io/open-datasets/FABDEM}"),
        ("ICESat-2 ATL08 v7", "Photon-derived land + canopy heights",
         "100~m segment, 17~m footprint", "Global, 91-day repeat",
         "HDF5", "Public domain", "NASA Earthdata via \\texttt{earthaccess}"),
        ("Sentinel-2 L2A", "Surface reflectance, 13 bands",
         "10--60 m", "Global, $\\sim$5-day revisit", "GeoTIFF (COG)",
         "Open (Copernicus)", "Microsoft Planetary Computer STAC"),
        ("Sentinel-1 RTC", "C-band SAR backscatter $\\sigma^{0}$",
         "10~m", "Global, $\\sim$12-day revisit", "GeoTIFF (COG)",
         "Open (Copernicus)", "Microsoft Planetary Computer STAC"),
        ("EGM2008 geoid", "Geoid undulation grid",
         "2.5$^{\\prime}$", "Global", "GeoTIFF", "Open (NGA)",
         "PROJ network mode (EPSG:4979$\\rightarrow$9518)"),
        ("CIGIDEN Licantén",
         "Sentinel-derived flood polygon, jun 2023 event",
         "Vector polygons", "Licantén (35$^{\\circ}$S), Mataquito mouth",
         "Shapefile", "Open (CC)",
         "Zenodo DOI \\texttt{10.5281/zenodo.13307972}"),
    ]
    cols = ["Source", "Type", "Native res.", "Coverage", "Format", "License", "Access"]
    df = pd.DataFrame(rows, columns=cols)

    tex = r"""\begin{table*}[t]
\centering
\caption{Data sources used in this study. ICESat-2 ATL08 provides ground truth; FABDEM is the DEM under correction; Sentinel-1/2 and EGM2008 feed the feature stack; CIGIDEN Licant\'en is used as flood reference in the discussion.}
\label{tab:data_sources}
\footnotesize
\begin{tabular}{p{2.4cm}p{3cm}p{2cm}p{2cm}p{1.5cm}p{1.5cm}p{3cm}}
\toprule
\textbf{Source} & \textbf{Type} & \textbf{Native res.} & \textbf{Coverage} & \textbf{Format} & \textbf{License} & \textbf{Access} \\
\midrule
"""
    for _, r in df.iterrows():
        tex += " & ".join(str(v) for v in r) + r" \\" + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    md = "# T1 — Data sources\n\n" + df.to_markdown(index=False) + "\n"
    write_pair("T1_data_sources", tex, md)


# ============================================================================
# T2: XGBoost hyperparameters
# ============================================================================
def t2_hyperparameters():
    print("T2: hyperparameters")
    m = json.loads((P1 / "samples_unified" / "mm_metrics.json").read_text())
    bp = m["best_params"]
    rows = [
        ("n\\_estimators", bp["n_estimators"], "Trees in ensemble"),
        ("max\\_depth", bp["max_depth"], "Maximum tree depth"),
        ("learning\\_rate (eta)", f"{bp['learning_rate']:.4f}",
         "Shrinkage rate (low $\\to$ more regularization)"),
        ("subsample", f"{bp['subsample']:.3f}", "Row subsample ratio per tree"),
        ("colsample\\_bytree", f"{bp['colsample_bytree']:.3f}",
         "Feature subsample ratio per tree"),
        ("min\\_child\\_weight", bp["min_child_weight"],
         "Min child node weight (regularization)"),
        ("reg\\_lambda (L2)", f"{bp['reg_lambda']:.3f}",
         "L2 weight regularization"),
        ("reg\\_alpha (L1)", f"{bp['reg_alpha']:.3f}",
         "L1 weight regularization"),
        ("gamma", f"{bp['gamma']:.3f}",
         "Min loss reduction for split (post-pruning)"),
        ("early\\_stopping\\_rounds", 30, "Stops if eval RMSE stops improving"),
        ("tree\\_method", "hist", "Histogram-based split finding"),
        ("Optuna trials", 100, "TPE sampler, objective = mean spatial-CV OOF RMSE"),
        ("random\\_state", 42, ""),
    ]
    cols = ["Parameter", "Value", "Description"]
    df = pd.DataFrame(rows, columns=cols)

    tex = r"""\begin{table}[t]
\centering
\caption{XGBoost hyperparameters selected by Optuna (TPE sampler, 100 trials) over spatial-block cross-validation. The combination favours deep but heavily regularised trees: high \texttt{reg\_lambda} and \texttt{min\_child\_weight} compensate for the increased depth.}
\label{tab:hyperparams}
\footnotesize
\begin{tabular}{lll}
\toprule
\textbf{Parameter} & \textbf{Value} & \textbf{Description} \\
\midrule
"""
    for _, r in df.iterrows():
        tex += " & ".join(str(v) for v in r) + r" \\" + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    md = "# T2 — XGBoost hyperparameters\n\n" + df.to_markdown(index=False) + "\n"
    write_pair("T2_hyperparameters", tex, md)


# ============================================================================
# T3: Per-tile metrics
# ============================================================================
def t3_per_tile():
    print("T3: per-tile metrics")
    df = pd.read_csv(P3A / "samples_unified" / "samples_p3a_with_mm_predictions.csv")
    df["err_raw"] = df["fabdem"] - df["h_te_orthometric"]
    df["err_corr"] = (df["fabdem"] + df["pred_residual_mm_model"]) - df["h_te_orthometric"]
    agg = df.groupby(["tile", "regime"]).agg(
        n=("lon", "count"),
        rmse_raw=("err_raw", lambda s: float(np.sqrt(np.mean(s**2)))),
        rmse_corr=("err_corr", lambda s: float(np.sqrt(np.mean(s**2)))),
        mae_raw=("err_raw", lambda s: float(np.mean(np.abs(s)))),
        mae_corr=("err_corr", lambda s: float(np.mean(np.abs(s)))),
        bias_raw=("err_raw", "mean"),
        bias_corr=("err_corr", "mean"),
    ).reset_index()
    agg["improve_pct"] = 100 * (agg.rmse_raw - agg.rmse_corr) / agg.rmse_raw
    # Sort: mediterranean first, then humid, by tile
    agg["regime_rank"] = agg.regime.map({"mediterranean": 0, "humid_temperate": 1})
    agg = agg.sort_values(["regime_rank", "tile"]).drop(columns="regime_rank")

    tex = r"""\begin{table*}[t]
\centering
\caption{Per-tile performance of the Mediterranean-trained XGBoost model. Tiles S35W071--S37W072 are in-sample (Mediterranean regime, training data); tiles S38W072--S39W073 are out-of-distribution (humid temperate forest). Improvement (\%) = (RMSE\textsubscript{raw} $-$ RMSE\textsubscript{corr}) / RMSE\textsubscript{raw} $\times$ 100.}
\label{tab:per_tile}
\footnotesize
\begin{tabular}{llrrrrrrrr}
\toprule
\textbf{Tile} & \textbf{Regime} & \textbf{n} &
\textbf{RMSE\textsubscript{raw}} & \textbf{RMSE\textsubscript{corr}} &
\textbf{MAE\textsubscript{raw}} & \textbf{MAE\textsubscript{corr}} &
\textbf{Bias\textsubscript{raw}} & \textbf{Bias\textsubscript{corr}} &
\textbf{$\Delta$\%} \\
& & & (m) & (m) & (m) & (m) & (m) & (m) & \\
\midrule
"""
    for _, r in agg.iterrows():
        regime_short = "Med" if r.regime == "mediterranean" else "HT"
        tex += (f"{r.tile} & {regime_short} & {int(r.n):,} & "
                f"{r.rmse_raw:.3f} & {r.rmse_corr:.3f} & "
                f"{r.mae_raw:.3f} & {r.mae_corr:.3f} & "
                f"{r.bias_raw:+.3f} & {r.bias_corr:+.3f} & "
                f"{r.improve_pct:+.1f}").replace("nan", "---") + r" \\" + "\n"
    # Aggregate row
    md = agg[agg.regime == "mediterranean"]
    ht = agg[agg.regime == "humid_temperate"]

    def overall(sub):
        sub_df = df[df.regime.isin(sub.regime.unique())] if len(sub) else df.iloc[:0]
        if len(sub_df) == 0:
            return None
        rmse_r = np.sqrt((sub_df.err_raw**2).mean())
        rmse_c = np.sqrt((sub_df.err_corr**2).mean())
        return {
            "n": len(sub_df),
            "rmse_raw": rmse_r, "rmse_corr": rmse_c,
            "mae_raw": sub_df.err_raw.abs().mean(),
            "mae_corr": sub_df.err_corr.abs().mean(),
            "bias_raw": sub_df.err_raw.mean(),
            "bias_corr": sub_df.err_corr.mean(),
            "improve_pct": 100 * (rmse_r - rmse_c) / rmse_r,
        }
    tex += r"\midrule" + "\n"
    for tag, sub in [("\\textit{Mediterranean total}", md), ("\\textit{Humid temperate total}", ht)]:
        o = overall(sub)
        if not o:
            continue
        tex += (f"{tag} & --- & {o['n']:,} & "
                f"{o['rmse_raw']:.3f} & {o['rmse_corr']:.3f} & "
                f"{o['mae_raw']:.3f} & {o['mae_corr']:.3f} & "
                f"{o['bias_raw']:+.3f} & {o['bias_corr']:+.3f} & "
                f"{o['improve_pct']:+.1f}") + r" \\" + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    md_out = "# T3 — Per-tile metrics\n\n" + agg.round(3).to_markdown(index=False) + "\n"
    write_pair("T3_per_tile", tex, md_out)


# ============================================================================
# T4: Stratification summary across 6 dimensions
# ============================================================================
def t4_stratification():
    print("T4: stratification across 6 dimensions")
    # Use Mediterranean spatial-CV results (the honest training metric)
    files = [
        ("Elevation",   "by_elevation"),
        ("Slope",       "by_slope"),
        ("HAND",        "by_hand"),
        ("Geomorphon",  "by_geomorphon"),
        ("NDVI",        "by_ndvi"),
        ("Tile",        "by_tile"),
    ]
    rows = []
    for dim, fname in files:
        df = pd.read_csv(STRAT / f"{fname}.csv", index_col=0)
        for idx, r in df.iterrows():
            if r.n < 10:
                continue  # skip near-empty strata
            rows.append({
                "dimension": dim, "stratum": str(idx),
                "n": int(r.n),
                "rmse_raw": float(r.rmse_raw),
                "rmse_corr": float(r.rmse_corr),
                "improve_pct": float(r.improve_pct),
                "bias_raw": float(r.bias_raw),
                "bias_corr": float(r.bias_corr),
            })
    df_all = pd.DataFrame(rows).sort_values(["dimension", "n"], ascending=[True, False])

    tex = r"""\begin{table*}[t]
\centering
\caption{Stratified performance on the Mediterranean dataset (spatial-CV out-of-fold predictions, n=77{,}501 footprints). RMSE improvement is largest in dense vegetation, alpine terrain, and intermediate HAND bands; smaller in floodplains where FABDEM is already accurate; near-zero or negative in sample-sparse flat strata. Bias (signed mean error: FABDEM $-$ ICESat-2 orthometric) is systematically positive in raw FABDEM and near zero post-correction.}
\label{tab:stratification}
\footnotesize
\begin{tabular}{llrrrrrr}
\toprule
\textbf{Dimension} & \textbf{Stratum} & \textbf{n} &
\textbf{RMSE\textsubscript{raw}} & \textbf{RMSE\textsubscript{corr}} &
\textbf{$\Delta$\%} & \textbf{Bias\textsubscript{raw}} & \textbf{Bias\textsubscript{corr}} \\
& & & (m) & (m) & & (m) & (m) \\
\midrule
"""
    cur_dim = None
    for _, r in df_all.iterrows():
        if cur_dim is None:
            cur_dim = r["dimension"]
        elif r["dimension"] != cur_dim:
            tex += r"\addlinespace[2pt]" + "\n"
            cur_dim = r["dimension"]
        dim_show = r["dimension"] if (
            df_all[df_all["dimension"] == r["dimension"]].iloc[0].equals(r)
        ) else ""
        s = r["stratum"].replace("_", "\\_").replace("&", "\\&").replace("<", "$<$").replace(">", "$>$")
        tex += (f"{dim_show} & {s} & {r['n']:,} & "
                f"{r['rmse_raw']:.3f} & {r['rmse_corr']:.3f} & "
                f"{r['improve_pct']:+.1f}\\,\\% & "
                f"{r['bias_raw']:+.3f} & {r['bias_corr']:+.3f}") + r" \\" + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    md_out = "# T4 — Stratification (Mediterranean spatial CV)\n\n" + df_all.round(3).to_markdown(index=False) + "\n"
    write_pair("T4_stratification", tex, md_out)


# ============================================================================
# T5: Literature comparison
# ============================================================================
def t5_literature():
    print("T5: literature comparison")
    rows = [
        # (study, region, method, baseline_RMSE, post_RMSE, delta, validation, openness)
        ("Hawker et al. (2022)", "Global",
         "Random Forest correction of Copernicus DSM (canopy/buildings)",
         "$\\sim$4.3", "$\\sim$2.5", "$\\sim$40\\%",
         "Airborne LiDAR samples", "Open data + paper"),
        ("Meadows et al. (2024)", "5 countries",
         "Benchmark of public DEMs, no correction",
         "n/a", "FABDEM $\\sim$2.5",
         "n/a", "GNSS / airborne LiDAR", "Open paper"),
        ("Wing et al. (2024)", "Global",
         "FABDEM as input to 30~m global flood model",
         "n/a", "n/a", "n/a",
         "Vietnam case study", "Open paper, model commercial"),
        ("FABDEM+ (Fathom, 2024)", "Global commercial",
         "Proprietary FABDEM enhancement + local data fusion",
         "$\\sim$2.5", "improved (undisclosed)", "n/a",
         "Internal validation", "Commercial product"),
        ("\\textbf{This study (M-M)}", "Chile 34--37$^{\\circ}$S (Mediterranean)",
         "XGBoost residual correction, ICESat-2 ATL08",
         "\\textbf{3.054}", "\\textbf{2.483}", "\\textbf{$-$18.7\\%}",
         "Spatial-block CV (10~km, K=5)", "\\textbf{All open}"),
        ("\\textbf{This study (OOD)}", "Chile 37--39$^{\\circ}$S (humid temperate)",
         "Same M-M-trained model, no retrain",
         "\\textbf{4.946}", "\\textbf{3.850}", "\\textbf{$-$22.2\\%}",
         "Held-out, different climate", "\\textbf{All open}"),
    ]
    cols = ["Study", "Region", "Method", "RMSE\\textsubscript{baseline} (m)",
            "RMSE\\textsubscript{post} (m)", "$\\Delta$", "Validation", "Openness"]

    tex = r"""\begin{table*}[t]
\centering
\caption{Comparison with related work on FABDEM-family bias correction and evaluation. Our work is the first published, open, regional ML correction of FABDEM with ICESat-2 ATL08 as ground truth, and the first to demonstrate cross-regime transferability.}
\label{tab:literature}
\footnotesize
\begin{tabular}{p{2.7cm}p{2.6cm}p{3.5cm}rrlp{2.7cm}p{2.2cm}}
\toprule
"""
    tex += " & ".join(f"\\textbf{{{c}}}" for c in cols) + r" \\" + "\n"
    tex += r"\midrule" + "\n"
    for r in rows:
        tex += " & ".join(str(v) for v in r) + r" \\" + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    # Markdown preview
    md_df = pd.DataFrame(rows, columns=[c.replace("\\textsubscript{","_").replace("}","").replace("\\textbf{","").replace("$\\Delta$","Δ") for c in cols])
    md_out = "# T5 — Literature comparison\n\n" + md_df.to_markdown(index=False) + "\n"
    write_pair("T5_literature", tex, md_out)


if __name__ == "__main__":
    print(f"Generating LaTeX tables into {TBL}")
    t1_data_sources()
    t2_hyperparameters()
    t3_per_tile()
    t4_stratification()
    t5_literature()
    print(f"\n✅ 5 tables (.tex + .md) generated")
    for f in sorted(TBL.glob("*.tex")):
        print(f"  - {f.name}")
