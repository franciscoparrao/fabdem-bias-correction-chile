# FABDEM bias correction over central-south Chile using ICESat-2 ATL08

End-to-end pipeline and analysis code for the paper:

> **Parra Ortiz, F.** (2026). *Stratified machine-learning bias correction of FABDEM transfers between contrasting Chilean climate regimes: evidence from ICESat-2 over Mediterranean and humid temperate watersheds*. ISPRS Journal of Photogrammetry and Remote Sensing (submitted).

## What this repo provides

A reproducible XGBoost residual-correction pipeline for the Forest And Buildings removed Copernicus DEM (FABDEM v1.2) trained on 135,350 ICESat-2 ATL08 v7 ground footprints over central-south Chile (34–39 °S). The pipeline reduces spatial-block cross-validated RMSE from 3.05 m to 2.48 m on the Mediterranean training regime (−18.7 %, Diebold–Mariano *p* < 0.001) and transfers to humid temperate Valdivian rainforest at −22.2 % RMSE without retraining. Two non-Chilean out-of-distribution tiles (Vietnam Mekong Delta tropical-wet; Atacama hyperarid) **empirically bound** the transferability claim — they degrade performance and reveal that favourable transfer requires Andean-style relief and a positive FABDEM canopy bias to coexist.

## Headline results

| Regime | n footprints | RMSE raw (m) | RMSE corr (m) | Δ% |
|---|---:|---:|---:|---:|
| Mediterranean (training, spatial-CV OOF) | 77 501 | 3.054 | 2.483 | **−18.7 %** |
| Humid temperate Chile (OOD favourable) | 57 849 | 4.946 | 3.850 | **−22.2 %** |
| Tropical wet Vietnam Mekong (OOD boundary) | 20 569 | 1.091 | 1.154 | +5.8 % |
| Hyperarid Atacama (OOD boundary) | 20 681 | 0.605 | 0.934 | +54.5 % |

Negative Δ% indicates RMSE reduction (improvement); positive indicates degradation. See the paper for stratified analysis, SHAP attribution, and the bounded-transferability interpretation.

## Repo layout

```
.
├── paper/
│   ├── isprs/               ← Elsevier elsarticle submission (main.tex + tables + figures)
│   ├── DRAFT_v1_*.md        ← Section-level Markdown drafts
│   ├── figures/             ← Figure generation scripts + PDF outputs
│   ├── tables/              ← LaTeX tables
│   ├── experiments/         ← Ablation + DL baseline scripts and results
│   └── compute_uncertainty.py
├── scale_p1/                ← 6 Mediterranean tiles (S35-S37, W071-W072)
│   ├── steps/01_fabdem.py … 07_train.py
│   ├── run_tile.py          ← per-tile orchestrator with checkpoints + RAM watchdog
│   └── run_p1.py            ← M-M training pipeline
├── scale_p3a/               ← 4 humid temperate tiles (S38-S39, W072-W073)
├── scale_p4/                ← 2 non-Chilean OOD tiles (Vietnam Mekong, Atacama)
├── pilot_teno/              ← Initial single-tile prototype
├── p2_validation/           ← Stratified validation + HAND inundation check
├── environment.yml          ← conda dependencies
├── Dockerfile               ← reproducibility container
├── LICENSE                  ← MIT (code); see below for data products
└── CITATION.cff
```

## Reproduce in three commands

```bash
git clone https://github.com/franciscoparrao/fabdem-bias-correction-chile
cd fabdem-bias-correction-chile
conda env create -f environment.yml && conda activate fabdem-bias-correction
python3 scale_p1/run_p1.py            # ≈ 12 hours, 8 GB RAM cap
```

The pipeline is **checkpoint-resumable**: re-running skips completed stages. A `psutil` watchdog enforces an 8 GB RAM ceiling per stage; the ceiling is configurable in `run_tile.py`.

### Authentication required

- `earthaccess` for NASA ICESat-2 ATL08 — register at https://urs.earthdata.nasa.gov
- `planetary-computer` for Sentinel-1/2 — no auth needed for public data
- `geemap`+`earthengine-api` for FABDEM via Google Earth Engine — `earthengine authenticate`

### External tool

The terrain and hydrology stages (`02_terrain.sh`) call the [**SurtGis**](https://github.com/franciscoparrao/surtgis) Rust geospatial library (author's own tool). The Dockerfile installs SurtGis automatically.

## Data products (Zenodo)

Released alongside this repo on Zenodo with persistent DOIs:

1. **Corrected DEM Cloud-Optimized GeoTIFF** (Mataquito + Maule, 6 tiles, ~282 MB) — *CC BY-NC-4.0* (inherited from FABDEM).
2. **Training dataset** — 135 350 footprints, 33 features, EGM2008-corrected ATL08 ground truth, per-tile metadata (Parquet, ~62 MB) — *CC BY-NC-4.0*.
3. **Trained XGBoost model** in booster JSON and scikit-learn joblib formats with hyperparameter manifest — *CC0*.

DOIs will be added on acceptance. For now, install Zenodo links manually from the corresponding deposit pages.

## Licensing

- **Code** (this repo): MIT.
- **Data products** (Zenodo): CC BY-NC-4.0 for DEM and training set (inheritance from FABDEM v1.2); CC0 for trained-model weights.
- **Paper** (manuscript text): © Author; final-version licence depends on journal terms.

## Reproducibility checklist

- Random seed 42 across Optuna, train/test partitions, SHAP subsample, bootstrap.
- 100 Optuna trials, hyperparameters in `scale_p1/samples_unified/mm_metrics.json`.
- Spatial-block CV at 10 km (primary); sensitivity at 5 km / 20 km in `paper/experiments/ablation_blocksize_results.json`.
- Tabular DL baselines (MLP + TabNet) in `paper/experiments/dl_baseline_results.json`.
- All figures regenerable from `paper/figures/*.py`.

## Contact

**Francisco Parra Ortiz**
Universidad de Santiago de Chile (USACH)
`francisco.parra.o@usach.cl`

Issues, questions, and reproductions welcome via GitHub Issues.
