# Paper outline — FABDEM-ML stratified bias correction for Chile

**Working title**: *Stratified machine-learning bias correction of FABDEM transfers between contrasting Chilean climate regimes: evidence from ICESat-2 over Mediterranean and humid temperate watersheds*

**Target venue**: International Journal of Digital Earth (primary) / NHESS (backup)
**Authors**: Francisco Parra Ortiz (sole author for now)
**Status**: 2026-05-15 — outline + flagship figures stage

---

## Three defensible claims (the contribution)

1. **FABDEM has a learnable, geographically heterogeneous bias** — not uniform global +1 m as commonly assumed
2. **A regional ML correction with ICESat-2 ATL08 reduces RMSE 18.7% under spatial-block cross-validation** (95% CI: 2.37–2.62 m vs 2.95–3.18 m baseline; Diebold–Mariano p < 0.001) — eliminating systematic positive bias
3. **The correction transfers between contrasting Chilean climate regimes without retraining** — Mediterranean-trained model achieves −22.2% RMSE on humid temperate forest (95% CI on corrected RMSE: 3.76–3.96 m). Multi-continental generalisation remains to be tested in future work.

---

## Abstract

→ See `ABSTRACT.md` (v3, 275 words, post tex-review HIGH fixes applied 2026-05-16).

---

## Section structure

### 1. Introduction (1.5 pages)

- 1.1 Global DEMs and persistent vertical bias: SRTM → MERIT → Copernicus → FABDEM
- 1.2 Hawker et al. (2022) FABDEM as state-of-the-art: methodology with RF, canopy/building removal
- 1.3 Limitations of global correction:
  - Single global RF model
  - No regional fine-tuning
  - Limited validation in non-temperate-forest training regions
  - Bias correction at 30 m is bias correction, not super-resolution
- 1.4 The ICESat-2 ATL08 opportunity: cm-level vertical precision, global coverage, 91-day repeat
- 1.5 Research questions:
  - RQ1: Does FABDEM bias have spatial structure that is regionally learnable?
  - RQ2: How does correction performance stratify across geomorphometric and climate regimes?
  - RQ3: Does a regional ML correction transfer between contrasting climate regimes (Mediterranean ↔ humid temperate) without retraining?
- 1.6 Contributions and paper structure

### 2. Study area (1 page)

- 2.1 Geographic context: Chile 34–39°S, ~700,000 km², coast-Andes transition 220 km wide
- 2.2 Two contrasting climate regimes:
  - **Mediterranean** (Mataquito + Maule, 6 tiles 34–37°S): dry summer, mixed land use (vineyard, agriculture, sclerophyll forest), NDVI 0.2–0.5 typical
  - **Humid temperate** (Bío-Bío + Araucanía, 4 tiles 37–39°S): year-round precipitation, dense Valdivian forest, NDVI 0.5–0.8 typical
- 2.3 The June 2023 atmospheric river event ("Vuelven los gigantes"):
  - Strongest precipitation in 30 years over central Chile
  - 21,000+ displaced, 2 deaths, 1,800 homes destroyed
  - CIGIDEN documented Mataquito + Maule basins
  - Motivates need for accurate DEMs in flood-prone Chilean basins

### 3. Data (2 pages)

- 3.1 **FABDEM v1.2**: source DEM at 30 m, EGM2008 vertical datum, downloaded via GEE asset `projects/sat-io/open-datasets/FABDEM`, 10 tiles × ~3700² ≈ 137M pixels
- 3.2 **ICESat-2 ATL08 v7**: ground truth, 50,000+ filtered land segments after quality filters (terrain_flg=1, n_te_photons≥30, h_te_unc<10m, snow=1, water=0, cloud≤1), via `earthaccess`
- 3.3 **Sentinel-2 L2A** austral summer 2022-23 composite (top-3 per MGRS tile <20% cloud), bands B02/B03/B04/B08/B11/B12 → indices NDVI, NDWI, NDMI, BSI, NDBI
- 3.4 **Sentinel-1 RTC** medians, VV/VH polarization to dB, plus ratio
- 3.5 **EGM2008 geoid correction**: via PROJ network mode, transformer EPSG:4979 → EPSG:9518. Geoid undulation N ≈ +22 m in study area, must be removed before computing residual
- 3.6 **SurtGis-derived features** (Rust, Rayon parallel): from `terrain all` and `hydrology all`:
  - Slope, aspect (sin/cos), hillshade, curvature, TPI, TRI, VRM, dev, convergence
  - Openness positive/negative, sky-view factor (SVF)
  - MRVBF, MRRTF, geomorphons (10 classes)
  - HAND (Height Above Nearest Drainage), TWI, flow direction/accumulation
  - Valley depth
- 3.7 Total: 33 features per footprint after EGM2008 correction, target = h_te_orthometric − fabdem

### 4. Methods (3 pages)

- 4.1 Pipeline overview (figure: F2 flowchart)
- 4.2 Tile-based parallel processing with psutil watchdog (8 GB RAM cap), checkpointing
- 4.3 ATL08 quality filtering and EGM2008 correction
- 4.4 Feature stack at footprint locations: CRS-aware sampling (FABDEM in EPSG:4326, satellite in UTM 19S)
- 4.5 XGBoost regression target = residual, with Optuna 100 trials TPE sampler
- 4.6 **Spatial-block cross-validation**: 10 km blocks (570 unique), GroupKFold K=5 — to avoid spatial autocorrelation leakage
- 4.7 Hyperparameter ranges and final selections
- 4.8 SHAP TreeExplainer for feature importance (sample 8,000)
- 4.9 **Out-of-distribution test**: train on Mediterranean, evaluate on humid temperate without retraining
- 4.10 Inference at full raster scale: aligned feature stack, per-pixel prediction, COG output with overviews
- 4.11 Compute environment: SurtGis (Rust) for terrain, Python (planetary-computer + stackstac) for satellite, XGBoost+Optuna for ML, ≤8 GB RAM peak

### 5. Results (4 pages)

- 5.1 **Overall**: RMSE 3.054 → 2.483 m (spatial CV, −18.7%), MAE 1.423 → 1.099 m (−22.8%)
  - Random CV inflation: 2.395 m → would over-claim by 4%
  - Gap leakage: 3.5% — generalization sound
- 5.2 **Stratification by 6 dimensions** (figure F4): table T4
  - Elevation: largest gain at >3000m (−33.8%) and <500m (−21.4%)
  - Slope: −18.9% steep, near-zero at flat (extrapolation failure)
  - HAND: largest gain at 10–50m (−24.4%), smaller in floodplain
  - NDVI: largest gain in dense (>0.6, −41.6%)
  - Geomorphon: peak −33%, spur −23%
  - Per-tile: −22% (S35W071) to −10% (S36W071 — Andes central, sample <2k)
- 5.3 **Bias elimination** (figure F7): histograms before/after
  - Raw FABDEM bias: +1.0 to +4.6 m across regimes
  - Corrected bias: −0.05 to +0.03 m (essentially zero)
- 5.4 **SHAP feature importance**:
  - Top 5 in Mediterranean: twi, vrm, openness_negative, tri, convergence
  - Highest non-terrain: ndvi (#6), ndwi (#10)
  - SHAP values match feature stack diversity
- 5.5 **Out-of-distribution generalisation** (figure F5, primary result):
  - M-M model on humid_temperate (n=57,849): RMSE 4.946 m (95% CI 4.84–5.06) → 3.850 m (95% CI 3.76–3.96), **−22.2%** absolute reduction 1.10 m
  - **Diebold–Mariano DM = 40.2 (p < 0.001)** → improvement statistically robust under true OOD evaluation
  - **Per-pixel sign test**: corrected better in 57.2% of points (33,080/57,849; p < 0.001)
  - Per humid_temperate tile: −16.3% to −29.2%
  - SHAP shift (figure F6): valley_depth rises from rank 8 to 4; openness_negative rises from #3 to #1
  - **Nearly all features (32 of 33)** show increased SHAP magnitude in humid temperate (ratio > 1); only flow_accumulation has ratio 0.98. Interpretation: a larger baseline error budget yields proportionally larger feature contributions, not a regime-specific re-weighting
- 5.6 **HAND-based flood mapping comparison** (preempt reviewer):
  - HAND threshold doesn't differentiate raw vs corrected (best IoU 0.12 raw → 0.10 corr)
  - Diagnostic: correction acts in uplands (−2.81 m mean) not in floodplains (−0.71 m)
  - Hydrodynamic validation (LISFLOOD-FP) is appropriate future work

### 6. Discussion (3 pages)

- 6.1 **Where ML bias correction matters**: the regime-dependent improvement pattern. Dense vegetation receives roughly twice the correction magnitude of floodplains — an association **consistent with** (not proof of) canopy-induced overestimation in the upstream Copernicus DSM that FABDEM's RF correction does not fully remove.
- 6.2 **Limitations and scope of transferability claim** (placed before positive interpretation to anchor reader expectations):
  - **Two regimes tested, both within Chile** (geographic span ~660 km, latitudinal span 34–39°S). The "transfers between regimes" claim is supported for the Mediterranean↔humid-temperate contrast; testing against tropical, arid (Atacama), or polar terrain is future work.
  - Sample-sparse strata extrapolate poorly (S36W071, flat geomorphons) — small-n regions inherit larger uncertainty.
  - Sentinel composite is single-season austral summer — no temporal/phenological features.
  - No validation against airborne LiDAR (not publicly available for the study basins). ICESat-2 ATL08 (cm-level vertical precision) serves as the highest-resolution available ground truth.
  - No hydrodynamic flood model validation (LISFLOOD-FP) — see HAND comparison in 5.6 for the preliminary check; full hydrodynamic evaluation reserved for follow-up work.
  - No feature-ablation study in main text; included as supplementary if requested by reviewers.
- 6.3 **Geographic transferability evidence (within scope of D2 limitation above)**: a model trained on Mediterranean Chile (NDVI median ≈0.3, dry summer) achieves −22% RMSE on humid temperate forest (NDVI median ≈0.6, year-round precipitation) at ~800 km separation. This is **consistent with** the model capturing physically grounded terrain–error relationships rather than memorising patterns specific to the training regime. Alternative explanations include: shared Andean morphology (foothills + cordillera in both regimes) — partially controlled but not fully ruled out.
- 6.4 **SHAP adaptation between regimes**: valley_depth and openness_negative climb in importance for the humid temperate case, while TWI cedes the top rank. This pattern is consistent with the model emphasising 3D shape descriptors more when vegetation density (and the associated upstream canopy bias signal) is high, and drainage features more in semi-arid conditions. Causal interpretation should remain hypothesis-grade given the N=2 regime test.
- 6.5 **Comparison with FABDEM+ (Fathom)**: ours is open, transparent, regional-adaptive, while FABDEM+ is commercial with proprietary tuning
- 6.6 **Implications for downstream applications**:
  - Hydraulic modelers using FABDEM in dense-vegetation or alpine regions: significant gain
  - Floodplain-only applications: marginal gain
  - Land surface models (evapotranspiration, slope stability): direct benefit
- 6.7 **Future work**:
  - Extend to all Chile (P3b, P3c per roadmap)
  - LISFLOOD-FP hydrodynamic validation
  - Foundation model (Prithvi, TerraMind) integration
  - Multi-event flood polygon dataset (paper 2)

### 7. Conclusions (0.5 pages)

- FABDEM has stratified, learnable bias with magnitude that depends on terrain and vegetation regime
- ICESat-2 ATL08 is scalable, cm-precision ground truth that enables regional refinement without airborne LiDAR
- A Mediterranean-trained correction generalises to humid temperate forest without retraining (−22% RMSE, DM p < 0.001), demonstrating transferability between contrasting Chilean climate regimes
- SHAP feature-importance shifts between regimes are consistent with physically grounded terrain–error learning rather than regional memorisation, though multi-continental validation is needed
- All artifacts (DEM COG, training dataset, trained model, pipeline code) are publicly released under CC BY-NC-4.0 (inherited from FABDEM)

### 8. Data and code availability

- **Corrected DEM (COG)**: Zenodo DOI [TBD]. Released under **CC BY-NC-4.0 (inherited from FABDEM)**.
- Training dataset (135k footprints × 33 features): Zenodo DOI [TBD]
- Trained model (XGBoost booster + sklearn joblib): Zenodo DOI [TBD]
- Code: GitHub [TBD] with Dockerfile + environment.yml; tagged release at submission
- **Software versions**: Python 3.12; XGBoost 3.1.1; Optuna 4.6.0 (TPE sampler); scikit-learn 1.8.0; SHAP 0.50.0; pyproj 3.6.x (PROJ network mode for EGM2008); SurtGis 0.7.0 (Rust)
- **Random seeds**: 42 across Optuna sampler, train/test splits, SHAP sampling, bootstrap (2000 resamples)
- **Block-size sensitivity** (supplementary): tested 5 km and 10 km blocks; RMSE differs by < 0.05 m → reported 10 km as primary
- ICESat-2 ATL08 v7 from NASA Earthdata (open, accessed via `earthaccess`)
- FABDEM v1.2 from University of Bristol (CC BY-NC-4.0)
- Sentinel-1/2 from Microsoft Planetary Computer (open, accessed via `pystac-client` + `stackstac`)
- EGM2008 geoid via PROJ network mode (open)

### References (50–80)

Key citations to include:
- Hawker et al. 2022 (FABDEM original)
- Meadows et al. 2024 (FABDEM benchmark, IJDE)
- Hawker et al. 2024 (LISFLOOD-FP + FABDEM Vietnam, NHESS)
- Wing et al. 2024 (FABDEM flood model, WRR)
- Neuenschwander et al. 2019 (ATL08 algorithm)
- Markus et al. 2017 (ICESat-2 mission) + Neuenschwander & Pitts 2019 (ATL08 product)
- Jasiewicz & Stepinski 2013 (geomorphons)
- Lundberg & Lee 2017 (SHAP)
- Chen & Guestrin 2016 (XGBoost)
- Akiba et al. 2019 (Optuna)
- Roberts et al. 2017 (spatial block CV)
- Hawker et al. 2018 (vertical accuracy)
- Yamazaki et al. 2017 (MERIT-DEM)
- CIGIDEN reports 2023

---

## Figures (8 flagship)

| # | Description | Section | Status |
|---|---|---|---|
| **F1** | Study area map: 10 tiles, 2 regímenes, AR jun 2023 affected area | 2 | TODO (matplotlib + cartopy) |
| **F2** | Pipeline flowchart: data → features → training → inference | 4 | TODO (boxes & arrows) |
| **F3** | Per-tile RMSE bar chart (raw vs corrected) | 5.1 | ✓ scripted |
| **F4** | Stratification 4-panel: NDVI/HAND/slope/elevation × regime | 5.2 | ✓ scripted |
| **F5** | OOD generalization: scatter improvement vs sample, by tile/regime | 5.5 | ✓ scripted |
| **F6** | SHAP regime comparison: feature importance shift bars | 5.4 & 6.3 | ✓ scripted |
| **F7** | Bias histogram: before/after, per regime | 5.3 | ✓ scripted |
| **F8** | mm_residual.tif map: spatial pattern of correction (+ inset Licantén) | 5.5 / 8.6 | TODO (rasterio + matplotlib) |

## Tables (5)

| # | Description | Section |
|---|---|---|
| T1 | Data sources summary | 3 |
| T2 | XGBoost hyperparameters (Optuna best) | 4 |
| T3 | Per-tile metrics | 5.1 |
| T4 | Stratification metrics × 6 dimensions | 5.2 |
| T5 | Comparison with FABDEM (raw), FABDEM+ (Fathom), Meadows et al. 2024 | 6.5 |

---

## Submission plan

- **Step 1** (this week): F3, F4, F5, F6, F7 figures generated
- **Step 2** (next week): F1, F2, F8 figures
- **Step 3** (~3 weeks): Draft v1 (5,000 words)
- **Step 4** (~5 weeks): Draft v2 + supplementary
- **Step 5** (~6 weeks): Submission IJDE + EarthArXiv preprint + Zenodo DOIs
- **Step 6** (~12-16 weeks after submission): Reviewer revisions
- **Target acceptance**: Q1-Q2 2027
