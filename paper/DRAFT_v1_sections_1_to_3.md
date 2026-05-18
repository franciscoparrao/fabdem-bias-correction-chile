# Draft v1 — Sections 1–3
**Working title**: *Stratified machine-learning bias correction of FABDEM transfers between contrasting Chilean climate regimes: evidence from ICESat-2 over Mediterranean and humid temperate watersheds*

**Status**: 2026-05-16 draft v1, sections 1–3 only (Intro + Study area + Data). Sections 4–8 to follow.
**Target length 1–3**: 4.5 manuscript pages (~2,500 words excluding citations).
**Style**: British English; third person; LaTeX-ready inline citations as `\citep{key}` and `\citet{key}`.

---

## 1. Introduction

Open-access global digital elevation models (DEMs) underpin essentially every contemporary application that requires terrain information at regional or continental scale, from flood inundation modelling \citep{Wing2024GlobalFlood, Hawker2024Vietnam} to land-surface processes, infrastructure planning, and earth-surface dynamics. Three decades of mission heritage — SRTM \citep{Farr2007SRTM}, ASTER GDEM, TanDEM-X \citep{Rizzoli2017TanDEMX}, ALOS World 3D, NASADEM, MERIT \citep{Yamazaki2017MERIT}, and Copernicus DEM — have progressively reduced global vertical error but have not eliminated it. Each successive product inherits, in modified form, residual biases from the underlying acquisition geometry, vegetation occlusion, and built-environment artefacts \citep{Schumann2018DEMSurvey, Hawker2018Accuracy}. The Forest And Buildings removed Copernicus DEM (FABDEM; \citealt{Hawker2022FABDEM}) is, at the time of writing, the most accurate publicly available bare-earth DEM at 30-m resolution, achieving an aggregate global root mean square error (RMSE) of approximately 2.5 m after a random-forest correction removes canopy and building height contributions from the Copernicus DSM. Independent benchmarks confirm FABDEM as the strongest of the freely available 30-m DEMs in flood-prone environments \citep{Meadows2024VerticalAccuracy}.

Despite this advance, two practically important questions remain only partially addressed. First, does FABDEM's aggregate ~2.5 m global RMSE conceal a *geographically heterogeneous* bias structure that could be exploited by targeted regional correction? Although Hawker et al.'s \citeyearpar{Hawker2022FABDEM} global random-forest model removes the largest systematic effects, it is by design a single global function: the same correction is applied everywhere, with no regional adaptation. If FABDEM's residual error in, say, dense Andean foothill forest follows a different geomorphometric signature than its error in semi-arid Mediterranean valley terrain, then a regionally fine-tuned correction could in principle further reduce RMSE in specific terrain types. Second, can such a regional correction *transfer* between climate regimes — that is, does a model trained on one terrain–vegetation regime usefully correct FABDEM in a contrasting regime *without retraining*? This is a transferability question with direct implications for cost and scalability: if a Chilean-trained correction is portable, the methodology can be applied to ungauged regions where local ground-truth is unavailable. Commercial alternatives such as FABDEM+ \citep{Fathom2024FABDEMPlus} exist but their methodology is proprietary, precluding independent evaluation of regional behaviour.

Validating either question at scale has historically been bottlenecked by access to high-precision ground-truth elevation. Airborne LiDAR is the gold standard but is expensive, geographically uneven, and frequently unavailable for the specific basins of interest. ICESat-2, launched in 2018, has begun to change this landscape. The ATLAS instrument acquires along-track photon-counting altimetry at cm-level vertical precision over a global, repeat-orbit footprint \citep{Markus2017ICESat2}, and the ATL08 land-and-vegetation product \citep{Neuenschwander2019ATL08} provides quality-controlled ground-classified segments at ~100 m along-track resolution with 17 m circular footprint. ATL08 thus offers a satellite-borne, cm-precision LiDAR-equivalent ground truth that scales globally and refreshes every ~91 days. The relevant cost is no longer access — it is engineering: extracting, quality-filtering, datum-correcting (ATL08 reports ellipsoidal heights; FABDEM uses orthometric EGM2008 heights), and feature-engineering at scale.

In this study we exploit ATL08 to construct a regional FABDEM correction and to test transferability across climate regimes. Specifically, we address three research questions:

- **RQ1**: Does FABDEM's bias have spatial structure that is regionally learnable from ICESat-2 ATL08 ground truth plus geomorphometric and satellite-derived features?
- **RQ2**: How does correction performance stratify across geomorphometric (slope, curvature, HAND), climatic (NDVI, elevation), and tile-level dimensions?
- **RQ3**: Does a regional ML correction transfer between contrasting climate regimes (Mediterranean ↔ humid temperate) *without retraining*?

We assemble 135,350 ICESat-2 ATL08 v7 ground-classified footprints over 10 1°×1° FABDEM tiles spanning central-south Chile (34–39°S), engineer 33 features per footprint (geomorphometric, hydrological, and Sentinel-1/2 derived), and train XGBoost regression \citep{Chen2016XGBoost} on the residual *h_te* − *fabdem* (after EGM2008 datum correction; \citealt{Pavlis2012EGM2008}) with Optuna hyperparameter optimisation \citep{Akiba2019Optuna} under spatial-block cross-validation \citep{Roberts2017SpatialCV} to suppress autocorrelation leakage. We then apply the Mediterranean-trained model to humid temperate forest tiles without retraining to evaluate cross-regime transferability. Feature attribution uses SHAP values \citep{Lundberg2017SHAP} to compare what the model emphasises in each regime. We report bootstrap 95% confidence intervals on all RMSE estimates and Diebold–Mariano tests of squared-error differentials.

The contributions of this work are:

1. **An honest regional benchmark for ML correction of FABDEM**: we reduce spatial-block cross-validated RMSE from 3.054 m (95% CI 2.95–3.18) to 2.483 m (2.37–2.62) — an absolute reduction of 0.57 m (−18.7%) that is highly statistically significant (Diebold–Mariano *p* < 0.001) — and eliminate a systematic +1.0 m positive bias.
2. **A stratified characterisation of where the correction matters**: we show RMSE gains of −41.6% in dense vegetation, −33.8% above 3000 m, and ≤−19.1% in already-accurate floodplains, with sample-sparse flat terrain extrapolating poorly.
3. **First demonstration, to our knowledge, of cross-regime transferability of an ML FABDEM correction**: the Mediterranean-trained model achieves a further −22.2% RMSE reduction on humid temperate forest *without retraining* (Diebold–Mariano *p* < 0.001). SHAP attribution shifts from drainage features to 3D shape descriptors in a manner consistent with the model capturing physically grounded terrain–error relationships rather than memorising regional patterns — although multi-continental generalisation remains to be tested.
4. **An open artefact release** — corrected COG, training dataset, trained model, and pipeline code (under CC BY-NC-4.0 inherited from FABDEM) — that positions ICESat-2 ATL08 as scalable LiDAR-equivalent ground truth for regional DEM refinement elsewhere.

The remainder of the paper is organised as follows. Section 2 describes the study area, including the contrasting climate regimes and the June 2023 atmospheric-river event that motivated the choice of basins. Section 3 details the data sources. Section 4 specifies the methodology and pipeline. Section 5 presents results, including stratified analysis and the out-of-distribution transferability test. Section 6 discusses where the correction matters and what the SHAP-attribution shifts imply about physical interpretability and scope of transferability. Section 7 concludes and Section 8 lists data and code availability.

---

## 2. Study area

The study area covers ~700,000 km² of central-south Chile between 34°S and 39°S (Figure F1). At these latitudes the country is a narrow ~220 km strip from Pacific coast to the Andean drainage divide, and the steep west–east climatic and topographic gradient creates a natural laboratory for terrain-correction studies: lowland coastal plain, agricultural Central Valley, pre-Andean foothills, and high Andes above 3000 m all occur within each 1° latitude band. We sample this transect with 10 contiguous 1°×1° FABDEM tiles (S35W071 to S39W073), grouped into two contrasting climate regimes for the experimental design.

The **Mediterranean regime** comprises six tiles (S35W071, S35W072, S36W071, S36W072, S37W071, S37W072) spanning the Mataquito and Maule river basins (34–37°S). Climate is Mediterranean *sensu stricto* — dry summers, wet winters concentrated in May–August — with mean annual precipitation increasing from ~400 mm in the north to ~900 mm at the southern boundary. Land cover is a mosaic of intensive agriculture (vineyards, fruit orchards, cereals) in the Central Valley, sclerophyll forest on coastal-range and pre-Andean slopes, and barren or sparsely vegetated alpine terrain above the tree line. The growing-season NDVI distribution is broad with median ≈0.3, reflecting this heterogeneity, with concentrated areas of bare ground (NDVI < 0.1) in the upper cordillera and dense canopy (NDVI > 0.5) only in irrigated valleys and remnant forest patches.

The **humid temperate regime** comprises four tiles (S38W072, S38W073, S39W072, S39W073) spanning the Bío-Bío and Araucanía regions (37–39°S). This is the northern margin of the Valdivian temperate rainforest biome: mean annual precipitation rises to 1,500–3,000 mm with year-round rainfall, and natural vegetation is dense evergreen-broadleaf forest dominated by *Nothofagus* species. Plantation forestry (*Pinus radiata*, *Eucalyptus*) covers a substantial fraction of the coastal range and central valley. Growing-season NDVI distribution is shifted upward, median ≈0.6, with very dense canopy (NDVI > 0.7) common. Topographically the regime mirrors the Mediterranean transect — coastal range, central valley, Andean foothills, high Andes — but with denser and more spatially uniform vegetation cover. The latitudinal separation of ~3° (≈330 km) between the two regimes' centroids and the contrast in vegetation cover (NDVI Δmedian ≈ 0.3) provide a clean test of cross-regime transferability.

The June 2023 atmospheric-river event over central Chile motivates the regional focus. Between 21 and 26 June 2023, an exceptionally intense frontal system — characterised by the Centro de Ciencia del Clima y la Resiliencia as the strongest of the last three decades \citep{Garreaud2024AR2023} — delivered persistent heavy precipitation over the Mediterranean regime in particular, with widespread river overflow in the Mataquito and Maule basins. CIGIDEN documented post-disaster flood extents at Licantén (Mataquito river mouth; \citealt{CIGIDEN2023Licanten, CIGIDEN2023MataquitoMaule}). Although the present paper focuses on DEM correction rather than flood mapping, this event provides direct policy relevance: improvement in DEM accuracy in flood-prone Chilean basins has demonstrable applications, and Section 5.6 uses the CIGIDEN Licantén polygons in a HAND-threshold comparison as a preliminary downstream check.

---

## 3. Data

We use five primary data sources — the bare-earth DEM under correction, the ground-truth altimetry, two satellite remote-sensing sources for feature engineering, and a geoid model for datum harmonisation — plus a derived geomorphometric feature stack computed from the DEM itself. Table T1 summarises the sources; the following subsections detail provenance, processing, and resulting feature count.

### 3.1 FABDEM v1.2

FABDEM v1.2 \citep{Hawker2022FABDEM} is the bare-earth DEM under correction. It is derived from Copernicus DEM GLO-30 (the upstream Digital Surface Model) by removing canopy and building height contributions with a random-forest regression trained on global LiDAR samples. Native horizontal resolution is 1 arcsec (~30 m at our latitudes) on a geographic WGS84 (EPSG:4326) grid, and the vertical datum is orthometric EGM2008. We download the 10 1°×1° tiles via the Google Earth Engine asset `projects/sat-io/open-datasets/FABDEM`, yielding ~137 M pixels across the study area. License is CC BY-NC-4.0, which the corrected product inherits.

### 3.2 ICESat-2 ATL08 v7

ICESat-2 \citep{Markus2017ICESat2} carries the ATLAS photon-counting laser altimeter (532 nm, 10 kHz pulse rate, six beams in three weak/strong pairs). The ATL08 v7 product \citep{Neuenschwander2019ATL08} aggregates ATL03 geolocated photons into ~100-m along-track land segments with a 17 m circular footprint and classifies returns into ground, canopy, and noise. We download all ATL08 v7 granules intersecting each tile between January 2018 and December 2025 via `earthaccess`, extract footprints inside each tile's bounding box, and apply the following quality filters (recommended by the ATL08 ATBD and consistent with prior published DEM-validation work):

- `terrain_flg == 1` (segment passed terrain quality screening);
- `n_te_photons ≥ 30` (sufficient photon support for the terrain estimate);
- `h_te_uncertainty < 10 m` (1-σ along-track uncertainty);
- `segment_snowcover == 1` and `segment_watermask == 0` (no snow, not water);
- `cloud_flag_atm ≤ 1` (clear-sky atmosphere).

After filtering, the retained sample is 77,501 footprints in the Mediterranean regime and 57,849 in humid temperate, for a combined dataset of 135,350 ground-classified footprints. The variable used as truth is `h_te_best_fit` (best-fit terrain elevation), converted from WGS84 ellipsoidal to EGM2008 orthometric heights via the PROJ pipeline (EPSG:4979 → EPSG:9518) using gridded EGM2008 \citep{Pavlis2012EGM2008} with PROJ network mode enabled. The geoid undulation over the study area ranges from approximately +18 to +24 m and applying it removes an apparent +21–22 m baseline offset between raw ATL08 ellipsoidal heights and FABDEM, leaving a residual mean error of approximately +1 m attributable to FABDEM itself.

### 3.3 Sentinel-2 L2A optical composites

We assemble single-season austral summer optical composites (December 2022 to February 2023) from Sentinel-2 L2A \citep{Drusch2012Sentinel2} via the Microsoft Planetary Computer STAC API. For each MGRS tile intersecting the study area we select the three least-cloudy granules with cloud cover below 20%, stack them with `stackstac` at 60 m resolution on the UTM 19S projection (EPSG:32719), and compute the per-pixel median across the time stack for bands B02 (blue), B03 (green), B04 (red), B08 (NIR), B11 (SWIR-1), and B12 (SWIR-2). From these we derive five spectral indices: NDVI (vegetation), NDWI (water/wetness), NDMI (moisture), BSI (bare-soil), and NDBI (built-up/non-vegetation). The 60-m output resolution is a deliberate trade-off: lowering Sentinel-2 from native 10–20 m to 60 m reduces memory footprint by an order of magnitude for the per-tile compute, with minimal information loss for point-sampling at ICESat-2 footprint locations (which are themselves spatial integrations over a 17 m footprint).

### 3.4 Sentinel-1 RTC SAR composites

We assemble parallel C-band SAR backscatter composites from Sentinel-1 RTC products \citep{Torres2012Sentinel1} via Planetary Computer. For each tile we retain up to 12 dual-polarisation (VV + VH) granules from the same December 2022–February 2023 window, compute the per-pixel median, convert to decibels (σ⁰ in dB), and derive the VV/VH ratio. Final outputs at 60 m UTM 19S are three rasters: `s1_vv_db`, `s1_vh_db`, and `s1_vv_vh_ratio`. SAR backscatter complements the optical composite by being weather-independent and sensitive to surface roughness and dielectric properties (vegetation moisture content), which are partially decoupled from optical vegetation greenness.

### 3.5 Geomorphometric and hydrological features (SurtGis)

From each tile's FABDEM raster we derive a stack of geomorphometric and hydrological terrain features using SurtGis \citep{SurtGis2026}, an open-source Rust geospatial library that exposes streaming, parallel implementations of standard algorithms. The two batch operations `surtgis terrain all` and `surtgis hydrology all` produce, in a single pass per tile:

- **Slope, aspect (sin/cos), hillshade** — standard first-derivative quantities;
- **Curvature (general, profile, planform), TPI, TRI, VRM, dev, convergence** — second-derivative and ruggedness descriptors;
- **Openness positive/negative, sky-view factor (SVF)** — 3D form descriptors capturing local concavity/convexity at longer length scales \citep{Yokoyama2002Openness};
- **MRVBF, MRRTF** — Multi-Resolution Valley Bottom / Ridge Top Flatness \citep{Gallant2003MRVBF};
- **Geomorphons** — 10-class landform classification \citep{Jasiewicz2013Geomorphons};
- **HAND** — Height Above Nearest Drainage \citep{Nobre2011HAND}, a drainage-relative elevation quantity widely used in flood-prone areas;
- **TWI** — Topographic Wetness Index \citep{Beven1979TOPMODEL};
- **Flow direction (D8 and D-infinity), flow accumulation (single and multi-flow), filled DEM, stream network** — standard hydrological derivatives;
- **Valley depth** — distance from each pixel to the nearest ridge top elevation.

In total, after de-duplication, this yields 25 geomorphometric and hydrological rasters per tile in the FABDEM native EPSG:4326 / 30 m grid. Each is sampled at footprint locations during the feature-stack assembly step.

### 3.6 Final feature stack

Combining the satellite-derived (5 Sentinel-2 indices + 3 Sentinel-1 SAR statistics) and DEM-derived (25 SurtGis outputs) features yields a **33-feature stack** per footprint after sampling. The regression target is the EGM2008-corrected residual

$$
y = h_{\text{te,orthometric}} - h_{\text{fabdem}}, \quad \text{units in metres,}
$$

with values close to zero indicating an accurate FABDEM pixel and large-magnitude values signalling correctable bias. The sign convention is that positive *y* corresponds to FABDEM underestimating the true ground surface; the empirically observed regional median is −1 m, consistent with the published global tendency of FABDEM to slightly *overestimate* bare-earth elevation in vegetated terrain — likely a residual artefact of the upstream Copernicus DSM canopy bias incompletely removed by the global RF model. Table T1 lists licences and access methods for every source.

---

## Word counts and notes (will be removed in final draft)

- Section 1 (Introduction): **~870 words** — within target of 1.5 pages (~750–900 words at IJDE typesetting density).
- Section 2 (Study area): **~510 words** — within 1-page target.
- Section 3 (Data): **~1,210 words** — slightly long for 2-page target (~1,000–1,100 words). Section 3.5 SurtGis listing can be tightened or moved to a supplementary table if compression is needed.
- **Total 1–3**: **~2,590 words** — slightly above 2,500 target; acceptable.

### Sections not yet written (deferred to next writing session)

- Section 4 (Methods): ~3 pages — pipeline, EGM2008, XGBoost, Optuna, spatial-block CV, SHAP, OOD test, inference
- Section 5 (Results): ~4 pages — overall metrics, stratification (table T4), bias elimination, SHAP, OOD generalisation, HAND comparison
- Section 6 (Discussion): ~3 pages — where it matters, transferability scope, SHAP adaptation interpretation, limitations, FABDEM+ comparison, implications
- Section 7 (Conclusions): ~0.5 page
- Section 8 (Data/code availability): ~0.5 page

### Open questions / decisions still pending

- **Sole authorship vs invited co-author**: outline still says sole; consider inviting (a) hydrologist familiar with Chilean basins for Section 2 review, or (b) ICESat-2 specialist
- **British vs American English**: this draft uses British; convert if NHESS submission preferred (Copernicus journals use British)
- **Whether to include a feature-ablation supplementary table**: would address one MEDIUM reviewer concern from tex-review
- **Whether to attempt Hawker outreach pre-submission**: the existence of `Hawker2024Vietnam` (validated in references) makes this more attractive — same group is actively publishing on FABDEM validation
