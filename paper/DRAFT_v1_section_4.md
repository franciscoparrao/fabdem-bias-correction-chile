# Draft v1 — Section 4 (Methods)
**Continuation of `DRAFT_v1_sections_1_to_3.md`**
**Target**: ~1,500 words / 3 pages

---

## 4. Methods

### 4.1 Pipeline architecture

The processing pipeline (Figure F2) is organised as a tile-parallel sequence of seven stages: (i) FABDEM tile retrieval, (ii) geomorphometric and hydrological feature derivation, (iii) Sentinel-1/2 composite assembly, (iv) ICESat-2 ATL08 granule acquisition, (v) footprint extraction and datum correction, (vi) feature-stack sampling at footprint locations, and (vii) regression target assembly. Stages (i)–(vi) execute independently per 1° tile inside isolated process trees; their outputs (per-tile sample tables) are concatenated at the end. Each stage is checkpointed: if a partial output already exists with valid metadata, the stage is skipped on re-execution, which makes the pipeline robust to mid-run interruptions and incremental data additions. A `psutil`-based watchdog monitors combined RAM of the running stage plus all child processes and aborts the stage cleanly if a configurable threshold (here 8 GB) is exceeded, preventing the long-running pipeline from competing for swap with the operating system. The watchdog never fired in production runs after the satellite-stage RAM optimisation described in Section 4.4 was applied.

### 4.2 Footprint extraction and EGM2008 datum correction

For each tile we download every ATL08 v7 granule whose bounding polygon intersects the tile geometry, then iterate through the six ATLAS beams (`gt1l`–`gt3r`). For each land-segment record we retain the longitude, latitude, best-fit terrain height (`terrain/h_te_best_fit`), per-segment uncertainty (`terrain/h_te_uncertainty`), and the quality flags listed in Section 3.2. Quality filtering retains records satisfying simultaneously: `terrain_flg = 1`, `n_te_photons ≥ 30`, `h_te_uncertainty < 10 m`, `segment_snowcover = 1`, `segment_watermask = 0`, `cloud_flag_atm ≤ 1`. These thresholds follow Neuenschwander and Pitts \citeyearpar{Neuenschwander2019ATL08} for surface terrain analysis and screen out water bodies, snow-covered surfaces, cloud-contaminated retrievals, and low-photon-count segments where the best-fit elevation is poorly constrained.

ATL08 reports heights above the WGS84 ellipsoid, whereas FABDEM uses orthometric heights referenced to the EGM2008 geoid \citep{Pavlis2012EGM2008}. Failing to harmonise the two leaves an apparent ~+22 m baseline offset that would entirely dominate the regression target and silently corrupt downstream analysis. We apply the conversion per footprint using the PROJ pipeline from EPSG:4979 (WGS84 3D ellipsoidal) to EPSG:9518 (WGS84 horizontal + EGM2008 vertical), with PROJ network mode enabled so that the EGM2008 grid is fetched on-demand. The geoid undulation varies by approximately 6 m across the latitudinal extent of the study area, so the correction is per-point rather than a constant offset. The corrected heights, denoted *h_te,ortho*, define the regression target as the signed residual *y = h_te,ortho − h_fabdem* in metres.

### 4.3 Feature-stack assembly with CRS-aware sampling

FABDEM and its derived terrain features (Section 3.5) reside on a geographic grid (EPSG:4326, ~30 m), while the Sentinel-1/2 composites (Section 3.3–3.4) reside on a projected grid (EPSG:32719 UTM 19S, 60 m). For each footprint we reproject the (lon, lat) coordinate pair into each source raster's native CRS and sample the underlying pixel value with nearest-neighbour interpolation via `rasterio`. This avoids unnecessary raster reprojection at the per-pixel scale, which would propagate resampling artefacts into the feature values. One feature, `stream_network`, is a sparse binary indicator produced by the SurtGis hydrology stream-extraction; for non-stream cells the SurtGis output is NaN, which is semantically equivalent to "no stream present" and is therefore reassigned to zero before regression to retain its informational value as a present/absent indicator.

### 4.4 XGBoost regression with Optuna hyperparameter tuning

We train gradient-boosted decision trees with XGBoost \citep{Chen2016XGBoost} on the 33-feature stack to predict the residual target *y*. Hyperparameters are tuned by Optuna \citep{Akiba2019Optuna} using a Tree-structured Parzen Estimator (TPE) sampler over 100 trials, with the objective function defined as the mean spatial-cross-validated out-of-fold RMSE (Section 4.5). The search space covers `max_depth` ∈ [3, 9], `learning_rate` ∈ [0.005, 0.2] (log scale), `subsample` ∈ [0.5, 1.0], `colsample_bytree` ∈ [0.5, 1.0], `min_child_weight` ∈ [1, 20], `reg_lambda` ∈ [10⁻³, 10²] (log scale), `reg_alpha` ∈ [10⁻⁴, 10] (log scale), and `gamma` ∈ [0, 5]. The number of boosting rounds is capped at 1,000 with early stopping after 30 rounds without validation-RMSE improvement. The final selected hyperparameters (Table T2) describe deep trees (`max_depth = 9`) coupled with aggressive regularisation (`min_child_weight = 14`, `reg_lambda ≈ 7.4`, `gamma ≈ 2.2`) and a low learning rate (≈ 0.011), a combination consistent with a smooth target manifold that benefits from depth but is prone to overfit under standard regularisation choices.

The regression operates on raw, untransformed features. XGBoost handles missing values internally by routing them along the default branch learned during split selection, which is convenient because Sentinel-1/2 composite pixels can be NaN where all input scenes are cloud-flagged. We rely on this behaviour rather than imputing values that would inject false information.

### 4.5 Spatial-block cross-validation

Per-pixel cross-validation with random folds is inappropriate here because ICESat-2 ATL08 footprints exhibit strong along-track spatial autocorrelation: adjacent 100 m segments share atmospheric conditions, drainage context, and reference orbit, making nearby footprints near-duplicates from the model's perspective. Random partitions would assign these clusters to both train and test sets and yield optimistic generalisation estimates \citep{Roberts2017SpatialCV}. We therefore use **spatial-block cross-validation**: footprints are projected to UTM 19S, integer-divided by 10,000 m to assign a 10 km × 10 km block index, and these block indices are used as group labels in scikit-learn's `GroupKFold` partitioning with *K* = 5 folds. The procedure ensures that all footprints within a 10 km tile are kept in the same fold, so that the model is always evaluated on geographically distinct ground from where it was trained. Across the Mediterranean dataset, 570 unique blocks are created from the 77,501 footprints, giving approximately 114 blocks per fold (with some imbalance from variable block density). To quantify the magnitude of autocorrelation leakage we also report random *K*-fold CV (which loses the spatial constraint) for direct comparison; the gap is small (3.5 % RMSE inflation under random CV), indicating that 10 km blocks are large enough to suppress local-track contamination without being so large that they remove genuine model capacity. A 5 km versus 10 km sensitivity test (Section 8, supplementary) yields RMSE differences below 0.05 m, supporting the 10 km choice as primary reporting block size.

### 4.6 Feature attribution via SHAP

Per-feature importance is computed using the SHAP TreeExplainer \citep{Lundberg2017SHAP}, which exploits the tree structure to compute exact Shapley values in polynomial time. To keep memory and computation tractable on the full 77,501-row Mediterranean training set we draw a uniform random sample of 8,000 footprints (seed 42) and compute SHAP values for the final, hyperparameter-tuned model trained on the complete training set. Reported feature importance is the mean absolute SHAP value across that sample, which preserves both the magnitude and the rank of feature contributions while remaining computationally inexpensive. The same procedure is repeated on a separate 8,000-row sample drawn from the humid temperate dataset to enable the cross-regime comparison reported in Section 5.

### 4.7 Out-of-distribution evaluation

To test whether the regional model generalises beyond its training regime, we apply the Mediterranean-trained XGBoost model directly to the humid temperate dataset without any retraining or fine-tuning. Specifically, we load the booster fit on the complete 77,501-footprint Mediterranean training set (the same model used for inference and SHAP attribution) and predict residuals for the 57,849 humid temperate footprints. The corrected elevation is computed as *h_corrected = h_fabdem + ŷ* and compared against the ATL08 ground truth via RMSE, MAE, signed bias, and per-pixel sign comparison. This is a true out-of-distribution test in the strict statistical sense: no humid temperate footprint or feature ever enters the training pipeline. Statistical significance of the squared-error reduction is assessed with a Diebold–Mariano test on the per-footprint squared-error differential, and per-pixel improvement frequency is tested with a one-sided binomial test against the null hypothesis of equal correction/degradation rates.

### 4.8 Raster-wide inference

To produce the distributable corrected DEM (Section 8) we apply the trained model to every pixel of the FABDEM raster across the six Mediterranean tiles. Each feature raster is opened lazily; for those already on the FABDEM grid (terrain and hydrology layers) values are read directly, while satellite layers in EPSG:32719 are reprojected on-the-fly to the FABDEM grid using bilinear resampling with `rasterio.warp.reproject`. A per-tile dense feature stack is assembled in float32, fed to the XGBoost booster via `xgb.DMatrix` with explicit `missing = NaN`, and the predicted residual is added to the corresponding FABDEM pixel to produce the corrected elevation. Output is written as a tiled Cloud-Optimized GeoTIFF with `deflate` compression, predictor 2, internal block size 512 × 512, and average-resampled overviews at decimation levels [2, 4, 8, 16, 32]. The six per-tile COGs are mosaicked into a single 7,422 × 11,133 raster (`mm_corrected.tif`, 282 MB) using SurtGis `mosaic`, which preserves the COG structure of the inputs.

### 4.9 Software, versions, and reproducibility

The Python environment used throughout is: Python 3.12, XGBoost 3.1.1, Optuna 4.6.0, scikit-learn 1.8.0, SHAP 0.50.0, rasterio 1.4.x, pyproj 3.6.x (PROJ network mode enabled for the EGM2008 grid), `earthaccess` 0.18.0 for ATL08, `pystac-client` and `stackstac` for Planetary Computer access. Terrain and hydrology features are computed with SurtGis 0.7.0 \citep{SurtGis2026}, an open-source Rust library that exposes streaming, parallel implementations of standard geomorphometric algorithms with bounded memory per tile. A fixed random seed (42) is used for the Optuna sampler, all train/test partitions, the SHAP subsampling, and the 2,000-resample bootstrap that produces the confidence intervals reported in Section 5. The complete pipeline, including the tile orchestrator with checkpointing and watchdog, is released under an open licence at the repository given in Section 8.

---

## Word counts

- 4.1: ~165 words
- 4.2: ~280 words
- 4.3: ~165 words
- 4.4: ~265 words
- 4.5: ~310 words
- 4.6: ~135 words
- 4.7: ~160 words
- 4.8: ~190 words
- 4.9: ~140 words

**Total Section 4: ~1,810 words** (target was 1,500; slightly over but within tolerable range for 3-page methods section in IJDE format).

### Notes for next writing session

- Section 5 (Results) — next priority. Has all numbers ready in `uncertainty_stats.json`, `mm_metrics.json`, stratification CSVs.
- Could trim 4.5 if word budget is tight (currently the longest subsection); the spatial-CV rationale paragraph can be compressed.
- Wilcox2016Atacama still unused (could fit in Section 6 Discussion as historical Chilean flood reference).
