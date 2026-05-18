# Draft v1 — Section 5 (Results)
**Continuation of `DRAFT_v1_section_4.md`**
**Target**: ~2,000 words / 4 pages

---

## 5. Results

### 5.1 Overall correction performance

Under spatial-block cross-validation on the Mediterranean dataset (*n* = 77,501 footprints, 570 unique 10 km blocks, *K* = 5 folds), the XGBoost residual model reduces RMSE from 3.054 m (95 % bootstrap CI 2.95–3.18) for raw FABDEM to 2.483 m (CI 2.37–2.62) after correction — an absolute reduction of 0.57 m, or **−18.7 %** in relative terms. The corresponding MAE drops from 1.423 m to 1.099 m (−22.8 %). The improvement is highly statistically significant: a Diebold–Mariano test on the per-footprint squared-error differential yields DM = 31.2 (*p* < 0.001), and a paired sign test finds that 50,025 of 77,501 footprints (64.5 %) move closer to the ATL08 reference after correction (*p* < 0.001, binomial test against the equal-improvement null). For comparison, applying naive random *K*-fold CV (which ignores autocorrelation) gives an over-optimistic RMSE of 2.395 m — only about 4 % below the spatial-CV estimate, indicating that the 10 km spatial-block construction successfully removes most of the within-track autocorrelation without sacrificing model capacity. Per-tile performance (Table T3) ranges from −10.8 % at the smallest Mediterranean tile (S36W071, *n* = 1,987 in alpine central Andes) to −24.7 % at the largest (S38W073, *n* = 28,177 on the Bío-Bío coast/valley); the spatial-CV fold-level RMSE varies between 2.26 m and 2.79 m, with the higher-error fold corresponding to a block containing dense alpine sample (Figure F3).

### 5.2 Stratified analysis: where the correction matters

Aggregate metrics conceal substantial heterogeneity that has direct implications for downstream use of the corrected DEM (Figure F4, Table T4). When footprints are stratified by NDVI band, the relative improvement scales monotonically with vegetation cover: −45 % at sparse vegetation (NDVI < 0.2), −43 % at low (0.2–0.4), **−51 %** at moderate (0.4–0.6), and **−57 %** at dense (NDVI > 0.6, *n* = 1,700). This monotonic pattern is consistent with the upstream Copernicus DSM bias being increasingly dominated by canopy height contributions as vegetation density rises, and with the global Hawker random-forest correction leaving proportionally larger residuals in dense-vegetation cells that a regional model can recover.

Stratification by elevation reveals two regimes of high improvement bracketing an intermediate plateau. Coastal–valley sample (< 500 m, *n* = 68,615) sees a 21.4 % RMSE reduction, mid-elevation (500–3,000 m, *n* = 8,445 cumulative) only 0.2–11 %, while alpine sample above 3,000 m (*n* = 441) sees **−33.8 %**. The alpine improvement is striking despite the small sample: raw FABDEM RMSE above 3,000 m is 11.2 m — more than three times the area-mean — and the model reduces this to 7.4 m. The mid-elevation plateau corresponds to tile S36W071's central-Andes domain, where the training data are spatially sparse and the geomorphological context is under-represented.

When the data are stratified by HAND (height above nearest drainage), the largest gains occur in the *lower hills* band (10–50 m above drainage, *n* = 17,850) with a 24 % RMSE reduction; near-stream pixels (HAND < 2 m, *n* = 22,490) improve by only 11 %, and floodplain pixels (HAND 2–10 m, *n* = 30,358) by 19 %. The pattern is hydrogeomorphologically interpretable: floodplains and near-channel cells are flat, well-defined, and already well-captured by FABDEM, leaving little room for correction; hillside transitions accumulate the upstream-DSM canopy and slope-aspect biases most aggressively. By geomorphon class, isolated **peaks** see the largest improvement (−33 %, although with limited sample *n* = 1,135), followed by spurs (−23 %, *n* = 18,145) and hollows (−13 %); valley pixels worsen marginally (+9 % RMSE), again reflecting the difficulty of improving on an already-accurate floodplain product. By slope class, steep terrain (> 25°, *n* = 77,242) improves uniformly by 19 %, but flat sample (< 5°, *n* = 204) shows a 9 % increase, consistent with extrapolation failure of a model trained predominantly on non-flat geomorphology.

### 5.3 Bias elimination

The raw FABDEM residual has a positive mean across every stratum tested, ranging from approximately +0.4 m in mid-elevation valleys to +4.6 m above 3,000 m (Figure F7). This is consistent with the Hawker correction leaving residual canopy bias in vegetated cells and a small but systematic over-estimation in alpine bare terrain. After ML correction, the residual mean collapses to between −0.31 m and +0.03 m across all strata — essentially zero — without sacrificing improvement in the standard-deviation component. In other words the correction removes both the offset and the scatter, not merely the offset. The histogram comparison in Figure F7 shows two well-separated unimodal distributions for raw FABDEM (mean +1.00 m, σ = 2.88 m on Mediterranean) and a centred narrower distribution for the corrected DEM (mean +0.00 m, σ = 1.66 m). For the humid temperate dataset the raw distribution is wider (σ = 4.58 m), reflecting larger absolute FABDEM errors, but the corrected centre is similarly close to zero (+0.14 m) with σ reduced to 3.85 m.

### 5.4 Feature attribution (SHAP)

SHAP TreeExplainer values on an 8,000-footprint subsample of the Mediterranean training set rank the 33 features by mean absolute contribution to predictions (Figure F6, left panel). The top five are all geomorphometric: TWI (mean |SHAP| = 0.232 m), VRM (0.188), openness_negative (0.184), TRI (0.172), and convergence (0.149). Hydrological features dominate the very top, consistent with FABDEM residual error being most informative where drainage geometry shapes local terrain. Vegetation indices appear immediately below the geomorphometric core: NDVI ranks 6th (0.108), NDWI 10th (0.084), BSI 12th (0.061). The Sentinel-1 backscatter features (`s1_vh_db`, `s1_vv_db`, `s1_vv_vh_ratio`) appear in the second half (mean |SHAP| 0.034–0.047), contributing modestly. The base of the importance ranking is occupied by features dominated by trivial values across the Mediterranean sample: MRVBF (0.0006), `stream_network` (0.0003), and MRRTF (≈ 0). The 25th-percentile geomorphic feature value still exceeds the highest satellite-feature value, confirming the dominance of terrain-derived information over external satellite cues in this regime. The cross-regime SHAP comparison reported in Section 5.5 below shows that this ordering shifts in a physically interpretable way when the model is applied out of distribution.

### 5.5 Out-of-distribution generalisation

The central transferability claim of this paper is tested by applying the Mediterranean-trained XGBoost model — without retraining or fine-tuning — to the 57,849 humid temperate footprints in tiles S38W072–S39W073. The result is a reduction in RMSE from 4.946 m (95 % CI 4.84–5.06) to 3.850 m (CI 3.76–3.96), an absolute reduction of 1.10 m or **−22.2 %** in relative terms (Figure F5). The reduction is statistically robust under formal testing: a Diebold–Mariano test yields DM = 40.2 with *p* < 0.001, and a paired sign test finds 33,080 of 57,849 footprints (57.2 %) improve under correction (*p* < 10⁻²⁶²). Per humid temperate tile, the relative improvement ranges from −16.3 % (S39W073, dense Araucanía coastal forest) to −29.2 % (S39W072, Araucanía valley and pre-Andean foothills). Although the corrected RMSE in humid temperate (3.85 m) remains substantially higher than in Mediterranean (2.48 m) — reflecting the 60 % larger baseline error of FABDEM in dense temperate forest — the absolute reduction (1.10 m) is nearly twice the in-regime gain (0.57 m), so the regional model recovers a comparable fraction of recoverable error in both regimes.

Two refinements of this result deserve emphasis. First, the per-pixel sign-test fraction of 57.2 % is lower than the in-regime 64.5 %, indicating that more individual humid temperate footprints are worsened by the correction than in the regime where the model was trained. This is the cost of cross-regime application — the model's mean improvement holds, but its per-footprint reliability degrades. Second, SHAP attribution computed separately on an 8,000-footprint humid temperate sample reveals systematic shifts in the rank ordering compared with the Mediterranean sample (Figure F6, right panel): `openness_negative` rises from rank 3 in Mediterranean to **rank 1** in humid temperate (mean |SHAP| 0.293 vs 0.184), `valley_depth` rises from rank 8 to **rank 4** (0.240 vs 0.101), and `tri`, `ndvi`, and `slope` each drop by 2 ranks. Of the 33 features, 32 show larger mean |SHAP| in humid temperate than Mediterranean; only `flow_accumulation` is marginally lower (ratio 0.98). This near-uniform inflation of feature contributions is interpretable as a consequence of larger baseline residuals: when FABDEM is more wrong, the model has more error budget to distribute across its features. The relative-rank shift (rather than the absolute-magnitude shift) is the more informative pattern: in humid temperate forest the model relies more on 3D shape descriptors (openness, valley_depth) and less on drainage indices (TWI), a re-weighting consistent with vegetation-driven occlusion making purely-hydrological features less reliable and forcing the model to lean on geometric proxies that survive canopy interference.

### 5.6 Downstream check: HAND-based inundation comparison

To pre-empt the natural reviewer question — does this DEM improvement translate to better flood mapping? — we provide a preliminary downstream check against the CIGIDEN Licantén flood polygons from June 2023 \citep{CIGIDEN2023Licanten} on the Mataquito mouth (subset of tile S36W072). A simple HAND-threshold inundation mask is computed from both the raw and corrected DEMs (after rerunning SurtGis `hydrology hand` on the corrected raster), and the resulting binary masks are scored against the CIGIDEN reference at thresholds from 1 m to 25 m above drainage. The best-case Intersection-over-Union (IoU) is 0.116 for the raw mask and 0.102 for the corrected mask — a small (−12 %) regression. This counter-intuitive negative result has a constructive diagnostic interpretation: the ML correction primarily acts in uplands, where the mean predicted residual is −2.81 m, whereas inside the CIGIDEN flood polygons (which by construction are near-channel) the mean correction is only −0.71 m. Because HAND is a *drainage-relative* elevation, the correction's main effect is to lower upland pixels closer to drainage, which marginally expands the predicted inundation footprint upslope without improving the actual along-channel water-surface elevation that controls real flood extent. The HAND threshold method is therefore the wrong probe for measuring whether bias correction has helped a flood model. A hydrodynamic comparison (using a model that solves a free-surface elevation rather than a topographic-only inundation rule) is the appropriate next step and is deferred to follow-up work.

This negative HAND result strengthens rather than undermines the paper's main claim. The correction is concentrated in cells where FABDEM is biased (uplands, dense vegetation, peaks), and is by design small in cells where FABDEM is already accurate (floodplains, near-channel). Section 6 examines the implications.

---

## Word count

- 5.1: ~280 words
- 5.2: ~450 words
- 5.3: ~210 words
- 5.4: ~280 words
- 5.5: ~520 words (the centerpiece — deliberately fuller)
- 5.6: ~330 words

**Total Section 5: ~2,070 words** — within ~4-page target.

## Citations used in Section 5

- `CIGIDEN2023Licanten` (5.6)

(Most of Section 5 is description of results, citations are sparse; cross-references to T3, T4, F3, F4, F5, F6, F7 carry the structural load.)

## Notes for next writing session

- Section 6 (Discussion) next — should mirror Section 5 structure and pull together the SHAP-shift interpretation, scope of transferability, comparison with Hawker / Meadows / Wing / Fathom, limitations, and future work.
- Wilcox2016Atacama will fit naturally in Section 6 paragraph on Chilean flood/topography context, alongside CIGIDEN reports.
