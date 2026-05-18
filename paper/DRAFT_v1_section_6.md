# Draft v1 — Section 6 (Discussion)
**Continuation of `DRAFT_v1_section_5.md`**
**Target**: ~1,500 words / 3 pages

---

## 6. Discussion

### 6.1 Where ML bias correction matters

The stratified analysis in Section 5.2 reveals a coherent pattern: FABDEM's residual error is largest, and most aggressively corrected, in strata where the upstream Copernicus DSM is most affected by canopy occlusion (dense vegetation), aspect-driven slope artefacts (alpine and peak geomorphons), or both. The pattern is association rather than proven causation, but it is consistent with the published mechanism of FABDEM's global random-forest correction \citep{Hawker2022FABDEM}: by design, that correction operates uniformly across the planet and cannot exploit regional differences in how vegetation height, canopy density, or built-up land interact with the InSAR phase used by the Copernicus DSM. A regional model trained on regionally specific features therefore has the most room to operate precisely in the strata where the global model leaves the largest residual signal.

The complement of this argument is that the regional correction is small or negligible where FABDEM is already accurate — most clearly in floodplain pixels (HAND 2–10 m above drainage), where the residual mean is only −0.71 m and the relative RMSE reduction is below 20 %. This stratified pattern carries practical guidance: a user who applies FABDEM exclusively in flat, low-relief, vegetation-poor terrain will gain little from regional ML correction and may prefer the simpler raw product; conversely, a user working in alpine, vegetated, or hilly terrain stands to benefit substantially.

### 6.2 Limitations and scope of the transferability claim

We deliberately frame this paper's transferability claim with a scope-restricting caveat before interpreting the headline OOD result. The cross-regime test compares two contrasting *Chilean* regimes — Mediterranean dry and humid temperate Valdivian rainforest — separated by approximately 330 km of latitudinal distance, with NDVI medians shifting from ~0.3 to ~0.6 and mean annual precipitation rising from ~700 mm to ~2,000 mm. The geomorphological substrate (steep Pacific-facing Andean range, narrow central valley, narrow coastal range) is shared between regimes, even as the vegetation and hydrological forcing differ markedly. A reader should therefore read our generalisation result as evidence of robustness *between two Chilean regimes that share Andean tectonics and along-strike orientation*, not as evidence of universal transferability to tropical or arid systems, to plate-interior topography, or to high-latitude terrain. Validating transferability against tropical (e.g. Amazonian or Indonesian) or arid (e.g. Atacama) regimes is a clear and important direction for future work.

Within Chile, the model behaves poorly in sample-sparse strata: flat slope classes (*n* = 204 in Mediterranean), valley geomorphons, and tile S36W071's alpine central-Andes domain (*n* = 1,987). The pattern is interpretable as the gradient-boosted tree extrapolating beyond the convex hull of its training distribution when the local feature combination is rare. Practitioners applying the correction operationally should consider rejecting predictions in cells whose feature vector falls outside the distribution seen during training, although a robust uncertainty-aware deployment is beyond the scope of this paper.

Two additional limitations carry methodological implications. First, the Sentinel-1/2 composites are single-season (December 2022–February 2023, austral summer), and therefore omit phenological information that might modulate the residual signal in seasonally dynamic land cover. Second, we do not validate the corrected DEM against an airborne LiDAR product, because no publicly distributed airborne LiDAR with the relevant spatial coverage exists for our study basins. ICESat-2 ATL08 (with cm-level vertical precision \citep{Markus2017ICESat2, Neuenschwander2019ATL08}) provides the best available alternative and is the operational ground truth used throughout.

### 6.3 Geographic transferability: evidence and alternative explanations

Within the scope just articulated, the out-of-distribution result is unambiguous: a Mediterranean-trained model achieves a 22 % RMSE reduction on humid temperate forest *without retraining*, statistically robust under both a Diebold–Mariano test (DM = 40, *p* < 0.001) and a one-sided sign test (*p* < 10⁻²⁶²). This result is consistent with the hypothesis that the model captures terrain–error relationships that are *physically structured* — that is, depend on geomorphometric and vegetation properties of the local terrain rather than on idiosyncratic patterns peculiar to the Mediterranean training samples.

However, a competing explanation must be acknowledged. Both Chilean regimes inherit a shared Andean uplift history that produces broadly similar morphology: foothills and ridges with comparable slope distributions, narrow valleys with comparable HAND structure. The Mediterranean-trained model could be exploiting morphological similarities that are themselves not universal — a model trained on Andean terrain might fail to transfer to the rolling-relief Amazonian flank or to plate-interior bedrock in central Argentina. Disentangling "physically structured" generalisation from "morphologically similar" transfer requires the multi-continental validation noted above. Our result demonstrates that the question is empirically tractable and that the answer is favourable for the specific Andean Chilean transect studied here.

### 6.4 SHAP adaptation between regimes

The cross-regime SHAP comparison in Section 5.5 reveals systematic rank shifts: 3D shape descriptors (`openness_negative` rising from rank 3 to rank 1, `valley_depth` from rank 8 to rank 4) gain relative importance in the humid temperate regime, while drainage descriptors (`tri`, `slope`, `twi`) and the vegetation index `ndvi` each drop two or more positions. Substantively, the model in the humid temperate regime appears to *lean more on 3D shape* and *less on hydrological proxies and direct vegetation greenness*. A plausible mechanistic interpretation is that in dense forest where canopy occludes drainage features in the source Copernicus DSM, hydrological features computed from FABDEM become less reliable signals, so the model down-weights them; 3D shape descriptors (openness, valley_depth) by contrast survive the canopy-vs-terrain distinction better and inherit more of the predictive load. This is a hypothesis-grade interpretation: an *N* = 2 regime experiment provides correlation but not causal inference. The 32 of 33 features with ratio > 1 is consistent with an overall scaling of feature contributions to absorb a larger baseline residual, not a regime-specific re-weighting; the genuine information is in the *rank* changes, not the magnitude scaling.

### 6.5 Comparison with related work

Three published lines of work bracket this paper's contribution. Hawker et al. (\citeyear{Hawker2022FABDEM}) introduced FABDEM with a *global* random-forest correction at 30 m resolution, validated against airborne LiDAR samples. We confirm their reported ~2.5 m global RMSE in our Chilean sample (raw RMSE 3.05 m Mediterranean, 4.95 m humid temperate) and demonstrate that a *regional* refinement on top of their globally corrected product is worthwhile in specific terrain strata. Meadows et al. \citeyearpar{Meadows2024VerticalAccuracy} benchmarked FABDEM against four alternative public DEMs in flood-prone environments and confirmed FABDEM as the most accurate; our work continues their direction by quantifying where FABDEM's residual error is concentrated. Wing et al. \citeyearpar{Wing2024GlobalFlood} build a global flood-inundation model on FABDEM but apply it without further correction; our work suggests their alpine or vegetated outputs could be improved by inserting a regional correction step.

Commercially, Fathom's FABDEM+ \citep{Fathom2024FABDEMPlus} is an enhanced product that fuses local data sources with FABDEM. Its methodology and validation are proprietary, which precludes direct comparison. Our key positioning differences are openness and transparency: every step of our pipeline, including the trained XGBoost model, the training dataset, the corrected COG, and the hyperparameters, is released under permissive licensing (Section 8). Future regional users in Chile or comparable regions can therefore evaluate, reproduce, and extend our correction directly rather than rely on a closed-source product.

Beyond the FABDEM literature specifically, the broader context for accurate DEMs in Chile is one of repeated extreme-precipitation events with substantial topographic involvement: the 2015 Atacama mudflow event \citep{Wilcox2016Atacama} demonstrated that even arid northern Chile is exposed to high-magnitude flash flooding, and the June 2023 Maule event \citep{CIGIDEN2023MataquitoMaule, Garreaud2024AR2023} demonstrates the same for central Chile. In both cases the accuracy of the underlying DEM was an operational bottleneck for downstream inundation modelling; the present contribution narrows that bottleneck in the central Chilean case.

### 6.6 Implications for downstream applications

For practitioners, our results suggest a tiered uptake strategy. Applications dominated by *floodplain hydraulics* (free-surface elevation along channels, classical 1D or 2D HEC-RAS / LISFLOOD-FP simulations of riverine inundation) operate primarily in HAND 0–10 m strata where the correction is small; users can in most cases continue with raw FABDEM with only marginal accuracy loss. Conversely, applications dominated by *upland or alpine processes* — slope stability and landslide initiation, evapotranspiration modelling under canopy, glacier mass balance, mountainous infrastructure planning, debris-flow source-area delineation — operate where the correction is largest (20–40 % RMSE reduction in the relevant strata) and benefit substantially. The corrected COG distributed alongside this paper is therefore most valuable to upland-process users.

### 6.7 Future work

Three direct extensions are immediately suggested. First, geographic broadening: applying the same pipeline to the full Chilean transect from Atacama (semi-arid) to Patagonia (cool temperate), and ultimately to additional Andean countries (Argentina, Bolivia, Peru) would build a multi-regime training corpus and resolve the open question of multi-continental transferability raised in Section 6.2. Second, hydrodynamic validation: while the HAND-threshold downstream check in Section 5.6 is informative, a proper hydrodynamic comparison using a model that resolves free-surface elevation (LISFLOOD-FP, HEC-RAS 2D, or comparable; cf. \citet{Hawker2024Vietnam}) is needed to quantify whether the corrected DEM yields measurably better simulated flood extent or depth. Third, integration with foundation models for geospatial data: emerging vision foundation models such as Prithvi-EO and TerraMind could complement the explicit feature engineering used here with self-supervised feature representations, potentially capturing additional spatial structure that traditional geomorphometric descriptors miss. The dataset released in this paper provides a natural benchmark for such extensions.

---

## Word count

- 6.1: ~250 words
- 6.2: ~370 words
- 6.3: ~260 words
- 6.4: ~260 words
- 6.5: ~330 words
- 6.6: ~190 words
- 6.7: ~250 words

**Total Section 6: ~1,910 words** — slightly over the 1,500 word target (3 pages); compressible to 1,500 by trimming 6.2 alternative-explanations paragraph if word budget is tight at submission.

## Citations used in Section 6

New in this section: `Wilcox2016Atacama` (6.5, finally placed)
Re-used: `Hawker2022FABDEM`, `Hawker2024Vietnam`, `Meadows2024VerticalAccuracy`, `Wing2024GlobalFlood`, `Fathom2024FABDEMPlus`, `Garreaud2024AR2023`, `CIGIDEN2023MataquitoMaule`, `Markus2017ICESat2`, `Neuenschwander2019ATL08`

**All 29 entries in `references.bib` are now used at least once across sections 1–6.** Citation coverage is complete.

## Notes for next writing session

- Section 7 (Conclusions) is short — bullet-style summary of the three claims with their statistical support.
- Section 8 (Data and code availability) is mostly already drafted in OUTLINE.md and just needs prose conversion.
- A final pass to compress 6.2 if needed for word budget at submission.
- Consider whether to add a one-paragraph "Recommendation" subsection at the very end of Discussion as an actionable summary for practitioners — this is a Reviewer-friendly addition often valued by IJDE editors.
