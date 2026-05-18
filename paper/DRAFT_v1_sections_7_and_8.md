# Draft v1 — Sections 7 + 8 (Conclusions + Data availability)
**Continuation of `DRAFT_v1_section_6.md`**
**Target**: ~500 words / 1 page combined

---

## 7. Conclusions

We have shown that FABDEM — the most accurate publicly available 30-m DEM at the time of writing — has a *learnable, spatially heterogeneous* residual bias over central-south Chile, and that a regional XGBoost correction trained on 135,350 ICESat-2 ATL08 ground-classified footprints can reduce that bias significantly. Three results substantiate the contribution.

First, under spatial-block cross-validation (10 km blocks, *K* = 5), the correction reduces RMSE from 3.054 m (95 % CI 2.95–3.18) to 2.483 m (CI 2.37–2.62) on the Mediterranean training regime — an absolute reduction of 0.57 m or −18.7 %, statistically significant under both a Diebold–Mariano test (DM = 31, *p* < 0.001) and a paired sign test (64.5 % of footprints improved, *p* < 0.001). The correction also eliminates a systematic +1.0 m positive bias that pervades the raw product, leaving residual mean errors within ±0.05 m across all major strata.

Second, the magnitude of correction is strongly stratified: −41.6 % RMSE in dense vegetation (NDVI > 0.6), −33.8 % above 3,000 m elevation, and −33.0 % at peak geomorphons, contrasted with only −19.1 % in floodplains where FABDEM is already accurate. This pattern provides actionable guidance: regional correction is most valuable for upland, alpine, and vegetated applications, and marginal for purely floodplain-hydraulic uses.

Third, the Mediterranean-trained model transfers to humid temperate Valdivian rainforest *without retraining*, achieving a 22.2 % RMSE reduction on 57,849 held-out footprints across 330 km of latitudinal separation and a 60 % shift in NDVI distribution (DM = 40, *p* < 0.001). The accompanying SHAP feature-importance comparison shows the model emphasising 3D shape descriptors (openness, valley_depth) relatively more in dense forest and drainage descriptors (TWI) relatively less — a pattern *consistent with* the model capturing physically structured terrain–error relationships rather than memorising regional training patterns, although multi-continental validation remains a clear next step.

The work positions ICESat-2 ATL08 as a scalable, cm-precision LiDAR-equivalent ground truth for regional DEM refinement and demonstrates that the open, reproducible workflow developed here is transferable to other geomorphometrically similar regions worldwide.

---

## 8. Data and code availability

All artefacts described in this paper are released as open products to support reproducibility and downstream use.

**Primary outputs (Zenodo, persistent DOI)**:
- *Corrected DEM COG* covering Mataquito + Maule (6 tiles, ~80 M pixels, 282 MB), released under **CC BY-NC-4.0** (inherited from FABDEM \citep{Hawker2022FABDEM}): [Zenodo DOI to be assigned at submission].
- *Training dataset* — 135,350 footprints with 33 features, EGM2008-corrected ICESat-2 ground truth, and per-tile metadata in a single Parquet file (~62 MB): [Zenodo DOI].
- *Trained XGBoost model* in both portable booster JSON and scikit-learn joblib formats, with bundled hyperparameter manifest: [Zenodo DOI].

**Code**: The complete pipeline (tile orchestrator with checkpointing and watchdog, feature engineering, training, SHAP analysis, raster inference, statistical testing) is released on GitHub at [URL to be assigned]. A tagged release matching the manuscript revision is preserved on Zenodo with a software DOI. A Dockerfile and `environment.yml` capture the exact dependency tree.

**Software versions**: Python 3.12, XGBoost 3.1.1, Optuna 4.6.0, scikit-learn 1.8.0, SHAP 0.50.0, rasterio 1.4.x, pyproj 3.6.x (PROJ network mode for EGM2008 grid), `earthaccess` 0.18.0, `pystac-client` and `stackstac` for Planetary Computer access, SurtGis 0.7.0 \citep{SurtGis2026}.

**Random seeds**: A fixed seed of 42 is used for the Optuna sampler, all train/test partitions, the SHAP subsampling, and the 2,000-resample bootstrap that produces the confidence intervals reported throughout. All numerical results are exactly reproducible from the released artefacts.

**Sensitivity**: A 5 km versus 10 km spatial-block sensitivity test is included as supplementary material; RMSE estimates differ by less than 0.05 m between block sizes.

**Upstream data sources**: FABDEM v1.2 is available from the University of Bristol under CC BY-NC-4.0 \citep{Hawker2022FABDEM}. ICESat-2 ATL08 v7 is openly distributed by NASA Earthdata. Sentinel-1 RTC and Sentinel-2 L2A composites are accessed via Microsoft Planetary Computer under open Copernicus licensing.

**Licensing summary**: Code is released under MIT licence. Data products (corrected DEM, training dataset) inherit CC BY-NC-4.0 from upstream FABDEM. Trained model weights are released under CC0.

---

## Acknowledgements (placeholder for final draft)

The author thanks CIGIDEN researchers (Gironás, Cienfuegos, Viollier, Hora) for releasing the Licantén flood polygon to Zenodo, and the FABDEM development team at the University of Bristol for the open release of FABDEM v1.2 \citep{Hawker2022FABDEM}. ICESat-2 ATL08 was accessed through the NASA Earthdata Cloud. Sentinel-1 and Sentinel-2 imagery were accessed through the Microsoft Planetary Computer. SurtGis terrain analyses were performed using the open-source SurtGis Rust library \citep{SurtGis2026}.

---

## Word counts (verification)

- Section 7 (Conclusions): ~395 words — slightly above 250 target but appropriate for a paper of this scope
- Section 8 (Data/code availability): ~370 words
- Acknowledgements: ~80 words
- **Total sections 7+8+ack: ~845 words**

## Cumulative draft v1 status (all sections)

| Section | Words | Status |
|---|---:|:---:|
| 1. Introduction | 870 | ✅ |
| 2. Study area | 510 | ✅ |
| 3. Data | 1,210 | ✅ |
| 4. Methods | 1,810 | ✅ |
| 5. Results | 1,847 | ✅ |
| 6. Discussion | 1,741 | ✅ |
| 7. Conclusions | 395 | ✅ |
| 8. Data/code availability | 370 | ✅ |
| Acknowledgements | 80 | ✅ |
| **TOTAL** | **8,833** | **DRAFT V1 COMPLETE** |

## Notes for final polish before submission

- 8,833 words is reasonable for IJDE (typical range 6,000–9,000 main text excluding refs). Slightly compressible if needed.
- Section 6.2 (limitations) is the longest subsection in Discussion at 370 words; can be trimmed to ~280 by combining the within-Chile and across-continent limitation paragraphs.
- The acknowledgements section currently includes only institutional/data acknowledgements; a final draft should add personal acknowledgements as appropriate.
- Zenodo DOIs and GitHub URLs to be filled in at submission time.
- Consider a final structural pass to ensure cross-references between sections are consistent (e.g., "Section 4.5" not "section 4.5" or "Sect. 4.5").
- LaTeX compile + reference check before submission.
