# Abstract — submission-ready draft v3 (post tex-review)

**Word count**: 275 (IJDE allows 250–300)
**Last revision**: 2026-05-16 — applied 6 HIGH-severity fixes from tex-review (D1–D7)

---

Global 30-m digital elevation models such as FABDEM achieve aggregate vertical RMSE near 2.5 m, but it remains unclear whether this average hides geographically heterogeneous bias that could be exploited for targeted regional correction, and whether such a correction transfers across distinct climate regimes. We address both questions using 135,350 ICESat-2 ATL08 v7 ground footprints over central-south Chile (34–39°S) as cm-precision satellite LiDAR ground truth, spanning two contrasting regimes: Mediterranean watersheds (Mataquito and Maule) and humid temperate Valdivian rainforest (Bío-Bío and Araucanía). For each footprint we engineer 33 features — geomorphometric (slope, curvature, openness, MRVBF, geomorphons), hydrological (HAND, TWI, flow accumulation, valley depth), and satellite-derived (Sentinel-1 RTC σ⁰, Sentinel-2 NDVI/NDWI/NDMI/BSI) — and train XGBoost regression on the residual h_te − fabdem after EGM2008 datum correction, with Optuna hyperparameter optimization and spatial-block cross-validation (10-km blocks, n = 570, K = 5 folds). Spatial-block cross-validation reduces RMSE from 3.054 m (95% CI: 2.95–3.18) to 2.483 m (2.37–2.62) — a 0.57 m absolute reduction (−18.7%; Diebold–Mariano p < 0.001) — and eliminates a systematic +1.0 m positive bias. Stratification reveals strongly differential gains: −41.6% in dense vegetation (NDVI > 0.6), −33.8% above 3000 m, but only −19.1% in floodplains (HAND 2–10 m) where FABDEM is already accurate. Applying the Mediterranean-trained model to humid temperate forest *without retraining* yields a further −22.2% RMSE reduction (4.946 → 3.850 m; CIs 4.84–5.06 → 3.76–3.96), with SHAP attribution shifting from drainage features to 3D shape descriptors (openness_negative, valley_depth) — consistent with the model capturing terrain–error relationships that transfer between contrasting regimes rather than memorising regional patterns. We release the corrected DEM as a Cloud-Optimized GeoTIFF (CC BY-NC-4.0, inherited from FABDEM), the 135k-footprint training dataset, trained model, and pipeline code as open artifacts.

---

## Changes vs. v2 draft (applied 2026-05-16)

| # | Fix dimension | v2 | v3 |
|---|---|---|---|
| 1 | D2/D1 overclaim | "evidence the model learned physical relationships, not regional patterns" | "consistent with the model capturing terrain–error relationships that transfer between contrasting regimes rather than memorising regional patterns" |
| 2 | D2 overgeneralisation | "across climate regimes" (sounded universal) | "between contrasting regimes" + framing limited to MD↔HT |
| 3 | D3 statistical rigour | RMSE point estimates only | + bootstrap 95% CIs + Diebold–Mariano test result |
| 4 | D6 informal register | "Honest cross-validation" | "Spatial-block cross-validation" |
| 5 | D7 license | Implicit | Explicit "CC BY-NC-4.0, inherited from FABDEM" |
| 6 | D5 reader context | "Valdivian forest" | "Valdivian rainforest" (more specific) |

## Keywords (5–7 for IJDE)

`FABDEM`, `bias correction`, `ICESat-2 ATL08`, `XGBoost`, `spatial-block cross-validation`, `Chile`, `out-of-distribution generalization`

## Highlights (5 bullets, for some journals)

- ICESat-2 ATL08 provides cm-precision satellite LiDAR ground truth at scale for 135,350 footprints over central-south Chile
- ML correction reduces FABDEM RMSE from 3.054 to 2.483 m (−18.7%, 95% CI 2.37–2.62, DM p < 0.001) under spatial-block cross-validation, eliminating +1.0 m positive bias
- Stratification reveals correction is largest in dense vegetation and alpine terrain; marginal in floodplains where FABDEM is already accurate
- Mediterranean-trained model generalises to humid temperate forest *without retraining* (−22.2% RMSE), with SHAP showing physically interpretable feature-importance shifts between regimes — consistent with terrain–error learning rather than regional memorisation
- Corrected DEM, training dataset, model, and pipeline released as open artifacts (CC BY-NC-4.0, inherited from FABDEM)

## Plain-language summary

Public elevation maps from satellites (digital elevation models, or DEMs) are essential for flood prediction, water management, and infrastructure planning, but they contain systematic errors that vary across landscapes. We used cm-precision laser measurements from NASA's ICESat-2 satellite to map and correct these errors over a 220 × 660 km transect of central Chile, finding that the errors are larger in dense forests and high mountains than in flat valleys. A machine-learning model trained on one Chilean climate zone (Mediterranean) successfully corrected elevation errors in a contrasting humid temperate forest zone 800 km south *without any retraining*. This is consistent with the model capturing real physical relationships between terrain shape and error rather than regional patterns specific to where it was trained. We release the corrected map, training data, and software openly so others can apply or extend the approach.
