# LISFLOOD-FP setup — feasibility notes
**Status**: Not started. Staged for future session.
**Date**: 2026-05-15

## Decision rationale (2026-05-15)

L3a stratified validation produced **paper-grade results** (RMSE −18.7%, bias eliminated systematically, stratified pattern clearly documented). The paper is defendible without LISFLOOD-FP. Therefore LISFLOOD-FP postponed to:
- P4 revision rounds if reviewers request it
- Follow-up paper / Tier 2 work

The L3a story already explains *why* HAND-based simple inundation doesn't show improvement (correction is in uplands, not floodplains), preempting the most obvious reviewer concern.

## Path if/when we commit to LISFLOOD-FP

### Software stack required

| Component | Source | Effort |
|---|---|---|
| LISFLOOD-FP 8.2 (recommended) | https://zenodo.org/records/13121102 | Compile from C++ source (~0.5-1 day) |
| LISFLOOD-FP 5.9 BMI (alternative) | https://github.com/openearth/lisflood-fp-bmi | Easier, older, OK for basic case |
| LFPtools | `pip install lfptools` or https://github.com/jsosa/LFPtools | Depends on TauDEM, gdalutils |
| TauDEM | https://github.com/dtarb/TauDEM | needs MPI, MPICH |

### Input data needed

- ✅ DEM raw (mm_fabdem_raw.tif)
- ✅ DEM corrected (mm_corrected.tif)
- ❌ River network shapefile (TauDEM can extract from DEM)
- ❌ River widths (Yamazaki dataset, or estimate from drainage area)
- ❌ River bank elevations (LFPtools `lfp-getbankelevs`)
- ❌ River bed depths (Manning's, or Yamazaki)
- ❌ **Discharge hydrograph June 21-26 2023, Río Mataquito** — from DGA estación fluviométrica
- ❌ Manning's n coefficient (calibration parameter)
- ❌ Boundary conditions (inflow, outflow)

### DGA stations to query (Mataquito basin)

Per [DGA Mapas2](https://mapas2.mop.gob.cl/) — 11 estaciones fluviométricas en Mataquito:
- Río Mataquito en Licantén
- Río Teno bajo Quebrada La Jaula (próximo a piloto E1a-Teno)
- Río Claro en El Valle
- Río Lontué en Sagrada Familia
- etc.

API DGA en `snia.mop.gob.cl/dgasat` permite descargar series temporales.

### Estimated total effort

| Phase | Days |
|---|---:|
| Install LISFLOOD-FP + LFPtools + TauDEM | 1 |
| Prepare river network + widths + depths (LFPtools) | 1-2 |
| Download + prepare DGA hydrographs | 0.5 |
| Configure simulation (boundary conditions, Manning) | 1 |
| Calibration runs | 1-2 |
| Production runs (raw vs corrected × 2-3 sub-cuencas) | 1 |
| Analysis + plots | 1 |
| **Total** | **6-9 days** |

### What LISFLOOD-FP would add to the paper (best case)

- Stronger "so what" — direct flood depth/extent comparison
- Reviewer-proof against "did you validate downstream?"
- Possible elevation to Water Resources Research / Geophysical Research Letters

### What we have without LISFLOOD-FP (current state)

- 18.7% RMSE reduction (paper-defensible)
- Strong stratification analysis (NDVI, HAND, slope, geomorphon, elevation, tile)
- Bias elimination story (FABDEM positive bias → near-zero post-correction)
- Per-tile improvement maps
- Caveat documented: floodplains see less improvement because FABDEM is already accurate there

This is sufficient for **NHESS** or **International Journal of Digital Earth** as Tier 2 submission.

## Recommendation

**Pause LISFLOOD-FP** until P4 (manuscript writing) reveals if reviewers/co-authors specifically request it, OR until we have funding/time for Tier 1 effort.

**Next priority** (per ROADMAP.md): P3 — Escalar a Chile central 30-40°S. This adds geographic generalizability which is the stronger paper argument.

Alternative considered (rejected for now): integrating with the [LISFLOOD-LISVAP](https://pypi.org/project/lisflood-lisvap/) pip package — but that's evapotranspiration, not flood inundation.
