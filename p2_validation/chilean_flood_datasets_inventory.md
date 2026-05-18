# Inventario de polígonos de inundación públicos para Chile
**Compilado**: 2026-05-15

## ✅ Confirmados descargables (esta semana)

### 1. CIGIDEN Licantén jun-2023 (Zenodo)
- **Formato**: SHP completo (shp/dbf/prj/cpg/shx)
- **Cobertura**: ~42 km², desembocadura Río Mataquito, 417 polígonos
- **Origen**: derivado de Sentinel via DLR; análisis Gironás/Viollier/Cienfuegos/Hora
- **DOI**: [10.5281/zenodo.13307972](https://zenodo.org/records/13307972)
- **CRS**: EPSG:32719
- **Estado**: ya bajado en `p2_validation/cigiden/`

### 2. Disaster Charter Activation #826 — Chile jun-2023 (varios maps)
- **URL**: https://disasterscharter.org/activations/flood-large-in-chile-activation-826-
- **Productos identificados** (18 total):
  - **Hualañé** flood impact (en Mataquito — dentro de nuestro estudio)
  - Ñiquen, Ñuble Region (flood + landslide)
  - Talhuenes, Maule (flood + landslide)
  - Doñihue, O'Higgins (flood + landslide)
- **Project manager**: Marcelo Durán (CIREN — Chile)
- **Requestor**: ONEMI
- **Esfuerzo descarga**: requiere request via Charter portal; 1-2 días
- **Formatos típicos**: GeoTIFF + shp + PDF reports

### 3. Disaster Charter — Atacama mar-2015 (Copiapó)
- **URL**: https://disasterscharter.org/activations/flood-in-chile (entry 2015)
- **Activación**: 25 marzo 2015, request ONEMI
- **Productos**: imagen de áreas inundadas en Copiapó (27 mar 2015)
- **Limitación**: products más limitados que en 2023; principalmente imagery, posible shp

### 4. Copernicus EMS EMSN053 — Southern Chile flood risk (2018-07)
- **URL**: https://emergency.copernicus.eu/mapping/list-of-components/EMSN053
- **JRC catalog**: https://data.jrc.ec.europa.eu/dataset/ac185f35-3b2a-433a-a381-cff0edeaf794
- **Cobertura**: Villa Santa Lucía, comuna Chaitén (Aysén/Los Lagos)
- **Formato directo**: `EMSN053_WGS84_V2.gdb_.zip` (GeoDatabase)
- **Trigger**: lluvia intensa 16-dic-2017
- **Esfuerzo**: descarga directa libre

### 5. Copernicus EMS EMSN037 — Multi-hazard Chile/Perú/Mozambique
- **URL**: https://emergency.copernicus.eu/mapping/ems/multiple-natural-hazards-risk-assessment-chile-peru-mozambique-emsn037
- **Cobertura**: incluye Arica (flood + landslide)
- **Formato**: shp/gdb/PDF

## 🌍 Datasets globales con cobertura Chile (acceso vía API/GEE)

### 6. Global Flood Database v1 (Cloud to Street + DFO) — vía GEE
- **GEE asset**: `GLOBAL_FLOOD_DB/MODIS_EVENTS/V1`
- **Portal**: https://global-flood-database.cloudtostreet.ai/
- **Cobertura**: **913 eventos 2000-2018**, 250m, MODIS-based
- **Chile**: filtrable por country/date; probables eventos:
  - 2002 Maule
  - 2008 Antofagasta
  - **2015 Atacama (mudflow Copiapó)** ← seguro presente, evento DFO #4267
  - Otros menores
- **Formato**: GeoTIFF directo desde GCS bucket
- **Esfuerzo**: 1-2 horas via GEE Python API

### 7. GFM — Global Flood Monitoring (Copernicus)
- **URL**: https://global-flood.emergency.copernicus.eu/
- **Cobertura**: **todo Sentinel-1 desde 2015** procesado automáticamente, near-real-time
- **Chile**: cobertura sistemática, todos los eventos post-2015
- **Formato**: GeoJSON polígonos vía API
- **Esfuerzo**: registrarse en portal + script query (medio día)

### 8. WorldFloods (ML4Floods / ISP-UV-ES)
- **Zenodo**: [10.5281/zenodo.8153514](https://zenodo.org/records/8153514)
- **HuggingFace**: https://huggingface.co/datasets/tacofoundation/worldfloods
- **Cobertura**: **509 flood extent maps, 144 events 2016-2023**, Sentinel-2 + máscaras
- **Chile**: NO confirmado explícitamente, requiere inspeccionar metadata por país
- **Formato**: vectorized SHP polygons + Sentinel-2 imagery
- **Esfuerzo**: descarga Zenodo (~1-2 GB) + filtro por Chile (medio día)

### 9. SAR archive 10-year (Nature Communications 2025)
- **Paper**: https://www.nature.com/articles/s41467-025-60973-1
- **Cobertura**: DL sobre Sentinel-1 archive 2015-2024 global
- **Chile**: cobertura completa
- **Datos**: posiblemente Zenodo asociado al paper (verificar SI)

## 🇨🇱 Fuentes chilenas adicionales

### 10. ARClim — Atlas de Riesgo Climático (MMA + CR2)
- **Portal**: https://arclim.mma.gob.cl/
- **Capas relevantes**:
  - [Inundaciones por desbordes de ríos](https://arclim.mma.gob.cl/atlas/view/desbordes_rios_cbit/)
  - [Asentamientos × inundación](https://arclim.mma.gob.cl/atlas/view/asentamientos_inundaciones/)
- **Naturaleza**: **escenarios climáticos / proyecciones**, no eventos pasados
- **Cobertura**: nacional
- **Formato**: posible WMS + shp; verificar descargas directas

### 11. CIREN — Probabilidad de Inundación Fluvial
- **URL**: https://experience.arcgis.com/experience/eedd0b9a835041efbaf04dcaae7bea6f/page/Probabilidad-de-Inundaci%C3%B3n-Fluvial-
- **Naturaleza**: mapa probabilístico nacional CIREN
- **Cobertura**: pre-Andean + Andean basins Chile
- **Metodología**: imágenes radar + agricultura (CIREN 2010)
- **Esfuerzo**: navegación visor + posible request datos

### 12. IDE Minagri — descarga directa de capas
- **Portal**: https://ide.minagri.gob.cl/geoweb/
- **Layers download**: https://ide.minagri.gob.cl/geoweb/index.php/descargas
- **Cobertura**: nacional, formato shp
- **Esfuerzo**: navegación + selección manual

### 13. DGA — mapas amenaza inundación Maule (entregados a municipios)
- **Origen**: 10 municipios región Maule (post jun-2023)
- **Acceso**: probable solicitud Ley Transparencia (LPP); no público directo
- **Cobertura**: Maule
- **Esfuerzo**: 1-3 semanas vía solicitud formal

### 14. Geoportal IDE Chile
- **URL**: https://www.geoportal.cl/geoportal/map/4
- **Capas relevantes presentes**:
  - Tsunami evacuation areas (SHOA)
  - Glaciares (DGA)
  - "Área de Riesgo" en planos comunales (PRC)
  - Tornados Talcahuano
- **Cobertura nacional**, formato shp

## 📚 Eventos chilenos documentados (probables polígonos disponibles)

| Año/mes | Evento | Región | Donde buscar |
|---|---|---|---|
| 2015-03 | **Atacama mudflow** (~31 muertos) | Atacama | DFO/GFD, Disaster Charter, GFM, paper Wilcox 2016 |
| 2017-02 | Incendios + costa | Maule/Biobío | EMSR (fire-mainly) |
| 2017-03 | Aluviones Atacama | Atacama | GFM, GFD |
| 2018-07 | Villa Santa Lucía aluvión | Aysén | **EMSN053** ✓ |
| 2019 | Quebrada Macul | RM | press only probably |
| 2023-06 | **Vuelven los gigantes** | Centro-sur | CIGIDEN Licantén ✓, Charter #826, GFM, DLR |
| 2023-08 | 2do temporal | Centro-sur | GFM, posible EMSR |
| 2024-06 | Sistema frontal | Centro-sur | GFM, futuro CIGIDEN |
| Post-2024 | continuos | global | GFM automático |

## ❌ NO disponibles (públicamente)

- **CIGIDEN Maipo + Constitución + Aysén 2023**: descritos en informes pero NO publicados como shp públicos (solo Licantén llegó a Zenodo)
- **EGUsphere preprint 2026 Central Chile SAR**: paper sin data release
- **RAPID (UConn)**: solo CONUS, NO Chile
- **Sen1Floods11 / SEN12-FLOOD**: NO Chile

## 🎯 Acción concreta (priorizada)

| Prioridad | Acción | Esfuerzo | Valor |
|---|---|---|---|
| 🥇 ALTA | GFD via GEE — filtrar eventos Chile 2000-2018 | 2 hr | Alto: 5-15 eventos históricos |
| 🥇 ALTA | GFM portal — request Chile 2015-2024 polygons | 0.5 día | Alto: 50+ eventos automáticos |
| 🥈 MEDIA | EMSN053 GeoDatabase download (Villa Santa Lucía 2017) | 30 min | Medio: 1 evento extremo |
| 🥈 MEDIA | Charter #826 request — Hualañé impact map | 1-2 días | Alto: dentro de nuestro estudio M-M |
| 🥉 BAJA | WorldFloods Zenodo + filtrar Chile | 1 día | Posible: confirmar coverage |
| 🥉 BAJA | CIREN visor Probabilidad Fluvial | 0.5 día | Bajo (no es evento, es probabilidad) |

## Implicación para el paper

Para el **paper 1 (FABDEM bias)**: ya tienes Licantén. No urgen los demás.

Para **paper 2 / future work**: el inventario revela que con **GFD-GEE + GFM Copernicus** podrías construir un **dataset chileno multi-evento (5-15 eventos pasados, 50+ recientes)** sin necesidad de pedir nada. Eso sí es paper en *Earth System Science Data* o *Scientific Data* — "first open multi-event flood inundation reference dataset for Chile, 2000-2025".

Este último punto es **el plot twist más valioso del rastreo**: la materia prima para una contribución mucho más grande que solo Licantén existe pública. Solo nadie la ha empaquetado todavía.
