# Roadmap: FABDEM-ML para Chile central

**Producto**: Paper metodológico (B) + DEM corregido distribuible (A) + Dataset abierto (C)
**Target window**: Submission Q3 2026 (~4 meses), aceptación esperada Q1-Q2 2027
**Journal objetivo**: NHESS (primera opción) o International Journal of Digital Earth

## Estado inicial · 2026-05-14

| Componente | Estado |
|---|---|
| Pipeline reproducible (SurtGis terrain+hydro + planetary-computer + ATL08+EGM2008 + XGBoost) | ✅ |
| Pilot Teno (28×22 km, 888 footprints, RMSE 1.56 m, mejora 23%) | ✅ |
| E1a tile 1° S36W072 (25,645 footprints, RMSE 2.46 m, mejora 18.4%, gap leakage ~2%) | ✅ |
| Watchdog RAM + checkpointing | ✅ |
| Inferencia raster-wide (DEM corregido como producto) | ❌ |
| Validación hidráulica downstream (LISFLOOD-FP) | ❌ |
| Cobertura geográfica > 1 tile | ❌ |
| Dataset/COG empaquetado en Zenodo | ❌ |
| Manuscript | ❌ |

---

## Fase 1 — Escalar a Maule + Mataquito completo
**Deadline**: 2026-06-04 (3 semanas)
**Hitos numéricos**: ~150k footprints, RMSE M-M < 2.6 m

| Tarea | Detalle | ETA |
|---|---|---|
| T1.1 | Procesar 5 tiles adicionales (S35W072, S35W071, S36W071, S37W072, S37W071) con pipeline E1a existente | 3 días |
| T1.2 | Bajar ATL08 sin tope (todos 2019-2024, ~150-200 granules) | 1 día |
| T1.3 | Stitching de samples por tile en dataset unificado | 1 día |
| T1.4 | Re-entrenar XGBoost con Optuna 100 trials sobre dataset M-M | 2 días |
| T1.5 | Comparar métricas vs E1a (spatial CV, SHAP) | 1 día |

**Salidas**: `samples_mm_full.csv` (~150k × 33), modelo `xgb_mm.pkl`, métricas + SHAP
**Recursos**: ~50 GB disco, 8-12 hr cómputo total
**Decision point**: si RMSE > 2.7 m → revisar antes de continuar

---

## Fase 2 — Inferencia raster + Validación LISFLOOD-FP
**Deadline**: 2026-07-02 (4 semanas)
**Hitos numéricos**: IoU corrected > IoU raw en ≥ 2 sub-cuencas del temporal jun-2023

| Tarea | Detalle | ETA |
|---|---|---|
| T2.1 | Inferencia XGBoost sobre el grid completo FABDEM-MM → COG corregido | 1 día |
| T2.2 | Bajar referencias CIGIDEN: zonificación Licantén (Zenodo), ortofotos post-evento, mapas DLR | 1 día |
| T2.3 | Setup LISFLOOD-FP (instalar, leer manual, tutorial básico) | 4-5 días |
| T2.4 | Configurar 2-3 sub-cuencas (Licantén, Constitución, San Javier): mallado, condiciones de borde, hidrogramas DGA | 5 días |
| T2.5 | Correr LISFLOOD-FP × 2 escenarios (FABDEM raw vs corrected) × N sub-cuencas | 3 días (mucha CPU) |
| T2.6 | Calcular IoU/Jaccard/F1 huella inundada vs ortofoto CIGIDEN | 2 días |
| T2.7 | Generar plots comparativos (huellas overlap, perfiles, distribución profundidades) | 2 días |

**Salidas**: `fabdem_ml_mm.tif` (COG), tabla validación hidráulica, 3 plots flagship
**Recursos**: LISFLOOD-FP necesita CPU intensiva (~10-30 hr); revisar versión LISFLOOD8 o FastFlood
**Decision point crítico**: si IoU corregido ≤ raw → diagnóstico profundo antes de seguir; el paper sin esto pierde fuerza considerablemente

**Outreach paralelo**: email a Hawker (Bristol) cuando T2.6 esté listo, con resultados preliminares

---

## Fase 3 — Escalar a Chile central (30-40°S)
**Deadline**: 2026-07-23 (3 semanas)
**Hitos numéricos**: ~500-800k footprints, cobertura ~700,000 km²

| Tarea | Detalle | ETA |
|---|---|---|
| T3.1 | Pipeline en loop sobre 100 tiles 1°×1° (paralelizar si posible) | 7-10 días cómputo |
| T3.2 | Bajar GEDI L2A selectivo (1 de cada 5 granules, stratified) | 5 días + 200 GB |
| T3.3 | Dataset combinado ATL08+GEDI | 1 día |
| T3.4 | Modelo final entrenado a escala nacional | 1 día (Optuna 200 trials) |
| T3.5 | Inferencia COG corregido Chile central | 2-3 días |

**Salidas**: `chile_central_dataset.parquet`, `chile_central_fabdem_ml.cog.tif`, modelo final
**Recursos**: 300-500 GB disco, 50-70 hr cómputo. **Considerar burst computing Hetzner** (~$30-80 por la corrida si supera capacidad local)
**Decision point**: si recursos insuficientes → restringir a 33-37°S (paper sigue siendo válido como "Chile mediterráneo + sub-húmedo")

---

## Fase 4 — Manuscript
**Deadline**: 2026-09-03 (6 semanas)

| Tarea | ETA |
|---|---|
| T4.1 Outline (intro, métodos, resultados, discusión) | 1 sem |
| T4.2 Figuras finales (mapa mejora regional, SHAP, scatter residuos, flood map comparison) | 1 sem |
| T4.3 Tablas comparativas + benchmark contra Hawker/Wing/Marsh | 0.5 sem |
| T4.4 Draft v1 (~30 págs) | 1.5 sem |
| T4.5 Revisión interna / colaborador | 1 sem |
| T4.6 Draft v2 + supplementary | 1 sem |

**Salidas**: manuscript submission-ready, supplementary, código limpio en GitHub
**Estructura propuesta**:
1. Intro: gap FABDEM en Chile + necesidad correcciones regionales
2. Data: FABDEM v1.2, ICESat-2 ATL08 v7, GEDI L2A, Sentinel-1/2, geología SERNAGEOMIN, EGM2008
3. Method: feature engineering, XGBoost + Optuna, spatial-block CV (10 km), EGM2008 correction
4. Results: métricas globales + por régimen + SHAP + LISFLOOD-FP downstream
5. Discussion: comparación con FABDEM raw/FABDEM+/literatura, limitaciones, transferibilidad
6. Conclusions + data/code availability

---

## Fase 5 — Submission + revisiones
**Deadline**: 2027-01-31 (decisión inicial), 2027-04-30 (aceptación esperada)

| Tarea | ETA |
|---|---|
| T5.1 Submission NHESS (primera opción) | día 0 |
| T5.2 Preprint EarthArXiv simultáneo | día 0 |
| T5.3 Email Hawker con preprint | día +1 |
| T5.4 Zenodo: dataset, COG, código (DOIs) | día +7 |
| T5.5 Ronda 1 revisores | +60 días |
| T5.6 Respuesta + revisión | +90 días |
| T5.7 Ronda 2 + final acceptance | +150-180 días |

**Salidas**: paper accepted, 3 DOIs Zenodo, repo GitHub final

---

## Tracks concurrentes (paralelos al critical path)

| Track | Acción | Esfuerzo |
|---|---|---|
| Lastarria Frontiers review | Aceptar y completar si calendario lo permite | 2-4 hr semana de revisión |
| no_supervisado_superficie (foundation models) | Separate project; posible spin-off al final si SR sub-30m emerge | Independiente |
| Hawker outreach | Email cordial con preprint en P2-P3 | 1 hr |
| Twitter/Mastodon visibility | Posts con cada hito grande | 30 min/post |

---

## Risk register

| Riesgo | P | Impacto | Mitigación |
|---|---|---|---|
| Cómputo insuficiente para escalado nacional | M | Alto | Burst Hetzner; restringir a 33-37°S |
| LISFLOOD-FP no muestra mejora con DEM corregido | M | Alto | Pivot métrica (IoU local en zonas críticas), o probar FastFlood alternativo |
| Scoop por grupo competidor (Brasil, Argentina) | B | Alto | EarthArXiv ASAP al final P3; preprint público |
| GEDI insuficiente en Chile central | B | M | Solo ATL08; documentar limitación |
| FABDEM v1.3 sale durante el proceso | M | M | Re-entrenar con nueva versión al final si llega antes de P3 |
| LiDAR público chileno aparece tarde | B | + | Bonus para extensión paper 2 (no afecta paper 1) |
| Carga concurrente Lastarria + foundation models | M | M | Lastarria review es discreto (1 sem); foundation models pausable |

---

## Decision points (calendarizados)

| Fecha | Pregunta | Si SÍ | Si NO |
|---|---|---|---|
| 2026-06-04 | ¿Métricas M-M consistentes con E1a? | Continuar P2 | Diagnóstico antes de seguir |
| 2026-07-02 | ¿LISFLOOD-FP valida la mejora? | Continuar P3 | Pivotar métrica o ajustar modelo |
| 2026-07-23 | ¿Cobertura nacional viable con recursos? | Submission Chile central completo | Restringir a sub-zona; paper sigue válido |
| 2026-09-03 | ¿Draft v2 sólido? | Submission octubre | Extender escritura 2-3 semanas |

---

## Success criteria al cierre del roadmap

- [ ] 1 paper bajo revisión en NHESS/IJDE/equivalente
- [ ] Zenodo DOI con dataset training (parquet + metadata)
- [ ] Zenodo DOI con COG DEM corregido Chile central
- [ ] GitHub repo público, reproducible (Dockerfile, env.yml, README)
- [ ] IoU corrected > IoU raw demostrado en ≥ 2 sub-cuencas
- [ ] Outreach Hawker realizado (independiente del resultado)
- [ ] Preprint EarthArXiv público con DOI

---

## Notas operativas

- **Cómputo**: workstation actual cubre P1, marginal P2-P3. Hetzner burst computing como respaldo
- **Storage**: ~500 GB total para P3 — verificar disponibilidad antes de iniciar
- **Backup**: dataset y modelos en al menos 2 ubicaciones (local + Zenodo + S3 propio)
- **Versionado**: tag git al cierre de cada fase
- **Memoria contextual**: este ROADMAP.md y `MEMORY.md` en `~/.claude/projects/.../memory/` son los anclajes de continuidad inter-sesión
