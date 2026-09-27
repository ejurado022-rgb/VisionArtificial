# VisionArtificial — Plan de Implementación

Formato por fase: **Objetivo · Archivos · Comportamiento logrado · Cómo se
verifica · Riesgo · Condición de salida.**

Dividido en dos pistas (ver `docs/SPEC.md` §3 para el porqué):
**Pista A** (días, entrega inmediata) deja el sistema funcionando de
punta a punta con un dataset mínimo viable. **Pista B** (semanas, después
de la entrega) completa el diseño experimental factorial y el análisis
estadístico exactamente como lo describe el PDF de la propuesta §3.4-3.5.

---

## Fase 0 — Preparación del proyecto y decisión de las 3 frutas

**Objetivo:** esqueleto instalable y testeable, con las 3 frutas objetivo
confirmadas, sin lógica todavía.

**Archivos:** `pyproject.toml`, `.gitignore`, `README.md`, `NOTAS.md`,
`docs/SPEC.md`, `docs/ARCHITECTURE.md`, `visionartificial/__init__.py`,
`tests/conftest.py`.

**Comportamiento logrado:** `pip install -e .[dev]` funciona; `pytest`
corre un test de humo; el paquete es importable.

**Riesgo:** ninguno.

**Condición de salida:** `pytest` verde, estructura creada según
`ARCHITECTURE.md`. Frutas confirmadas: manzana, banano, naranja (ver
`docs/SPEC.md` §2).

---

## Fase 1 — Dataset mínimo viable

**Objetivo:** tener un dataset etiquetado y dividido, suficiente para
entrenar un modelo real (PDF §3.2, adaptado en escala para Pista A).

**Archivos:** `visionartificial/dataset.py`, `data/raw/`, `data/labeled/`,
`data/splits/{train,val,test}` (todo gitignored).

**Comportamiento logrado:** combinación de fotos propias (variedad de
iluminación/ángulo/agrupamiento casera) + dataset público de Roboflow
Universe de las mismas 3 clases, etiquetado en Roboflow, exportado en
formato YOLO, dividido 70/20/10.

**Cómo se verifica:** conteo de imágenes por clase y por split (script o
test en `dataset.py`); revisión manual de una muestra de anotaciones.

**Riesgo:** dataset chico/desbalanceado — mitigado combinando con el
dataset público de Roboflow.

**Condición de salida:** `data/splits/{train,val,test}` con imágenes +
labels en formato YOLO, conteo por clase documentado en `docs/SPEC.md`.

---

## Fase 2 — Entrenamiento del modelo

**Objetivo:** reentrenar YOLOv8 (transfer learning sobre pesos COCO) con
el dataset de la Fase 1 (PDF §3.3.2).

**Archivos:** script de entrenamiento (`scripts/train.py` o notebook),
`models/best.pt` (gitignored), `visionartificial/config.py`.

**Comportamiento logrado:** modelo `.pt` que detecta las 3 clases con
confianza razonable.

**Cómo se verifica:** métricas de entrenamiento reportadas por
`ultralytics` (precision/recall/mAP en validación) documentadas en
`docs/SPEC.md`.

**Riesgo:** dataset chico puede dar mAP bajo — aceptable para Pista A, se
refuerza en Pista B con el dataset ampliado del diseño experimental.

**Condición de salida:** pesos entrenados, cargables desde `detector.py`.

---

## Fase 3 — Núcleo de detección y conteo

**Objetivo:** `detector.py` + `counter.py` sobre el modelo de la Fase 2
(PDF §3.3.2-3.3.3).

**Archivos:** `visionartificial/detector.py`, `visionartificial/counter.py`,
`tests/test_detector.py`, `tests/test_counter.py`.

**Comportamiento logrado:** `detectar(imagen) -> list[Deteccion]`;
`contar(detecciones) -> dict[str, int]` que resuelve cajas solapadas del
mismo objeto sin duplicar el conteo.

**Cómo se verifica:**
- test: detecciones sintéticas sin solapamiento -> conteo = número de
  detecciones.
- test: dos cajas muy solapadas de la misma clase -> cuentan como una.
- test: detecciones de clases distintas solapadas -> se cuentan por
  separado.

**Naturaleza:** `counter.py` es lógica pura, sin IO — se prueba con datos
sintéticos, sin depender de tener un modelo entrenado.

**Riesgo:** bajo.

**Condición de salida:** las pruebas anteriores en verde.

---

## Fase 4 — Métricas básicas y CLI de demo

**Objetivo:** cerrar el ciclo con métricas (PDF §3.5, versión reducida) y
un CLI ejecutable de punta a punta.

**Archivos:** `visionartificial/metrics.py`, `visionartificial/report.py`,
`visionartificial/cli.py`, `visionartificial/__main__.py`,
`tests/test_metrics.py`, `tests/test_cli.py`.

**Comportamiento logrado:** `metrics.py` calcula precisión/recall/F1/mAP y
tiempo por imagen sobre `data/splits/test`; `cli.py` toma una imagen,
corre preprocess -> detector -> counter -> report, imprime resultado y
guarda una imagen anotada.

**Cómo se verifica:**
- test: métricas sobre un mini-set sintético con conteo esperado conocido.
- test de humo: `python -m visionartificial <imagen>` corre sin errores y
  genera la imagen anotada.

**Condición de salida:** se puede ejecutar el CLI de punta a punta sobre
una foto real con las 3 frutas mezcladas y el resultado es correcto a
simple vista. **Esto es lo que se muestra en la entrega inmediata.**

---

## Pista B — Diseño experimental completo (después de la entrega, semanas)

### Fase 5 — Diseño experimental factorial 3×3×3

**Objetivo:** montar el setup físico y capturar las 810 imágenes de
prueba que especifica el PDF §3.4.1.

**Archivos:** `data/experimentos/<iluminacion>_<agrupamiento>_<angulo>/`
(27 carpetas, 30 imágenes cada una).

**Comportamiento logrado:** iluminación ajustada con lámparas LED
regulables y verificada con luxómetro; ángulo fijado con soporte
calibrado y verificado con goniómetro (PDF §3.4.2-3.4.3); revisión manual
del 10% de las anotaciones antes de correr los experimentos.

**Riesgo:** es trabajo físico de captura — no paralelizable con código;
por eso se hace después de la entrega inmediata, no antes.

**Condición de salida:** 810 imágenes organizadas y etiquetadas.

---

### Fase 6 — `experiment.py`

**Objetivo:** correr el modelo sobre las 27 carpetas y exportar el CSV de
métricas por imagen (PDF §3.4.2).

**Archivos:** `visionartificial/experiment.py`, `tests/test_experiment.py`.

**Comportamiento logrado:** recorre cada carpeta, corre
`detector.py`+`counter.py`, calcula métricas por imagen, escribe un CSV
con columnas `iluminacion, agrupamiento, angulo, precision, recall, f1,
tiempo_ms, conteo_real, conteo_predicho`.

**Cómo se verifica:** test con un subconjunto sintético de carpetas ->
CSV con el número exacto de filas esperado.

**Condición de salida:** CSV de 810 filas generado sobre el dataset real
de la Fase 5.

---

### Fase 7 — `analysis.py`

**Objetivo:** análisis estadístico sobre el CSV de la Fase 6 (PDF §3.5).

**Archivos:** `visionartificial/analysis.py`, `tests/test_analysis.py`.

**Comportamiento logrado:** matrices de confusión por experimento, ANOVA
de tres factores (iluminación, agrupamiento, ángulo) sobre cada métrica
dependiente, pruebas post-hoc de Tukey (α=0.05) cuando el ANOVA da
efectos significativos, regresión múltiple para predecir precisión/recall
en configuraciones no evaluadas.

**Cómo se verifica:** test con datos sintéticos donde el efecto de un
factor es conocido de antemano -> el ANOVA debe detectarlo.

**Naturaleza:** depende de `statsmodels`/`scikit-learn` (extra
`analysis` en `pyproject.toml`).

**Condición de salida:** resultados del ANOVA y Tukey documentados,
coherentes con lo observado visualmente en los datos.

---

### Fase 8 — Informe de resultados

**Objetivo:** generar las visualizaciones y tablas que exige el PDF
§3.4.4-3.5 para la sección de resultados del documento académico final.

**Archivos:** `visionartificial/report.py` (extendido), salidas en
`outputs/` (gitignored).

**Comportamiento logrado:** gráficos de barras con barras de error,
boxplots de tiempo de procesamiento y métricas, tablas resumen con
medias/desviaciones estándar/valores p por combinación experimental.

**Condición de salida:** confirmación del usuario de que los gráficos y
tablas están listos para pegar en el documento académico.

---

## Resumen de dependencias entre fases

```
Fase 0 (preparación + 3 frutas)
  └─▶ Fase 1 (dataset mínimo viable)
        └─▶ Fase 2 (entrenamiento YOLOv8)
              └─▶ Fase 3 (detector + counter)
                    └─▶ Fase 4 (metrics + CLI)  ◀── entrega inmediata (Pista A)
                          └─▶ Fase 5 (captura 810 imgs, 27 condiciones)
                                └─▶ Fase 6 (experiment.py -> CSV)
                                      └─▶ Fase 7 (analysis.py: ANOVA/Tukey)
                                            └─▶ Fase 8 (informe de resultados)
```

## Primera fase a implementar

**Fase 0 — Preparación del proyecto.** Las 3 frutas ya quedaron
confirmadas (manzana, banano, naranja) en `docs/SPEC.md` §2.
