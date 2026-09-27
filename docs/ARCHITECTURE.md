# VisionArtificial — Arquitectura técnica v1

Basado en la arquitectura de 5 módulos descrita en el PDF §3.3.1
("Ilustración 1. Arquitectura del sistema propuesto para la clasificación
de frutas"), mapeada a módulos Python concretos.

## 1. Vista general

```
imagen -> [preprocess] -> [detector: YOLOv8] -> [counter: NMS + agregación] -> [metrics/report]
                                                                                      |
                                                                              (Pista B) [experiment] -> [analysis]
```

## 2. Módulos

### `config.py`
Constantes del proyecto: lista de clases (`["manzana", "banano", "naranja"]`),
rutas por defecto (`data/splits`, `models/best.pt`), umbrales de confianza
e IoU para NMS. Sin lógica, solo datos — igual rol que `config.py` en
`VerificadorInformacion` (contenido "editorial" del proyecto).

### `dataset.py`
Gestión del dataset: helpers para descargar/exportar desde Roboflow (SDK
`roboflow`, opcional) y para verificar que `data/splits/{train,val,test}`
tiene la estructura YOLO esperada (imágenes + `.txt` de labels con
coordenadas normalizadas). PDF §3.2.

### `preprocess.py` (diferido)
Operaciones OpenCV sobre una imagen antes de pasarla al modelo:
normalización, ajuste de contraste, reducción de ruido. PDF §3.3.1
(módulo de preprocesamiento). Funciones puras `imagen -> imagen`. No se
implementó en Pista A: `ultralytics` ya hace letterboxing/normalización
internamente antes de la inferencia, y no era condición de salida de
ninguna fase de `docs/PLAN.md`. Se retoma si en Pista B se necesita
controlar explícitamente contraste/ruido por condición experimental.

### `detector.py`
Wrapper delgado sobre `ultralytics.YOLO`: carga los pesos entrenados
(`models/best.pt`) y expone `detectar(imagen) -> list[Deteccion]` donde
`Deteccion` tiene clase, bbox (x1,y1,x2,y2), confianza. Corresponde al
"Núcleo de Detección YOLO" del PDF §3.3.2.

### `counter.py`
Recibe las detecciones crudas de `detector.py`, aplica supresión de
no-máximos (NMS ya lo hace `ultralytics` internamente, pero aquí se
resuelve agregación adicional si hace falta) y devuelve un conteo por
clase (`dict[str, int]`). Corresponde al "Módulo de Conteo y Validación"
del PDF §3.3.3. Lógica pura, sin IO — se testea con detecciones armadas a
mano (incluyendo el caso de dos cajas solapadas del mismo objeto).

### `metrics.py`
Calcula precisión, recall, F1, mAP y tiempo de procesamiento por imagen
sobre un split de test, comparando detecciones contra las anotaciones
ground-truth. PDF §3.5 (versión reducida en Pista A; en Pista B se
extiende con matrices de confusión completas por experimento).

### `report.py`
Convierte los resultados de `metrics.py` en texto/tablas legibles y en la
imagen anotada (cajas + etiquetas + conteo dibujado) que genera el CLI.

### `cli.py`
Orquestación: `python -m visionartificial <ruta_imagen>` ->
preprocess -> detector -> counter -> report. Única capa de interacción
directa, con una función de entrada inyectable para poder testear sin
I/O real (mismo patrón que el CLI de `VerificadorInformacion` y
`AutoOrganizador`).

### `experiment.py` (Pista B)
Recorre las 27 carpetas de condiciones del diseño factorial 3×3×3
(iluminación × agrupamiento × ángulo), corre `detector.py` +
`counter.py` sobre cada imagen, y exporta un CSV con las métricas por
imagen y condición. PDF §3.4.1-3.4.3.

### `analysis.py` (Pista B)
Sobre el CSV de `experiment.py`: matrices de confusión por experimento,
ANOVA de tres factores, pruebas post-hoc de Tukey, regresión múltiple.
PDF §3.5. Depende de `statsmodels`/`scikit-learn` (extra `analysis` en
`pyproject.toml`), no se instala por defecto en Pista A.

## 3. Formato de datos

- **Dataset:** formato YOLO estándar — un `.txt` por imagen con líneas
  `clase x_centro y_centro ancho alto` normalizadas a [0,1] (PDF §3.2.4).
- **Detección interna (`Deteccion`):** dataclass con `clase: str`,
  `bbox: tuple[float, float, float, float]`, `confianza: float`.
- **CSV de experimentos (Pista B):** una fila por imagen, columnas
  `iluminacion, agrupamiento, angulo, precision, recall, f1, tiempo_ms,
  conteo_real, conteo_predicho` (PDF §3.4.2).

## 4. Decisiones de diseño

- Se usa `ultralytics` (paquete oficial de YOLOv8) en vez de reimplementar
  YOLO desde cero — el PDF ya asume YOLOv8 como base arquitectónica
  (§3.3.2) y no hay razón para no usar la implementación de referencia.
- `counter.py` y `metrics.py` son lógica pura (sin IO), para poder
  testearlos con datos sintéticos sin depender de tener un modelo
  entrenado — igual que `scoring.py` es el componente central y más
  testeado en `VerificadorInformacion`.
- Los artefactos pesados (dataset, pesos `.pt`) nunca viven en el
  repositorio de código — se gestionan en Roboflow (dataset) y se
  regeneran localmente (pesos, vía entrenamiento).
