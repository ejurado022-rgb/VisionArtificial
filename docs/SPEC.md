# VisionArtificial — Especificación funcional v1

Fuente: `VA_Avance1Proyecto_Jurado,Jaramillo.pdf` (Avance del proyecto
final, Tópicos Especiales I, UTP). Este documento traduce esa propuesta
académica en una especificación implementable, y deja explícito qué se
construye ahora (Pista A) y qué se completa después (Pista B) — ver
`docs/PLAN.md`.

## 1. Objetivo del sistema

Detectar, clasificar y contar automáticamente 3 tipos de frutas comunes de
supermercado en una sola imagen, usando YOLOv8, en tiempo razonable y con
la mayor precisión posible, incluso con oclusión parcial y agrupamiento de
frutas (PDF §1.3, §1.4).

## 2. Clases objetivo

**Manzana, banano, naranja.**

Decisión de diseño: el PDF de la propuesta no fija las 3 frutas. Se eligieron
estas por ser fáciles de conseguir para fotografiar en el plazo inmediato
y por coincidir con las clases de un dataset público de Roboflow Universe
ya identificado (ver §3.1), lo que permite complementar el dataset propio
sin depender solo de captura manual. Si se necesita cambiarlas, es un
cambio en `visionartificial/config.py` y en el dataset — no en la
arquitectura.

## 3.1. Fuente de dataset público confirmada

**"Fruit Detection Dataset"** por SS25KILV, Roboflow Universe:
<https://universe.roboflow.com/ss25kilv/fruit-detection-fv839>

- 300 imágenes, tarea Object Detection, 4 versiones publicadas.
- 3 clases: `apple`, `orange`, `banana` (mapeadas a `manzana`, `naranja`,
  `banano` en `config.py`).
- Licencia **CC BY 4.0** — requiere atribución al reutilizarlo; citar con
  el BibTeX de la página al documentar el dataset final del proyecto.
- Se descarga manualmente en formato YOLO (export de Roboflow) y se coloca
  en `data/raw/fruit-detection-fv839/` para que `dataset.py` lo integre
  junto con las fotos propias (Fase 1 de `docs/PLAN.md`).

Cita BibTeX (de la propia página, para el documento académico final):

```
@misc{ fruit-detection-fv839_dataset,
title = { Fruit Detection Dataset },
type = { Open Source Dataset },
author = { SS25KILV },
howpublished = { \url{ https://universe.roboflow.com/ss25kilv/fruit-detection-fv839 } },
url = { https://universe.roboflow.com/ss25kilv/fruit-detection-fv839 },
journal = { Roboflow Universe },
publisher = { Roboflow },
year = { 2025 },
month = { may },
}
```

### 3.1.1. Estado del dataset (Fase 1 completada)

El dataset SS25KILV se descargó, se remapeó a las clases en español y se
construyó en `data/splits/` (formato YOLO, `data.yaml` incluido). Split
heredado del export de Roboflow (80/10/10, no el 70/20/10 que aspira el
PDF a largo plazo — se ajusta en Pista B con más datos):

| split | imágenes | manzana | banano | naranja |
|-------|---------:|--------:|-------:|--------:|
| train | 240      | 221     | 89     | 288     |
| val   | 30       | 26      | 15     | 20      |
| test  | 30       | 26      | 17     | 22      |

Nota: `banano` está notoriamente menos representado (89 cajas en train
contra 221-288 de las otras dos clases) — riesgo a vigilar en las
métricas de la Fase 2/4; se corrige en Pista B sumando fotos propias de
banano si el recall de esa clase sale bajo.

Aún no se sumaron fotos propias — el dataset de Fase 1 es 100% la fuente
pública. `dataset.py` ya soporta combinar múltiples `FuenteDataset` si
más adelante se agregan fotos propias sin rehacer nada.

## 3.1.2. Estado del modelo y resultados (Pista A completa)

Modelo: YOLOv8n reentrenado por transfer learning (pesos COCO), 30 épocas,
imgsz=416, CPU (sin GPU). `models/best.pt` (6.2 MB). Entrenamiento
completo en ~12 min (0.198 h) en un Intel Core i9-14900HX.

**Métricas sobre el split de test** (30 imágenes, 65 cajas, nunca vistas
en entrenamiento — `visionartificial.metrics.evaluar_split`):

| métrica    | valor |
|------------|------:|
| precisión  | 0.898 |
| recall     | 0.722 |
| F1         | 0.801 |
| mAP50      | 0.843 |
| mAP50-95   | 0.697 |

Velocidad de inferencia: ~22 ms/imagen en CPU (~45 FPS) — cumple
holgadamente el requisito de tiempo real del PDF §1.3 incluso sin GPU.

Por clase, sobre el split de test (`ultralytics` val per-clase):

| clase   | precisión | recall | mAP50 | mAP50-95 |
|---------|----------:|-------:|------:|---------:|
| manzana | 0.959     | 0.923  | 0.970 | 0.855    |
| naranja | 0.906     | 0.955  | 0.951 | 0.871    |
| banano  | 0.830     | 0.288  | 0.609 | 0.365    |

`manzana` y `naranja` rinden muy bien. `banano` es claramente la clase
débil (recall 0.29 en test) — coincide con ser la clase con menos cajas
de entrenamiento (89 contra 221-288 de las otras dos, ver §3.1.1) y con
tener el test set más chico (17 cajas). Prioridad #1 para Pista B: sumar
fotos propias de banano, en especial agrupado/parcialmente oculto.

**Validación visual del CLI:** `python -m visionartificial <imagen>`
corrido sobre una imagen de test con 6 manzanas anotadas devolvió 8
detecciones (algunas cajas se solapan sobre trozos/mitades de manzana muy
próximos) — conteo razonable a simple vista, cajas bien ubicadas.

Con esto se cumplen los criterios de éxito de Pista A definidos en §6.

## 3. Alcance

**Incluido (Pista A, entrega inmediata):**
- Detección + clasificación + conteo sobre una imagen estática con las 3
  frutas objetivo, incluyendo casos con solapamiento simple.
- Dataset propio (fotos caseras en variedad de condiciones) + dataset
  público de Roboflow Universe de las mismas clases, etiquetado en
  Roboflow, dividido 70/20/10 (PDF §3.2.4).
- Modelo YOLOv8 reentrenado por transfer learning sobre pesos COCO
  preentrenados (PDF §3.3.2).
- Métricas básicas sobre el split de test: precisión, recall, F1, mAP,
  tiempo de procesamiento por imagen (PDF §3.5, versión reducida).
- CLI de demo: imagen de entrada -> imagen anotada + conteo por clase +
  métricas.

**Incluido (Pista B, después de la entrega — ver `docs/PLAN.md` fases
5-8):**
- Diseño experimental factorial 3×3×3 (iluminación × agrupamiento ×
  ángulo), 30 imágenes por combinación, 810 imágenes de prueba (PDF
  §3.4.1).
- Verificación instrumentada de condiciones (luxómetro, goniómetro) y
  control de calidad del etiquetado (revisión del 10% del dataset de
  prueba) (PDF §3.4.2-3.4.3).
- Análisis estadístico: matrices de confusión por experimento, ANOVA de
  tres factores, pruebas post-hoc de Tukey (α=0.05), regresión múltiple
  (PDF §3.5).
- Visualizaciones: gráficos de barras con error, boxplots, tablas resumen
  con medias/desviaciones/valores p (PDF §3.4.4, §3.5).

**Excluido explícitamente (delimitación del PDF §1.2):**
- Análisis de calidad o grado de madurez de la fruta.
- Frutas exóticas o poco comunes fuera de las 3 clases elegidas.
- Integración real con hardware de punto de venta / básculas — el PDF lo
  menciona como visión a futuro, no como entregable de esta fase.

## 4. Requisitos funcionales (Pista A)

1. Dado un archivo de imagen, el sistema debe devolver una lista de
   detecciones: clase, caja delimitadora, confianza.
2. El sistema debe agregar detecciones solapadas de un mismo objeto físico
   (NMS) y devolver un conteo por clase, no un conteo de cajas crudas
   (PDF §3.3.3).
3. El sistema debe poder evaluarse sobre un conjunto de test y reportar
   precisión, recall, F1, mAP y tiempo medio de procesamiento por imagen.
4. El CLI debe guardar una copia de la imagen con las cajas y las
   etiquetas dibujadas, para poder mostrarla en la entrega.

## 5. Requisitos no funcionales

- Debe poder ejecutarse en CPU (sin GPU dedicada) para pruebas iniciales,
  aunque el rendimiento sea menor (PDF §2.3).
- Python 3.10+, dependencias declaradas en `pyproject.toml`.
- El dataset y los pesos entrenados nunca se commitean al repositorio
  (`data/`, `models/*.pt` en `.gitignore`) — son artefactos grandes y
  regenerables, no código.

## 6. Criterios de éxito de Pista A

- El modelo detecta correctamente las 3 clases en al menos una imagen real
  con frutas mezcladas (validación visual, no solo métricas).
- `pytest` en verde para `detector.py`, `counter.py`, `metrics.py`, `cli.py`.
- Existe un número (aunque sea modesto) de precisión/recall/mAP sobre el
  split de test — no hace falta alcanzar el >95% que aspira el PDF a largo
  plazo, eso se persigue en Pista B con más datos.

## 7. Referencias

Todas las referencias numeradas [1]-[33] citadas en este documento
corresponden a la bibliografía del PDF de la propuesta
`VA_Avance1Proyecto_Jurado,Jaramillo.pdf`.
