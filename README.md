# VisionArtificial

Sistema de visión artificial basado en YOLOv8 para la detección,
clasificación y conteo automático de tres frutas comunes de supermercado
(**manzana, banano, naranja**) a partir de una sola imagen.

Proyecto desarrollado por **Ernesto Jurado** y **Tomás Jaramillo** para la
asignatura Tópicos Especiales I, Universidad Tecnológica de Panamá.

## Estado actual

Sistema funcionando de punta a punta: dataset construido, modelo YOLOv8n
entrenado, CLI de detección + conteo. Métricas sobre el split de test:
precisión 0.90, recall 0.72, mAP50 0.84 (detalle en `docs/SPEC.md` §3.1.2).

El diseño experimental factorial y el análisis estadístico completo
(sección 3.4-3.5 de la propuesta) quedan como siguiente etapa —
requieren captura de imágenes en condiciones controladas.

## Documentación

- [`NOTAS.md`](NOTAS.md) — decisiones de diseño y cómo trabajo en el repo.
- [`docs/SPEC.md`](docs/SPEC.md) — especificación funcional y alcance.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — arquitectura técnica.
- [`docs/PLAN.md`](docs/PLAN.md) — plan de implementación por fases.

## Instalación

```bash
pip install -e .[dev]
pytest
```

Para entrenar o correr el modelo hace falta además el extra `ml`:

```bash
pip install -e .[dev,ml]
```

## Uso

```bash
python -m visionartificial ruta/a/imagen.jpg
```
