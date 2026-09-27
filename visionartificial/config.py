"""Constantes del proyecto: clases objetivo, rutas y umbrales.

Contenido "editorial" del proyecto (igual rol que config.py en
VerificadorInformacion) - sin logica, solo datos. Ver docs/SPEC.md S2.
"""

from __future__ import annotations

from pathlib import Path

# Orden fijo: el indice de cada clase en esta lista es el indice usado en
# las etiquetas YOLO (archivos .txt) despues de construir data/splits.
CLASSES: list[str] = ["manzana", "banano", "naranja"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "best.pt"

# Umbrales de inferencia (PDF S3.3.3 - modulo de conteo y validacion).
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
