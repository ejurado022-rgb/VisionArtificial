"""Metricas basicas sobre el split de test (PDF S3.5, version reducida
para Pista A). Reusa la validacion nativa de ultralytics (precision,
recall, mAP) en vez de reimplementar el calculo de mAP a mano, y mide el
tiempo de inferencia por imagen con el propio Detector del proyecto.
"""

from __future__ import annotations

import time
from pathlib import Path

from ultralytics import YOLO

from visionartificial.config import DEFAULT_MODEL_PATH, SPLITS_DIR
from visionartificial.detector import Detector


def evaluar_split(
    modelo_path: str | Path = DEFAULT_MODEL_PATH,
    data_yaml: str | Path = SPLITS_DIR / "data.yaml",
    split: str = "test",
) -> dict[str, float]:
    """Precision, recall, mAP50 y mAP50-95 sobre `split`, via ultralytics."""
    modelo = YOLO(str(modelo_path))
    resultado = modelo.val(data=str(data_yaml), split=split, verbose=False)

    precision = float(resultado.box.mp)
    recall = float(resultado.box.mr)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": float(resultado.box.map50),
        "map50_95": float(resultado.box.map),
    }


def medir_tiempo_por_imagen(detector: Detector, imagenes: list[Path]) -> float:
    """Tiempo medio de inferencia (segundos) por imagen sobre `imagenes`."""
    if not imagenes:
        return 0.0
    inicio = time.perf_counter()
    for imagen in imagenes:
        detector.detectar(imagen)
    total = time.perf_counter() - inicio
    return total / len(imagenes)
