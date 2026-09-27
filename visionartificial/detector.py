"""Nucleo de deteccion YOLO (PDF S3.3.2): wrapper delgado sobre
ultralytics.YOLO que carga los pesos entrenados y expone
`detectar(imagen) -> list[Deteccion]`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from visionartificial.config import CONF_THRESHOLD, DEFAULT_MODEL_PATH, IOU_THRESHOLD
from visionartificial.types import Deteccion


def _resultado_a_detecciones(resultado: Any) -> list[Deteccion]:
    """Convierte un `Results` de ultralytics en `list[Deteccion]`.

    Funcion pura separada de `Detector` para poder testearla con un
    objeto `resultado` sintetico, sin necesitar pesos entrenados reales.
    """
    detecciones = []
    for box in resultado.boxes:
        idx_clase = int(box.cls[0])
        clase = resultado.names[idx_clase]
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        confianza = float(box.conf[0])
        detecciones.append(Deteccion(clase=clase, bbox=(x1, y1, x2, y2), confianza=confianza))
    return detecciones


class Detector:
    """Carga un modelo YOLOv8 entrenado y detecta frutas en una imagen."""

    def __init__(
        self,
        modelo_path: str | Path = DEFAULT_MODEL_PATH,
        conf: float = CONF_THRESHOLD,
        iou: float = IOU_THRESHOLD,
    ) -> None:
        self.conf = conf
        self.iou = iou
        self._modelo = YOLO(str(modelo_path))

    def detectar(self, imagen: str | Path) -> list[Deteccion]:
        """Corre inferencia sobre una imagen y devuelve sus detecciones."""
        resultados = self._modelo.predict(
            source=str(imagen), conf=self.conf, iou=self.iou, verbose=False
        )
        detecciones: list[Deteccion] = []
        for resultado in resultados:
            detecciones.extend(_resultado_a_detecciones(resultado))
        return detecciones
