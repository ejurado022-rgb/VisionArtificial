"""Tipos de datos compartidos entre detector.py, counter.py y metrics.py.

Separado de detector.py para que counter.py (logica pura) no dependa de
ultralytics solo para importar el tipo de dato. Ver docs/ARCHITECTURE.md S2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Deteccion:
    """Una deteccion cruda de YOLO sobre una imagen.

    bbox en formato absoluto de pixeles (x1, y1, x2, y2), x1<x2, y1<y2.
    """

    clase: str
    bbox: tuple[float, float, float, float]
    confianza: float

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
