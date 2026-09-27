"""Conteo automatico de frutas a partir de detecciones YOLO.

Modulo de Conteo y Validacion (PDF S3.3.3): agrega detecciones que
corresponden al mismo objeto fisico (cajas muy solapadas de la misma
clase) para no contarlas dos veces. Logica pura, sin IO - se testea con
detecciones sinteticas (docs/PLAN.md Fase 3).
"""

from __future__ import annotations

from visionartificial.config import IOU_THRESHOLD
from visionartificial.types import Deteccion


def iou(bbox_a: tuple[float, float, float, float], bbox_b: tuple[float, float, float, float]) -> float:
    """Intersection-over-Union de dos cajas (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    interseccion = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if interseccion == 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - interseccion
    if union == 0.0:
        return 0.0
    return interseccion / union


def agregar(
    detecciones: list[Deteccion], iou_threshold: float = IOU_THRESHOLD
) -> list[Deteccion]:
    """Supresion de no-maximos por clase: entre cajas de una misma clase
    que se solapan mas que `iou_threshold`, conserva solo la de mayor
    confianza (se asume que son el mismo objeto fisico detectado dos
    veces). Cajas de clases distintas nunca se suprimen entre si, aunque
    se solapen.
    """
    resultado: list[Deteccion] = []
    por_clase: dict[str, list[Deteccion]] = {}
    for det in detecciones:
        por_clase.setdefault(det.clase, []).append(det)

    for dets_clase in por_clase.values():
        pendientes = sorted(dets_clase, key=lambda d: d.confianza, reverse=True)
        conservadas: list[Deteccion] = []
        while pendientes:
            actual = pendientes.pop(0)
            conservadas.append(actual)
            pendientes = [
                d for d in pendientes if iou(actual.bbox, d.bbox) <= iou_threshold
            ]
        resultado.extend(conservadas)

    return resultado


def contar(
    detecciones: list[Deteccion], iou_threshold: float = IOU_THRESHOLD
) -> dict[str, int]:
    """Cuenta frutas por clase, resolviendo duplicados por solapamiento."""
    agregadas = agregar(detecciones, iou_threshold=iou_threshold)
    conteo: dict[str, int] = {}
    for det in agregadas:
        conteo[det.clase] = conteo.get(det.clase, 0) + 1
    return conteo
