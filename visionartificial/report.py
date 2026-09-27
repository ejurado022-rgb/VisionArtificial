"""Presentacion de resultados: texto legible y la imagen anotada que
genera el CLI (PDF S3.3.3 - modulo de validacion y reporte).
"""

from __future__ import annotations

from pathlib import Path

import cv2

from visionartificial.types import Deteccion

_COLOR_POR_CLASE = {
    "manzana": (0, 0, 220),  # BGR: rojo
    "banano": (0, 210, 255),  # amarillo
    "naranja": (0, 140, 255),  # naranja
}
_COLOR_DEFECTO = (255, 255, 255)


def formatear_reporte(
    conteo: dict[str, int], metricas: dict[str, float] | None = None
) -> str:
    """Texto legible con el conteo por clase y, si se pasan, las metricas."""
    lineas = ["Conteo de frutas detectadas:"]
    total = sum(conteo.values())
    if not conteo:
        lineas.append("  (ninguna fruta detectada)")
    for clase, cantidad in sorted(conteo.items()):
        lineas.append(f"  {clase}: {cantidad}")
    lineas.append(f"  total: {total}")

    if metricas:
        lineas.append("")
        lineas.append("Metricas sobre el split de test:")
        for nombre, valor in metricas.items():
            lineas.append(f"  {nombre}: {valor:.3f}")

    return "\n".join(lineas)


def dibujar_detecciones(imagen_path: str | Path, detecciones: list[Deteccion]):
    """Devuelve la imagen (array BGR de OpenCV) con cajas y etiquetas dibujadas."""
    imagen = cv2.imread(str(imagen_path))
    if imagen is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {imagen_path}")

    for det in detecciones:
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
        color = _COLOR_POR_CLASE.get(det.clase, _COLOR_DEFECTO)
        cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 2)
        etiqueta = f"{det.clase} {det.confianza:.2f}"
        cv2.putText(
            imagen,
            etiqueta,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return imagen


def guardar_imagen_anotada(
    imagen_path: str | Path, detecciones: list[Deteccion], destino: str | Path
) -> Path:
    """Dibuja las detecciones y guarda el resultado en `destino`."""
    imagen_anotada = dibujar_detecciones(imagen_path, detecciones)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), imagen_anotada)
    return destino
