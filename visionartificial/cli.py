"""CLI de demo: imagen -> deteccion + conteo + imagen anotada (Fase 4).

Uso: python -m visionartificial ruta/a/imagen.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from visionartificial.config import DEFAULT_MODEL_PATH
from visionartificial.counter import contar
from visionartificial.detector import Detector
from visionartificial.report import formatear_reporte, guardar_imagen_anotada


def ejecutar(
    imagen_path: str | Path,
    modelo_path: str | Path = DEFAULT_MODEL_PATH,
    destino_anotada: str | Path | None = None,
) -> str:
    """Corre detector -> counter -> report sobre una imagen y devuelve el
    texto del reporte. Guarda la imagen anotada junto a la original si no
    se especifica `destino_anotada`.
    """
    detector = Detector(modelo_path=modelo_path)
    detecciones = detector.detectar(imagen_path)
    conteo = contar(detecciones)

    imagen_path = Path(imagen_path)
    if destino_anotada is None:
        destino_anotada = imagen_path.with_name(
            f"{imagen_path.stem}_anotada{imagen_path.suffix}"
        )
    guardar_imagen_anotada(imagen_path, detecciones, destino_anotada)

    reporte = formatear_reporte(conteo)
    return f"{reporte}\n\nImagen anotada guardada en: {destino_anotada}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detecta, clasifica y cuenta frutas en una imagen."
    )
    parser.add_argument("imagen", help="Ruta a la imagen de entrada")
    parser.add_argument(
        "--modelo", default=str(DEFAULT_MODEL_PATH), help="Ruta a los pesos .pt entrenados"
    )
    parser.add_argument(
        "--salida", default=None, help="Ruta de salida para la imagen anotada"
    )
    args = parser.parse_args(argv)

    print(ejecutar(args.imagen, modelo_path=args.modelo, destino_anotada=args.salida))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
