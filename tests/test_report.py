import cv2
import numpy as np

from visionartificial.report import formatear_reporte, guardar_imagen_anotada
from visionartificial.types import Deteccion


def test_formatear_reporte_sin_detecciones():
    texto = formatear_reporte({})
    assert "ninguna fruta detectada" in texto
    assert "total: 0" in texto


def test_formatear_reporte_con_conteo_y_metricas():
    conteo = {"manzana": 2, "naranja": 1}
    metricas = {"precision": 0.9, "recall": 0.8}
    texto = formatear_reporte(conteo, metricas)
    assert "manzana: 2" in texto
    assert "naranja: 1" in texto
    assert "total: 3" in texto
    assert "precision: 0.900" in texto


def test_guardar_imagen_anotada_crea_archivo(tmp_path):
    imagen_path = tmp_path / "entrada.jpg"
    imagen = np.zeros((100, 100, 3), dtype="uint8")
    cv2.imwrite(str(imagen_path), imagen)

    detecciones = [Deteccion("manzana", (10.0, 10.0, 50.0, 50.0), 0.9)]
    destino = tmp_path / "salida" / "anotada.jpg"

    resultado = guardar_imagen_anotada(imagen_path, detecciones, destino)

    assert resultado == destino
    assert destino.exists()
