import cv2
import numpy as np

from visionartificial import cli
from visionartificial.types import Deteccion


class _DetectorFalso:
    def __init__(self, modelo_path=None, conf=None, iou=None):
        pass

    def detectar(self, imagen_path):
        return [
            Deteccion("manzana", (10.0, 10.0, 50.0, 50.0), 0.9),
            Deteccion("naranja", (60.0, 60.0, 90.0, 90.0), 0.8),
        ]


def test_ejecutar_genera_reporte_y_guarda_imagen_anotada(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "Detector", _DetectorFalso)

    imagen_path = tmp_path / "entrada.jpg"
    cv2.imwrite(str(imagen_path), np.zeros((100, 100, 3), dtype="uint8"))

    salida = cli.ejecutar(imagen_path)

    assert "manzana: 1" in salida
    assert "naranja: 1" in salida
    assert "total: 2" in salida
    assert (tmp_path / "entrada_anotada.jpg").exists()


def test_main_smoke(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "Detector", _DetectorFalso)
    imagen_path = tmp_path / "entrada.jpg"
    cv2.imwrite(str(imagen_path), np.zeros((100, 100, 3), dtype="uint8"))

    codigo = cli.main([str(imagen_path)])

    assert codigo == 0
    salida_impresa = capsys.readouterr().out
    assert "Conteo de frutas detectadas" in salida_impresa
