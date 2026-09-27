from types import SimpleNamespace

from visionartificial.detector import _resultado_a_detecciones


class _FakeTensor(list):
    """Imita lo suficiente de un tensor de ultralytics: iterable y __getitem__."""


def _fake_box(idx_clase: float, xyxy: tuple[float, float, float, float], conf: float):
    return SimpleNamespace(
        cls=_FakeTensor([idx_clase]),
        xyxy=[_FakeTensor(list(xyxy))],
        conf=_FakeTensor([conf]),
    )


def _fake_resultado(boxes, names):
    return SimpleNamespace(boxes=boxes, names=names)


def test_resultado_a_detecciones_convierte_boxes_a_deteccion():
    names = {0: "manzana", 1: "banano", 2: "naranja"}
    boxes = [
        _fake_box(0, (10.0, 20.0, 30.0, 40.0), 0.91),
        _fake_box(2, (50.0, 60.0, 70.0, 80.0), 0.77),
    ]
    resultado = _fake_resultado(boxes, names)

    detecciones = _resultado_a_detecciones(resultado)

    assert len(detecciones) == 2
    assert detecciones[0].clase == "manzana"
    assert detecciones[0].bbox == (10.0, 20.0, 30.0, 40.0)
    assert detecciones[0].confianza == 0.91
    assert detecciones[1].clase == "naranja"


def test_resultado_a_detecciones_sin_boxes_devuelve_lista_vacia():
    resultado = _fake_resultado([], {0: "manzana"})
    assert _resultado_a_detecciones(resultado) == []
