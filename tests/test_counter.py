from visionartificial.counter import contar, iou
from visionartificial.types import Deteccion


def test_iou_cajas_identicas_es_uno():
    caja = (0.0, 0.0, 10.0, 10.0)
    assert iou(caja, caja) == 1.0


def test_iou_cajas_sin_overlap_es_cero():
    assert iou((0.0, 0.0, 5.0, 5.0), (100.0, 100.0, 110.0, 110.0)) == 0.0


def test_contar_sin_solapamiento_cuenta_cada_deteccion():
    detecciones = [
        Deteccion("manzana", (0.0, 0.0, 10.0, 10.0), 0.9),
        Deteccion("manzana", (50.0, 50.0, 60.0, 60.0), 0.8),
        Deteccion("naranja", (100.0, 100.0, 110.0, 110.0), 0.7),
    ]
    assert contar(detecciones) == {"manzana": 2, "naranja": 1}


def test_contar_dos_cajas_solapadas_misma_clase_cuentan_una_vez():
    detecciones = [
        Deteccion("banano", (0.0, 0.0, 10.0, 10.0), 0.9),
        Deteccion("banano", (0.5, 0.5, 10.5, 10.5), 0.6),  # casi la misma caja
    ]
    assert contar(detecciones) == {"banano": 1}


def test_contar_cajas_solapadas_de_clases_distintas_se_cuentan_por_separado():
    detecciones = [
        Deteccion("manzana", (0.0, 0.0, 10.0, 10.0), 0.9),
        Deteccion("naranja", (0.0, 0.0, 10.0, 10.0), 0.85),
    ]
    assert contar(detecciones) == {"manzana": 1, "naranja": 1}


def test_contar_lista_vacia():
    assert contar([]) == {}
