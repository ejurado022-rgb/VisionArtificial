from pathlib import Path

import yaml

from visionartificial.dataset import FuenteDataset, construir_splits, contar_por_clase

CLASES_TEST = ["manzana", "banano", "naranja"]


def _crear_fuente_fake(tmp_path: Path, nombre: str, nombres_clases_fuente: list[str]) -> Path:
    """Crea una fuente con la forma de un export de Roboflow: train/valid/test,
    cada split con 1 imagen + 1 label, mas su propio data.yaml.
    """
    directorio = tmp_path / nombre
    (directorio / "train" / "images").mkdir(parents=True)
    (directorio / "train" / "labels").mkdir(parents=True)
    (directorio / "valid" / "images").mkdir(parents=True)
    (directorio / "valid" / "labels").mkdir(parents=True)
    (directorio / "test" / "images").mkdir(parents=True)
    (directorio / "test" / "labels").mkdir(parents=True)

    (directorio / "data.yaml").write_text(
        yaml.safe_dump({"names": nombres_clases_fuente, "nc": len(nombres_clases_fuente)}),
        encoding="utf-8",
    )

    for split_fuente, split_destino_stub in (("train", "a"), ("valid", "b"), ("test", "c")):
        img = directorio / split_fuente / "images" / f"img_{split_destino_stub}.jpg"
        img.write_bytes(b"fake-image-bytes")
        lbl = directorio / split_fuente / "labels" / f"img_{split_destino_stub}.txt"
        # una caja de la clase 0 (primer nombre de nombres_clases_fuente)
        lbl.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    return directorio


def test_construir_splits_copia_imagenes_y_remapea_clases(tmp_path):
    fuente_dir = _crear_fuente_fake(tmp_path, "fuente1", ["apple", "banana", "orange"])
    fuente = FuenteDataset(
        directorio=fuente_dir,
        mapa_clases={"apple": "manzana", "banana": "banano", "orange": "naranja"},
    )
    destino = tmp_path / "splits"

    resumen = construir_splits([fuente], destino, clases=CLASES_TEST)

    assert (destino / "train" / "images" / "fuente1__img_a.jpg").exists()
    assert (destino / "val" / "images" / "fuente1__img_b.jpg").exists()
    assert (destino / "test" / "images" / "fuente1__img_c.jpg").exists()

    # la clase 0 de la fuente ("apple") debe haberse remapeado a "manzana"
    label_train = (destino / "train" / "labels" / "fuente1__img_a.txt").read_text()
    assert label_train.startswith("0 ")  # manzana es indice 0 en CLASES_TEST

    assert resumen["train"]["n_imagenes"] == 1
    assert resumen["train"]["cajas"]["manzana"] == 1
    assert resumen["val"]["cajas"]["manzana"] == 1
    assert resumen["test"]["cajas"]["manzana"] == 1


def test_construir_splits_combina_dos_fuentes_sin_colisionar(tmp_path):
    fuente1_dir = _crear_fuente_fake(tmp_path, "fuente1", ["apple", "banana", "orange"])
    fuente2_dir = _crear_fuente_fake(tmp_path, "fuente2", ["banana", "apple", "orange"])
    fuentes = [
        FuenteDataset(
            directorio=fuente1_dir,
            mapa_clases={"apple": "manzana", "banana": "banano", "orange": "naranja"},
        ),
        FuenteDataset(
            directorio=fuente2_dir,
            mapa_clases={"apple": "manzana", "banana": "banano", "orange": "naranja"},
        ),
    ]
    destino = tmp_path / "splits"

    resumen = construir_splits(fuentes, destino, clases=CLASES_TEST)

    # cada fuente aporta una imagen de clase 0 ("apple" fuente1, "banana" fuente2)
    # -> fuente1 aporta manzana, fuente2 aporta banano (su indice 0 es "banana")
    assert resumen["train"]["n_imagenes"] == 2
    assert resumen["train"]["cajas"]["manzana"] == 1
    assert resumen["train"]["cajas"]["banano"] == 1
    assert (destino / "train" / "images" / "fuente1__img_a.jpg").exists()
    assert (destino / "train" / "images" / "fuente2__img_a.jpg").exists()


def test_construir_splits_ignora_clases_de_la_fuente_sin_mapeo(tmp_path):
    fuente_dir = _crear_fuente_fake(tmp_path, "fuente1", ["apple", "kiwi"])
    # etiqueta la unica caja como clase 1 ("kiwi"), que no tiene mapeo
    for split in ("train", "valid", "test"):
        lbl = fuente_dir / split / "labels"
        for f in lbl.iterdir():
            f.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    fuente = FuenteDataset(directorio=fuente_dir, mapa_clases={"apple": "manzana"})
    destino = tmp_path / "splits"

    resumen = construir_splits([fuente], destino, clases=CLASES_TEST)

    # la imagen se copia pero su label queda vacio (ninguna caja mapeable)
    assert (destino / "train" / "images" / "fuente1__img_a.jpg").exists()
    assert (destino / "train" / "labels" / "fuente1__img_a.txt").read_text() == ""
    assert sum(resumen["train"]["cajas"].values()) == 0


def test_contar_por_clase_sobre_splits_ya_construidos(tmp_path):
    fuente_dir = _crear_fuente_fake(tmp_path, "fuente1", ["apple", "banana", "orange"])
    fuente = FuenteDataset(
        directorio=fuente_dir,
        mapa_clases={"apple": "manzana", "banana": "banano", "orange": "naranja"},
    )
    destino = tmp_path / "splits"
    construir_splits([fuente], destino, clases=CLASES_TEST)

    resumen = contar_por_clase(destino, clases=CLASES_TEST)
    assert resumen["train"]["n_imagenes"] == 1
    assert resumen["val"]["n_imagenes"] == 1
    assert resumen["test"]["n_imagenes"] == 1
