"""Gestion del dataset: combina fuentes (Roboflow export, fotos propias)
en la estructura data/splits/{train,val,test} que usa el resto del
pipeline. Ver docs/ARCHITECTURE.md S2 (dataset.py) y docs/PLAN.md Fase 1.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from visionartificial.config import CLASSES

# Un export de Roboflow nombra el split de validacion "valid"; el resto
# del proyecto usa "val" (convencion de ultralytics). Este mapa traduce
# entre los nombres de carpeta de la fuente y los nombres de split del
# proyecto.
SPLITS_DESTINO = ("train", "val", "test")
_SPLITS_FUENTE_POR_DEFECTO = {"train": "train", "val": "valid", "test": "test"}


@dataclass(frozen=True)
class FuenteDataset:
    """Una fuente de imagenes etiquetadas para combinar en data/splits.

    `mapa_clases` traduce el nombre de clase tal como aparece en el
    `data.yaml` de la fuente hacia un nombre presente en `config.CLASSES`
    (p. ej. {"apple": "manzana"}). Una clase de la fuente ausente del
    mapa se ignora (sus cajas no se copian) en vez de fallar, para poder
    reusar datasets publicos con clases de sobra.
    """

    directorio: Path
    mapa_clases: dict[str, str]
    carpetas_split: dict[str, str] = field(
        default_factory=lambda: dict(_SPLITS_FUENTE_POR_DEFECTO)
    )
    prefijo: str | None = None

    def nombre(self) -> str:
        return self.prefijo or self.directorio.name


def _leer_clases_fuente(directorio: Path) -> list[str]:
    data_yaml = directorio / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"No se encontro data.yaml en {directorio}")
    contenido = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    return list(contenido["names"])


def _mapa_indices(
    fuente: FuenteDataset, clases_destino: list[str]
) -> dict[int, int]:
    nombres_fuente = _leer_clases_fuente(fuente.directorio)
    mapa: dict[int, int] = {}
    for idx_fuente, nombre_fuente in enumerate(nombres_fuente):
        nombre_destino = fuente.mapa_clases.get(nombre_fuente)
        if nombre_destino is None:
            continue
        if nombre_destino not in clases_destino:
            raise ValueError(
                f"'{nombre_destino}' (mapeado desde '{nombre_fuente}' en "
                f"{fuente.directorio}) no esta en clases_destino={clases_destino}"
            )
        mapa[idx_fuente] = clases_destino.index(nombre_destino)
    return mapa


def _copiar_split(
    fuente: FuenteDataset,
    split_fuente: str,
    split_destino: str,
    destino: Path,
    mapa_idx: dict[int, int],
) -> int:
    img_src_dir = fuente.directorio / split_fuente / "images"
    lbl_src_dir = fuente.directorio / split_fuente / "labels"
    if not img_src_dir.exists():
        return 0

    img_dst_dir = destino / split_destino / "images"
    lbl_dst_dir = destino / split_destino / "labels"
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)

    n_copiadas = 0
    for img_path in sorted(img_src_dir.iterdir()):
        if not img_path.is_file():
            continue
        lbl_path = lbl_src_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        nombre_nuevo = f"{fuente.nombre()}__{img_path.name}"
        shutil.copy2(img_path, img_dst_dir / nombre_nuevo)

        lineas_nuevas = []
        for linea in lbl_path.read_text(encoding="utf-8").splitlines():
            if not linea.strip():
                continue
            partes = linea.split()
            idx_original = int(partes[0])
            idx_nuevo = mapa_idx.get(idx_original)
            if idx_nuevo is None:
                continue
            lineas_nuevas.append(" ".join([str(idx_nuevo), *partes[1:]]))

        lbl_dst_path = lbl_dst_dir / (Path(nombre_nuevo).stem + ".txt")
        lbl_dst_path.write_text("\n".join(lineas_nuevas), encoding="utf-8")
        n_copiadas += 1

    return n_copiadas


def construir_splits(
    fuentes: list[FuenteDataset],
    destino: Path,
    clases: list[str] | None = None,
) -> dict[str, dict]:
    """Combina una o mas fuentes en destino/{train,val,test}.

    Devuelve un resumen por split: numero de imagenes y cajas por clase,
    util para verificar la Fase 1 (docs/PLAN.md) antes de entrenar.
    """
    clases = list(clases or CLASSES)
    destino.mkdir(parents=True, exist_ok=True)

    for split_destino in SPLITS_DESTINO:
        for fuente in fuentes:
            mapa_idx = _mapa_indices(fuente, clases)
            split_fuente = fuente.carpetas_split.get(split_destino, split_destino)
            _copiar_split(fuente, split_fuente, split_destino, destino, mapa_idx)

    _escribir_data_yaml(destino, clases)
    return contar_por_clase(destino, clases)


def _escribir_data_yaml(destino: Path, clases: list[str]) -> None:
    contenido = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(clases),
        "names": clases,
    }
    (destino / "data.yaml").write_text(
        yaml.safe_dump(contenido, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def contar_por_clase(splits_dir: Path, clases: list[str] | None = None) -> dict[str, dict]:
    """Cuenta imagenes y cajas por clase en cada split ya construido."""
    clases = list(clases or CLASSES)
    resumen: dict[str, dict] = {}
    for split in SPLITS_DESTINO:
        lbl_dir = splits_dir / split / "labels"
        cajas = {clase: 0 for clase in clases}
        n_imagenes = 0
        if lbl_dir.exists():
            for lbl_path in lbl_dir.iterdir():
                n_imagenes += 1
                for linea in lbl_path.read_text(encoding="utf-8").splitlines():
                    if not linea.strip():
                        continue
                    idx = int(linea.split()[0])
                    cajas[clases[idx]] += 1
        resumen[split] = {"n_imagenes": n_imagenes, "cajas": cajas}
    return resumen
