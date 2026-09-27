"""Fase 2 (docs/PLAN.md): reentrena YOLOv8 por transfer learning sobre
data/splits/data.yaml y copia los mejores pesos a models/best.pt.

Uso:
    py scripts/train.py
    py scripts/train.py --epochs 50 --imgsz 512
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO  # noqa: E402

from visionartificial.config import MODELS_DIR, SPLITS_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument(
        "--modelo-base",
        default="yolov8n.pt",
        help="Pesos preentrenados de partida (transfer learning, PDF S3.3.2).",
    )
    args = parser.parse_args()

    data_yaml = SPLITS_DIR / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(
            f"No existe {data_yaml}. Corre la Fase 1 (visionartificial.dataset) primero."
        )

    modelo = YOLO(args.modelo_base)
    resultado = modelo.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(MODELS_DIR / "runs"),
        name="fase2",
        exist_ok=True,
    )

    mejor_pt = Path(resultado.save_dir) / "weights" / "best.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    destino = MODELS_DIR / "best.pt"
    shutil.copy2(mejor_pt, destino)
    print(f"\nModelo entrenado copiado a: {destino}")


if __name__ == "__main__":
    main()
