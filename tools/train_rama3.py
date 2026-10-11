#!/usr/bin/env python3
"""
train_rama3.py - entrena una variante de la rama 3 fuera del notebook (la Mac con MPS o una GPU NVIDIA).

Hace lo mismo que la seccion de entrenamiento de notebooks/rama3_finetune.ipynb: arma la estructura images/ +
labels/ con preparar_yolo_ds (controla los numeros del equipo) y llama a entrenar() de fisheye_aug.py.

Asi se entreno la variante base, en la Mac (MPS), desde la raiz del repo:

  caffeinate -i uv run python tools/train_rama3.py --variante base --epochs 30 --device mps \
      --imagenes rgb_images --ds outputs/rama3/yolo_ds --salida outputs/rama3/mac \
      --modelo yolo11m.pt --cache ram 2>&1 | tee outputs/rama3/mac/train_base.log

Con una GPU NVIDIA es lo mismo con --device 0 (sin caffeinate). En Colab lo llama notebooks/rama3_colab.ipynb.
La extension se entrena con --variante radial. Si se corren las dos variantes a la vez, cada una necesita su propia
estructura (--ds) para no escribir al mismo tiempo el mismo labels.cache.
Las etiquetas y las listas por defecto son las de outputs/yolo_ds (convert_to_yolo.py).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from fisheye_aug import AUG_RADIAL, CALIBRACION, entrenar, preparar_yolo_ds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", choices=["base", "radial"], required=True)
    ap.add_argument("--epochs", type=int, required=True, help="el presupuesto acordado con la rama 4")
    ap.add_argument("--device", default="0", help="0, 1, mps o cpu")
    ap.add_argument("--imagenes", type=Path, required=True, help="carpeta rgb_images del dataset")
    ap.add_argument("--etiquetas", type=Path, default=RAIZ / "outputs/yolo_ds/labels")
    ap.add_argument("--listas", type=Path, default=RAIZ / "outputs/yolo_ds", help="carpeta con train/val/test.txt")
    ap.add_argument("--ds", type=Path, required=True, help="donde armar images/ + labels/ + data.yaml")
    ap.add_argument("--salida", type=Path, required=True, help="project de ultralytics")
    ap.add_argument("--modelo", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cache", default="ram", help="ram, disk o false")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraccion", type=float, default=1.0)
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--calib", type=Path, default=CALIBRACION, help="calibration.json para el centro optico")
    a = ap.parse_args()

    cache = False if a.cache.lower() in {"false", "no", "0", "none"} else a.cache
    data, conteo = preparar_yolo_ds(a.listas, a.etiquetas, a.imagenes, a.ds)
    print(f"[{a.variante}] dataset {data}: {conteo}", flush=True)
    if a.variante == "radial":
        print(f"[{a.variante}] augmentacion radial: {AUG_RADIAL} | centros: {a.calib}", flush=True)

    t0 = time.time()
    best = entrenar(a.variante, data, a.salida, a.epochs, modelo=a.modelo, imgsz=a.imgsz, batch=a.batch,
                    device=a.device, cache=cache, fraction=a.fraccion, close_mosaic=a.close_mosaic,
                    workers=a.workers, calib=a.calib)
    print(f"[{a.variante}] listo en {(time.time() - t0) / 3600:.2f} h -> {best}", flush=True)


if __name__ == "__main__":
    main()
