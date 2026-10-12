#!/usr/bin/env python3
"""
hacer_video.py - arma un video con imagenes de WoodScape para probar la app como si fuera una camara en vivo.

El release de 8.2K no tiene secuencias contiguas (DATASET_SPLIT.md, seccion 6): cada imagen es otra escena, asi
que el video es una sucesion de fotos y no un recorrido. Sirve para medir si el detector llega a procesar al ritmo
de la camara, no para seguimiento entre frames. Cada imagen se repite --repetir frames: a 30 fps y 15 repeticiones
la camara "ve" cada escena medio segundo.

Por defecto usa las imagenes del test del equipo (splits/test_1000.txt), que ningun modelo vio al entrenar.
Desde la raiz del repo:

  uv run python tools/hacer_video.py --camara FV               # outputs/videos/test_FV.mp4
  uv run python tools/hacer_video.py --camara MVR --n 60 --repetir 30

Ademas del mp4 escribe <nombre>.json con el stem de cada imagen y las repeticiones, para que la app use la
calibracion de cada imagen en la proyeccion cilindrica.
"""
import argparse
import json
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parents[1]
CAMARAS = ("FV", "RV", "MVL", "MVR")


def carpeta_imagenes():
    """rgb_images de Kaggle en la raiz, o el test del Drive (CV2/) si no esta"""
    for d in (RAIZ / "rgb_images", RAIZ / "CV2/split_v1/drive_package/test_set/rgb_images"):
        if d.is_dir():
            return d
    return RAIZ / "rgb_images"


def abrir_writer(ruta, fps, tam):
    """h264 (avc1) si opencv lo trae, que se reproduce en cualquier lado; si no, mpeg-4 (mp4v)"""
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*codec), fps, tam)
        if w.isOpened():
            return w, codec
    raise RuntimeError(f"opencv no pudo abrir un VideoWriter para {ruta}")


def main():
    ap = argparse.ArgumentParser(description="video de una camara de WoodScape para el modo en vivo de la app")
    ap.add_argument("--camara", choices=CAMARAS, default="FV")
    ap.add_argument("--lista", type=Path, default=RAIZ / "splits/test_1000.txt",
                    help="imagenes a usar, un nombre por linea (por defecto el test del equipo)")
    ap.add_argument("--imagenes", type=Path, default=None, help="carpeta con los png (por defecto rgb_images/)")
    ap.add_argument("--n", type=int, default=None, help="cantidad de imagenes (por defecto todas las de la camara)")
    ap.add_argument("--fps", type=float, default=30, help="fps de la camara simulada")
    ap.add_argument("--repetir", type=int, default=15, help="frames seguidos de cada imagen")
    ap.add_argument("--salida", type=Path, default=None, help="por defecto outputs/videos/test_<camara>.mp4")
    args = ap.parse_args()

    imagenes = args.imagenes or carpeta_imagenes()
    nombres = [l.strip() for l in args.lista.read_text().splitlines() if l.strip().endswith(f"_{args.camara}.png")]
    nombres = sorted(nombres)[: args.n]  # el orden de los indices no tiene estructura temporal, da lo mismo
    faltan = [n for n in nombres if not (imagenes / n).is_file()]
    if faltan:
        raise SystemExit(f"faltan {len(faltan)} imagenes en {imagenes} (por ejemplo {faltan[0]})")
    if not nombres:
        raise SystemExit(f"no hay imagenes de {args.camara} en {args.lista}")

    salida = args.salida or RAIZ / f"outputs/videos/test_{args.camara}.mp4"
    salida.parent.mkdir(parents=True, exist_ok=True)
    primera = cv2.imread(str(imagenes / nombres[0]))
    alto, ancho = primera.shape[:2]
    writer, codec = abrir_writer(salida, args.fps, (ancho, alto))
    for i, n in enumerate(nombres, 1):
        img = cv2.imread(str(imagenes / n))
        if img.shape[:2] != (alto, ancho):
            raise SystemExit(f"{n} es de {img.shape[1]}x{img.shape[0]} y el video de {ancho}x{alto}")
        for _ in range(args.repetir):
            writer.write(img)
        if i % 50 == 0:
            print(f"{i}/{len(nombres)}", flush=True)
    writer.release()

    meta = {"camara": args.camara, "fps": args.fps, "repetir": args.repetir, "ancho": ancho, "alto": alto,
            "lista": args.lista.name, "stems": [n.removesuffix(".png") for n in nombres]}
    salida.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    seg = len(nombres) * args.repetir / args.fps
    print(f"{salida} | {len(nombres)} imagenes x {args.repetir} = {len(nombres) * args.repetir} frames | "
          f"{args.fps:g} fps | {seg:.0f} s | {codec} | {salida.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
