#!/usr/bin/env python3
"""
build_calibration.py - junta las calibraciones de Valeo (un json por imagen) en config/calibration.json.

WoodScape trae un json de calibracion por imagen, pero en las 8234 hay solo 23 juegos distintos de
parametros intrinsecos. Este script guarda cada juego una vez y, por imagen, su centro optico, su camara
y el indice de su juego:

  juegos:   [{k1, k2, k3, k4, aspect_ratio, width, height}, ...]   en el orden en que aparecen
  imagenes: {stem: {cx, cy, cam, juego}}                           cx = cx_offset + width/2 - 0.5

El modelo de la lente es rho(theta) = k1 th + k2 th^2 + k3 th^3 + k4 th^4. Lo usan eval_radial.py (theta de
cada caja) y fisheye_aug.py (centro optico). Uso, desde la raiz del repo:

  uv run python tools/build_calibration.py --calib-dir CV2/calibration --out config/calibration.json
"""
import argparse
import json
from pathlib import Path

CLAVES_JUEGO = ("k1", "k2", "k3", "k4", "aspect_ratio", "width", "height")


def main():
    ap = argparse.ArgumentParser(description="consolida las calibraciones de Valeo en un solo json")
    ap.add_argument("--calib-dir", required=True, help="carpeta con un json de Valeo por imagen (<stem>.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    archivos = sorted(Path(args.calib_dir).glob("*.json"))
    if not archivos:
        raise SystemExit(f"no hay json en {args.calib_dir}")
    juegos, indice, imagenes = [], {}, {}
    for p in archivos:
        c = json.loads(p.read_text())
        i = c["intrinsic"]
        # k1..k4 con 6 decimales, como el archivo versionado; aspect_ratio y el tamano tal cual
        juego = {k: (round(i[k], 6) if k in ("k1", "k2", "k3", "k4") else i[k]) for k in CLAVES_JUEGO}
        clave = tuple(juego.values())
        if clave not in indice:
            indice[clave] = len(juegos)
            juegos.append(juego)
        imagenes[p.stem] = {"cx": round(i["cx_offset"] + i["width"] / 2 - 0.5, 3),
                            "cy": round(i["cy_offset"] + i["height"] / 2 - 0.5, 3),
                            "cam": c.get("name", p.stem.split("_")[-1]), "juego": indice[clave]}

    salida = {"fuente": "WoodScape calibration (Valeo)",
              "nota_centro_optico": "cx = cx_offset + width/2 - 0.5 ; cy = cy_offset + height/2 - 0.5",
              "modelo": "rho(theta) = k1*th + k2*th^2 + k3*th^3 + k4*th^4",
              "juegos": juegos, "imagenes": imagenes}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # fin de linea CRLF como el archivo versionado (se genero en Windows): regenerarlo no deja diferencias en git
    with open(args.out, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(json.dumps(salida, indent=1, ensure_ascii=False))
    print(f"{len(imagenes)} imagenes, {len(juegos)} juegos de parametros -> {args.out}")


if __name__ == "__main__":
    main()
