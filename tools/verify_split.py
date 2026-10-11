#!/usr/bin/env python3
"""
verify_split.py — Verifica que una copia local de WoodScape coincida con el split oficial.

Para los que ya bajaron el dataset por su cuenta: no hace falta descargar las
imagenes de nuevo. Este script compara la copia local contra manifest.csv y
avisa si falta algo o si algun archivo difiere (mirrors distintos, recompresion,
descarga incompleta).

Uso:
  python verify_split.py --root /ruta/woodscape --split-dir ./split_v1
  python verify_split.py --root /ruta/woodscape --split-dir ./split_v1 --fast

--fast omite el sha256 y solo chequea presencia de archivos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def resolve_root(root: Path, images_dir: str, max_depth: int = 2) -> Path:
    """Permite apuntar a una carpeta padre; busca la que contiene images_dir."""
    if (root / images_dir).is_dir():
        return root
    if not root.is_dir():
        sys.exit(f"[ERROR] No existe la ruta: {root}")

    candidates, frontier = [], [(root, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        try:
            children = sorted(d for d in current.iterdir() if d.is_dir())
        except PermissionError:
            continue
        for child in children:
            if (child / images_dir).is_dir():
                candidates.append(child)
            else:
                frontier.append((child, depth + 1))

    if len(candidates) == 1:
        print(f"[info] Raiz del dataset detectada en: {candidates[0]}")
        return candidates[0]
    if len(candidates) > 1:
        opciones = "\n  ".join(str(c) for c in candidates)
        sys.exit(f"[ERROR] Hay mas de una carpeta con '{images_dir}':\n  {opciones}")
    sys.exit(f"[ERROR] No se encontro '{images_dir}' dentro de {root}.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--images-dir", default="rgb_images")
    ap.add_argument("--split-dir", required=True, type=Path)
    ap.add_argument("--fast", action="store_true", help="Omite el sha256")
    args = ap.parse_args()

    manifest = args.split_dir / "manifest.csv"
    if not manifest.is_file():
        sys.exit(f"[ERROR] No se encuentra {manifest}")

    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    root = resolve_root(args.root, args.images_dir)
    img_dir = root / args.images_dir
    if not img_dir.is_dir():
        sys.exit(f"[ERROR] No se encuentra {img_dir}")

    missing, mismatched = [], []
    for i, r in enumerate(rows, 1):
        p = img_dir / r["filename"]
        if not p.is_file():
            missing.append(r["filename"])
            continue
        if not args.fast and r.get("sha256"):
            if sha256_of(p) != r["sha256"]:
                mismatched.append(r["filename"])
        if i % 1000 == 0:
            print(f"    {i}/{len(rows)}", flush=True)

    print(f"\nArchivos en manifest : {len(rows)}")
    print(f"Faltantes            : {len(missing)}")
    print(f"Con hash distinto    : {len(mismatched)}")
    print(f"Distribucion splits  : {dict(Counter(r['split'] for r in rows))}")

    if missing:
        print(f"\nEjemplos faltantes: {missing[:5]}")
    if mismatched:
        print(f"\nEjemplos con hash distinto: {mismatched[:5]}")
        print("Probablemente tu copia viene de otro mirror. Usa las imagenes del Drive "
              "para esos archivos, o el equipo entero pasa al mismo mirror.")

    if missing or mismatched:
        sys.exit(1)

    print("\n[OK] Tu copia local coincide con el split oficial. Podes entrenar.")
    print("     Recordatorio: quarantine.txt no se usa ni para entrenar ni para evaluar.")


if __name__ == "__main__":
    main()
