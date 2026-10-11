#!/usr/bin/env python3
"""
make_drive_package.py — Arma el paquete liviano para subir a Drive.

Copia las 1000 imagenes del hold-out mas sus anotaciones, junto con los
archivos del split, en una sola carpeta lista para comprimir y subir.

Las imagenes se copian bit a bit, sin recomprimir: si alguien las pasa a JPEG
para ahorrar espacio, cambian los pixeles, dejan de coincidir con los sha256
del manifest y las metricas dejan de ser comparables entre ramas.

Uso:
  python make_drive_package.py --root D:\\CV2 --split-dir D:\\CV2\\split_v1
  python make_drive_package.py --root D:\\CV2 --split-dir D:\\CV2\\split_v1 --zip
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

SPLIT_FILES = ("train_pool.txt", "quarantine.txt", "manifest.csv",
               "split_config.json", "leakage_audit.csv", "README_DATASET.md")


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024 or unit == "GB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} GB"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def resolve_root(root: Path, images_dir: str, max_depth: int = 2) -> Path:
    if (root / images_dir).is_dir():
        return root
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
        print(f"[info] Raiz detectada: {candidates[0]}")
        return candidates[0]
    if len(candidates) > 1:
        sys.exit("[ERROR] Varias candidatas: " + ", ".join(str(c) for c in candidates))
    sys.exit(f"[ERROR] No se encontro '{images_dir}' dentro de {root}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--images-dir", default="rgb_images")
    ap.add_argument("--ann-dir", default="instance_annotations")
    ap.add_argument("--split-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="Por defecto <split-dir>\\drive_package")
    ap.add_argument("--zip", action="store_true", help="Comprime la carpeta al final")
    ap.add_argument("--sin-imagenes", action="store_true",
                    help="Solo listas y manifest, sin copiar las imagenes")
    ap.add_argument("--verify", action="store_true",
                    help="Recalcula sha256 de lo copiado contra el manifest")
    args = ap.parse_args()

    root = resolve_root(args.root, args.images_dir)
    out = args.out or (args.split_dir / "drive_package")

    # --- lista de test
    listas = sorted(args.split_dir.glob("test_*.txt"))
    if not listas:
        sys.exit(f"[ERROR] No se encontro test_*.txt en {args.split_dir}")
    lista = listas[0]
    nombres = [n.strip() for n in lista.read_text(encoding="utf-8").splitlines() if n.strip()]
    print(f"[info] {lista.name}: {len(nombres)} imagenes")

    # --- sha esperados
    esperado = {}
    man = args.split_dir / "manifest.csv"
    if man.is_file():
        for r in csv.DictReader(man.open(encoding="utf-8")):
            if r.get("sha256"):
                esperado[r["filename"]] = r["sha256"]

    img_src = root / args.images_dir
    ann_src = root / args.ann_dir
    img_dst = out / "test_set" / args.images_dir
    ann_dst = out / "test_set" / args.ann_dir
    img_dst.mkdir(parents=True, exist_ok=True)
    if ann_src.is_dir():
        ann_dst.mkdir(parents=True, exist_ok=True)
    else:
        print(f"[AVISO] No existe {ann_src}: se copian solo las imagenes.")

    # --- chequeo de espacio previo
    necesario = 0
    for nombre in (nombres if not args.sin_imagenes else []):
        src = img_src / nombre
        if src.is_file():
            necesario += src.stat().st_size
    if ann_src.is_dir():
        necesario = int(necesario * 1.02)  # margen por las anotaciones
    libre = shutil.disk_usage(out.anchor or str(out)).free
    print(f"[info] Espacio necesario: {human(necesario)} | libre en destino: {human(libre)}")
    if libre < necesario * 1.1:
        print(f"\n[ERROR] No hay espacio suficiente en {out.anchor or out}.")
        print(f"        Hacen falta ~{human(int(necesario * 1.1))} y hay {human(libre)}.")
        print("        Opciones:")
        print("          - Liberar espacio y reintentar (la copia se reanuda).")
        print("          - Mandar el paquete a otro disco con --out, por ejemplo:")
        print("              --out C:\\CV2_drive_package")
        print("          - Subir solo los .txt y el manifest, sin imagenes: --sin-imagenes")
        sys.exit(1)

    total, faltantes_img, faltantes_ann, malos, saltadas = 0, [], [], [], 0
    for i, nombre in enumerate(nombres if not args.sin_imagenes else [], 1):
        src = img_src / nombre
        if not src.is_file():
            faltantes_img.append(nombre)
            continue
        dst = img_dst / nombre
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            saltadas += 1
        else:
            shutil.copy2(src, dst)
        total += dst.stat().st_size
        if args.verify and nombre in esperado and sha256_of(dst) != esperado[nombre]:
            malos.append(nombre)

        if ann_src.is_dir():
            stem = Path(nombre).stem
            hallado = False
            for cand in ann_src.glob(stem + ".*"):
                shutil.copy2(cand, ann_dst / cand.name)
                total += (ann_dst / cand.name).stat().st_size
                hallado = True
            if not hallado:
                faltantes_ann.append(nombre)

        if i % 200 == 0:
            print(f"    {i}/{len(nombres)}  ({human(total)})")

    # --- archivos del split
    copiados = []
    for f in (lista.name,) + SPLIT_FILES:
        p = args.split_dir / f
        if p.is_file():
            shutil.copy2(p, out / f)
            total += p.stat().st_size
            copiados.append(f)

    print(f"\n=== Paquete armado en {out} ===")
    copiadas = 0 if args.sin_imagenes else len(nombres) - len(faltantes_img)
    print(f"  Imagenes copiadas   : {copiadas}" + (f" ({saltadas} ya estaban)" if saltadas else ""))
    print(f"  Anotaciones         : {0 if args.sin_imagenes else len(nombres) - len(faltantes_ann)}")
    print(f"  Archivos del split  : {', '.join(copiados)}")
    print(f"  Peso total          : {human(total)}")
    if faltantes_img:
        print(f"  [AVISO] Imagenes no encontradas: {len(faltantes_img)} "
              f"(ej: {faltantes_img[:3]})")
    if faltantes_ann:
        print(f"  [AVISO] Sin anotacion: {len(faltantes_ann)} (ej: {faltantes_ann[:3]})")
    if malos:
        print(f"  [ERROR] Copias con sha256 distinto: {len(malos)} -> {malos[:3]}")
    elif args.verify:
        print("  [OK] Todas las copias coinciden con el manifest.")

    if args.zip:
        zpath = out.with_suffix(".zip")
        print(f"\n[info] Comprimiendo en {zpath.name} (los PNG ya estan comprimidos, "
              f"asi que no va a achicarse mucho)...")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(out))
        print(f"[OK] {zpath} ({human(zpath.stat().st_size)})")

    print("\nSubir a Drive: la carpeta completa.")
    print("No recomprimir las imagenes: cambian los sha256 y rompen la comparabilidad.")


if __name__ == "__main__":
    main()
