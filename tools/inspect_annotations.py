#!/usr/bin/env python3
"""
inspect_annotations.py — Muestra el esquema real de instance_annotations
y cuenta las etiquetas presentes.

Sirve para dos cosas:
  1. Ver como estan estructurados los JSON antes de escribir el conversor
     poligono -> bounding box.
  2. Tener el histograma de etiquetas, que es lo que define cuales son las
     tres clases definitivas del trabajo.

Uso:
  python inspect_annotations.py --root D:\\CV2 --sample 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

LABEL_KEYS = ("tags", "tag", "label", "labels", "class", "category",
              "class_name", "name", "type")


def resolve_root(root: Path, needle: str, max_depth: int = 2) -> Path:
    if (root / needle).is_dir():
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
            if (child / needle).is_dir():
                candidates.append(child)
            else:
                frontier.append((child, depth + 1))
    if len(candidates) == 1:
        print(f"[info] Raiz detectada: {candidates[0]}")
        return candidates[0]
    if len(candidates) > 1:
        sys.exit("[ERROR] Varias candidatas: " + ", ".join(str(c) for c in candidates))
    sys.exit(f"[ERROR] No se encontro '{needle}' dentro de {root}")


def outline(node, indent: int = 0, max_items: int = 3, max_depth: int = 6) -> str:
    """Resumen legible de la estructura, truncando listas largas."""
    pad = "  " * indent
    if indent > max_depth:
        return pad + "...\n"
    if isinstance(node, dict):
        out = ""
        for k, v in list(node.items())[:12]:
            if isinstance(v, (dict, list)):
                out += f"{pad}{k}:\n" + outline(v, indent + 1, max_items, max_depth)
            else:
                shown = repr(v)
                if len(shown) > 70:
                    shown = shown[:70] + "..."
                out += f"{pad}{k}: {shown}\n"
        return out
    if isinstance(node, list):
        out = f"{pad}[lista de {len(node)} elementos]\n"
        for item in node[:max_items]:
            out += outline(item, indent + 1, max_items, max_depth)
        if len(node) > max_items:
            out += f"{pad}  ... ({len(node) - max_items} mas)\n"
        return out
    shown = repr(node)
    if len(shown) > 70:
        shown = shown[:70] + "..."
    return f"{pad}{shown}\n"


def collect_labels(node, counter: Counter) -> None:
    """Busca recursivamente valores bajo claves que parezcan etiquetas."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k in LABEL_KEYS:
                if isinstance(v, str):
                    counter[v] += 1
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, str):
                            counter[item] += 1
            collect_labels(v, counter)
    elif isinstance(node, list):
        for item in node:
            collect_labels(item, counter)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--ann-dir", default="instance_annotations")
    ap.add_argument("--sample", type=int, default=3,
                    help="Cuantos JSON mostrar completos")
    ap.add_argument("--count-all", action="store_true",
                    help="Recorre TODOS los JSON para el histograma (mas lento)")
    args = ap.parse_args()

    root = resolve_root(args.root, args.ann_dir)
    ann_dir = root / args.ann_dir
    files = sorted(ann_dir.glob("*.json"))
    if not files:
        files = sorted(p for p in ann_dir.iterdir() if p.is_file())
    if not files:
        sys.exit(f"[ERROR] {ann_dir} esta vacio")

    print(f"\n=== {len(files)} archivos en {ann_dir.name} ===")
    print(f"Extensiones: {dict(Counter(p.suffix for p in files))}")
    print(f"Primeros nombres: {[p.name for p in files[:3]]}")

    print(f"\n=== ESQUEMA (primeros {args.sample} archivos) ===")
    for p in files[:args.sample]:
        print(f"\n--- {p.name} ---")
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception as e:
            print(f"  [no es JSON valido: {e}]")
            print("  Primeros 300 caracteres:")
            print("  " + p.read_text(encoding="utf-8", errors="ignore")[:300])
            continue
        print(outline(data), end="")

    scope = files if args.count_all else files[:1500]
    print(f"\n=== ETIQUETAS ({len(scope)} archivos) ===")
    counter: Counter = Counter()
    empty = 0
    for i, p in enumerate(scope, 1):
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            empty += 1
            continue
        collect_labels(data, counter)
        if i % 500 == 0:
            print(f"    {i}/{len(scope)}")

    if not counter:
        print("  No se detectaron etiquetas con las claves habituales.")
        print(f"  Claves buscadas: {LABEL_KEYS}")
        print("  Mirar el esquema de arriba y avisar cual es la clave correcta.")
    else:
        total = sum(counter.values())
        print(f"\n  {'etiqueta':<28} {'instancias':>10} {'%':>7}")
        print("  " + "-" * 47)
        for label, n in counter.most_common(40):
            print(f"  {label:<28} {n:>10} {100 * n / total:>6.1f}%")
        print(f"\n  Total instancias: {total}")
    if empty:
        print(f"  Archivos ilegibles: {empty}")

    if not args.count_all and len(files) > len(scope):
        print(f"\n[info] Muestra parcial. Para el conteo completo: --count-all")


if __name__ == "__main__":
    main()
