#!/usr/bin/env python3
"""
convert_to_yolo.py — Convierte instance_annotations (poligonos) a cajas YOLO.

Esquema de entrada (WoodScape):
  {
    "annotation": [
       {"id": "...", "segmentation": [[x, y], [x, y], ...],
        "tags": ["car"], "z_order": 0}, ...
    ],
    "image_width": 1280, "image_height": 966, ...
  }

Que hace
--------
- Lee image_width/image_height del propio JSON: no abre ni una imagen.
- Toma el minimo y maximo de cada poligono como bounding box, y la recorta al
  borde de la imagen.
- Filtra cajas degeneradas (--min-area, --min-side).
- Mapea etiquetas segun class_map.json.
- Parte train_pool en train/val POR GRUPOS (columna group_id del manifest),
  estratificado por camara, para no repetir el error de partir por imagen.
- Escribe listas de rutas en lugar de copiar imagenes: no duplica gigabytes.

Salida (en --out-dir)
---------------------
  labels\\<stem>.txt     una linea por objeto: clase cx cy w h (normalizado)
  train.txt, val.txt, test.txt   rutas absolutas de imagenes
  data.yaml             config para Ultralytics
  reporte.csv           por imagen: objetos por clase, descartes
  reporte.txt           resumen legible

Uso:
  python convert_to_yolo.py --root D:\\CV2 --split-dir D:\\CV2\\split_v1 ^
      --out-dir D:\\CV2\\yolo_ds --class-map D:\\CV2\\Scripts\\class_map.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


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
        log(f"[info] Raiz detectada: {candidates[0]}")
        return candidates[0]
    if len(candidates) > 1:
        sys.exit("[ERROR] Varias candidatas: " + ", ".join(str(c) for c in candidates))
    sys.exit(f"[ERROR] No se encontro '{images_dir}' dentro de {root}")


def collect_points(node, acc: list) -> None:
    """Extrae recursivamente pares (x, y) numericos de cualquier anidamiento."""
    if isinstance(node, (list, tuple)):
        if (len(node) == 2 and all(isinstance(v, (int, float)) for v in node)):
            acc.append((float(node[0]), float(node[1])))
            return
        for item in node:
            collect_points(item, acc)


def desenvolver(data):
    """WoodScape envuelve el contenido en una clave con el nombre del archivo.

    Devuelve el nodo que realmente contiene 'annotation', soportando tanto
    {"00000_FV.json": {...}} como {...} directo.
    """
    if isinstance(data, dict):
        if "annotation" in data or "image_width" in data:
            return data
        for v in data.values():
            if isinstance(v, dict) and ("annotation" in v or "image_width" in v):
                return v
    return data


def tags_of(ann: dict):
    t = ann.get("tags")
    if isinstance(t, str):
        return [t]
    if isinstance(t, list):
        return [x for x in t if isinstance(x, str)]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--images-dir", default="rgb_images")
    ap.add_argument("--ann-dir", default="instance_annotations")
    ap.add_argument("--split-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--class-map", type=Path, default=None)
    ap.add_argument("--min-area", type=float, default=300.0,
                    help="Area minima de la caja en pixeles")
    ap.add_argument("--min-side", type=float, default=8.0,
                    help="Lado minimo de la caja en pixeles")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--splits", default="train,val,test",
                    help="Cuales generar. Con solo el paquete de Drive: --splits test")
    ap.add_argument("--seed", type=int, default=20261012)
    args = ap.parse_args()

    root = resolve_root(args.root, args.images_dir)
    img_dir = root / args.images_dir
    ann_dir = root / args.ann_dir
    if not ann_dir.is_dir():
        sys.exit(f"[ERROR] No existe {ann_dir}")

    # --- mapeo de clases
    cm_path = args.class_map or (Path(__file__).parent / "class_map.json")
    if not cm_path.is_file():
        sys.exit(f"[ERROR] No se encuentra el mapeo de clases: {cm_path}")
    cm = json.loads(cm_path.read_text(encoding="utf-8"))
    clases = cm["clases"]
    mapeo = {k: v for k, v in cm["mapeo"].items() if v is not None}
    excluidas = {k for k, v in cm["mapeo"].items() if v is None}
    cls_id = {c: i for i, c in enumerate(clases)}
    for tag, c in mapeo.items():
        if c not in cls_id:
            sys.exit(f"[ERROR] '{tag}' apunta a la clase '{c}', que no esta en 'clases'.")
    log(f"[info] Clases: {clases}")
    log(f"[info] Etiquetas mapeadas: {len(mapeo)} | excluidas explicitamente: {len(excluidas)}")

    # --- splits del manifest
    man = args.split_dir / "manifest.csv"
    if not man.is_file():
        sys.exit(f"[ERROR] No se encuentra {man}")
    filas = list(csv.DictReader(man.open(encoding="utf-8")))
    por_split = defaultdict(list)
    for r in filas:
        por_split[r["split"]].append(r)
    log(f"[info] Manifest: { {k: len(v) for k, v in por_split.items()} }")

    # --- train/val por GRUPOS, estratificado por camara
    import random
    rng = random.Random(args.seed)
    grupos = defaultdict(list)
    for r in por_split["train_pool"]:
        grupos[r["group_id"]].append(r)
    g_por_cam = defaultdict(list)
    for g, rs in grupos.items():
        g_por_cam[Counter(r["camera"] for r in rs).most_common(1)[0][0]].append(g)

    val_rows, train_rows = [], []
    for cam, gs in sorted(g_por_cam.items()):
        gs = sorted(gs)
        rng.shuffle(gs)
        objetivo = int(round(args.val_frac * sum(len(grupos[g]) for g in gs)))
        acum = 0
        for g in gs:
            if acum < objetivo:
                val_rows.extend(grupos[g])
                acum += len(grupos[g])
            else:
                train_rows.extend(grupos[g])
    log(f"[info] train={len(train_rows)} | val={len(val_rows)} | "
        f"test={len(por_split['test'])} (val por grupos, estratificado por camara)")

    # --- conversion
    lbl_dir = args.out_dir / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    hist = Counter()
    tags_vistos = Counter()
    sin_objetos, descartes_chicas, total_cajas = 0, 0, 0
    recortadas = 0
    reporte = []
    faltan_json = []

    pedidos = {s.strip() for s in args.splits.split(",") if s.strip()}
    if pedidos - {"train", "val", "test"}:
        sys.exit(f"[ERROR] --splits solo acepta train, val, test.")
    todas = []
    if "train" in pedidos:
        todas += [(r, "train") for r in train_rows]
    if "val" in pedidos:
        todas += [(r, "val") for r in val_rows]
    if "test" in pedidos:
        todas += [(r, "test") for r in por_split["test"]]
    if pedidos != {"train", "val", "test"}:
        log(f"[info] Generando solo: {sorted(pedidos)}")

    for i, (r, split) in enumerate(todas, 1):
        stem = Path(r["filename"]).stem
        jp = ann_dir / f"{stem}.json"
        if not jp.is_file():
            faltan_json.append(r["filename"])
            continue
        try:
            data = json.loads(jp.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            faltan_json.append(r["filename"])
            continue

        data = desenvolver(data)
        W = float(data.get("image_width") or 0)
        H = float(data.get("image_height") or 0)
        if W <= 0 or H <= 0:
            faltan_json.append(r["filename"])
            continue

        lineas, por_clase = [], Counter()
        for ann in data.get("annotation", []) or []:
            if not isinstance(ann, dict):
                continue
            etiquetas = tags_of(ann)
            for t in etiquetas:
                tags_vistos[t] += 1
            destino = next((mapeo[t] for t in etiquetas if t in mapeo), None)
            if destino is None:
                continue

            pts = []
            collect_points(ann.get("segmentation"), pts)
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, x2 = max(0.0, min(xs)), min(W, max(xs))
            y1, y2 = max(0.0, min(ys)), min(H, max(ys))
            if min(xs) < 0 or max(xs) > W or min(ys) < 0 or max(ys) > H:
                recortadas += 1
            w, h = x2 - x1, y2 - y1
            if w < args.min_side or h < args.min_side or w * h < args.min_area:
                descartes_chicas += 1
                continue

            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            lineas.append(f"{cls_id[destino]} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
            por_clase[destino] += 1
            hist[destino] += 1
            total_cajas += 1

        (lbl_dir / f"{stem}.txt").write_text("\n".join(lineas) + ("\n" if lineas else ""),
                                             encoding="utf-8")
        if not lineas:
            sin_objetos += 1
        reporte.append({"filename": r["filename"], "split": split, "camera": r["camera"],
                        **{c: por_clase.get(c, 0) for c in clases}})

        if i % 1000 == 0:
            log(f"    {i}/{len(todas)}")

    # --- listas de imagenes
    def escribir_lista(nombre: str, rows):
        p = args.out_dir / nombre
        with p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(str((img_dir / r["filename"]).resolve()) + "\n")
        return p

    if "train" in pedidos:
        escribir_lista("train.txt", train_rows)
    if "val" in pedidos:
        escribir_lista("val.txt", val_rows)
    if "test" in pedidos:
        escribir_lista("test.txt", por_split["test"])

    partes = ["# Generado por convert_to_yolo.py",
              f"path: {args.out_dir.resolve()}"]
    for s in ("train", "val", "test"):
        if s in pedidos:
            partes.append(f"{s}: {s}.txt")
    partes.append(f"nc: {len(clases)}")
    partes.append(f"names: {clases}")
    yaml = "\n".join(partes) + "\n"
    (args.out_dir / "data.yaml").write_text(yaml, encoding="utf-8")

    # --- reportes
    with (args.out_dir / "reporte.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["filename", "split", "camera"] + clases)
        w.writeheader()
        w.writerows(reporte)

    no_mapeadas = [(t, n) for t, n in tags_vistos.most_common()
                   if t not in mapeo and t not in excluidas]

    resumen = []
    resumen.append(f"Imagenes procesadas : {len(reporte)}")
    resumen.append(f"Cajas generadas     : {total_cajas}")
    resumen.append(f"Sin objetos         : {sin_objetos} "
                   f"({100 * sin_objetos / max(1, len(reporte)):.1f}%)")
    resumen.append(f"Descartadas x tamano: {descartes_chicas} "
                   f"(min_area={args.min_area}, min_side={args.min_side})")
    resumen.append(f"Poligonos recortados al borde: {recortadas}")
    resumen.append("")
    resumen.append("Cajas por clase:")
    for c in clases:
        resumen.append(f"  {c:<14} {hist.get(c, 0):>8} "
                       f"({100 * hist.get(c, 0) / max(1, total_cajas):.1f}%)")
    resumen.append("")
    resumen.append("Cajas por split:")
    for s in ("train", "val", "test"):
        rows = [r for r in reporte if r["split"] == s]
        n = sum(r[c] for r in rows for c in clases)
        resumen.append(f"  {s:<6} imagenes={len(rows):>6}  cajas={n:>8}")
    if faltan_json:
        resumen.append("")
        resumen.append(f"[AVISO] Sin JSON legible: {len(faltan_json)} "
                       f"(ej: {faltan_json[:3]})")
    if no_mapeadas:
        resumen.append("")
        resumen.append("Etiquetas presentes pero NO mapeadas (semanticas o pendientes):")
        for t, n in no_mapeadas[:15]:
            resumen.append(f"  {t:<32} {n}")

    texto = "\n".join(resumen)
    (args.out_dir / "reporte.txt").write_text(texto, encoding="utf-8")
    log("\n" + texto)
    log(f"\n[OK] Dataset YOLO en {args.out_dir.resolve()}")
    log("     Las imagenes NO se copiaron: data.yaml apunta a las originales.")


if __name__ == "__main__":
    main()
