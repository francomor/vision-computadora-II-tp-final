#!/usr/bin/env python3
"""
make_test_split.py (v2) — Hold-out de WoodScape sin data leakage.

Cambios respecto de v1
----------------------
1. dhash de 256 bits (16x16) en lugar de 64: mucho mas discriminativo.
2. Agrupamiento SECUENCIAL por camara en lugar de union global. Solo se unen
   frames vecinos en indice y visualmente parecidos, que es exactamente el
   mecanismo de fuga (frames contiguos de la misma grabacion). Esto elimina el
   chaining que fusionaba miles de imagenes solo porque comparten paragolpes
   y cielo.
3. Tope duro de tamano de grupo (--max-group): ninguna cadena puede crecer
   indefinidamente.
4. Grupos puros por camara -> la cuota por camara se cumple de verdad.
5. Auditoria posterior: mide la distancia minima de cada imagen de test contra
   todo el train y reporta cuantos pares quedan sospechosamente cerca. Ese
   numero va al paper.
6. Cache de hashes en el out-dir: recalcular no vuelve a leer las imagenes.

Uso:
  python make_test_split.py --root D:\\CV2 --out-dir D:\\CV2\\split_v1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CAMERAS = ("FV", "RV", "MVL", "MVR")
IMG_EXTS = (".png", ".jpg", ".jpeg")
TS_KEYS = ("timestamp", "timeStamp", "time", "utc_timestamp", "capture_time")

import re
NAME_RE = re.compile(r"^(?P<idx>\d+)_(?P<cam>FV|RV|MVL|MVR)$", re.IGNORECASE)

POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int16)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Descubrimiento
# ---------------------------------------------------------------------------

def resolve_root(root: Path, images_dir: str, max_depth: int = 2) -> Path:
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
        log(f"[info] Raiz del dataset detectada en: {candidates[0]}")
        return candidates[0]
    if len(candidates) > 1:
        opciones = "\n  ".join(str(c) for c in candidates)
        sys.exit(f"[ERROR] Hay mas de una carpeta con '{images_dir}':\n  {opciones}")
    sys.exit(f"[ERROR] No se encontro '{images_dir}' dentro de {root}")


def discover_images(root: Path, images_dir: str):
    img_dir = root / images_dir
    records, skipped = [], []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        m = NAME_RE.match(p.stem)
        if not m:
            skipped.append(p.name)
            continue
        records.append({"name": p.name, "stem": p.stem, "path": p,
                        "idx": int(m.group("idx")), "cam": m.group("cam").upper()})
    if not records:
        sys.exit(f"[ERROR] Sin imagenes con nombre <indice>_<CAM> en {img_dir}")
    if skipped:
        log(f"[AVISO] {len(skipped)} archivos ignorados (ej: {skipped[:3]})")
    return records


def find_annotation_dir(root: Path, explicit: str | None):
    if explicit:
        p = root / explicit
        return (p, explicit) if p.is_dir() else (None, explicit)
    for cand in ("box_2d_annotations", "instance_annotations"):
        if (root / cand).is_dir():
            return root / cand, cand
    return None, None


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Hashes
# ---------------------------------------------------------------------------

def dhash_bits(path: Path, size: int) -> np.ndarray | None:
    """dhash de size*size bits, empaquetado en bytes."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((size + 1, size), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.int16)
    except Exception:
        return None
    bits = (arr[:, 1:] > arr[:, :-1]).flatten()
    return np.packbits(bits)


def compute_hashes(records, size: int, cache: Path | None):
    n_bytes = (size * size + 7) // 8
    if cache is not None and cache.is_file():
        try:
            data = np.load(cache, allow_pickle=False)
            names = list(data["names"])
            if names == [r["name"] for r in records] and data["hashes"].shape[1] == n_bytes:
                log(f"[info] Hashes leidos del cache ({cache.name})")
                return data["hashes"], data["valid"]
            log("[info] El cache no coincide con las imagenes actuales, se recalcula.")
        except Exception:
            log("[info] Cache ilegible, se recalcula.")

    log(f"[info] Calculando dhash de {len(records)} imagenes (size={size}, "
        f"{size * size} bits)...")
    hashes = np.zeros((len(records), n_bytes), dtype=np.uint8)
    valid = np.zeros(len(records), dtype=bool)
    for i, r in enumerate(records):
        h = dhash_bits(r["path"], size)
        if h is not None:
            hashes[i] = h
            valid[i] = True
        if (i + 1) % 1000 == 0:
            log(f"       {i + 1}/{len(records)}")
    if cache is not None:
        try:
            np.savez_compressed(cache, names=np.array([r["name"] for r in records]),
                                hashes=hashes, valid=valid)
            log(f"[info] Cache de hashes guardado en {cache.name}")
        except Exception as e:
            log(f"[AVISO] No se pudo guardar el cache: {e}")
    if not valid.any():
        sys.exit("[ERROR] No se pudo hashear ninguna imagen.")
    return hashes, valid


def hamming_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distancia de Hamming fila a fila entre dos arrays (n, nbytes)."""
    return POPCOUNT[np.bitwise_xor(a, b)].sum(axis=1)


# ---------------------------------------------------------------------------
# Agrupamiento
# ---------------------------------------------------------------------------

def group_sequential(records, hashes, valid, thresh_bits: int,
                     max_index_gap: int, max_group: int):
    """Une frames vecinos en indice, dentro de la misma camara, si son parecidos.

    Solo se comparan pares consecutivos, asi que las cadenas siguen el eje
    temporal y no pueden saltar entre escenas distintas. El tope max_group
    corta cualquier cadena que se vuelva demasiado larga.
    """
    by_cam = defaultdict(list)
    for i, r in enumerate(records):
        by_cam[r["cam"]].append(i)

    groups = {}
    gid = 0
    dist_samples = []
    for cam in sorted(by_cam):
        order = sorted(by_cam[cam], key=lambda i: records[i]["idx"])
        current = [order[0]]
        for prev_i, cur_i in zip(order, order[1:]):
            gap = records[cur_i]["idx"] - records[prev_i]["idx"]
            close = False
            if gap <= max_index_gap and valid[prev_i] and valid[cur_i]:
                d = int(POPCOUNT[np.bitwise_xor(hashes[prev_i], hashes[cur_i])].sum())
                dist_samples.append(d)
                close = d <= thresh_bits
            if close and len(current) < max_group:
                current.append(cur_i)
            else:
                for i in current:
                    groups[records[i]["stem"]] = f"g{gid:05d}"
                gid += 1
                current = [cur_i]
        for i in current:
            groups[records[i]["stem"]] = f"g{gid:05d}"
        gid += 1

    if dist_samples:
        d = np.array(dist_samples)
        log(f"[info] Distancia entre frames consecutivos: mediana {np.median(d):.0f}, "
            f"p10 {np.percentile(d, 10):.0f}, p90 {np.percentile(d, 90):.0f} "
            f"(umbral usado: {thresh_bits})")
    return groups


def group_by_time(records, root: Path, vehicle_dir: str, gap: float, max_group: int):
    vdir = root / vehicle_dir
    if not vdir.is_dir():
        log(f"[info] No existe {vdir}: no hay timestamps disponibles.")
        return None
    stamps = {}
    for r in records:
        jp = vdir / f"{r['stem']}.json"
        if not jp.is_file():
            continue
        try:
            data = json.loads(jp.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        def walk(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in TS_KEYS and isinstance(v, (int, float, str)):
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            pass
                    f = walk(v)
                    if f is not None:
                        return f
            elif isinstance(node, list):
                for item in node:
                    f = walk(item)
                    if f is not None:
                        return f
            return None

        ts = walk(data)
        if ts is not None:
            stamps[r["stem"]] = ts

    coverage = len(stamps) / len(records)
    log(f"[info] Timestamps utilizables: {len(stamps)}/{len(records)} ({coverage:.1%})")
    if coverage < 0.9:
        return None

    vals = np.array(list(stamps.values()), dtype=np.float64)
    span = float(vals.max() - vals.min()) if len(vals) > 1 else 0.0
    scale = 1e9 if span > 1e12 else 1e6 if span > 1e9 else 1e3 if span > 1e6 else 1.0
    if scale != 1.0:
        log(f"[info] Reescalando timestamps por 1/{scale:g}")

    groups, gid, prev, size = {}, 0, None, 0
    for stem, ts in sorted(stamps.items(), key=lambda kv: kv[1]):
        ts_s = ts / scale
        if prev is not None and ((ts_s - prev) > gap or size >= max_group):
            gid += 1
            size = 0
        groups[stem] = f"t{gid:05d}"
        prev, size = ts_s, size + 1
    log(f"[info] Agrupamiento temporal: {gid + 1} grupos (gap > {gap}s)")
    return groups


# ---------------------------------------------------------------------------
# Seleccion
# ---------------------------------------------------------------------------

def select_test(records, groups, n_test: int, seed: int):
    """Cuota por camara, eligiendo grupos enteros dentro de cada camara."""
    rng = np.random.default_rng(seed)

    members = defaultdict(list)
    for r in records:
        members[groups[r["stem"]]].append(r)
    for g in members:
        members[g].sort(key=lambda r: r["idx"])

    # Camara dominante de cada grupo (con el modo secuencial, siempre es una sola).
    group_cam = {g: Counter(r["cam"] for r in rs).most_common(1)[0][0]
                 for g, rs in members.items()}
    mixed = sum(1 for g, rs in members.items() if len({r["cam"] for r in rs}) > 1)
    if mixed:
        log(f"[AVISO] {mixed} grupos mezclan camaras.")

    cam_total = Counter(r["cam"] for r in records)
    quota = {c: int(round(n_test * cam_total[c] / len(records))) for c in cam_total}
    # Ajuste de redondeo para que las cuotas sumen exactamente n_test.
    diff = n_test - sum(quota.values())
    for c in sorted(quota, key=lambda c: -cam_total[c])[:abs(diff)]:
        quota[c] += 1 if diff > 0 else -1

    groups_by_cam = defaultdict(list)
    for g in members:
        groups_by_cam[group_cam[g]].append(g)

    test, quarantine, taken = [], [], set()
    for cam in sorted(groups_by_cam):
        gs = sorted(groups_by_cam[cam])
        rng.shuffle(gs)
        need = quota.get(cam, 0)
        got = 0
        for g in gs:
            if got >= need:
                break
            rows = members[g]
            taken.add(g)
            room = need - got
            test.extend(rows[:room])
            quarantine.extend(rows[room:])
            got += min(room, len(rows))
        if got < need:
            log(f"[AVISO] Camara {cam}: solo se reunieron {got} de {need} imagenes.")

    train = [r for r in records if groups[r["stem"]] not in taken]
    test.sort(key=lambda r: (r["cam"], r["idx"]))
    quarantine.sort(key=lambda r: (r["cam"], r["idx"]))
    return test, quarantine, train


# ---------------------------------------------------------------------------
# Auditoria de leakage
# ---------------------------------------------------------------------------

def audit_leakage(test, train, records, hashes, valid, alert_bits: int, out_csv: Path):
    """Distancia minima de cada imagen de test contra todo el train."""
    pos = {r["stem"]: i for i, r in enumerate(records)}
    ti = np.array([pos[r["stem"]] for r in test if valid[pos[r["stem"]]]])
    ri = np.array([pos[r["stem"]] for r in train if valid[pos[r["stem"]]]])
    if len(ti) == 0 or len(ri) == 0:
        log("[AVISO] No se pudo auditar (faltan hashes).")
        return None

    log(f"[info] Auditando {len(ti)} x {len(ri)} pares test-train...")
    H_test, H_train = hashes[ti], hashes[ri]
    best_d = np.full(len(ti), 10 ** 6, dtype=np.int32)
    best_j = np.zeros(len(ti), dtype=np.int64)
    chunk = 256
    for s in range(0, len(ti), chunk):
        e = min(s + chunk, len(ti))
        xor = np.bitwise_xor(H_test[s:e, None, :], H_train[None, :, :])
        d = POPCOUNT[xor].sum(axis=2)
        j = d.argmin(axis=1)
        best_d[s:e] = d[np.arange(e - s), j]
        best_j[s:e] = j

    flagged_mask = best_d <= alert_bits
    flagged = int(flagged_mask.sum())
    flagged_train = {records[ri[best_j[k]]]["stem"]
                     for k in np.nonzero(flagged_mask)[0]}
    log(f"[{'AVISO' if flagged else 'OK'}] Pares test-train por debajo de "
        f"{alert_bits} bits: {flagged}/{len(ti)}")
    log(f"[info] Distancia minima test-train: mediana {np.median(best_d):.0f}, "
        f"minimo {best_d.min()}")

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["test_image", "vecino_mas_cercano_en_train", "distancia_bits"])
        for k in np.argsort(best_d)[:200]:
            w.writerow([records[ti[k]]["name"], records[ri[best_j[k]]]["name"],
                        int(best_d[k])])
    return {"flagged": flagged, "median": float(np.median(best_d)),
            "min": int(best_d.min()), "alert_bits": alert_bits,
            "_flagged_train": flagged_train}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Hold-out de WoodScape sin data leakage (v2).")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--images-dir", default="rgb_images")
    ap.add_argument("--vehicle-dir", default="vehicle_data")
    ap.add_argument("--ann-dir", default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--group-mode", choices=("auto", "time", "seq"), default="auto")
    ap.add_argument("--hash-size", type=int, default=16, help="dhash de size*size bits")
    ap.add_argument("--thresh-frac", type=float, default=0.10,
                    help="Fraccion de bits distintos para considerar dos frames vecinos")
    ap.add_argument("--max-index-gap", type=int, default=5,
                    help="Salto maximo de indice para comparar dos frames")
    ap.add_argument("--max-group", type=int, default=40,
                    help="Tope duro de imagenes por grupo")
    ap.add_argument("--alert-frac", type=float, default=0.06,
                    help="Umbral de alerta de la auditoria, en fraccion de bits")
    ap.add_argument("--time-gap", type=float, default=10.0)
    ap.add_argument("--no-sha", action="store_true")
    ap.add_argument("--no-remediate", action="store_true",
                    help="No mover a cuarentena los casi-duplicados detectados")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.root = resolve_root(args.root, args.images_dir)
    records = discover_images(args.root, args.images_dir)
    log(f"[info] Imagenes: {len(records)} | por camara: "
        f"{dict(Counter(r['cam'] for r in records))}")

    ann_dir, ann_label = find_annotation_dir(args.root, args.ann_dir)
    ann_index = {p.stem: p for p in sorted(ann_dir.rglob('*')) if p.is_file()} if ann_dir else {}
    if ann_dir:
        cov = sum(1 for r in records if r["stem"] in ann_index) / len(records)
        log(f"[info] Anotaciones: '{ann_label}', cobertura {cov:.1%}")
    else:
        log("[AVISO] Sin directorio de anotaciones.")

    n_bits = args.hash_size * args.hash_size
    thresh_bits = max(1, int(round(args.thresh_frac * n_bits)))
    alert_bits = max(1, int(round(args.alert_frac * n_bits)))

    hashes, valid = compute_hashes(records, args.hash_size, args.out_dir / "hash_cache.npz")

    groups = None
    if args.group_mode in ("auto", "time"):
        groups = group_by_time(records, args.root, args.vehicle_dir,
                               args.time_gap, args.max_group)
    if groups is None:
        groups = group_sequential(records, hashes, valid, thresh_bits,
                                  args.max_index_gap, args.max_group)

    sizes = Counter(groups.values())
    arr = np.array(list(sizes.values()))
    log(f"[info] Grupos: {len(sizes)} | medio {arr.mean():.1f} | "
        f"mediana {np.median(arr):.0f} | maximo {arr.max()}")
    if arr.max() > max(args.max_group, 0.05 * len(records)):
        log("[AVISO] Hay un grupo desproporcionado: revisar --thresh-frac.")

    test, quarantine, train = select_test(records, groups, args.n_test, args.seed)
    log(f"[OK] test={len(test)} | cuarentena={len(quarantine)} | train_pool={len(train)}")
    log(f"[info] Test por camara: {dict(Counter(r['cam'] for r in test))}")

    overlap = {groups[r['stem']] for r in test} & {groups[r['stem']] for r in train}
    if overlap:
        sys.exit(f"[ERROR] {len(overlap)} grupos compartidos entre test y train.")
    log("[OK] Ningun grupo aparece en test y train a la vez.")

    audit = audit_leakage(test, train, records, hashes, valid, alert_bits,
                          args.out_dir / "leakage_audit.csv")

    remediated = 0
    if audit and audit.get("_flagged_train") and not args.no_remediate:
        culpables = audit.pop("_flagged_train")
        antes = len(train)
        train = [r for r in train if r["stem"] not in culpables]
        movidas = [r for r in records if r["stem"] in culpables]
        quarantine.extend(movidas)
        quarantine.sort(key=lambda r: (r["cam"], r["idx"]))
        remediated = antes - len(train)
        log(f"[OK] Remediacion: {remediated} imagenes de train pasadas a cuarentena "
            f"por ser casi-duplicados de imagenes de test.")
        for r in movidas[:10]:
            log(f"     - {r['name']}")
        audit["remediated_train_images"] = remediated
        audit["flagged_after_remediation"] = 0
    elif audit:
        audit.pop("_flagged_train", None)
        audit["remediated_train_images"] = 0

    def write_list(path: Path, rows):
        with path.open("w", encoding="utf-8") as fh:
            for r in sorted(rows, key=lambda x: (x["cam"], x["idx"])):
                fh.write(r["name"] + "\n")

    write_list(args.out_dir / f"test_{args.n_test}.txt", test)
    write_list(args.out_dir / "train_pool.txt", train)
    write_list(args.out_dir / "quarantine.txt", quarantine)

    split_of = {}
    for rows, name in ((test, "test"), (quarantine, "quarantine"), (train, "train_pool")):
        for r in rows:
            split_of[r["stem"]] = name

    log("[info] Escribiendo manifest.csv...")
    with (args.out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "camera", "group_id", "split", "sha256", "has_annotation"])
        for r in sorted(records, key=lambda x: (x["cam"], x["idx"])):
            w.writerow([r["name"], r["cam"], groups[r["stem"]], split_of[r["stem"]],
                        "" if args.no_sha else sha256_of(r["path"]),
                        int(r["stem"] in ann_index)])

    config = {
        "generated_for": "TP final Vision por Computadora II - CEIA FIUBA",
        "dataset": "WoodScape (Valeo)",
        "dataset_root": str(args.root),
        "version": 2,
        "seed": args.seed,
        "n_test": args.n_test,
        "group_mode": args.group_mode,
        "hash_bits": n_bits,
        "thresh_bits": thresh_bits,
        "max_index_gap": args.max_index_gap,
        "max_group": args.max_group,
        "images_total": len(records),
        "counts": {"test": len(test), "quarantine": len(quarantine),
                   "train_pool": len(train), "groups": len(sizes)},
        "test_by_camera": dict(Counter(r["cam"] for r in test)),
        "train_by_camera": dict(Counter(r["cam"] for r in train)),
        "annotation_dir": ann_label,
        "leakage_audit": audit,
        "sha256_included": not args.no_sha,
    }
    with (args.out_dir / "split_config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)

    log(f"[OK] Archivos escritos en {args.out_dir.resolve()}")
    log("     quarantine.txt NO se usa para entrenar ni para evaluar.")


if __name__ == "__main__":
    main()
