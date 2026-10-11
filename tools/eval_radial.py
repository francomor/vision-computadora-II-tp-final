#!/usr/bin/env python3
"""
eval_radial.py - mAP@50 y mAP@50-95 global y por anillo de angulo de incidencia theta, con el esquema de COCO.

Reconstruccion del evaluador del equipo (el original no quedo en el repo). Replica pycocotools
(evaluateImg + accumulate) poniendo theta en el lugar del area, como hace areaRng: en cada anillo
  * los objetos de referencia de otros anillos se marcan ignore (no suman al recall),
  * una deteccion que coincide con un objeto ignorado no cuenta ni como acierto ni como error,
  * una deteccion sin pareja cuyo centro cae fuera del anillo se ignora.
Igual que COCO: hasta 100 detecciones por imagen y clase, IoU de 0.50 a 0.95 cada 0.05, 101 puntos de
recall, y el mAP es el promedio sobre las clases que tienen objetos en ese anillo. Solo usa numpy.

theta sale del centro de cada caja invirtiendo rho(theta) = k1 th + k2 th^2 + k3 th^3 + k4 th^4 con la
calibracion de su imagen (la misma cuenta que theta() de fisheye_utils.py).

Uso, desde la raiz del repo:

  uv run python tools/eval_radial.py --yolo-dir outputs/yolo_ds --split test \\
      --pred-json outputs/rama1/pred_yolo11m.json --out outputs/eval_radial/rama1 \\
      --calibration config/calibration.json --rings-file config/rings.json --cruce-tamano --etiqueta rama1

Predicciones: --pred-json con una lista de cajas {image, class, score, bbox} (bbox en xyxy de la fisheye
original; tambien acepta las claves stem, bbox_xyxy, class_id y category_id) o --pred-dir con labels de
Ultralytics guardados con save_txt + save_conf (clase cx cy w h conf, normalizados).
El ground truth es el de convert_to_yolo.py: <yolo-dir>/<split>.txt y <yolo-dir>/labels/.

Deja en --out: metricas.json (filas con clase, rango, n_gt, AP50 y AP50_95; la clase "TODAS" es el mAP)
y metricas.txt con las mismas tablas.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
IOUS = np.linspace(0.5, 0.95, 10)
RECALLS = np.linspace(0.0, 1.0, 101)
TAMANOS = {"chico": (0.0, 32.0 ** 2), "mediano": (32.0 ** 2, 96.0 ** 2), "grande": (96.0 ** 2, 1e10)}


# ---------------------------------------------------------------- calibracion y theta

def cargar_calibracion(ruta):
    """{stem: (cx, cy, k1, k2, k3, k4, aspect_ratio, W, H)} desde config/calibration.json o una carpeta de Valeo"""
    ruta = Path(ruta)
    out = {}
    if ruta.is_dir():
        for p in sorted(ruta.glob("*.json")):
            i = json.loads(p.read_text())["intrinsic"]
            out[p.stem] = (i["cx_offset"] + i["width"] / 2 - 0.5, i["cy_offset"] + i["height"] / 2 - 0.5,
                           i["k1"], i["k2"], i["k3"], i["k4"], i["aspect_ratio"], i["width"], i["height"])
        return out
    c = json.loads(ruta.read_text())
    for stem, e in c["imagenes"].items():
        j = c["juegos"][e["juego"]]
        out[stem] = (e["cx"], e["cy"], j["k1"], j["k2"], j["k3"], j["k4"], j["aspect_ratio"], j["width"], j["height"])
    return out


_TABLAS = {}


def _tabla(k):
    """theta [rad] y rho [px] muestreados, solo el tramo donde rho crece (para poder invertirlo)"""
    if k not in _TABLAS:
        th = np.linspace(0, np.deg2rad(120), 4001)
        rho = k[0] * th + k[1] * th ** 2 + k[2] * th ** 3 + k[3] * th ** 4
        baja = np.diff(rho) <= 0
        n = int(np.argmax(baja)) + 1 if baja.any() else len(rho)
        _TABLAS[k] = (th[:n], rho[:n])
    return _TABLAS[k]


def theta(cajas, cal):
    """angulo de incidencia en grados del centro de cada caja xyxy"""
    cx, cy, k1, k2, k3, k4, ar = cal[:7]
    u = (cajas[:, 0] + cajas[:, 2]) / 2 - cx
    v = ((cajas[:, 1] + cajas[:, 3]) / 2 - cy) / ar
    th, rho = _tabla((k1, k2, k3, k4))
    return np.rad2deg(np.interp(np.hypot(u, v), rho, th))


# ---------------------------------------------------------------- lectura

def leer_yolo(p, W, H, con_conf):
    """cajas xyxy en pixeles, clases y (si hay) confianzas de un .txt de Ultralytics"""
    filas = []
    if p.is_file():
        for linea in p.read_text().splitlines():
            v = linea.split()
            if len(v) >= 5:
                filas.append([float(x) for x in v[:6]] if con_conf else [float(x) for x in v[:5]])
    a = np.array(filas, dtype=np.float64).reshape(-1, 6 if con_conf else 5)
    if con_conf and len(filas) and any(len(f) < 6 for f in filas):
        sys.exit(f"{p}: faltan las confianzas (guardar con save_conf=True)")
    cajas = np.stack([(a[:, 1] - a[:, 3] / 2) * W, (a[:, 2] - a[:, 4] / 2) * H,
                      (a[:, 1] + a[:, 3] / 2) * W, (a[:, 2] + a[:, 4] / 2) * H], axis=1)
    return cajas, a[:, 0].astype(int), (a[:, 5] if con_conf else None)


def leer_pred_json(ruta, formato):
    """{stem: [(bbox xyxy, clase, score)]} desde una lista de cajas en json"""
    datos = json.loads(Path(ruta).read_text())
    if isinstance(datos, dict):
        datos = datos.get("predicciones", datos.get("annotations", []))
    preds = defaultdict(list)
    for d in datos:
        imagen = next(d[k] for k in ("image", "stem", "image_id", "file_name") if k in d)
        b = [float(x) for x in next(d[k] for k in ("bbox_xyxy", "bbox") if k in d)]
        if formato == "xywh":
            b = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
        clase = int(next(d[k] for k in ("class", "class_id", "category_id") if k in d))
        score = float(next(d[k] for k in ("score", "conf") if k in d))
        preds[Path(str(imagen)).stem].append((b, clase, score))
    return preds


# ---------------------------------------------------------------- COCO con theta

def iou_matriz(d, g):
    """IoU (D, G) entre cajas xyxy, sin el +1 de VOC (igual que pycocotools)"""
    if len(d) == 0 or len(g) == 0:
        return np.zeros((len(d), len(g)))
    ix = np.clip(np.minimum(d[:, None, 2], g[None, :, 2]) - np.maximum(d[:, None, 0], g[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(d[:, None, 3], g[None, :, 3]) - np.maximum(d[:, None, 1], g[None, :, 1]), 0, None)
    inter = ix * iy
    ad = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    union = ad[:, None] + ag[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def fuera(feats, condiciones):
    """True donde algun rasgo cae fuera de su rango [lo, hi] (inclusivo, como areaRng)"""
    m = np.zeros(len(next(iter(feats.values()))), dtype=bool)
    for nombre, lo, hi in condiciones:
        f = feats[nombre]
        m |= (f < lo) | (f > hi)
    return m


def evaluar_imagen(e, condiciones):
    """evaluateImg de pycocotools para una imagen y una clase, con el rango dado por condiciones"""
    g_ig = fuera(e["g"], condiciones) if len(e["gt"]) else np.zeros(0, bool)
    orden = np.argsort(g_ig, kind="mergesort")  # no ignorados primero
    g_ig = g_ig[orden]
    iou = e["iou"][:, orden]
    D, G, T = len(e["dt"]), len(orden), len(IOUS)
    dtm = np.zeros((T, D), dtype=bool)
    dt_ig = np.zeros((T, D), dtype=bool)
    if D and G:
        filas = iou.tolist()
        ig = g_ig.tolist()
        for ti, t in enumerate(IOUS):
            usado = [False] * G
            for d in range(D):
                mejor, m, fila = min(t, 1 - 1e-10), -1, filas[d]
                for g in range(G):
                    if usado[g]:
                        continue
                    if m > -1 and not ig[m] and ig[g]:
                        break
                    if fila[g] < mejor:
                        continue
                    mejor, m = fila[g], g
                if m == -1:
                    continue
                dt_ig[ti, d] = ig[m]
                dtm[ti, d] = True
                usado[m] = True
    if D:
        dt_ig |= ~dtm & fuera(e["d"], condiciones)[None, :]
    return e["score"], dtm, dt_ig, int((~g_ig).sum())


def acumular(resultados):
    """accumulate de pycocotools: precision (T, R) de una clase, o None si no hay objetos no ignorados"""
    npig = sum(r[3] for r in resultados)
    if npig == 0:
        return None
    scores = np.concatenate([r[0] for r in resultados]) if resultados else np.zeros(0)
    orden = np.argsort(-scores, kind="mergesort")
    dtm = np.concatenate([r[1] for r in resultados], axis=1)[:, orden]
    dt_ig = np.concatenate([r[2] for r in resultados], axis=1)[:, orden]
    tp = np.cumsum(dtm & ~dt_ig, axis=1).astype(float)
    fp = np.cumsum(~dtm & ~dt_ig, axis=1).astype(float)
    prec = np.zeros((len(IOUS), len(RECALLS)))
    for t in range(len(IOUS)):
        nd = tp.shape[1]
        if nd == 0:
            continue
        rc = tp[t] / npig
        pr = tp[t] / (fp[t] + tp[t] + np.spacing(1))
        pr = np.maximum.accumulate(pr[::-1])[::-1]  # envolvente de la precision
        idx = np.searchsorted(rc, RECALLS, side="left")
        prec[t] = np.where(idx < nd, pr[np.minimum(idx, nd - 1)], 0.0)
    return prec


def map_rango(por_clase, condiciones, n_clases):
    """AP50 y AP50-95 por clase y el promedio sobre las clases con objetos (la fila TODAS)"""
    filas = {}
    for k in range(n_clases):
        res = [evaluar_imagen(e, condiciones) for e in por_clase[k]]
        prec = acumular(res)
        n_gt = sum(r[3] for r in res)
        filas[k] = (n_gt, None, None) if prec is None else (n_gt, float(prec[0].mean()), float(prec.mean()))
    validas = [v for v in filas.values() if v[1] is not None]
    todas = (sum(v[0] for v in filas.values()),
             float(np.mean([v[1] for v in validas])) if validas else None,
             float(np.mean([v[2] for v in validas])) if validas else None)
    return filas, todas


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="mAP global y por anillo de theta con el esquema ignore de COCO")
    ap.add_argument("--yolo-dir", required=True, help="carpeta de convert_to_yolo.py (<split>.txt y labels/)")
    ap.add_argument("--split", default="test")
    pred = ap.add_mutually_exclusive_group(required=True)
    pred.add_argument("--pred-json", help="lista de cajas {image, class, score, bbox}")
    pred.add_argument("--pred-dir", help="labels de Ultralytics con save_txt + save_conf")
    ap.add_argument("--bbox", choices=("xyxy", "xywh"), default="xyxy", help="formato de bbox en --pred-json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibration", default=str(RAIZ / "config/calibration.json"),
                    help="config/calibration.json o una carpeta con un json de Valeo por imagen")
    ap.add_argument("--rings-file", default=None, help="cortes congelados (config/rings.json); sin esto usa los terciles del gt")
    ap.add_argument("--class-map", default=str(RAIZ / "config/class_map.json"))
    ap.add_argument("--etiqueta", default="")
    ap.add_argument("--cruce-tamano", action="store_true", help="agrega el cruce anillo x tamano (rangos de area de COCO)")
    ap.add_argument("--conf-min", type=float, default=0.0, help="descarta predicciones con score menor")
    ap.add_argument("--max-dets", type=int, default=100, help="detecciones por imagen y clase, como maxDets de COCO")
    args = ap.parse_args()

    clases = json.loads(Path(args.class_map).read_text(encoding="utf-8"))["clases"]
    cal = cargar_calibracion(args.calibration)
    yolo = Path(args.yolo_dir)
    stems = sorted(Path(l.strip()).stem for l in (yolo / f"{args.split}.txt").read_text().splitlines() if l.strip())
    sin_cal = [s for s in stems if s not in cal]
    if sin_cal:
        sys.exit(f"{len(sin_cal)} imagenes sin calibracion (ej: {sin_cal[:3]})")

    if args.pred_json:
        preds = leer_pred_json(args.pred_json, args.bbox)
    else:
        preds = {}
        for s in stems:
            W, H = cal[s][7], cal[s][8]
            c, k, sc = leer_yolo(Path(args.pred_dir) / f"{s}.txt", W, H, con_conf=True)
            preds[s] = [(b, int(kk), float(ss)) for b, kk, ss in zip(c.tolist(), k, sc)]
    ajenas = sum(len(v) for s, v in preds.items() if s not in set(stems))
    if ajenas:
        print(f"[aviso] {ajenas} predicciones de imagenes que no estan en {args.split}, se ignoran")

    # gt, predicciones, theta, area e IoU por imagen y clase (una sola vez)
    por_clase = [[] for _ in clases]
    th_gt_todos, n_gt, n_pred = [], 0, 0
    for s in stems:
        W, H = cal[s][7], cal[s][8]
        g_cajas, g_cls, _ = leer_yolo(yolo / "labels" / f"{s}.txt", W, H, con_conf=False)
        p = [x for x in preds.get(s, []) if x[2] >= args.conf_min]
        d_cajas = np.array([x[0] for x in p], dtype=np.float64).reshape(-1, 4)
        d_cls = np.array([x[1] for x in p], dtype=int)
        d_sc = np.array([x[2] for x in p], dtype=np.float64)
        g_th, d_th = theta(g_cajas, cal[s]), theta(d_cajas, cal[s])
        th_gt_todos.append(g_th)
        n_gt += len(g_cajas)
        for k in range(len(clases)):
            gi, di = np.flatnonzero(g_cls == k), np.flatnonzero(d_cls == k)
            di = di[np.argsort(-d_sc[di], kind="mergesort")][: args.max_dets]
            if len(gi) == 0 and len(di) == 0:
                continue  # como pycocotools: la imagen no aporta nada a esta clase
            gc, dc = g_cajas[gi], d_cajas[di]
            n_pred += len(di)
            por_clase[k].append({
                "gt": gc, "dt": dc, "score": d_sc[di], "iou": iou_matriz(dc, gc),
                "g": {"theta": g_th[gi], "area": (gc[:, 2] - gc[:, 0]) * (gc[:, 3] - gc[:, 1])},
                "d": {"theta": d_th[di], "area": (dc[:, 2] - dc[:, 0]) * (dc[:, 3] - dc[:, 1])},
            })
    th_gt = np.concatenate(th_gt_todos)

    if args.rings_file:
        cortes = [float(c) for c in json.loads(Path(args.rings_file).read_text())["cortes"]]
    else:
        q = np.quantile(th_gt, [1 / 3, 2 / 3])
        cortes = [0.0, float(q[0]), float(q[1]), 1e9]
        print(f"[aviso] sin --rings-file: cortes calculados sobre este gt ({q[0]:.2f}, {q[1]:.2f}); "
              "no son comparables con los congelados del equipo")
    nombres = ["centro", "medio", "periferia"] if len(cortes) == 4 else [f"anillo_{i}" for i in range(len(cortes) - 1)]
    rangos = {"global": []} | {n: [("theta", cortes[i], cortes[i + 1])] for i, n in enumerate(nombres)}

    filas, resumen = [], {}
    for rango, cond in rangos.items():
        por_k, todas = map_rango(por_clase, cond, len(clases))
        for k, (n, a50, a5095) in por_k.items():
            filas.append({"clase": clases[k], "rango": rango, "n_gt": n, "AP50": a50, "AP50_95": a5095})
        filas.append({"clase": "TODAS", "rango": rango, "n_gt": todas[0], "AP50": todas[1], "AP50_95": todas[2]})
        resumen[rango] = todas

    cruce = []
    if args.cruce_tamano:
        for n in nombres:
            for t, (lo, hi) in TAMANOS.items():
                _, todas = map_rango(por_clase, rangos[n] + [("area", lo, hi)], len(clases))
                cruce.append({"rango": n, "tamano": t, "n_gt": todas[0], "AP50": todas[1], "AP50_95": todas[2]})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"etiqueta": args.etiqueta, "split": args.split, "n_imagenes": len(stems), "n_gt": n_gt,
            "n_pred_evaluadas": n_pred, "cortes": cortes, "conf_min": args.conf_min, "max_dets": args.max_dets,
            "esquema": "COCO (ignore fuera del anillo), theta del centro de la caja"}
    (out / "metricas.json").write_text(json.dumps({**meta, "filas": filas, "cruce_tamano": cruce}, indent=1), encoding="utf-8")

    fmt = lambda v: "   -  " if v is None else f"{v:.4f}"  # noqa: E731
    lineas = [f"eval_radial | {args.etiqueta} | {args.split}: {len(stems)} imagenes, {n_gt} cajas de gt, "
              f"{n_pred} predicciones | cortes theta {cortes[1]:.2f} / {cortes[2]:.2f}", "",
              f"{'rango':<10} {'clase':<12} {'n_gt':>6} {'AP50':>8} {'AP50-95':>8}"]
    for f in filas:
        lineas.append(f"{f['rango']:<10} {f['clase']:<12} {f['n_gt']:>6} {fmt(f['AP50']):>8} {fmt(f['AP50_95']):>8}")
    if cruce:
        lineas += ["", "cruce anillo x tamano (mAP de todas las clases)", f"{'rango':<10} {'tamano':<8} {'n_gt':>6} {'AP50':>8} {'AP50-95':>8}"]
        lineas += [f"{c['rango']:<10} {c['tamano']:<8} {c['n_gt']:>6} {fmt(c['AP50']):>8} {fmt(c['AP50_95']):>8}" for c in cruce]
    (out / "metricas.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas[:3]))
    for rango, (n, a50, a5095) in resumen.items():
        print(f"{rango:<10} {'TODAS':<12} {n:>6} {fmt(a50):>8} {fmt(a5095):>8}")
    print(f"-> {out / 'metricas.json'}")


if __name__ == "__main__":
    main()
