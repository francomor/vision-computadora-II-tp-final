#!/usr/bin/env python3
"""
validar_eval_radial.py - compara eval_radial.py contra pycocotools, global y anillo por anillo.

pycocotools no sabe de anillos, pero su areaRng hace exactamente el esquema que queremos sobre el campo
"area". El truco es poner theta en ese campo (al ground truth y a las detecciones) y pedir
areaRng = [lo, hi] de cada anillo. Para el rango global se usa el area real. Si las dos
implementaciones coinciden a 4 decimales, eval_radial reproduce COCO.

pycocotools no es dependencia del proyecto; se usa solo para esta prueba:

  uv run --with pycocotools python tools/validar_eval_radial.py --yolo-dir outputs/yolo_ds --split test \\
      --pred-json outputs/rama1/pred_yolo11m.json --metricas outputs/eval_radial/rama1/metricas.json
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_radial as er  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-dir", required=True)
    ap.add_argument("--split", default="test")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred-json")
    g.add_argument("--pred-dir")
    ap.add_argument("--metricas", required=True, help="metricas.json de eval_radial.py con las mismas predicciones")
    ap.add_argument("--calibration", default=str(er.RAIZ / "config/calibration.json"))
    args = ap.parse_args()

    met = json.loads(Path(args.metricas).read_text())
    cortes, conf_min, max_dets = met["cortes"], met["conf_min"], met["max_dets"]
    cal = er.cargar_calibracion(args.calibration)
    yolo = Path(args.yolo_dir)
    stems = sorted(Path(l.strip()).stem for l in (yolo / f"{args.split}.txt").read_text().splitlines() if l.strip())
    if args.pred_json:
        preds = er.leer_pred_json(args.pred_json, "xyxy")
    else:
        preds = {}
        for s in stems:
            c, k, sc = er.leer_yolo(Path(args.pred_dir) / f"{s}.txt", cal[s][7], cal[s][8], con_conf=True)
            preds[s] = [(b, int(kk), float(ss)) for b, kk, ss in zip(c.tolist(), k, sc)]

    imagenes, anns, dets = [], [], []
    for i, s in enumerate(stems, 1):
        W, H = cal[s][7], cal[s][8]
        imagenes.append({"id": i, "width": W, "height": H})
        gc, gk, _ = er.leer_yolo(yolo / "labels" / f"{s}.txt", W, H, con_conf=False)
        for b, k, th in zip(gc, gk, er.theta(gc, cal[s])):
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(k) + 1, "iscrowd": 0,
                         "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                         "area": (b[2] - b[0]) * (b[3] - b[1]), "theta": float(th)})
        p = [x for x in preds.get(s, []) if x[2] >= conf_min]
        dc = np.array([x[0] for x in p], dtype=np.float64).reshape(-1, 4)
        for (b, k, sc), th in zip(p, er.theta(dc, cal[s])):
            dets.append({"image_id": i, "category_id": k + 1, "score": sc,
                         "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "theta": float(th)})

    clases = sorted({a["category_id"] for a in anns} | {d["category_id"] for d in dets})
    base = {"images": imagenes, "categories": [{"id": c} for c in clases]}

    def coco(campo_area):
        gt = COCO()
        gt.dataset = {**base, "annotations": [{**a, "area": a[campo_area]} for a in anns]}
        with contextlib.redirect_stdout(io.StringIO()):
            gt.createIndex()
            dt = gt.loadRes([{k: v for k, v in d.items() if k != "theta"} for d in dets])
        for d_ann, d in zip(dt.dataset["annotations"], dets):
            if campo_area == "theta":
                d_ann["area"] = d["theta"]  # loadRes pone w*h; para los anillos va theta
        return gt, dt

    def ap(campo_area, lo, hi):
        gt, dt = coco(campo_area)
        e = COCOeval(gt, dt, "bbox")
        e.params.areaRng, e.params.areaRngLbl, e.params.maxDets = [[lo, hi]], ["r"], [max_dets]
        with contextlib.redirect_stdout(io.StringIO()):
            e.evaluate()
            e.accumulate()
        p = e.eval["precision"][:, :, :, 0, 0]
        return float(np.mean(p[0][p[0] > -1])), float(np.mean(p[p > -1]))

    nombres = ["centro", "medio", "periferia"]
    rangos = {"global": ("area", 0, 1e10)} | {n: ("theta", cortes[i], cortes[i + 1]) for i, n in enumerate(nombres)}
    propias = {f["rango"]: (f["AP50"], f["AP50_95"]) for f in met["filas"] if f["clase"] == "TODAS"}
    peor = 0.0
    print(f"{'rango':<10} {'AP50 eval_radial':>17} {'AP50 pycoco':>12} {'AP50-95 eval_radial':>20} {'AP50-95 pycoco':>15}")
    for rango, (campo, lo, hi) in rangos.items():
        c50, c5095 = ap(campo, lo, hi)
        m50, m5095 = propias[rango]
        peor = max(peor, abs(c50 - m50), abs(c5095 - m5095))
        print(f"{rango:<10} {m50:>17.5f} {c50:>12.5f} {m5095:>20.5f} {c5095:>15.5f}")
    print(f"diferencia maxima: {peor:.2e} -> {'COINCIDEN' if peor < 5e-5 else 'NO COINCIDEN'} a 4 decimales")
    sys.exit(0 if peor < 5e-5 else 1)


if __name__ == "__main__":
    main()
