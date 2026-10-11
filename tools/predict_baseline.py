#!/usr/bin/env python3
"""
predict_baseline.py - inferencia de un YOLO pre-entrenado en COCO sobre un split (ramas 1 y 2), con latencia.

Hace lo mismo que la inferencia de notebooks/rama1_baseline.ipynb (y, con --proyeccion cilindrica, la de
notebooks/rama2_cilindrica.ipynb), pero desde la terminal y dejando las predicciones en el formato que lee
eval_radial.py:

  # rama 1: directo sobre la fisheye
  uv run python tools/predict_baseline.py --yolo-dir outputs/yolo_ds --split test --modelo yolo11m.pt \\
      --out outputs/pred/rama1 --coco-map config/coco_map.json
  # rama 2: proyeccion cilindrica de fisheye_utils.py y cajas de vuelta a la fisheye
  uv run python tools/predict_baseline.py --yolo-dir outputs/yolo_ds --split test --modelo yolo11m.pt \\
      --out outputs/pred/rama2 --coco-map config/coco_map.json --proyeccion cilindrica
  # despues
  uv run python tools/eval_radial.py --yolo-dir outputs/yolo_ds --pred-json outputs/pred/rama1/pred.json ...

Deja en --out:
  pred.json      una entrada por caja: image, class (id del TP), score, bbox (xyxy en la fisheye original)
  latencia.json  ms por imagen (media, p50 y p95) de pre (proyeccion), detector y post (clases y vuelta de
                 las cajas), con diez pasadas de calentamiento y la lectura del disco fuera de la medicion
El umbral de confianza por defecto es 0.001: el mAP necesita la cola de baja confianza (regla 5 de la GUIA).
La proyeccion cilindrica necesita la carpeta CV2/ del Drive (calibracion por imagen), como fisheye_utils.py.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]


def hardware(device):
    import torch
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if platform.system() == "Darwin":
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if r.stdout.strip():
            return r.stdout.strip()
    return platform.processor() or platform.machine()


def resumen(ms):
    ms = np.asarray(ms)
    return {"media": float(ms.mean()), "p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95))}


def main():
    ap = argparse.ArgumentParser(description="inferencia de un YOLO de COCO con las clases del TP (ramas 1 y 2)")
    ap.add_argument("--yolo-dir", required=True, help="carpeta de convert_to_yolo.py, con <split>.txt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imagenes", default=None, help="carpeta donde buscar las imagenes si las rutas de la lista no existen aca")
    ap.add_argument("--modelo", default="yolo11m.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--coco-map", default=str(RAIZ / "config/coco_map.json"))
    ap.add_argument("--proyeccion", choices=("fisheye", "cilindrica"), default="fisheye")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="por defecto cuda, mps o cpu, el que haya")
    ap.add_argument("--n", type=int, default=None, help="solo las primeras n imagenes, para probar")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    rutas = [Path(l.strip()) for l in (Path(args.yolo_dir) / f"{args.split}.txt").read_text().splitlines() if l.strip()]
    if args.imagenes:
        rutas = [r if r.is_file() else Path(args.imagenes) / r.name for r in rutas]
    faltan = [r for r in rutas if not r.is_file()]
    if faltan:
        sys.exit(f"{len(faltan)} imagenes no existen (ej: {faltan[0]}); usar --imagenes")
    rutas = rutas[: args.n] if args.n else rutas

    cm = json.loads(Path(args.coco_map).read_text(encoding="utf-8"))
    modelo = YOLO(args.modelo)
    coco_a_tp = {int(i): cm["clases"].index(cm["mapeo"][n]) for i, n in modelo.names.items() if cm["mapeo"].get(n)}
    print(f"{args.modelo} | {args.proyeccion} | device {device} ({hardware(device)}) | {len(rutas)} imagenes de {args.split}")
    print("clases: " + ", ".join(f"{modelo.names[i]}->{cm['clases'][k]}" for i, k in sorted(coco_a_tp.items())))

    cil = args.proyeccion == "cilindrica"
    if cil:
        sys.path.insert(0, str(RAIZ))
        from fisheye_utils import caja_a_fisheye, clave_calib, mapas
        claves = {r.stem: clave_calib(r.stem, r.stem.split("_")[-1]) for r in rutas}
        for c in set(claves.values()):
            mapas(c)  # los mapas antes, para que no ensucien la latencia

    def detectar(img):
        b = modelo.predict(img, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)[0].boxes
        return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)

    img0 = cv2.imread(str(rutas[0]))
    entrada0 = cv2.remap(img0, *mapas(claves[rutas[0].stem]), interpolation=cv2.INTER_LINEAR) if cil else img0
    for _ in range(10):  # calentamiento
        detectar(entrada0)

    registros, ms = [], {"pre": [], "detector": [], "post": []}
    t_inicio = time.perf_counter()
    for k, r in enumerate(rutas, 1):
        img = cv2.imread(str(r))
        H, W = img.shape[:2]
        t0 = time.perf_counter()
        if cil:
            mx, my = mapas(claves[r.stem])
            entrada = cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR)
        else:
            entrada = img
        t1 = time.perf_counter()
        xyxy, conf, cls = detectar(entrada)
        t2 = time.perf_counter()
        m = np.isin(cls, list(coco_a_tp))
        xyxy, conf, cls = xyxy[m], conf[m], np.array([coco_a_tp[c] for c in cls[m]], dtype=int)
        if cil:
            xyxy, ok = caja_a_fisheye(xyxy, mx, my, W, H)
            xyxy, conf, cls = xyxy[ok], conf[ok], cls[ok]
        t3 = time.perf_counter()
        ms["pre"].append(1e3 * (t1 - t0))
        ms["detector"].append(1e3 * (t2 - t1))
        ms["post"].append(1e3 * (t3 - t2))
        registros += [{"image": r.name, "class": int(c), "score": round(float(s), 5), "bbox": [round(float(v), 2) for v in b]}
                      for b, s, c in zip(xyxy, conf, cls)]
        if k % 250 == 0 or k == len(rutas):
            print(f"  {k}/{len(rutas)} ({time.perf_counter() - t_inicio:.0f} s)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pred.json").write_text(json.dumps(registros))
    total = np.array(ms["pre"]) + np.array(ms["detector"]) + np.array(ms["post"])
    lat = {"modelo": args.modelo, "proyeccion": args.proyeccion, "device": device, "hardware": hardware(device),
           "imgsz": args.imgsz, "conf": args.conf, "n_imagenes": len(rutas),
           **{f"ms_{k}": resumen(v) for k, v in ms.items()}, "ms_total": resumen(total),
           "fps": float(1e3 / total.mean())}
    (out / "latencia.json").write_text(json.dumps(lat, indent=1))
    print(f"{len(registros)} cajas -> {out / 'pred.json'} | {lat['ms_total']['media']:.1f} ms por imagen ({lat['fps']:.0f} FPS)")


if __name__ == "__main__":
    main()
