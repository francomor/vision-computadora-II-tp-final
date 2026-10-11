"""
finetune_utils.py - fine-tuning de los 8 modelos de la rama 2 sobre la proyeccion cilindrica (rama 4).

Rama 4 (proyeccion="cilindrica"): reentrena sobre la cilindrica de fisheye_utils.
proyeccion="fisheye" arma lo mismo sin rectificar, pero NO se uso para los resultados: la rama 3 se
entreno con fisheye_aug.py (tools/train_rama3.py), con los mismos valores de HP_YOLO salvo el batch
fisico (16) y la semilla (0). Como Ultralytics acumula gradientes hasta un batch nominal de 64, el
optimizador (AdamW, lr 0.001429) y el weight decay son los mismos en las dos ramas.

Uso desde un notebook en notebooks/ (igual que fisheye_utils):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent))
    import finetune_utils as ft

La proyeccion es la de fisheye_utils (mapas, clave_calib, caja_a_fisheye) sin tocar, para que la
comparacion rama 2 vs rama 4 aisle solo el efecto del fine-tuning.

Que hay:
- poligonos_tp / etiquetas_cilindricas / mapa_instancias: objetos de una anotacion con el mismo filtro
  que cargar_gt, llevados a la cilindrica por su POLIGONO (no por la caja, que se inflaria en la periferia).
- construir_split: arma images/, labels/ y masks/ por split. Con proyeccion="fisheye" arma lo mismo sin
  rectificar (alternativa, no usada para la rama 3).
- MODELOS_RAMA4, HP_YOLO, HP_TV: los modelos de la rama 2 y los hiperparametros (HP_YOLO coincide con la rama 3, ver arriba).
- entrenar: YOLO11 con Ultralytics y Faster/Mask R-CNN con un loop propio. Reanudables, eligen el
  checkpoint por mAP@50-95 en validacion (el fitness de Ultralytics 8.4 es exactamente ese).
- detector_ft / predecir_test: inferencia sobre el test con el mismo flujo y la misma medicion de
  latencia que la rama 2 (o directo sobre la fisheye en la rama 3), sin traducir clases COCO: los
  modelos ya salen con las clases del TP.
- a_pred_json: predicciones al formato de eval_radial.py.
"""
import contextlib
import json
import math
import os
import random
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import torch
from supervision.metrics import MeanAveragePrecision
from torchvision.models import detection as tvd
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from ultralytics import YOLO

from fisheye_utils import (
    CLASES, CONF, DEVICE, IMGSZ, MAPEO, MIN_AREA, MIN_SIDE, caja_a_fisheye, cilindrica, clave_calib,
    collect_points, desenvolver, mapas, tags_of,
)

MIN_LADO_CIL = 2  # una caja de menos de 2 px en la cilindrica es un artefacto del muestreo
SEMILLA = 42

# ---------------------------------------------------------------- modelos e hiperparametros
# Los mismos 8 modelos y en el mismo orden que MODELOS de fisheye_utils (rama 2).
# la rama 3 (fisheye_aug.py) usa los mismos valores de HP_YOLO salvo batch fisico y semilla: si se cambian aca, avisar.

YOLOS = [f"yolo11{t}" for t in "nsmlx"]
TV = {
    "fasterrcnn_mobilenet_v3": (tvd.fasterrcnn_mobilenet_v3_large_fpn, tvd.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
    "fasterrcnn_r50_v2": (tvd.fasterrcnn_resnet50_fpn_v2, tvd.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
    "maskrcnn_r50_v2": (tvd.maskrcnn_resnet50_fpn_v2, tvd.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
}
MODELOS_RAMA4 = YOLOS + list(TV)
# los ocho, ordenados de menor a mayor costo de entrenamiento: si la corrida no termina, quedan los
# modelos completos que mas entran y no uno solo a medio hacer
MODELOS_FINETUNE = ["yolo11n", "yolo11s", "fasterrcnn_mobilenet_v3", "yolo11m", "yolo11l",
                    "fasterrcnn_r50_v2", "maskrcnn_r50_v2", "yolo11x"]
assert sorted(MODELOS_FINETUNE) == sorted(MODELOS_RAMA4)

HP_YOLO = dict(epochs=30, imgsz=IMGSZ, patience=10, seed=SEMILLA, deterministic=True, workers=4)
# batch por modelo, pensado para que entre en los 6 GB de una GTX 1660 sin AMP y sin desbordar a la memoria
# compartida de Windows (que es mucho mas lenta). Ultralytics acumula gradientes hasta un batch nominal de
# 64, asi que el batch fisico casi no cambia el resultado (la rama 3 uso 16).
BATCH_YOLO = {"yolo11n": 8, "yolo11s": 4, "yolo11m": 2, "yolo11l": 2, "yolo11x": 1}

HP_TV = dict(epochs=12, momentum=0.9, weight_decay=1e-4, warmup_iters=300, patience=4, flip=0.5,
             seed=SEMILLA, workers=4)
BATCH_TV = {"fasterrcnn_mobilenet_v3": 2, "fasterrcnn_r50_v2": 2, "maskrcnn_r50_v2": 1}
LR_TV_BASE = 0.02  # receta de torchvision para 16 imagenes por lote; se escala lineal con el batch


def lr_tv(nombre, batch=None):
    return LR_TV_BASE * (batch or BATCH_TV[nombre]) / 16

# placas donde la precision mixta da perdidas NaN o mAP cero; es la misma lista que usa Ultralytics
GPU_SIN_AMP = re.compile(r"(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)",
                         re.IGNORECASE)


def usar_amp(device):
    """precision mixta solo en CUDA y fuera de las placas con problemas conocidos (GTX 16xx, entre otras)"""
    return device == "cuda" and torch.cuda.is_available() and not GPU_SIN_AMP.search(torch.cuda.get_device_name(0))


def semilla(s=SEMILLA):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ---------------------------------------------------------------- etiquetas

def poligonos_tp(ann_path):
    """[(clase, pts Nx2 float32, z_order)] de una anotacion, con el mismo filtro que cargar_gt.

    El filtro se aplica sobre la caja en la fisheye, igual que convert_to_yolo, asi el conjunto de
    objetos es exactamente el mismo que en las otras ramas.
    """
    data = desenvolver(json.loads(Path(ann_path).read_text(encoding="utf-8", errors="ignore")))
    W, H = float(data["image_width"]), float(data["image_height"])
    salida = []
    for ann in data.get("annotation", []) or []:
        destino = next((MAPEO[t] for t in tags_of(ann) if t in MAPEO), None)
        pts = []
        collect_points(ann.get("segmentation"), pts)
        if destino is None or len(pts) < 2:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x1, x2 = max(0.0, min(xs)), min(W, max(xs))
        y1, y2 = max(0.0, min(ys)), min(H, max(ys))
        w, h = x2 - x1, y2 - y1
        if w < MIN_SIDE or h < MIN_SIDE or w * h < MIN_AREA:
            continue
        salida.append((destino, np.array(pts, dtype=np.float32), float(ann.get("z_order") or 0)))
    return salida, W, H


def etiquetas_cilindricas(poligonos, mx, my, W, H):
    """cajas xyxy en la cilindrica, clases, indices de los poligonos etiquetados, perdidos y degenerados.

    Cada poligono se rasteriza en una mascara fisheye y pasa por el mismo remap que la imagen (vecino
    mas cercano, para no inventar pixeles). Si la mascara queda vacia, el objeto cae fuera de la banda
    de la cilindrica y no hay nada que etiquetar.
    """
    cajas, clases, idx, perdidos, degenerados = [], [], [], 0, 0
    mascara = np.zeros((int(H), int(W)), dtype=np.uint8)
    for i, (clase, pts, _z) in enumerate(poligonos):
        mascara[:] = 0
        cv2.fillPoly(mascara, [np.round(pts).astype(np.int32)], 1)
        cil = cv2.remap(mascara, mx, my, interpolation=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ys, xs = np.nonzero(cil)
        if len(xs) == 0:
            perdidos += 1
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        if (x2 - x1) < MIN_LADO_CIL or (y2 - y1) < MIN_LADO_CIL:
            degenerados += 1
            continue
        cajas.append([x1, y1, x2, y2])
        clases.append(clase)
        idx.append(i)
    return (np.array(cajas, dtype=np.float32).reshape(-1, 4), np.array(clases, dtype=int),
            idx, perdidos, degenerados)


def etiquetas_fisheye(poligonos, W, H):
    """lo mismo sin rectificar: la caja del poligono recortada al borde, igual que convert_to_yolo"""
    cajas, clases = [], []
    for clase, pts, _z in poligonos:
        cajas.append([max(0.0, pts[:, 0].min()), max(0.0, pts[:, 1].min()),
                      min(W, pts[:, 0].max()), min(H, pts[:, 1].max())])
        clases.append(clase)
    return (np.array(cajas, dtype=np.float32).reshape(-1, 4), np.array(clases, dtype=int),
            list(range(len(poligonos))))


def mapa_instancias(poligonos, idx, W, H, mx=None, my=None):
    """mapa uint16 donde el pixel vale k+1 si pertenece al objeto de la linea k de la etiqueta.

    Se pinta en orden de z_order (los de adelante al final) y, a igual z, de mayor a menor area, asi
    las oclusiones quedan como en la escena. Solo lo usa Mask R-CNN; las cajas no salen de aca.
    """
    inst = np.zeros((int(H), int(W)), dtype=np.uint16)
    area = lambda p: cv2.contourArea(np.round(p).astype(np.float32))  # noqa: E731
    orden = sorted(range(len(idx)), key=lambda k: (poligonos[idx[k]][2], -area(poligonos[idx[k]][1])))
    for k in orden:
        cv2.fillPoly(inst, [np.round(poligonos[idx[k]][1]).astype(np.int32)], k + 1)
    if mx is None:
        return inst
    return cv2.remap(inst, mx, my, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def a_yolo(cajas, clases, wc, hc):
    """lineas 'clase cx cy w h' normalizadas al tamano de la imagen"""
    lineas = [f"{c} {(x1 + x2) / 2 / wc:.6f} {(y1 + y2) / 2 / hc:.6f} {(x2 - x1) / wc:.6f} {(y2 - y1) / hc:.6f}"
              for (x1, y1, x2, y2), c in zip(cajas, clases)]
    return "\n".join(lineas) + ("\n" if lineas else "")


def leer_yolo(path, w, h):
    """cajas xyxy en pixeles y clases desde un .txt YOLO"""
    cajas, clases = [], []
    p = Path(path)
    if p.is_file():
        for linea in p.read_text(encoding="utf-8").splitlines():
            partes = linea.split()
            if len(partes) < 5:
                continue
            c, cx, cy, bw, bh = int(partes[0]), *map(float, partes[1:5])
            cajas.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            clases.append(c)
    return np.array(cajas, dtype=np.float32).reshape(-1, 4), np.array(clases, dtype=np.int64)


# ---------------------------------------------------------------- dataset

def leer_lista(path):
    """rutas absolutas de imagen desde train.txt / val.txt de yolo_ds"""
    return [Path(l.strip()) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def camara_de(stem):
    return stem.split("_")[-1].upper()


def subconjunto(rutas, n):
    """n rutas repartidas entre las cuatro camaras, deterministico (para el modo de prueba)"""
    if not n:
        return rutas
    por_cam = defaultdict(list)
    for r in rutas:
        por_cam[camara_de(r.stem)].append(r)
    salida = []
    for cam in sorted(por_cam):
        salida += por_cam[cam][: max(1, n // len(por_cam))]
    return salida


def _enlazar(origen, destino):
    """hard link (no ocupa espacio y conserva los bytes); si el disco no lo soporta, copia"""
    try:
        os.link(origen, destino)
    except OSError:
        shutil.copy2(origen, destino)


def construir_split(rutas, split, out_dir, proyeccion="cilindrica", progreso=None):
    """genera out_dir/{images,labels,masks}/<split> en formato YOLO.

    proyeccion="cilindrica" (rama 4): la imagen pasa por la proyeccion de la rama 2 y las etiquetas
    salen de remapear los poligonos. proyeccion="fisheye" (rama 3): la imagen se enlaza sin tocar y
    las etiquetas son las de convert_to_yolo. Las anotaciones se buscan en <raiz>/instance_annotations.

    Agrupa por calibracion para calcular cada mapa una sola vez, y es reanudable: si una imagen ya
    tiene imagen, etiqueta y mascara, la saltea.
    """
    if proyeccion not in ("cilindrica", "fisheye"):
        raise ValueError(f"proyeccion desconocida: {proyeccion}")
    out_dir = Path(out_dir)
    d_img, d_lbl, d_msk = (out_dir / d / split for d in ("images", "labels", "masks"))
    for d in (d_img, d_lbl, d_msk):
        d.mkdir(parents=True, exist_ok=True)

    grupos, sin_calib = defaultdict(list), []
    for r in rutas:
        if proyeccion == "fisheye":
            grupos[None].append(r)
            continue
        try:
            grupos[clave_calib(r.stem, camara_de(r.stem))].append(r)
        except FileNotFoundError:
            sin_calib.append(r.name)
    if sin_calib:
        print(f"[aviso] {len(sin_calib)} imagenes sin calibracion, se omiten (ej: {sin_calib[:3]})")

    filas, hechas = [], 0
    total = sum(len(v) for v in grupos.values())
    params_png = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    for clave, rs in grupos.items():
        mx = my = None
        for r in rs:
            f_img, f_lbl, f_msk = d_img / f"{r.stem}.png", d_lbl / f"{r.stem}.txt", d_msk / f"{r.stem}.png"
            if f_img.is_file() and f_lbl.is_file() and f_msk.is_file():
                n = sum(1 for l in f_lbl.read_text().splitlines() if l.strip())
                filas.append({"stem": r.stem, "camara": camara_de(r.stem), "cajas": n,
                              "perdidos": np.nan, "degenerados": np.nan, "reanudada": True})
            else:
                ann = r.parent.parent / "instance_annotations" / f"{r.stem}.json"
                polis, W, H = poligonos_tp(ann)
                if proyeccion == "cilindrica":
                    if mx is None:
                        mx, my = mapas(clave)
                    img = cv2.imread(str(r))
                    if img is None:
                        print(f"[aviso] no se pudo leer {r}")
                        continue
                    cajas, clases, idx, perdidos, degen = etiquetas_cilindricas(polis, mx, my, W, H)
                    salida = cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR)
                    hc, wc = salida.shape[:2]
                    cv2.imwrite(str(f_img), salida, params_png)
                    inst = mapa_instancias(polis, idx, W, H, mx, my)
                else:
                    cajas, clases, idx = etiquetas_fisheye(polis, W, H)
                    perdidos = degen = 0
                    if not f_img.is_file():
                        _enlazar(r, f_img)
                    wc, hc = W, H
                    inst = mapa_instancias(polis, idx, W, H)
                cv2.imwrite(str(f_msk), inst, params_png)
                f_lbl.write_text(a_yolo(cajas, clases, wc, hc), encoding="utf-8")
                filas.append({"stem": r.stem, "camara": camara_de(r.stem), "cajas": len(cajas),
                              "perdidos": perdidos, "degenerados": degen, "reanudada": False})
            hechas += 1
            if progreso is not None:
                progreso(hechas, total)
        if proyeccion == "cilindrica":
            mapas.cache_clear()  # un grupo por vez en memoria
    return pd.DataFrame(filas)


def escribir_yaml(out_dir, clases=CLASES):
    """data.yaml de Ultralytics con la estructura images/<split> + labels/<split>"""
    out_dir = Path(out_dir)
    p = out_dir / "data.yaml"
    p.write_text(f"path: {out_dir.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
                 f"nc: {len(clases)}\nnames: {list(clases)}\n", encoding="utf-8")
    return p


class DatasetDeteccion(torch.utils.data.Dataset):
    """imagenes de out_dir/images/<split> con sus cajas (y mascaras) para los modelos de torchvision.

    Devuelve (imagen float RGB en [0, 1], target, stem). Las clases van corridas en uno porque en
    torchvision el 0 es el fondo. La normalizacion y el resize los hace el propio modelo.
    """

    def __init__(self, raiz, split, mascaras=False, flip=0.0):
        raiz = Path(raiz)
        self.d_img, self.d_lbl, self.d_msk = (raiz / d / split for d in ("images", "labels", "masks"))
        self.stems = sorted(p.stem for p in self.d_img.glob("*.png"))
        if not self.stems:
            raise FileNotFoundError(f"no hay imagenes en {self.d_img}")
        self.mascaras, self.flip = mascaras, flip

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        img = cv2.imread(str(self.d_img / f"{stem}.png"), cv2.IMREAD_COLOR)[:, :, ::-1]
        h, w = img.shape[:2]
        cajas, clases = leer_yolo(self.d_lbl / f"{stem}.txt", w, h)
        masks = None
        if self.mascaras:
            inst = cv2.imread(str(self.d_msk / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            masks = np.stack([inst == k + 1 for k in range(len(cajas))]) if len(cajas) else np.zeros((0, h, w), bool)
        if self.flip and random.random() < self.flip:
            img = img[:, ::-1]
            cajas[:, [0, 2]] = w - cajas[:, [2, 0]]
            if masks is not None:
                masks = masks[:, :, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255)
        target = {"boxes": torch.from_numpy(cajas), "labels": torch.from_numpy(clases + 1)}
        if masks is not None:
            target["masks"] = torch.from_numpy(np.ascontiguousarray(masks)).to(torch.uint8)
        return x, target, stem


def colar(lote):
    return tuple(zip(*lote))


# ---------------------------------------------------------------- torchvision

def construir_tv(nombre, pesos_coco=True, min_size=None):
    """detector de torchvision con la cabeza reemplazada por las 3 clases del TP (+ fondo).

    Con pesos_coco parte de los mismos pesos COCO que la rama 2. min_size cambia el resize interno:
    dejarlo en None para que la resolucion sea la misma que en la rama 2 (solo sirve para pruebas en CPU).
    """
    constructor, pesos = TV[nombre]
    kw = dict(box_score_thresh=CONF)
    if min_size:
        kw.update(min_size=min_size, max_size=int(min_size * 1333 / 800))
    modelo = constructor(weights=pesos if pesos_coco else None, weights_backbone=None, **kw)
    nc = len(CLASES) + 1
    modelo.roi_heads.box_predictor = FastRCNNPredictor(modelo.roi_heads.box_predictor.cls_score.in_features, nc)
    if getattr(modelo.roi_heads, "mask_predictor", None) is not None:
        modelo.roi_heads.mask_predictor = MaskRCNNPredictor(
            modelo.roi_heads.mask_predictor.conv5_mask.in_channels, 256, nc)
    return modelo


def _autocast(device, activo):
    return torch.autocast(device_type="cuda") if activo else contextlib.nullcontext()


@torch.inference_mode()
def evaluar_tv(modelo, loader, device, amp=False):
    """mAP@50 y mAP@50-95 sobre un loader, con supervision como en la rama 2"""
    modelo.eval()
    P, T = [], []
    for imgs, tgts, _ in loader:
        with _autocast(device, amp):
            salidas = modelo([i.to(device) for i in imgs])
        for o, t in zip(salidas, tgts):
            P.append(sv.Detections(xyxy=o["boxes"].float().cpu().numpy(), confidence=o["scores"].float().cpu().numpy(),
                                   class_id=o["labels"].cpu().numpy().astype(int) - 1))
            T.append(sv.Detections(xyxy=t["boxes"].numpy(), class_id=t["labels"].numpy().astype(int) - 1))
    r = MeanAveragePrecision().update(P, T).compute()
    return float(r.map50), float(r.map50_95)


def entrenar_tv(nombre, ds_dir, out_dir, hp=None, device=DEVICE, pesos_coco=True, min_size=None, log=print):
    """fine-tuning de un detector de torchvision. Reanudable; guarda el mejor por mAP@50-95 en val.

    Devuelve (ruta de best.pt, DataFrame con una fila por epoca).
    """
    hp = {**HP_TV, **(hp or {})}
    hp.setdefault("batch", BATCH_TV[nombre])
    hp.setdefault("lr", lr_tv(nombre, hp["batch"]))
    run = Path(out_dir) / "train" / nombre
    run.mkdir(parents=True, exist_ok=True)
    best, last, fin, hist = run / "best.pt", run / "last.pt", run / "TERMINADO", run / "results.csv"
    if fin.exists() and best.exists():
        log(f"[{nombre}] ya entrenado, uso {best}")
        return best, pd.read_csv(hist)

    semilla(hp["seed"])
    mascaras = nombre.startswith("maskrcnn")
    ds_tr = DatasetDeteccion(ds_dir, "train", mascaras=mascaras, flip=hp["flip"])
    ds_va = DatasetDeteccion(ds_dir, "val")
    g = torch.Generator()
    g.manual_seed(hp["seed"])
    kw = dict(collate_fn=colar, num_workers=hp["workers"], pin_memory=device == "cuda",
              persistent_workers=hp["workers"] > 0)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=hp["batch"], shuffle=True, generator=g, **kw)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=hp["batch"], shuffle=False, **kw)

    modelo = construir_tv(nombre, pesos_coco=pesos_coco, min_size=min_size).to(device)
    opt = torch.optim.SGD([p for p in modelo.parameters() if p.requires_grad], lr=hp["lr"],
                          momentum=hp["momentum"], weight_decay=hp["weight_decay"])
    total = hp["epochs"] * len(dl_tr)

    def factor(it):  # warmup lineal y coseno hasta el 1% del lr inicial
        return min(1.0, (it + 1) / hp["warmup_iters"]) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * it / total)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    amp = usar_amp(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ep0, mejor, sin_mejora = 0, -1.0, 0
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        modelo.load_state_dict(ck["modelo"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        ep0, mejor, sin_mejora = ck["epoca"] + 1, ck["mejor"], ck["sin_mejora"]
        log(f"[{nombre}] reanudo desde la epoca {ep0 + 1}")

    log(f"[{nombre}] train={len(ds_tr)} val={len(ds_va)} | epocas={hp['epochs']} batch={hp['batch']} "
        f"lr={hp['lr']} | device={device} amp={amp}")
    for ep in range(ep0, hp["epochs"]):
        modelo.train()
        t0, suma, n, saltados = time.perf_counter(), 0.0, 0, 0
        for imgs, tgts, _ in dl_tr:
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in tgts]
            with _autocast(device, amp):
                perdida = sum(modelo(imgs, tgts).values())
            opt.zero_grad(set_to_none=True)
            if not math.isfinite(perdida.item()):
                saltados += 1  # un lote con perdida no finita se saltea en vez de romper la corrida
                sched.step()
                continue
            scaler.scale(perdida).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            suma, n = suma + perdida.item(), n + 1
        if saltados > 0.1 * len(dl_tr):
            raise RuntimeError(f"[{nombre}] {saltados} lotes con perdida no finita: bajar el lr")
        t_train = time.perf_counter() - t0

        t1 = time.perf_counter()
        m50, m5095 = evaluar_tv(modelo, dl_va, device, amp)
        t_val = time.perf_counter() - t1

        if m5095 > mejor:
            mejor, sin_mejora = m5095, 0
            torch.save(modelo.state_dict(), best)
        else:
            sin_mejora += 1
        fila = pd.DataFrame([{"epoch": ep + 1, "perdida": suma / max(n, 1), "map50": m50, "map50_95": m5095,
                              "lr": opt.param_groups[0]["lr"], "s_train": t_train, "s_val": t_val,
                              "n_train": len(ds_tr), "n_val": len(ds_va), "lotes_saltados": saltados}])
        fila.to_csv(hist, mode="a", header=not hist.exists(), index=False)
        torch.save({"modelo": modelo.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoca": ep, "mejor": mejor, "sin_mejora": sin_mejora}, last)
        log(f"[{nombre}] epoca {ep + 1}/{hp['epochs']} | perdida {suma / max(n, 1):.3f} | val mAP50 {m50:.3f} "
            f"mAP50-95 {m5095:.3f} | {t_train / 60:.1f} + {t_val / 60:.1f} min")
        if sin_mejora >= hp["patience"]:
            log(f"[{nombre}] sin mejora en {hp['patience']} epocas, corto")
            break

    fin.write_text(json.dumps({"mejor_map50_95_val": mejor, "hp": hp}, indent=2), encoding="utf-8")
    return best, pd.read_csv(hist)


# ---------------------------------------------------------------- yolo

def entrenar_yolo(nombre, yaml_path, out_dir, hp=None, batch=None, device=DEVICE, plots=True, log=print):
    """fine-tuning de un YOLO11 con Ultralytics. Reanudable; Ultralytics guarda best.pt por mAP@50-95 en val.

    Devuelve (ruta de best.pt, DataFrame con una fila por epoca y los tiempos de train y val).
    """
    hp = {**HP_YOLO, **(hp or {})}
    batch = batch or BATCH_YOLO.get(nombre, 16)
    proyecto = Path(out_dir) / "train"
    run = proyecto / nombre
    best, last, fin = run / "weights/best.pt", run / "weights/last.pt", run / "TERMINADO"
    if fin.exists() and best.exists():
        log(f"[{nombre}] ya entrenado, uso {best}")
        return best, _historia_yolo(run)

    # tiempos por epoca separando entrenamiento y validacion, para estimar la corrida completa
    marcas, tiempos = {}, []

    def ini_ep(_):
        marcas["t0"] = time.perf_counter()

    def fin_train(_):
        marcas["t1"] = time.perf_counter()

    def fin_fit(tr):  # Ultralytics tambien lo dispara en la evaluacion final: solo cuenta tras una epoca
        t1 = marcas.pop("t1", None)
        if t1 is None:
            return
        tiempos.append({"epoch": tr.epoch + 1, "s_train": t1 - marcas["t0"], "s_val": time.perf_counter() - t1})

    if last.exists():
        log(f"[{nombre}] reanudo desde {last}")
        modelo = YOLO(str(last))
        kw = dict(resume=True)
    else:
        modelo = YOLO(f"{nombre}.pt")
        kw = dict(data=str(yaml_path), project=str(proyecto), name=nombre, exist_ok=True, device=device,
                  batch=batch, plots=plots, **hp)
    modelo.add_callback("on_train_epoch_start", ini_ep)
    modelo.add_callback("on_train_epoch_end", fin_train)
    modelo.add_callback("on_fit_epoch_end", fin_fit)
    log(f"[{nombre}] epocas={hp['epochs']} batch={batch} imgsz={hp['imgsz']} | device={device}")
    try:
        modelo.train(**kw)
    except AssertionError as e:  # Ultralytics avisa asi que no queda nada por reanudar
        if "nothing to resume" not in str(e) or not best.exists():
            raise
    if tiempos:
        pd.DataFrame(tiempos).to_csv(run / "tiempos.csv", mode="a", header=not (run / "tiempos.csv").exists(), index=False)
    fin.write_text(json.dumps({"hp": hp, "batch": batch}, indent=2), encoding="utf-8")
    return best, _historia_yolo(run)


def _historia_yolo(run):
    """results.csv de Ultralytics con las columnas renombradas como las de entrenar_tv"""
    p = Path(run) / "results.csv"
    if not p.is_file():
        return pd.DataFrame()
    h = pd.read_csv(p)
    h.columns = h.columns.str.strip()
    h = h.rename(columns={"metrics/mAP50(B)": "map50", "metrics/mAP50-95(B)": "map50_95"})
    h["perdida"] = h[[c for c in h.columns if c.startswith("train/")]].sum(axis=1)
    t = Path(run) / "tiempos.csv"
    if t.is_file():
        h = h.merge(pd.read_csv(t).drop_duplicates("epoch", keep="last"), on="epoch", how="left")
    return h


def entrenar(nombre, ds_dir, out_dir, device=DEVICE, log=print, **kw):
    """entrena cualquiera de los 8 modelos sobre un dataset armado con construir_split"""
    if nombre in YOLOS:
        return entrenar_yolo(nombre, Path(ds_dir) / "data.yaml", out_dir, device=device, log=log, **kw)
    if nombre in TV:
        return entrenar_tv(nombre, ds_dir, out_dir, device=device, log=log, **kw)
    raise ValueError(f"modelo desconocido: {nombre}. Opciones: {MODELOS_RAMA4}")


def ruta_best(nombre, out_dir):
    run = Path(out_dir) / "train" / nombre
    return run / "weights/best.pt" if nombre in YOLOS else run / "best.pt"


def terminado(nombre, out_dir):
    """True si el entrenamiento de ese modelo termino (no solo si hay un best.pt a mitad de camino)"""
    return (Path(out_dir) / "train" / nombre / "TERMINADO").exists() and ruta_best(nombre, out_dir).exists()


def historia(nombre, out_dir):
    """curva de entrenamiento ya guardada, sin entrenar: results.csv de Ultralytics o del loop propio"""
    run = Path(out_dir) / "train" / nombre
    if nombre in YOLOS:
        return _historia_yolo(run)
    p = run / "results.csv"
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()


def resumen_dataset(ds_dir):
    """imagenes y cajas por split de un dataset armado con construir_split, leyendo solo las etiquetas"""
    filas = []
    for split in ("train", "val"):
        d = Path(ds_dir) / "labels" / split
        txts = list(d.glob("*.txt")) if d.is_dir() else []
        filas.append({"split": split, "imagenes": len(txts),
                      "cajas": sum(sum(1 for l in f.read_text().splitlines() if l.strip()) for f in txts)})
    return pd.DataFrame(filas).set_index("split")


# ---------------------------------------------------------------- inferencia sobre el test

def detector_ft(nombre, peso, device=DEVICE, conf=CONF, imgsz=IMGSZ):
    """detectar(img_bgr) -> xyxy, conf, clase del TP, para un modelo ya fine-tuneado"""
    if nombre in YOLOS:
        modelo = YOLO(str(peso))

        def detectar(img):
            r = modelo.predict(img, imgsz=imgsz, conf=conf, device=device, verbose=False)[0].boxes
            return r.xyxy.cpu().numpy(), r.conf.cpu().numpy(), r.cls.cpu().numpy().astype(int)
        return detectar

    modelo = construir_tv(nombre, pesos_coco=False)
    modelo.load_state_dict(torch.load(peso, map_location="cpu", weights_only=True))
    modelo.eval().to(device)

    @torch.inference_mode()
    def detectar(img):
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1).float().div_(255).to(device)
        o = modelo([x])[0]
        return o["boxes"].cpu().numpy(), o["scores"].cpu().numpy(), o["labels"].cpu().numpy().astype(int) - 1
    return detectar


def predecir_test(nombre, detectar, test, salidas, proyeccion="cilindrica", progreso=None):
    """mismo flujo y misma medicion de latencia que predecir() de la rama 2, con cache en disco.

    Con proyeccion="fisheye" (rama 3) detecta directo sobre la imagen original: no hay pre ni
    reproyeccion, y las columnas *_cil repiten las de la fisheye.
    """
    ruta = Path(salidas) / f"preds_{nombre}_{len(test)}.pkl"
    if ruta.exists():
        return pd.read_pickle(ruta)
    cil_activa = proyeccion == "cilindrica"
    if cil_activa:
        for fila in test.drop_duplicates("calib").itertuples():
            mapas(fila.calib)  # los mapas antes, para que no ensucien la latencia
    img0 = cilindrica(test.iloc[0]) if cil_activa else cv2.imread(str(test.iloc[0].ruta))
    for _ in range(10):  # calentamiento
        detectar(img0)

    filas = []
    for i, fila in enumerate(test.itertuples(), 1):
        img = cv2.imread(str(fila.ruta))
        t0 = time.perf_counter()
        cil = cilindrica(fila, img) if cil_activa else img
        t1 = time.perf_counter()
        xyxy, conf, cls = detectar(cil)
        t2 = time.perf_counter()
        if cil_activa:
            fish, ok = caja_a_fisheye(xyxy, *mapas(fila.calib), fila.W, fila.H)
        else:
            fish, ok = xyxy, np.ones(len(xyxy), dtype=bool)
        t3 = time.perf_counter()
        filas.append({"stem": fila.stem, "cajas_cil": xyxy, "conf_cil": conf, "clases_cil": cls,
                      "cajas": fish[ok], "conf": conf[ok], "clases": cls[ok],
                      "ms_pre": 1e3 * (t1 - t0), "ms_inf": 1e3 * (t2 - t1), "ms_post": 1e3 * (t3 - t2)})
        if progreso is not None:
            progreso(i, len(test))
    preds = pd.DataFrame(filas)
    preds.to_pickle(ruta)
    return preds


def a_pred_json(preds, out_path):
    """predicciones (stem, cajas xyxy fisheye, conf, clases) -> json de eval_radial.py"""
    datos = [{"image": f"{f.stem}.png", "class": int(c), "score": float(s), "bbox": [float(v) for v in b]}
             for f in preds.itertuples() for b, s, c in zip(f.cajas, f.conf, f.clases)]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(datos), encoding="utf-8")
    return len(datos)


def iou_xyxy(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------- estimacion de tiempos

def estimar_horas(historia, n_prueba_train, n_prueba_val, epocas, n_train=6148, n_val=1085):
    """horas de una corrida completa extrapolando la ultima epoca medida (la primera trae calentamiento).

    Es una cota gruesa: con pocas imagenes pesan los costos fijos de cada epoca. Sirve para decidir
    que modelos entran en el tiempo disponible, no para planificar al minuto.
    """
    if "s_train" not in historia or historia["s_train"].dropna().empty:
        return np.nan
    u = historia.dropna(subset=["s_train"]).iloc[-1]
    return epocas * (u["s_train"] / n_prueba_train * n_train + u["s_val"] / n_prueba_val * n_val) / 3600
