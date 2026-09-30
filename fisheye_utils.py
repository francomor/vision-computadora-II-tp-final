"""
fisheye_utils.py - funciones compartidas por las tres ramas del TP (WoodScape fisheye).

Uso desde un notebook en notebooks/:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent))
    from fisheye_utils import cargar_test, MODELOS, evaluar, tabla_resultados

Que hay:
- cargar_test / cargar_gt: el test del Drive con el mismo gt que convert_to_yolo.py
  (class_map.json, min_area=300, min_side=8), la calibracion de cada imagen y su centro optico.
- get_mapping / fisheye_to_cylindrical / mapas / cilindrica: proyeccion fisheye -> cilindrica
  (tutorial de Plaut), con los mapas cacheados y recortados a la banda del horizonte.
- caja_a_fisheye: lleva cajas de la cilindrica a la fisheye.
- MODELOS / detector_yolo / detector_tv: detectores COCO con las clases ya traducidas a las del TP.
- evaluar / tabla_resultados: map50, map50-95 y map estratificado por excentricidad radial.
- dibujar: cajas con supervision.

Chequeo rapido del entorno: uv run python fisheye_utils.py
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # algunas ops de torchvision no tienen kernel en mps

import json
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import torch
from pyquaternion import Quaternion
from supervision.metrics import MeanAveragePrecision
from torchvision.models import detection as tvd
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent
TEST_DIR = RAIZ / "CV2/split_v1/drive_package/test_set"
CALIB_DIR = RAIZ / "CV2/calibration"
MANIFEST = RAIZ / "CV2/split_v1/manifest.csv"
CLASS_MAP = RAIZ / "CV2/Scripts/class_map.json"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
IMGSZ = 640  # misma resolucion para las tres ramas (regla 5)
CONF = 0.01  # umbral bajo: el map necesita la curva completa de precision-recall
MIN_AREA, MIN_SIDE = 300, 8  # mismo filtro que convert_to_yolo.py
HFOV, VFOV = np.deg2rad(190), np.deg2rad(143)  # cubren todo el campo de la lente
ELEV_ARRIBA, ELEV_ABAJO = np.deg2rad(45), np.deg2rad(45)  # banda alrededor del horizonte de la cilindrica
ANILLOS = [0, 1 / 3, 2 / 3, 1.0]
NOMBRES_ANILLOS = ["centro", "medio", "periferia"]

sys.path.insert(0, str(RAIZ / "CV2/Scripts"))
from convert_to_yolo import collect_points, desenvolver, tags_of  # noqa: E402

_cm = json.loads(CLASS_MAP.read_text(encoding="utf-8"))
CLASES = _cm["clases"]
MAPEO = {k: CLASES.index(v) for k, v in _cm["mapeo"].items() if v is not None}

# ids coco de ultralytics (80 clases) y de torchvision (91, con huecos) -> ids del tp
COCO_YOLO = {2: 0, 5: 0, 7: 0, 0: 1, 1: 2, 3: 2}
COCO_TV = {3: 0, 6: 0, 8: 0, 1: 1, 2: 2, 4: 2}


# ---------------------------------------------------------------- datos

def cargar_gt(stem, ann_dir=TEST_DIR / "instance_annotations"):
    """cajas xyxy y clases del tp para una imagen, igual que convert_to_yolo"""
    data = desenvolver(json.loads((Path(ann_dir) / f"{stem}.json").read_text(encoding="utf-8", errors="ignore")))
    W, H = float(data["image_width"]), float(data["image_height"])
    cajas, clases = [], []
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
        cajas.append([x1, y1, x2, y2])
        clases.append(destino)
    return np.array(cajas, dtype=np.float32).reshape(-1, 4), np.array(clases, dtype=int)


def clave_calib(stem, camara):
    """json de calibracion como string (hasheable): por imagen si existe, si no la primera de su camara"""
    p = CALIB_DIR / f"{stem}.json"
    if not p.is_file():
        p = next(iter(sorted(CALIB_DIR.glob(f"*_{camara}.json"))), None)
        if p is None:
            raise FileNotFoundError(f"no hay calibraciones de {camara} en {CALIB_DIR}")
    c = json.loads(p.read_text())
    return json.dumps({"intrinsic": c["intrinsic"], "extrinsic": {"quaternion": c["extrinsic"]["quaternion"]}}, sort_keys=True)


def cargar_test(n_imagenes=None):
    """dataframe del test: una fila por imagen con ruta, gt, calibracion y centro optico"""
    man = pd.read_csv(MANIFEST)
    test = man[man["split"] == "test"][["filename", "camera"]].reset_index(drop=True)
    test["stem"] = test["filename"].str.replace(".png", "", regex=False)
    test["ruta"] = test["filename"].map(lambda f: TEST_DIR / "rgb_images" / f)
    presentes = test["ruta"].map(Path.is_file)
    print(f"test: {len(test)} imagenes en el manifest | presentes: {presentes.sum()} | faltan {(~presentes).sum()}")
    test = test[presentes].reset_index(drop=True)
    if n_imagenes:
        test = test.groupby("camera").head(n_imagenes // 4).reset_index(drop=True)

    gts = [cargar_gt(s) for s in test["stem"]]
    test["gt_cajas"] = [g[0] for g in gts]
    test["gt_clases"] = [g[1] for g in gts]
    n_cajas = sum(len(c) for c in test["gt_clases"])
    if len(test) == 1000:
        assert n_cajas == 11910, f"el gt no coincide con el readme del equipo: {n_cajas}"

    test["calib"] = [clave_calib(s, c) for s, c in zip(test["stem"], test["camera"])]
    test["calib_propia"] = [(CALIB_DIR / f"{s}.json").is_file() for s in test["stem"]]
    intr = test["calib"].map(lambda c: json.loads(c)["intrinsic"])
    test["cx0"] = intr.map(lambda i: i["cx_offset"] + i["width"] / 2 - 0.5)  # centro optico real
    test["cy0"] = intr.map(lambda i: i["cy_offset"] + i["height"] / 2 - 0.5)
    test["W"] = intr.map(lambda i: i["width"])
    test["H"] = intr.map(lambda i: i["height"])
    return test


# ---------------------------------------------------------------- proyeccion cilindrica
# get_mapping y fisheye_to_cylindrical son las del tutorial de plaut, sin cambios

def get_mapping(calib, hfov=np.deg2rad(190), vfov=np.deg2rad(143)):
    """
    Compute the pixel mapping from a fisheye image to a cylindrical image
    :param calib: calibration in WoodScape format, as a dictionary
    :param hfov: horizontal field of view, in radians
    :param vfov: vertical field of view, in radians
    :return: horizontal and vertical mapping
    """
    # Prepare intrinsic and extrinsic matrices for the cylindrical image
    R = Quaternion(w=calib['extrinsic']['quaternion'][3],
                   x=calib['extrinsic']['quaternion'][0],
                   y=calib['extrinsic']['quaternion'][1],
                   z=calib['extrinsic']['quaternion'][2]).rotation_matrix.T
    rdf_to_flu = np.array([[0, 0, 1],
                           [-1, 0, 0],
                           [0, -1, 0]], dtype=np.float64)
    R = R @ rdf_to_flu  # Rotation from vehicle to camera includes FLU-to-RDF. Remove FLU-to-RDF from R.
    azimuth = np.arccos(R[2, 2] / np.sqrt(R[0, 2] ** 2 + R[2, 2] ** 2))  # azimuth angle parallel to the ground
    if R[0, 2] < 0:
        azimuth = 2*np.pi - azimuth
    tilt = -np.arccos(np.sqrt(R[0, 2]**2 + R[2, 2]**2))  # elevation to the ground plane
    Ry = np.array([[np.cos(azimuth), 0, np.sin(azimuth)],
                     [0, 1, 0],
                     [-np.sin(azimuth), 0, np.cos(azimuth)]]).T
    R = R @ Ry  # now forward axis is parallel to the ground, but in the direction of the camera (not vehicle's forward)
    f = calib['intrinsic']['k1']
    h, w = int(2*f*np.tan(vfov/2)), int(f*hfov)  # cylindrical image has a different size than the fisheye image
    K = np.array([[f, 0, w/2],
                  [0, f, f * np.tan(vfov/2 + tilt)],
                  [0, 0, 1]], dtype=np.float32)  # intrinsic matrix for the cylindrical projection
    K_inv = np.linalg.inv(K)
    # Create pixel grid and compute a ray for every pixel
    xv, yv = np.meshgrid(range(w), range(h), indexing='xy')
    p = np.stack([xv, yv, np.ones_like(xv)])  # pixel homogeneous coordinates
    p = p.transpose(1, 2, 0)[:, :, :, np.newaxis]
    r = K_inv @ p  # r is in cylindrical coordinates
    r /= r[:, :, [2], :]  # r is now in cylindrical coordinates with unit cylindrical radius
    # Convert to Cartesian coordinates
    r[:, :, 2, :] = np.cos(r[:, :, 0, :])
    r[:, :, 0, :] = np.sin(r[:, :, 0, :])
    r[:, :, 1, :] = r[:, :, 1, :]
    r = R @ r  # extrinsic rotation from an upright cylinder to the camera axis
    theta = np.arccos(r[:, :, [2], :] / np.linalg.norm(r, axis=2, keepdims=True))  # compute incident angle
    # project the ray onto the fisheye image according to the fisheye model and intrinsic calibration
    c_X = calib['intrinsic']['cx_offset'] + calib['intrinsic']['width'] / 2 - 0.5
    c_Y = calib['intrinsic']['cy_offset'] + calib['intrinsic']['height'] / 2 - 0.5
    k1, k2, k3, k4 = [calib['intrinsic']['k%d' % i] for i in range(1, 5)]
    rho = k1 * theta + k2 * theta ** 2 + k3 * theta ** 3 + k4 * theta ** 4
    chi = np.linalg.norm(r[:, :, :2, :], axis=2, keepdims=True)
    u = np.true_divide(rho * r[:, :, [0], :], chi, out=np.zeros_like(chi), where=(chi != 0))  # horizontal
    v = np.true_divide(rho * r[:, :, [1], :], chi, out=np.zeros_like(chi), where=(chi != 0))  # vertical
    mapx = u[:, :, 0, 0] + c_X
    mapy = v[:, :, 0, 0] * calib['intrinsic']['aspect_ratio'] + c_Y
    return mapx, mapy


def fisheye_to_cylindrical(image, calib):
    """
    Warp a fisheye image to a cylindrical image
    :param image: fisheye image, as a numpy array
    :param calib: calibration in WoodScape format, as a dictionary
    :return: cylindrical image
    """
    mapx, mapy = get_mapping(calib)
    return cv2.remap(image, mapx.astype(np.float32), mapy.astype(np.float32), interpolation=cv2.INTER_LINEAR)


def fila_horizonte(calib, vfov=VFOV):
    """fila de la cilindrica donde cae el horizonte, mismas cuentas que get_mapping"""
    q = calib["extrinsic"]["quaternion"]
    R = Quaternion(w=q[3], x=q[0], y=q[1], z=q[2]).rotation_matrix.T @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    tilt = -np.arccos(np.sqrt(R[0, 2] ** 2 + R[2, 2] ** 2))
    return calib["intrinsic"]["k1"] * np.tan(vfov / 2 + tilt)


@lru_cache(maxsize=None)
def mapas(clave, hfov=HFOV, vfov=VFOV, arriba=ELEV_ARRIBA, abajo=ELEV_ABAJO):
    """mapx, mapy en float32 para cv2.remap, recortados a la banda del horizonte; uno por calibracion"""
    calib = json.loads(clave)
    mx, my = get_mapping(calib, hfov, vfov)
    f, cy = calib["intrinsic"]["k1"], fila_horizonte(calib, vfov)
    y0, y1 = max(0, int(cy - f * np.tan(arriba))), min(mx.shape[0], int(cy + f * np.tan(abajo)))
    return mx[y0:y1].astype(np.float32), my[y0:y1].astype(np.float32)


def cilindrica(fila, img=None):
    """imagen bgr fisheye -> cilindrica con los mapas cacheados; fila es una fila de cargar_test"""
    img = cv2.imread(str(fila.ruta)) if img is None else img
    return cv2.remap(img, *mapas(fila.calib), interpolation=cv2.INTER_LINEAR)


def caja_a_fisheye(cajas, mapx, mapy, W, H, n=20):
    """reproyecta cajas xyxy de la cilindrica a la fisheye muestreando n puntos por lado"""
    # ponytail: caja axis-aligned del contorno curvo, en la periferia infla el area; cajas rotadas si hace falta
    hc, wc = mapx.shape
    t = np.linspace(0, 1, n)
    salida = np.zeros((len(cajas), 4), dtype=np.float32)
    validas = np.zeros(len(cajas), dtype=bool)
    for i, (x1, y1, x2, y2) in enumerate(cajas):
        xs = np.concatenate([x1 + (x2 - x1) * t, np.full(n, x2), x2 - (x2 - x1) * t, np.full(n, x1)])
        ys = np.concatenate([np.full(n, y1), y1 + (y2 - y1) * t, np.full(n, y2), y2 - (y2 - y1) * t])
        c = np.clip(np.round(xs).astype(int), 0, wc - 1)
        f = np.clip(np.round(ys).astype(int), 0, hc - 1)
        u, v = mapx[f, c], mapy[f, c]
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if ok.sum() < 2:
            continue  # la caja cae entera fuera de la lente
        salida[i] = [u[ok].min(), v[ok].min(), u[ok].max(), v[ok].max()]
        validas[i] = True
    return salida, validas


# ---------------------------------------------------------------- detectores

def detector_yolo(peso, imgsz=IMGSZ, conf=CONF):
    """(detectar, mapa_coco) para un yolo de ultralytics; detectar(img_bgr) -> xyxy, conf, clase_coco"""
    modelo = YOLO(peso)

    def detectar(img):
        r = modelo.predict(img, imgsz=imgsz, conf=conf, device=DEVICE, verbose=False)[0].boxes
        return r.xyxy.cpu().numpy(), r.conf.cpu().numpy(), r.cls.cpu().numpy().astype(int)
    return detectar, COCO_YOLO


def detector_tv(constructor, pesos, conf=CONF):
    """lo mismo para un detector de torchvision (resize propio, min_size=800)"""
    modelo = constructor(weights=pesos, box_score_thresh=conf).eval().to(DEVICE)
    prep = pesos.transforms()

    @torch.inference_mode()
    def detectar(img):
        x = prep(torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1)).to(DEVICE)
        o = modelo([x])[0]
        return o["boxes"].cpu().numpy(), o["scores"].cpu().numpy(), o["labels"].cpu().numpy()
    return detectar, COCO_TV


def a_clases_tp(xyxy, conf, cls, coco):
    """filtra las clases coco que no son del tp y traduce el resto"""
    m = np.isin(cls, list(coco))
    return xyxy[m], conf[m], np.array([coco[c] for c in cls[m]], dtype=int)


# los mismos modelos para todas las ramas; cada valor se llama para cargar el modelo
MODELOS = {
    **{f"yolo11{t}": (lambda t=t: detector_yolo(f"yolo11{t}.pt")) for t in "nsmlx"},
    "fasterrcnn_mobilenet_v3": lambda: detector_tv(tvd.fasterrcnn_mobilenet_v3_large_fpn, tvd.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
    "fasterrcnn_r50_v2": lambda: detector_tv(tvd.fasterrcnn_resnet50_fpn_v2, tvd.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
    "maskrcnn_r50_v2": lambda: detector_tv(tvd.maskrcnn_resnet50_fpn_v2, tvd.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
}


# ---------------------------------------------------------------- metricas

def radio(cajas, fila):
    """distancia normalizada (por la media diagonal) del centro de cada caja al centro optico"""
    cx, cy = (cajas[:, 0] + cajas[:, 2]) / 2, (cajas[:, 1] + cajas[:, 3]) / 2
    return np.hypot(cx - fila.cx0, cy - fila.cy0) / np.hypot(fila.W / 2, fila.H / 2)


def evaluar(p, filas, anillo=None):
    """map50 y map50-95 de las predicciones p sobre las filas del test, opcionalmente solo un anillo radial

    p es un dataframe con columnas stem, cajas (xyxy en la fisheye), conf y clases (ids del tp).
    """
    p = p.set_index("stem")
    lo, hi = (ANILLOS[anillo], ANILLOS[anillo + 1]) if anillo is not None else (-np.inf, np.inf)
    P, T = [], []
    for fila in filas.itertuples():
        q = p.loc[fila.stem]
        rg, rp = radio(fila.gt_cajas, fila), radio(q["cajas"], fila)
        mg, mp = (rg >= lo) & (rg < hi), (rp >= lo) & (rp < hi)
        T.append(sv.Detections(xyxy=fila.gt_cajas[mg], class_id=fila.gt_clases[mg]))
        P.append(sv.Detections(xyxy=q["cajas"][mp].astype(np.float32), class_id=q["clases"][mp], confidence=q["conf"][mp].astype(np.float32)))
    r = MeanAveragePrecision().update(P, T).compute()
    return r.map50, r.map50_95


def tabla_resultados(preds, filas):
    """una fila por modelo: map global, map por anillo y latencias medias (columnas ms_* que existan)"""
    salida = []
    for nombre, p in preds.items():
        m50, m5095 = evaluar(p, filas)
        f = {"modelo": nombre, "mAP50": m50, "mAP50-95": m5095}
        for i, a in enumerate(NOMBRES_ANILLOS):
            f[f"mAP50 {a}"], f[f"mAP50-95 {a}"] = evaluar(p, filas, anillo=i)
        ms = [k for k in p.columns if k.startswith("ms_")]
        f.update({k: p[k].mean() for k in ms})
        f["ms_total"] = sum(f[k] for k in ms)
        f["fps"] = 1e3 / f["ms_total"] if f["ms_total"] else np.nan
        salida.append(f)
    return pd.DataFrame(salida).set_index("modelo")


# ---------------------------------------------------------------- visualizacion

_cajas_ann = sv.BoxAnnotator(thickness=2)
_etiquetas_ann = sv.LabelAnnotator(text_scale=0.5, text_padding=2)


def dibujar(img, cajas, clases, conf=None):
    """dibuja cajas con supervision sobre una copia de la imagen bgr"""
    det = sv.Detections(xyxy=np.asarray(cajas, dtype=np.float32).reshape(-1, 4), class_id=np.asarray(clases, dtype=int),
                        confidence=None if conf is None else np.asarray(conf, dtype=np.float32))
    etiquetas = [CLASES[c] for c in det.class_id] if conf is None else [f"{CLASES[c]} {p:.2f}" for c, p in zip(det.class_id, det.confidence)]
    out = _cajas_ann.annotate(img.copy(), det)
    return _etiquetas_ann.annotate(out, det, labels=etiquetas)


if __name__ == "__main__":
    # chequeo del entorno y de la logica con 8 imagenes
    test = cargar_test(8)
    gt = pd.DataFrame({"stem": test["stem"], "cajas": test["gt_cajas"], "clases": test["gt_clases"],
                       "conf": [np.ones(len(c), dtype=np.float32) for c in test["gt_clases"]]})
    assert min(evaluar(gt, test)) > 0.99, "el gt como prediccion tiene que dar map 1"

    fila = test.iloc[0]
    mx, my = mapas(fila.calib)
    fi, ci = mx.shape[0] // 2, mx.shape[1] // 2
    caja, ok = caja_a_fisheye(np.array([[ci - 10, fi - 10, ci + 10, fi + 10]]), mx, my, fila.W, fila.H)
    assert ok[0] and caja[0, 0] <= mx[fi, ci] <= caja[0, 2] and caja[0, 1] <= my[fi, ci] <= caja[0, 3]
    print(f"ok | device: {DEVICE} | cilindrica: {cilindrica(fila).shape[1]}x{cilindrica(fila).shape[0]}")
