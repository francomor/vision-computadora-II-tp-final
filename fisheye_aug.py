"""
fisheye_aug.py - rama 3 del TP: fine-tuning sobre las fisheye, sin rectificar.

Uso desde un notebook en notebooks/:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent))
    from fisheye_aug import AUG_RADIAL, distorsionar, preparar_yolo_ds, entrenar

Que hay:
- mapas_radiales / puntos_adelante / distorsionar_cajas / distorsionar: augmentacion de distorsion radial
  anclada en la esquina mas lejana, r_src = r (1 + k r^2) / (1 + k). k > 0 lente mas fisheye, k < 0 menos.
- DistorsionRadial / YOLODatasetRadial / TrainerFisheye: la aplican dentro de Ultralytics, imagen por imagen y
  antes del mosaico. Con radial=None el trainer es el de Ultralytics sin cambios (variante base).
- preparar_yolo_ds: arma images/ y labels/ con enlaces simbolicos para que Ultralytics encuentre las etiquetas.
- entrenar: el mismo entrenamiento para las dos variantes, desde la terminal (tools/train_rama3.py) o en Colab
  (notebooks/rama3_colab.ipynb).
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # algunas ops no tienen kernel en mps

import json
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer

RAIZ = Path(__file__).resolve().parent
CLASS_MAP = RAIZ / "config/class_map.json"
CALIBRACION = RAIZ / "config/calibration.json"  # centro optico de las 8234 imagenes (drive del equipo)

# variante "radial": probabilidad de distorsionar cada imagen y rango de k (uniforme)
AUG_RADIAL = {"p": 0.5, "k_min": -0.2, "k_max": 0.3}
K_LIMITE = -1 / 3  # con k <= -1/3 la transformacion deja de ser monotona y la imagen se pliega

# numeros de control de convert_to_yolo.py (guia del equipo): (imagenes, cajas) por split
CONTROL = {"train": (6148, 74909), "val": (1085, 13176), "test": (1000, 11910)}


# ---------------------------------------------------------------- distorsion radial

@lru_cache(maxsize=8)
def _grilla(w, h):
    """coordenadas de cada pixel (float64), cacheadas por tamano"""
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    return u, v


def radio_max(w, h, cx, cy):
    """distancia del centro a la esquina mas lejana; con ella r = 1 en esa esquina y r <= 1 en toda la imagen"""
    return max(np.hypot(x - cx, y - cy) for x in (0, w - 1) for y in (0, h - 1))


def mapas_radiales(w, h, k, cx, cy):
    """mapx, mapy para cv2.remap: para cada pixel de salida, de que pixel de entrada se lee.

    r_src = r (1 + k r^2) / (1 + k), con r normalizado por radio_max. La esquina mas lejana queda fija.
    """
    R = radio_max(w, h, cx, cy)
    u, v = _grilla(w, h)
    du, dv = u - cx, v - cy
    s = (1 + k * (du * du + dv * dv) / (R * R)) / (1 + k)
    return (cx + du * s).astype(np.float32), (cy + dv * s).astype(np.float32)


def puntos_adelante(x, y, k, cx, cy, R):
    """lleva puntos de la imagen original a la distorsionada (la inversa de r_src(r), por interpolacion)"""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rs = np.hypot(x - cx, y - cy) / R
    r = np.linspace(0.0, 1.0, 2049)
    f = r * (1 + k * r * r) / (1 + k)  # r_src(r), creciente si k > -1/3
    rd = np.interp(rs, f, r)  # el r de salida tal que r_src(r) = rs
    escala = np.divide(rd, rs, out=np.full_like(rs, 1 + k), where=rs > 1e-9)  # cerca del centro rd/rs -> 1 + k
    return cx + (x - cx) * escala, cy + (y - cy) * escala


def distorsionar_cajas(cajas, k, w, h, centro=None, n=16):
    """cajas xyxy (px) de la original -> cajas xyxy en la distorsionada, y mascara de las que siguen siendo validas.

    Muestreo n puntos por lado, los llevo a la distorsionada y tomo el minimo y el maximo (como caja_a_fisheye de la
    rama 2). La caja de un rectangulo deformado es algo mas grande que la del objeto deformado.
    """
    cajas = np.asarray(cajas, dtype=np.float64).reshape(-1, 4)
    if len(cajas) == 0:
        return cajas.astype(np.float32), np.zeros(0, dtype=bool)
    cx, cy = centro if centro is not None else ((w - 1) / 2, (h - 1) / 2)
    R = radio_max(w, h, cx, cy)
    t = np.linspace(0, 1, n)
    x1, y1, x2, y2 = (cajas[:, i:i + 1] for i in range(4))
    xs = np.concatenate([x1 + (x2 - x1) * t, np.repeat(x2, n, 1), x1 + (x2 - x1) * t, np.repeat(x1, n, 1)], axis=1)
    ys = np.concatenate([np.repeat(y1, n, 1), y1 + (y2 - y1) * t, np.repeat(y2, n, 1), y1 + (y2 - y1) * t], axis=1)
    xd, yd = puntos_adelante(xs, ys, k, cx, cy, R)
    salida = np.stack([xd.min(1), yd.min(1), xd.max(1), yd.max(1)], axis=1)
    salida[:, [0, 2]] = salida[:, [0, 2]].clip(0, w)
    salida[:, [1, 3]] = salida[:, [1, 3]].clip(0, h)  # con k > 0 lo que esta cerca del borde puede salir de cuadro
    validas = ((salida[:, 2] - salida[:, 0]) >= 2) & ((salida[:, 3] - salida[:, 1]) >= 2)
    return salida.astype(np.float32), validas


def distorsionar(img, cajas, k, centro=None):
    """aplica la distorsion radial k a una imagen bgr y a sus cajas xyxy (px). devuelve imagen, cajas y validas"""
    h, w = img.shape[:2]
    cx, cy = centro if centro is not None else ((w - 1) / 2, (h - 1) / 2)
    mx, my = mapas_radiales(w, h, k, cx, cy)
    salida = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    cajas, validas = distorsionar_cajas(cajas, k, w, h, (cx, cy))
    return salida, cajas, validas


# ---------------------------------------------------------------- integracion con ultralytics

class DistorsionRadial:
    """augmentacion sobre el dict de etiquetas de ultralytics, antes de armar las Instances.

    En ese punto label['img'] es la imagen ya redimensionada (lado largo = imgsz) y label['bboxes'] son xywh
    normalizadas. Con probabilidad p sortea k ~ U[k_min, k_max] y distorsiona alrededor del centro optico de la
    imagen (de calibration.json, escalado al tamano redimensionado); si no lo encuentra usa el centro geometrico.
    """

    def __init__(self, p=0.5, k_min=-0.2, k_max=0.3, calib=None):
        assert k_min > K_LIMITE, f"k_min tiene que ser mayor que {K_LIMITE:.3f}"
        self.p, self.k_min, self.k_max = p, k_min, k_max
        self.centros = {}
        if calib and Path(calib).is_file():
            datos = json.loads(Path(calib).read_text())["imagenes"]
            self.centros = {s: (d["cx"], d["cy"]) for s, d in datos.items()}

    def __call__(self, label):
        if random.random() >= self.p or label.get("bbox_format") != "xywh" or not label.get("normalized", False):
            return label
        k = random.uniform(self.k_min, self.k_max)
        img = label["img"]
        h, w = img.shape[:2]
        h0, w0 = label["ori_shape"]
        c = self.centros.get(Path(label["im_file"]).stem)
        centro = (c[0] * w / w0, c[1] * h / h0) if c else ((w - 1) / 2, (h - 1) / 2)

        b = np.asarray(label["bboxes"], dtype=np.float64).reshape(-1, 4)
        xyxy = np.stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
                         (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h], axis=1)
        label["img"], nuevas, ok = distorsionar(img, xyxy, k, centro)
        if len(b):
            nuevas = nuevas[ok]
            label["bboxes"] = np.stack([(nuevas[:, 0] + nuevas[:, 2]) / 2 / w, (nuevas[:, 1] + nuevas[:, 3]) / 2 / h,
                                        (nuevas[:, 2] - nuevas[:, 0]) / w, (nuevas[:, 3] - nuevas[:, 1]) / h],
                                       axis=1).astype(np.float32)
            label["cls"] = label["cls"][ok]
            segs = label.get("segments", [])
            if len(segs):  # las etiquetas del TP son solo cajas; esto cubre etiquetas con poligonos
                R = radio_max(w, h, *centro)
                movidos = []
                for s, valido in zip(segs, ok):
                    if valido:
                        xs, ys = puntos_adelante(s[:, 0] * w, s[:, 1] * h, k, *centro, R)
                        movidos.append(np.stack([xs.clip(0, w) / w, ys.clip(0, h) / h], 1).astype(np.float32))
                label["segments"] = movidos
        return label


class YOLODatasetRadial(YOLODataset):
    """YOLODataset que distorsiona cada imagen de entrenamiento antes del mosaico.

    update_labels_info es el ultimo paso de get_image_and_label, que llaman tanto __getitem__ como el mosaico para
    las otras tres imagenes: cada tile se distorsiona alrededor de su propio centro optico. Al estar en el dataset y
    no en la lista de transforms, sigue activa cuando ultralytics apaga el mosaico (close_mosaic). Solo actua con
    augment=True, o sea en train.
    """

    radial = None

    def update_labels_info(self, label):
        if self.augment and self.radial is not None:
            label = self.radial(label)
        return super().update_labels_info(label)


ENV_RADIAL = "RAMA3_RADIAL"  # config de la augmentacion para los procesos que lanza ultralytics con varias GPUs


class TrainerFisheye(DetectionTrainer):
    """DetectionTrainer que usa YOLODatasetRadial para train si hay configuracion radial.

    La configuracion viaja en la variable de entorno RAMA3_RADIAL: con varias GPUs ultralytics relanza el
    entrenamiento en procesos nuevos que vuelven a importar esta clase, y un atributo puesto en tiempo de ejecucion
    se perderia. Sin configuracion es exactamente el trainer de ultralytics.
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        ds = super().build_dataset(img_path, mode, batch)
        cfg = json.loads(os.environ.get(ENV_RADIAL, "null"))
        if mode == "train" and cfg:
            ds.__class__ = YOLODatasetRadial  # misma instancia (y mismo cache), solo cambia update_labels_info
            ds.radial = DistorsionRadial(**cfg)
        return ds


# ---------------------------------------------------------------- datos y entrenamiento

def preparar_yolo_ds(listas, etiquetas, imagenes, salida, splits=("train", "val", "test"), controlar=True):
    """arma la estructura que espera ultralytics sin copiar ni recomprimir imagenes.

        salida/images/<split>/<archivo>.png   enlace simbolico a la imagen original
        salida/labels/<split>/<archivo>.txt   copia de la etiqueta de convert_to_yolo.py
        salida/data.yaml

    Ultralytics busca la etiqueta de cada imagen reemplazando /images/ por /labels/ en su ruta. Con las imagenes en
    rgb_images/ y las etiquetas en outputs/yolo_ds/labels/ no encuentra ninguna y entrena todo como fondo.
    listas: carpeta con train.txt, val.txt y test.txt de convert_to_yolo.py (solo se usa el nombre de archivo).
    """
    listas, etiquetas, imagenes, salida = (Path(p) for p in (listas, etiquetas, imagenes, salida))
    clases = json.loads(CLASS_MAP.read_text(encoding="utf-8"))["clases"]
    conteo = {}
    for split in splits:
        nombres = [Path(l.strip()).name for l in (listas / f"{split}.txt").read_text().splitlines() if l.strip()]
        dir_img, dir_lbl = salida / "images" / split, salida / "labels" / split
        dir_img.mkdir(parents=True, exist_ok=True)
        dir_lbl.mkdir(parents=True, exist_ok=True)
        n_cajas = 0
        for nombre in nombres:
            origen = (imagenes / nombre).resolve()
            if not origen.is_file():
                raise FileNotFoundError(f"falta la imagen {origen}")
            enlace = dir_img / nombre
            if enlace.is_symlink() and Path(os.readlink(enlace)) != origen:
                enlace.unlink()  # enlace armado en otra maquina
            if not enlace.is_symlink():
                enlace.symlink_to(origen)
            et = etiquetas / f"{Path(nombre).stem}.txt"
            texto = et.read_text() if et.is_file() else ""  # sin archivo = imagen sin objetos (fondo)
            (dir_lbl / et.name).write_text(texto)
            n_cajas += sum(1 for l in texto.splitlines() if l.strip())
        conteo[split] = (len(nombres), n_cajas)

    partes = ["# Generado por fisheye_aug.preparar_yolo_ds", f"path: {salida.resolve()}"]
    partes += [f"{s}: images/{s}" for s in splits]
    partes += [f"nc: {len(clases)}", f"names: {clases}"]
    (salida / "data.yaml").write_text("\n".join(partes) + "\n")

    if controlar:
        for s, n in conteo.items():
            assert n == CONTROL[s], f"{s}: {n} (imagenes, cajas) y el equipo tiene {CONTROL[s]}"
    return salida / "data.yaml", conteo


def entrenar(variante, data, salida, epochs, modelo="yolo11m.pt", imgsz=640, batch=16, device=None, cache="ram",
             fraction=1.0, close_mosaic=10, workers=8, seed=0, radial=None, calib=CALIBRACION, **extra):
    """entrena una variante de la rama 3 y devuelve la ruta a su best.pt.

    'base'  : fine-tuning con el trainer de ultralytics sin cambios y sus augmentaciones por defecto (punto C).
    'radial': lo mismo + distorsion radial (AUG_RADIAL, o lo que se pase en radial).
    Todo lo demas es identico. best.pt es la epoca con mejor mAP@50-95 de val (el fitness de ultralytics), que es el
    criterio de checkpoint acordado por el equipo. device puede ser '0,1' para repartir una corrida en dos GPUs.
    """
    from ultralytics import YOLO

    if variante not in ("base", "radial"):
        raise ValueError(f"variante desconocida: {variante}")
    if variante == "radial":
        cfg = {**AUG_RADIAL, **(radial or {}), "calib": str(calib) if calib else None}
        os.environ[ENV_RADIAL] = json.dumps(cfg)
        trainer = TrainerFisheye
    else:
        os.environ.pop(ENV_RADIAL, None)
        trainer = None  # el de ultralytics
    YOLO(str(modelo)).train(
        trainer=trainer, data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=device, cache=cache,
        fraction=fraction, close_mosaic=close_mosaic, workers=workers, seed=seed, deterministic=True,
        project=str(Path(salida).resolve()), name=f"rama3_{variante}", exist_ok=True, **extra,
    )
    return Path(salida).resolve() / f"rama3_{variante}" / "weights" / "best.pt"
