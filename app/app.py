"""
app/app.py - demo web de deteccion sobre imagenes fisheye, de a una imagen o con un video como camara en vivo.

Correr desde la raiz del repo (las dependencias ya estan en pyproject.toml):
    uv run python app/app.py               # abrir http://localhost:8000

Prueba sin navegador:
    curl -F imagen=@00002_FV.png -F modelo=base http://localhost:8000/detectar -o salida.jpg
    curl -F imagen=@00002_FV.png -F modelo=coco -F proyeccion=cilindrica http://localhost:8000/detectar -o salida.jpg

Preprocesamiento (campo proyeccion; si no se manda, el que corresponde al modelo):
- fisheye: la imagen va directo al modelo (ramas 1 y 3).
- cilindrica: el de la rama 2. La imagen pasa a la proyeccion cilindrica con la calibracion de WoodScape (banda
  de +-45 grados alrededor del horizonte), se detecta ahi y las cajas vuelven a la fisheye. Usa las mismas
  funciones de fisheye_utils.py que las ramas 2 y 4 (mapas, clave_calib, caja_a_fisheye), asi que necesita la
  carpeta CV2/ del Drive del equipo. La calibracion sale del nombre del archivo (00002_FV.png usa
  CV2/calibration/00002_FV.json); si el nombre no la trae, se usa la primera de la camara elegida.
  Con vista=cilindrica devuelve la imagen proyectada con las cajas sin reproyectar.

Video en vivo (GET /en_vivo): reproduce un video de outputs/videos/ como si fuera la camara del auto. El frame que
toca lo decide el reloj de la camara: si el pipeline tarda mas que un frame, los que pasaron se descartan. Devuelve
un stream MJPEG con las detecciones y los fps, la latencia y los descartes dibujados arriba. Los videos del test
se arman con tools/hacer_video.py, que deja al lado un json con la imagen de cada frame para usar su calibracion.

coco + fisheye es la rama 1, coco + cilindrica la rama 2, base y radial la rama 3 y cilindrica la rama 4.
La augmentacion radial es de entrenamiento, asi que base y radial se usan igual aca.
"""
import json
import re
import shutil
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
AQUI = Path(__file__).resolve().parent
VIDEOS = REPO / "outputs/videos"
sys.path.insert(0, str(REPO))  # para importar fisheye_utils

# modelos que se pueden elegir en la pagina: nombre -> (pesos, descripcion, {proyeccion: rama del 2x2})
# la primera proyeccion es la que se usa por defecto con ese modelo; si cambian las rutas, tocar aca
MODELOS = {
    "coco": (REPO / "yolo11m.pt", "COCO sin adaptar", {"fisheye": "rama 1", "cilindrica": "rama 2"}),
    "base": (REPO / "outputs/rama3/mac/rama3_base/weights/best.pt", "fine-tuning fisheye", {"fisheye": "rama 3"}),
    "radial": (REPO / "outputs/rama3/colab/rama3_radial/weights/best.pt", "fine-tuning + distorsión radial",
               {"fisheye": "rama 3, distorsión radial"}),
    "cilindrica": (REPO / "outputs/rama4/train/yolo11m/weights/best.pt", "fine-tuning + corrección cilíndrica",
                   {"cilindrica": "rama 4"}),
}
PROYECCIONES = ("fisheye", "cilindrica")
CAMARAS = ("FV", "RV", "MVL", "MVR")
EXT_VIDEO = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
IMGSZ = 640  # la resolucion con la que se entrenaron todas las ramas

# clases del equipo; las de COCO se llevan a estas para que todos los modelos se vean igual
COLORES = {"vehicle": (255, 140, 0), "person": (60, 60, 230), "two_wheeler": (40, 190, 40)}  # BGR
COCO_A_EQUIPO = {
    "car": "vehicle", "bus": "vehicle", "truck": "vehicle",
    "person": "person",
    "bicycle": "two_wheeler", "motorcycle": "two_wheeler",
}


def elegir_device():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():  # la Mac
        return "mps"
    return "cpu"


DEVICE = elegir_device()
_cargados = {}
_lock = threading.Lock()  # un modelo de ultralytics no se debe usar desde dos hilos a la vez


def ruta_pesos(nombre):
    pesos = MODELOS[nombre][0]
    if pesos.is_file():
        return str(pesos)
    if nombre == "coco":
        return "yolo11m.pt"  # ultralytics lo baja la primera vez
    return None


def cargar(nombre):
    if nombre not in _cargados:
        pesos = ruta_pesos(nombre)
        if pesos is None:
            raise HTTPException(404, f"no encuentro los pesos de '{nombre}' en {MODELOS[nombre][0]}")
        modelo = YOLO(pesos)
        # la primera inferencia es mucho mas lenta; se hace aca para que no ensucie la latencia que se muestra
        modelo.predict(np.zeros((IMGSZ, IMGSZ, 3), np.uint8), imgsz=IMGSZ, device=DEVICE, verbose=False)
        _cargados[nombre] = modelo
    return _cargados[nombre]


def clases_a_usar(modelo):
    """None si el modelo ya tiene las clases del equipo; si es el de COCO, solo los ids que se llevan a ellas."""
    if "car" not in modelo.names.values():
        return None
    return [i for i, n in modelo.names.items() if n in COCO_A_EQUIPO]


def validar(modelo, proyeccion):
    """proyeccion a usar (la del modelo si viene vacia), o HTTPException si algo no existe"""
    if modelo not in MODELOS:
        raise HTTPException(400, f"modelo desconocido: {modelo}; opciones: {list(MODELOS)}")
    proyeccion = proyeccion or next(iter(MODELOS[modelo][2]))
    if proyeccion not in PROYECCIONES:
        raise HTTPException(400, f"proyección desconocida: {proyeccion}; opciones: {list(PROYECCIONES)}")
    return proyeccion


# ---------------------------------------------------------------- preprocesamiento de la rama 2

def hay_cilindrica():
    """la proyeccion necesita fisheye_utils.py y la carpeta CV2/ del Drive (calibraciones y class_map)"""
    return (REPO / "fisheye_utils.py").is_file() and (REPO / "CV2/calibration").is_dir() \
        and (REPO / "CV2/Scripts/class_map.json").is_file()


def fisheye_utils():
    """el modulo de la rama 2; se importa recien cuando se pide la cilindrica, asi la app anda sin CV2/"""
    if not hay_cilindrica():
        raise HTTPException(503, "la proyección cilíndrica necesita fisheye_utils.py y la carpeta CV2/ del Drive "
                                 "del equipo (CV2/calibration y CV2/Scripts) en la raíz del repo")
    import fisheye_utils as fu
    return fu


def calibracion(archivo, camara, tam=None):
    """(clave de calibracion, archivo usado): la de la imagen si el nombre la trae, si no la primera de su camara.

    Es la misma regla que clave_calib de fisheye_utils (la que usaron las ramas 2 y 4 para el test). Con tam
    (alto, ancho) verifica que la imagen sea del tamano de la calibracion.
    """
    fu = fisheye_utils()
    m = re.search(r"(\d+)_(FV|RV|MVL|MVR)", Path(archivo or "").stem, re.I)
    stem, cam_nombre = (f"{m.group(1)}_{m.group(2).upper()}", m.group(2).upper()) if m else (None, None)
    if camara == "auto":
        if cam_nombre is None:
            raise HTTPException(400, "no puedo sacar la cámara del nombre del archivo (se espera algo como "
                                     "00002_FV.png); elegila en la página o mandá camara=FV|RV|MVL|MVR")
        camara = cam_nombre
    if camara not in CAMARAS:
        raise HTTPException(400, f"cámara desconocida: {camara}; opciones: auto, {', '.join(CAMARAS)}")
    if stem is None or cam_nombre != camara or not (fu.CALIB_DIR / f"{stem}.json").is_file():
        stem = None  # el nombre no sirve: la primera calibracion de la camara
    usado = f"{stem}.json" if stem else next(iter(sorted(fu.CALIB_DIR.glob(f"*_{camara}.json"))), Path("-")).name
    try:
        clave = fu.clave_calib(stem or "", camara)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    intr = json.loads(clave)["intrinsic"]
    W, H = int(intr["width"]), int(intr["height"])
    if tam is not None and tuple(tam) != (H, W):
        raise HTTPException(400, f"la imagen es de {tam[1]}x{tam[0]} y la calibración de {W}x{H}: la proyección "
                                 "cilíndrica solo sirve con imágenes de WoodScape sin redimensionar")
    return clave, usado


@lru_cache(maxsize=None)
def zona_cilindrica(clave):
    """mascara (H, W) de la fisheye que entra en la banda cilindrica; misma cuenta que cubiertas() de la rama 2"""
    fu = fisheye_utils()
    mx, my = fu.mapas(clave)
    intr = json.loads(clave)["intrinsic"]
    W, H = int(intr["width"]), int(intr["height"])
    zona = np.zeros((H, W), np.uint8)
    ok = (mx >= 0) & (mx < W - 1) & (my >= 0) & (my < H - 1)
    zona[np.round(my[ok]).astype(int), np.round(mx[ok]).astype(int)] = 1
    return cv2.morphologyEx(zona, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))  # tapo los huecos del muestreo


def preparar(clave):
    """calcula y cachea los mapas y la zona de una calibracion (la primera vez tarda ~1 s)"""
    with _lock:
        fisheye_utils().mapas(clave)
        zona_cilindrica(clave)


def marcar_zona(img, zona):
    """oscurece lo que la cilindrica no ve y marca el borde de la banda"""
    out = img.copy()
    out[zona == 0] = (out[zona == 0] * 0.45).astype(np.uint8)
    contornos, _ = cv2.findContours(zona, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contornos, -1, (230, 230, 230), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- deteccion y dibujo

def dibujar(img, cajas, clases, confs):
    """cajas xyxy con su clase del equipo y su confianza; devuelve la imagen y el conteo por clase"""
    out = img.copy()
    grosor = max(2, round(sum(img.shape[:2]) / 1000))
    escala, trazo = grosor / 3, max(1, grosor // 2)
    conteo = {}
    for (x1, y1, x2, y2), clase, p in zip(np.asarray(cajas).round().astype(int).tolist(), clases, confs):
        conteo[clase] = conteo.get(clase, 0) + 1
        color = COLORES.get(clase, (160, 160, 160))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, grosor)

        texto = f"{clase} {p:.2f}"
        (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, trazo)
        alto = th + base + 4
        y0 = y1 - alto if y1 - alto >= 0 else y1  # la etiqueta va arriba de la caja, o adentro si no entra
        cv2.rectangle(out, (x1, y0), (x1 + tw + 4, y0 + alto), color, -1)
        cv2.putText(out, texto, (x1 + 2, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), trazo,
                    cv2.LINE_AA)
    return out, conteo


def detectar_en(img, modelo, conf, proyeccion, clave=None, vista="fisheye"):
    """pipeline completo sobre una imagen bgr: (imagen dibujada, conteo por clase, tiempos en ms)"""
    cil = proyeccion == "cilindrica"
    t = {"remap": 0.0, "reproyeccion": 0.0}
    with _lock:
        m = cargar(modelo)
        if cil:
            fu = fisheye_utils()
            mx, my = fu.mapas(clave)  # cacheados por calibracion: la primera vez tarda ~1 s y no entra en la latencia
            t0 = time.perf_counter()
            entrada = cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR)  # igual que cilindrica() de la rama 2
            t["remap"] = 1e3 * (time.perf_counter() - t0)
        else:
            entrada = img
        r = m.predict(entrada, imgsz=IMGSZ, conf=conf, device=DEVICE, classes=clases_a_usar(m), verbose=False)[0]

    cajas = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clases = np.array([COCO_A_EQUIPO.get(m.names[c], m.names[c]) for c in r.boxes.cls.int().tolist()])
    if cil and vista == "cilindrica":
        salida, conteo = dibujar(entrada, cajas, clases, confs)
    elif cil:
        intr = json.loads(clave)["intrinsic"]
        t0 = time.perf_counter()
        # las cajas vuelven a la fisheye, como en las ramas 2 y 4
        fish, ok = fu.caja_a_fisheye(cajas, mx, my, int(intr["width"]), int(intr["height"]))
        t["reproyeccion"] = 1e3 * (time.perf_counter() - t0)
        salida, conteo = dibujar(marcar_zona(img, zona_cilindrica(clave)), fish[ok], clases[ok], confs[ok])
    else:
        salida, conteo = dibujar(img, cajas, clases, confs)
    v = r.speed  # ms: preprocess, inference, postprocess (los de ultralytics)
    t["inferencia"] = v["inference"]
    t["total"] = t["remap"] + v["preprocess"] + v["inference"] + v["postprocess"] + t["reproyeccion"]
    return salida, conteo, t


# ---------------------------------------------------------------- video en vivo

class CamaraSimulada:
    """un video que se reproduce al ritmo de una camara: el frame que toca lo decide el reloj, no el detector.

    Si el pipeline tarda mas que un frame, los frames que pasaron mientras tanto se descartan, como pasaria con la
    camara del auto. Al llegar al final el video vuelve a empezar.
    """

    def __init__(self, ruta, fps=0):
        self.cap = cv2.VideoCapture(str(ruta))
        if not self.cap.isOpened():
            raise HTTPException(400, f"opencv no puede abrir {ruta.name}")
        self.fps = fps or self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.tam = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.i = -1  # indice del ultimo frame leido dentro del archivo
        self.reloj = -1  # frames de la camara desde que arranco (no vuelve a cero)
        self.t0 = None

    def _avanzar(self, leer):
        if not self.cap.grab():  # fin del video: vuelvo al principio
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.i = -1
            if not self.cap.grab():
                raise HTTPException(400, "no pude leer frames del video")
        self.i += 1
        return self.cap.retrieve()[1] if leer else None

    def siguiente(self):
        """(frame, indice en el archivo, frames descartados desde el anterior)"""
        if self.t0 is None:
            self.t0 = time.perf_counter()
        toca = int((time.perf_counter() - self.t0) * self.fps)
        if toca <= self.reloj:  # el pipeline va mas rapido que la camara: espero el proximo frame
            time.sleep(max(0.0, (self.reloj + 1) / self.fps - (time.perf_counter() - self.t0)))
            toca = self.reloj + 1
        descartados = toca - self.reloj - 1
        for _ in range(descartados):
            self._avanzar(leer=False)
        frame = self._avanzar(leer=True)
        self.reloj = toca
        return frame, self.i, descartados

    def cerrar(self):
        self.cap.release()


class Medidor:
    """fps procesados, latencias (promedio movil) y frames descartados de una transmision"""

    def __init__(self, fps_camara):
        self.fps_camara = fps_camara
        self.procesados = self.descartados = 0
        self.fps = self.inferencia = self.ciclo = None
        self.t_ultimo = None

    def _ema(self, previo, nuevo, a=0.1):
        return nuevo if previo is None else (1 - a) * previo + a * nuevo

    def sumar(self, descartados, inferencia_ms, ciclo_ms):
        ahora = time.perf_counter()
        if self.t_ultimo is not None:
            self.fps = self._ema(self.fps, 1 / max(ahora - self.t_ultimo, 1e-6))
        self.t_ultimo = ahora
        self.procesados += 1
        self.descartados += descartados
        self.inferencia = self._ema(self.inferencia, inferencia_ms)
        self.ciclo = self._ema(self.ciclo, ciclo_ms)

    def resumen(self):
        total = self.procesados + self.descartados
        return {"fps_camara": self.fps_camara, "fps_procesados": self.fps, "inferencia_ms": self.inferencia,
                "ciclo_ms": self.ciclo, "procesados": self.procesados, "descartados": self.descartados,
                "descartados_pct": 100 * self.descartados / total if total else 0.0}


def hud(img, lineas):
    """texto blanco sobre una franja oscura arriba a la izquierda (opencv no dibuja tildes)"""
    escala = max(0.5, img.shape[1] / 1600)
    trazo = max(1, round(escala * 2))
    alto = int(32 * escala)
    ancho = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, escala, trazo)[0][0] for l in lineas) + 20
    franja = img[: alto * len(lineas) + 12, :ancho]
    franja[:] = (franja * 0.35).astype(np.uint8)
    for k, l in enumerate(lineas):
        cv2.putText(img, l, (10, alto * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), trazo, cv2.LINE_AA)
    return img


def cuadro_mensaje(texto, tam=(540, 960)):
    """un frame negro con un mensaje, para avisos y errores dentro del stream"""
    img = np.zeros((*tam, 3), np.uint8)
    palabras, lineas, actual = texto.split(), [], ""
    for p in palabras:
        if len(actual) + len(p) > 60:
            lineas.append(actual)
            actual = ""
        actual += (" " if actual else "") + p
    lineas.append(actual)
    for k, l in enumerate(lineas):
        cv2.putText(img, l, (30, 60 + 36 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
    return img


def parte_mjpeg(img, calidad=80):
    _, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, calidad])
    return b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" \
        + jpg.tobytes() + b"\r\n"


def sin_tildes(texto):
    return texto.translate(str.maketrans("áéíóúñÁÉÍÓÚÑ", "aeiounAEIOUN"))


def ruta_video(nombre):
    ruta = (VIDEOS / nombre).resolve()
    if ruta.parent != VIDEOS.resolve() or not ruta.is_file():
        raise HTTPException(404, f"no encuentro el video {nombre} en {VIDEOS}")
    return ruta


def meta_video(ruta):
    """el json que deja tools/hacer_video.py al lado del mp4 (stems de cada imagen), o None"""
    p = ruta.with_suffix(".json")
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


_vivo = {"id": 0, "estado": {}}  # una transmision por vez: la nueva corta la anterior


async def _transmitir(request, mi_id, video, modelo, conf, proyeccion, camara, vista, fps):
    cam = None
    try:
        try:
            proyeccion = validar(modelo, proyeccion)
            ruta = ruta_video(video)
            meta = meta_video(ruta)
            cam = CamaraSimulada(ruta, fps)
            stems, repetir = (meta["stems"], meta.get("repetir", 1)) if meta else (None, 1)
            claves = {}
            if proyeccion == "cilindrica":  # una calibracion por imagen si hay json, si no la de la camara elegida
                for s in stems or [None]:
                    archivo = f"{s}.png" if s else video
                    claves[s] = calibracion(archivo, camara, cam.tam)[0]
                distintas = sorted(set(claves.values()))
                for k, c in enumerate(distintas):
                    yield parte_mjpeg(cuadro_mensaje(f"Preparando la proyeccion cilindrica: calibracion {k + 1} de "
                                                     f"{len(distintas)} (solo la primera vez)"))
                    await run_in_threadpool(preparar, c)
            if modelo not in _cargados:
                yield parte_mjpeg(cuadro_mensaje(f"Cargando el modelo {modelo}..."))
                await run_in_threadpool(cargar, modelo)
        except HTTPException as e:
            yield parte_mjpeg(cuadro_mensaje(sin_tildes(f"No se puede transmitir: {e.detail}")))
            return

        rama = MODELOS[modelo][2].get(proyeccion, "fuera del 2x2")
        med = Medidor(cam.fps)

        def paso():
            frame, i, descartados = cam.siguiente()
            t0 = time.perf_counter()
            stem = stems[min(i // repetir, len(stems) - 1)] if stems else None
            clave = claves.get(stem, claves.get(None)) if claves else None
            salida, conteo, t = detectar_en(frame, modelo, conf, proyeccion, clave, vista)
            med.sumar(descartados, t["inferencia"], 1e3 * (time.perf_counter() - t0))
            e = med.resumen()
            hud(salida, [
                sin_tildes(f"{modelo} | {rama} | {proyeccion} | camara {cam.fps:g} fps"),
                f"procesa {e['fps_procesados'] or 0:.1f} fps | inferencia {e['inferencia_ms']:.0f} ms | "
                f"ciclo {e['ciclo_ms']:.0f} ms | descartados {e['descartados_pct']:.0f}%",
                (stem or video) + "  " + "  ".join(f"{k}: {v}" for k, v in conteo.items()),
            ])
            _vivo["estado"] = {"video": video, "frame": i, "imagen": stem, "modelo": modelo, "proyeccion": proyeccion,
                               "rama": rama, "detecciones": conteo, **e}
            return parte_mjpeg(salida)

        while _vivo["id"] == mi_id and not await request.is_disconnected():
            yield await run_in_threadpool(paso)
    finally:
        if cam is not None:
            cam.cerrar()


# ---------------------------------------------------------------- api

app = FastAPI(title="Detección fisheye - YOLO11")


@app.get("/")
def pagina():
    return FileResponse(AQUI / "index.html")


@app.get("/modelos")
def modelos():
    return {
        "device": DEVICE,
        "imgsz": IMGSZ,
        "cilindrica_disponible": hay_cilindrica(),
        "camaras": list(CAMARAS),
        "modelos": [
            {"nombre": n, "descripcion": d, "disponible": ruta_pesos(n) is not None, "ramas": ramas,
             "proyeccion": next(iter(ramas))}
            for n, (_, d, ramas) in MODELOS.items()
        ],
    }


@app.post("/detectar")
def detectar(imagen: UploadFile = File(...), modelo: str = Form("base"), conf: float = Form(0.25),
             proyeccion: str = Form(""), camara: str = Form("auto"), vista: str = Form("fisheye")):
    proyeccion = validar(modelo, proyeccion)
    img = cv2.imdecode(np.frombuffer(imagen.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "el archivo no es una imagen que OpenCV pueda leer")

    headers = {"X-Proyeccion": proyeccion}
    clave = None
    if proyeccion == "cilindrica":
        clave, headers["X-Calibracion"] = calibracion(imagen.filename, camara, img.shape[:2])
    salida, conteo, t = detectar_en(img, modelo, conf, proyeccion, clave, vista)

    _, jpg = cv2.imencode(".jpg", salida, [cv2.IMWRITE_JPEG_QUALITY, 90])  # solo para mostrar
    headers.update({
        "X-Latencia-Ms": f"{t['inferencia']:.1f}",
        "X-Remap-Ms": f"{t['remap']:.1f}",
        "X-Reproyeccion-Ms": f"{t['reproyeccion']:.1f}",
        "X-Total-Ms": f"{t['total']:.1f}",
        "X-Detecciones": json.dumps(conteo),
    })
    return Response(jpg.tobytes(), media_type="image/jpeg", headers=headers)


@app.get("/videos")
def videos():
    salida = []
    for p in sorted(VIDEOS.glob("*")) if VIDEOS.is_dir() else []:
        if p.suffix.lower() not in EXT_VIDEO:
            continue
        cap = cv2.VideoCapture(str(p))
        meta = meta_video(p)
        salida.append({"nombre": p.name, "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                       "fps": cap.get(cv2.CAP_PROP_FPS), "ancho": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       "alto": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "camara": meta["camara"] if meta else None,
                       "imagenes": len(meta["stems"]) if meta else None})
        cap.release()
    return {"carpeta": str(VIDEOS), "videos": salida}


@app.post("/videos")
def subir_video(video: UploadFile = File(...)):
    nombre = re.sub(r"[^\w.-]", "_", Path(video.filename or "video.mp4").name)
    if Path(nombre).suffix.lower() not in EXT_VIDEO:
        raise HTTPException(400, f"formato no soportado; opciones: {', '.join(EXT_VIDEO)}")
    VIDEOS.mkdir(parents=True, exist_ok=True)
    destino, k = VIDEOS / nombre, 1
    while destino.exists():  # no piso los videos que ya estan
        destino = VIDEOS / f"{Path(nombre).stem}_{k}{Path(nombre).suffix}"
        k += 1
    with destino.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    return {"nombre": destino.name}


@app.get("/en_vivo")
async def en_vivo(request: Request, video: str, modelo: str = "base", conf: float = 0.25, proyeccion: str = "",
                  camara: str = "auto", vista: str = "fisheye", fps: float = 0):
    _vivo["id"] += 1
    gen = _transmitir(request, _vivo["id"], video, modelo, conf, proyeccion, camara, vista, fps)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@app.post("/en_vivo/detener")
def detener():
    _vivo["id"] += 1
    return {"ok": True}


@app.get("/en_vivo/estado")
def estado():
    return _vivo["estado"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
