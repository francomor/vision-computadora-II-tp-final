"""
demo/app.py - demo web de deteccion: se sube una imagen y devuelve la misma imagen con las cajas dibujadas.

Correr desde la raiz del repo (entorno de uv):
    uv add fastapi "uvicorn[standard]" python-multipart
    uv run python demo/app.py               # abrir http://localhost:8000

Prueba sin navegador:
    curl -F imagen=@foto.png -F modelo=base http://localhost:8000/detectar -o salida.jpg

Solo hace inferencia: la augmentacion radial es de entrenamiento, asi que base y radial se usan igual aca.
"""
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
AQUI = Path(__file__).resolve().parent

# modelos que se pueden elegir en la pagina (nombre -> pesos, descripcion); si cambian las rutas, tocar aca
MODELOS = {
    "coco": (REPO / "yolo11m.pt", "COCO sin adaptar (baseline)"),
    
    "base": (REPO / "outputs/rama3/mac/rama3_base/weights/best.pt", "fine-tuning fisheye"),
    "radial": (REPO / "outputs/rama3/colab/rama3_radial/weights/best.pt", "fine-tuning + distorsión radial"),
    "cilindrica": (REPO / "outputs/rama4/train/yolo11m/weights/best.pt", "fine-tuning + corrección cilindrica"),
    
}
IMGSZ = 640  # la resolucion con la que se entrenaron todas las ramas

# clases del equipo; las de COCO se llevan a estas para que los tres modelos se vean igual
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
    pesos, _ = MODELOS[nombre]
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


def dibujar(img, resultado, names):
    out = img.copy()
    grosor = max(2, round(sum(img.shape[:2]) / 1000))
    escala, trazo = grosor / 3, max(1, grosor // 2)
    conteo = {}
    cajas = resultado.boxes
    for (x1, y1, x2, y2), c, p in zip(cajas.xyxy.int().tolist(), cajas.cls.int().tolist(), cajas.conf.tolist()):
        clase = COCO_A_EQUIPO.get(names[c], names[c])
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


app = FastAPI(title="Detección fisheye - YOLO11")


@app.get("/")
def pagina():
    return FileResponse(AQUI / "index.html")


@app.get("/modelos")
def modelos():
    return {
        "device": DEVICE,
        "imgsz": IMGSZ,
        "modelos": [
            {"nombre": n, "descripcion": d, "disponible": ruta_pesos(n) is not None} for n, (_, d) in MODELOS.items()
        ],
    }


@app.post("/detectar")
def detectar(imagen: UploadFile = File(...), modelo: str = Form("base"), conf: float = Form(0.25)):
    if modelo not in MODELOS:
        raise HTTPException(400, f"modelo desconocido: {modelo}; opciones: {list(MODELOS)}")
    img = cv2.imdecode(np.frombuffer(imagen.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "el archivo no es una imagen que OpenCV pueda leer")

    with _lock:
        m = cargar(modelo)
        r = m.predict(img, imgsz=IMGSZ, conf=conf, device=DEVICE, classes=clases_a_usar(m), verbose=False)[0]

    salida, conteo = dibujar(img, r, m.names)
    ok, jpg = cv2.imencode(".jpg", salida, [cv2.IMWRITE_JPEG_QUALITY, 90])  # solo para mostrar
    v = r.speed  # ms: preprocess, inference, postprocess
    headers = {
        "X-Latencia-Ms": f"{v['inference']:.1f}",
        "X-Total-Ms": f"{v['preprocess'] + v['inference'] + v['postprocess']:.1f}",
        "X-Detecciones": json.dumps(conteo),
    }
    return Response(jpg.tobytes(), media_type="image/jpeg", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
