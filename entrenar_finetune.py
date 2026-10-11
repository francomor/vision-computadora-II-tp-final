"""
entrenar_finetune.py - corre la rama 4 de punta a punta, fuera del notebook.

Para entrenamientos largos conviene esto y no el notebook: deja un log en disco, se puede cortar y
retomar donde quedo, sigue con el proximo modelo si uno falla, y en Windows los DataLoader con
workers andan mejor desde un script que desde Jupyter.

    uv run python entrenar_finetune.py --prueba          # rama 4: chequeo corto del flujo + estimacion de horas
    uv run python entrenar_finetune.py                   # rama 4: dataset, fine-tuning de los 8 y test
    uv run python entrenar_finetune.py --modelos yolo11m fasterrcnn_r50_v2
    uv run python entrenar_finetune.py --solo-dataset    # solo arma el dataset
    uv run python entrenar_finetune.py --sin-test        # entrena y no toca el test
    uv run python entrenar_finetune.py --rama 3          # alternativa sobre la fisheye, NO usada (ver abajo)

Los modelos van de menor a mayor costo (MODELOS_FINETUNE) y cada uno se evalua sobre el test apenas
termina de entrenar, asi que si la corrida se corta, todo lo terminado ya tiene sus resultados.

Rama 4 escribe en CV2/yolo_ds_cilindrica y outputs/rama4; --rama 3 en CV2/yolo_ds_fisheye y outputs/rama3.
La rama 3 de los resultados se entreno con tools/train_rama3.py (fisheye_aug.py) y guarda sus pesos y
metricas en outputs/rama3: no correr --rama 3 en esa misma carpeta.
El notebook notebooks/rama4_cilindrica_finetune.ipynb no entrena: levanta lo que deja este script.
"""
import argparse
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

YOLO_DS = RAIZ / "CV2/yolo_ds"  # train.txt, val.txt y test.txt de convert_to_yolo.py
EVAL_RADIAL = RAIZ / "CV2/Scripts/eval_radial.py"
CALIBRACION = RAIZ / "CV2/calibration.json"

# modo prueba: pocas imagenes y dos epocas, la segunda sirve para medir sin el calentamiento
PRUEBA = dict(train=96, val=32, test=24, epocas=2)
PROYECCION = {3: "fisheye", 4: "cilindrica"}


def carpetas(rama, prueba):
    sufijo = "_prueba" if prueba else ""
    return RAIZ / f"CV2/yolo_ds_{PROYECCION[rama]}{sufijo}", RAIZ / f"outputs/rama{rama}{sufijo}"


class Log:
    """imprime y deja todo en un archivo, para revisar a la manana lo que paso a la noche"""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg=""):
        linea = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
        print(linea, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(linea + "\n")


def barra(log, etiqueta, cada=500):
    def progreso(hechas, total):
        if hechas % cada == 0 or hechas == total:
            log(f"   {etiqueta}: {hechas}/{total}")
    return progreso


def main():
    ap = argparse.ArgumentParser(description="Fine-tuning de los 8 modelos: rama 4 (cilindrica) o rama 3 (fisheye)")
    ap.add_argument("--rama", type=int, choices=(3, 4), default=4,
                    help="4 (por defecto). 3 es una alternativa no usada: la rama 3 oficial es tools/train_rama3.py")
    ap.add_argument("--prueba", action="store_true", help="corrida corta para verificar el flujo y estimar tiempos")
    ap.add_argument("--modelos", nargs="+", default=None, help="por defecto MODELOS_FINETUNE (los 8, de menor a mayor costo)")
    ap.add_argument("--solo-dataset", action="store_true")
    ap.add_argument("--sin-test", action="store_true", help="no predice ni evalua sobre el test")
    ap.add_argument("--epocas-yolo", type=int, default=None, help="pisa HP_YOLO['epochs'] (avisar a la rama 3)")
    ap.add_argument("--epocas-tv", type=int, default=None, help="pisa HP_TV['epochs'] (avisar a la rama 3)")
    args = ap.parse_args()

    import torch
    import finetune_utils as r4
    from fisheye_utils import DEVICE, cargar_test, tabla_resultados

    ds_dir, salidas = carpetas(args.rama, args.prueba)
    proyeccion = PROYECCION[args.rama]
    log = Log(salidas / "log.txt")
    modelos = args.modelos or r4.MODELOS_FINETUNE
    desconocidos = [m for m in modelos if m not in r4.MODELOS_RAMA4]
    if desconocidos:
        sys.exit(f"modelos desconocidos: {desconocidos}. Opciones: {r4.MODELOS_RAMA4}")

    log("=" * 70)
    log(f"rama {args.rama} ({proyeccion}) {'(PRUEBA) ' if args.prueba else ''}| modelos: {modelos}")
    gpu = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB)" \
        if torch.cuda.is_available() else "sin CUDA"
    log(f"device: {DEVICE} | {gpu} | torch {torch.__version__}")
    if DEVICE == "cpu" and not args.prueba:
        log("[aviso] sin GPU el fine-tuning lleva dias. Revisar la instalacion de torch con CUDA.")

    # ------------------------------------------------------------ 1. dataset
    t0 = time.perf_counter()
    resumen = []
    for split in ("train", "val"):
        lista = YOLO_DS / f"{split}.txt"
        if not lista.is_file():
            sys.exit(f"falta {lista}: correr antes convert_to_yolo.py (07_convert_to_yolo.bat)")
        rutas = r4.leer_lista(lista)
        if args.prueba:
            rutas = r4.subconjunto(rutas, PRUEBA[split])
        faltan = [r for r in rutas if not r.is_file()]
        if faltan:
            sys.exit(f"{len(faltan)} imagenes de {split} no existen (ej: {faltan[0]}). Revisar el dataset de Kaggle.")
        log(f"dataset {split}: {len(rutas)} imagenes -> {ds_dir}")
        df = r4.construir_split(rutas, split, ds_dir, proyeccion=proyeccion, progreso=barra(log, split))
        nuevas = df[~df["reanudada"]]
        resumen.append({"split": split, "imagenes": len(df), "cajas": int(df["cajas"].sum()),
                        "fuera de banda": int(nuevas["perdidos"].sum()) if len(nuevas) else None})
    r4.escribir_yaml(ds_dir)
    log(f"dataset listo en {(time.perf_counter() - t0) / 60:.1f} min\n{pd.DataFrame(resumen).to_string(index=False)}")
    if args.solo_dataset:
        return

    # ------------------------------------------------------------ 2. fine-tuning y test, modelo por modelo
    hp_yolo, hp_tv = {}, {}
    if args.epocas_yolo:
        hp_yolo["epochs"] = args.epocas_yolo
    if args.epocas_tv:
        hp_tv["epochs"] = args.epocas_tv
    if args.prueba:
        hp_yolo.update(epochs=PRUEBA["epocas"], patience=100)
        hp_tv.update(epochs=PRUEBA["epocas"], patience=100, warmup_iters=10)
    if (args.epocas_yolo or args.epocas_tv) and not args.prueba:
        log(f"[aviso] epocas cambiadas ({hp_yolo or ''} {hp_tv or ''}): la rama 3 tiene que usar las mismas")

    test = None
    if not args.sin_test:
        test = cargar_test(PRUEBA["test"] if args.prueba else None)
    usar_theta = (not args.prueba and EVAL_RADIAL.is_file() and CALIBRACION.is_file()
                  and (YOLO_DS / "rings.json").is_file())

    historias, fallidos = {}, []
    for nombre in modelos:
        t = time.perf_counter()
        try:
            if nombre in r4.YOLOS:
                _, hist = r4.entrenar(nombre, ds_dir, salidas, log=log, hp=hp_yolo, plots=not args.prueba)
            else:
                _, hist = r4.entrenar(nombre, ds_dir, salidas, log=log, hp=hp_tv)
            historias[nombre] = hist
            log(f"[{nombre}] entrenamiento en {(time.perf_counter() - t) / 60:.1f} min")
        except Exception:  # un modelo que falla (p.ej. sin memoria) no frena a los demas
            fallidos.append(nombre)
            log(f"[{nombre}] FALLO el entrenamiento, sigo con el proximo:\n{traceback.format_exc()}")
            continue
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if test is None:
            continue
        # test apenas termina: si la corrida se corta mas adelante, este modelo ya queda completo
        try:
            detectar = r4.detector_ft(nombre, r4.ruta_best(nombre, salidas))
            preds = r4.predecir_test(nombre, detectar, test, salidas, proyeccion=proyeccion,
                                     progreso=barra(log, f"test {nombre}", 250))
            n = r4.a_pred_json(preds, salidas / "pred_json" / f"{nombre}.json")
            log(f"[{nombre}] test listo, {n} detecciones")
            if usar_theta:
                r = subprocess.run([sys.executable, str(EVAL_RADIAL), "--yolo-dir", str(YOLO_DS), "--split", "test",
                                    "--pred-json", str(salidas / "pred_json" / f"{nombre}.json"),
                                    "--out", str(salidas / "eval_radial" / nombre), "--calibration", str(CALIBRACION),
                                    "--rings-file", str(YOLO_DS / "rings.json"),
                                    "--etiqueta", f"rama{args.rama}_{nombre}", "--cruce-tamano"],
                                   capture_output=True, text=True)
                log(f"[{nombre}] eval_radial {'ok' if r.returncode == 0 else 'FALLO: ' + r.stderr[-500:]}")
        except Exception:
            fallidos.append(nombre)
            log(f"[{nombre}] FALLO en el test:\n{traceback.format_exc()}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.prueba and historias:
        filas = []
        for nombre, hist in historias.items():
            epocas = (args.epocas_yolo or r4.HP_YOLO["epochs"]) if nombre in r4.YOLOS \
                else (args.epocas_tv or r4.HP_TV["epochs"])
            filas.append({"modelo": nombre, "epocas": epocas,
                          "horas estimadas": r4.estimar_horas(hist, PRUEBA["train"], PRUEBA["val"], epocas)})
        est = pd.DataFrame(filas)
        est.to_csv(salidas / "estimacion_horas.csv", index=False)
        log("estimacion para la corrida completa (cota gruesa, con early stopping puede ser menos):\n"
            f"{est.round(2).to_string(index=False)}\n   total: {est['horas estimadas'].sum():.1f} h")

    # ------------------------------------------------------------ 3. tabla con todo lo que haya en disco
    if test is not None:
        preds = {m: pd.read_pickle(salidas / f"preds_{m}_{len(test)}.pkl") for m in r4.MODELOS_FINETUNE
                 if (salidas / f"preds_{m}_{len(test)}.pkl").exists()}
        if preds:
            res = tabla_resultados(preds, test)
            res.to_csv(salidas / "resultados.csv")
            log(f"resultados (metrica de fisheye_utils, la misma de la rama 2):\n{res.round(3).to_string()}")
            if args.prueba:
                log("ojo: con dos epocas y pocas imagenes estas metricas no significan nada, la prueba solo verifica el flujo")
    if not usar_theta and not args.prueba:
        log("eval_radial se omitio: falta eval_radial.py, calibration.json o rings.json en CV2/")

    if fallidos:
        log(f"modelos con errores: {sorted(set(fallidos))} (ver el detalle arriba en este log)")
    log("listo")


if __name__ == "__main__":  # obligatorio en Windows para los DataLoader con workers
    main()
