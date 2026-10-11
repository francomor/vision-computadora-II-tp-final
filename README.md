# Detección de objetos en imágenes fisheye automotrices

Trabajo práctico final de **Visión por Computadora II** (CEIA, FIUBA), grupo 1.

Integrantes:

- Franco Marcelo Morero
- Nicolás Granato
- Marcos Levi Riveros Koloszwa
- Tadeo Riveros Koloszwa

El paper con el análisis completo está en `paper_fisheye_ieee.pdf`.

---

## 1. Objetivo

Las cámaras de visión envolvente (*surround view*) de los vehículos usan lentes
*fisheye* de casi 190°. Esas lentes deforman los objetos según su ángulo respecto del
eje óptico. Un peatón a 20° y el mismo peatón a 85° se ven muy distintos en la imagen.

Los detectores disponibles están preentrenados en imágenes sin esa deformación. El
trabajo mide cuánto pierde un detector preentrenado hacia la periferia de la imagen
y qué estrategia recupera mejor ese desempeño.

## 2. Diseño

Se cruzan dos estrategias en un diseño 2×2. Las cuatro ramas usan YOLO11m, los
mismos datos, las mismas clases y la misma cantidad de épocas.

| | sin *fine-tuning* | con *fine-tuning* |
|---|---|---|
| **sin rectificar** | Rama 1: detector preentrenado | Rama 3: *fine-tuning* |
| **rectificado** | Rama 2: proyección cilíndrica | Rama 4: ambas |

![diagrama](vpc2_fisheye.png)

Rectificar significa pasar la imagen a una proyección cilíndrica antes de detectar.
Se eligió la cilíndrica porque la proyección de perspectiva obliga a recortar la
periferia, que es la zona en estudio. Las cajas de las ramas 2 y 4 se llevan de vuelta
a la imagen *fisheye* original, así las cuatro ramas se miden igual.

La métrica es el mAP por anillos del ángulo de incidencia θ: centro (< 68,3°), medio
y periferia (> 80,7°). Cada anillo tiene 3970 objetos de la prueba. El evaluador
(`tools/eval_radial.py`) sigue el esquema de COCO y se validó contra `pycocotools`.

## 3. Resultados

Las 1000 imágenes de prueba, en la misma máquina (Apple M4 Max). Centro, medio y
periferia son mAP@50. La caída es la pérdida del centro a la periferia.

| rama | mAP@50 | mAP@50-95 | centro | medio | periferia | caída | latencia |
|---|---|---|---|---|---|---|---|
| 1, sin adaptar | 0,378 | 0,239 | 0,541 | 0,432 | 0,170 | 69% | 25,0 ms |
| 2, proyección cilíndrica | 0,418 | 0,155 | 0,528 | 0,420 | 0,316 | 40% | 26,6 ms |
| 3, *fine-tuning* | **0,659** | **0,430** | **0,701** | **0,670** | **0,603** | **14%** | 22,9 ms |
| 4, cilíndrica + *fine-tuning* | 0,522 | 0,172 | 0,642 | 0,522 | 0,412 | 36% | 28,6 ms |

- El *fine-tuning* sobre la imagen *fisheye* (rama 3) es la mejor estrategia. Sube la
  periferia de 0,170 a 0,603 y no agrega latencia.
- Rectificar ayuda al detector preentrenado (+0,147 en la periferia), pero le resta
  0,191 al modelo con *fine-tuning*. La causa es la vuelta de las cajas a la imagen
  original: la caja queda más grande que el objeto y baja la IoU.
- El aumento de datos con distorsión radial no cambia el resultado con las lentes del
  conjunto de datos, pero reduce a la mitad la pérdida cuando la lente cambia.

Las tablas por tamaño de objeto, cámara y clase, y los efectos del diseño están en
el paper.

## 4. Datos

[WoodScape](https://woodscape.valeo.com/) de Valeo
([repositorio](https://github.com/valeoai/WoodScape),
[paper](https://arxiv.org/abs/1905.01489)): 8234 imágenes de cuatro cámaras *fisheye*
con anotaciones de instancia y calibración por imagen.

- Imágenes y anotaciones: Kaggle, `subarnadasgupta/woodscapes`.
- Calibración: repositorio oficial de Valeo.
- Partición, calibración consolidada y las 1000 imágenes de prueba: [carpeta del equipo en
  Google Drive](https://drive.google.com/drive/folders/1Lu6oGsKqbPjERQKhMCfpQjAXET4tPYzV) (carpeta `CV2/`).

Las 43 etiquetas de WoodScape se agrupan en tres clases (`config/class_map.json`):

| id | clase | etiquetas de origen |
|---|---|---|
| 0 | `vehicle` | car, van, bus, truck, trailer, caravan |
| 1 | `person` | person, rider |
| 2 | `two_wheeler` | bicycle, motorcycle |

La partición tiene 6148 imágenes de entrenamiento, 1085 de validación y 1000 de
prueba, estratificadas por cámara. Se auditó contra imágenes casi duplicadas entre
entrenamiento y prueba (detalle en `DATASET_SPLIT.md`).

## 5. Instalación

Requisitos: Python `>=3.11,<3.13` y [uv](https://docs.astral.sh/uv/). Las
dependencias están en `pyproject.toml` y `uv.lock`.

```bash
uv sync
```

Preparar los datos:

```bash
# 1. Descargar la carpeta CV2/ del Drive del equipo y dejarla en la raíz del repo:
#    https://drive.google.com/drive/folders/1Lu6oGsKqbPjERQKhMCfpQjAXET4tPYzV
#    Trae las 1000 imágenes de prueba, la partición y la calibración. Alcanza para
#    evaluar con los pesos incluidos.
#    Para entrenar hacen falta además las imágenes de entrenamiento, que salen de
#    Kaggle (subarnadasgupta/woodscapes). Al descomprimir debe quedar
#    <raiz>/rgb_images y <raiz>/instance_annotations

# 2. Verificar que la copia local coincide con la partición
uv run python tools/verify_split.py --root <raiz> --split-dir splits/

# 3. Generar las etiquetas YOLO (no copia imágenes)
uv run python tools/convert_to_yolo.py --root <raiz> --split-dir splits/ \
    --out-dir outputs/yolo_ds --class-map config/class_map.json
```

El paso 3 debe producir 99 995 cajas sobre 8233 imágenes.

## 6. Uso

Cada rama tiene su notebook en `notebooks/`. Los mismos pasos también corren por scripts:

```bash
# Ramas 1 y 2: inferencia con YOLO11m de COCO (--proyeccion cilindrica para la rama 2)
uv run python tools/predict_baseline.py --yolo-dir outputs/yolo_ds \
    --modelo yolo11m.pt --out outputs/pred/rama1 --coco-map config/coco_map.json

# Evaluación por anillo (cualquier rama)
uv run python tools/eval_radial.py --yolo-dir outputs/yolo_ds \
    --pred-json outputs/pred/rama1/pred.json --out outputs/resultados/rama1 \
    --calibration config/calibration.json --rings-file config/rings.json \
    --cruce-tamano --etiqueta rama1

# Rama 3: fine-tuning, 30 épocas (--device 0 con GPU NVIDIA, --variante radial para la extensión)
uv run python tools/train_rama3.py --variante base --epochs 30 --device mps \
    --imagenes <raiz>/rgb_images --ds outputs/rama3/yolo_ds --salida outputs/rama3/mac \
    --modelo yolo11m.pt --cache ram

# Rama 4: proyección cilíndrica, fine-tuning y evaluación
uv run python entrenar_finetune.py --modelos yolo11n yolo11s yolo11m
```

Sin GPU local, la rama 3 se puede entrenar en Colab con `notebooks/rama3_colab.ipynb`.

Para evaluar sin entrenar, el repositorio incluye los pesos finales:

| rama | pesos |
|---|---|
| 3 | `outputs/rama3/mac/rama3_base/weights/best.pt` |
| 3, distorsión radial | `outputs/rama3/colab/rama3_radial/weights/best.pt` |
| 4 | `outputs/rama4/train/yolo11m/weights/best.pt` |

`notebooks/comparacion_2x2.ipynb` reproduce la tabla de resultados con esos pesos.
Los notebooks esperan además la carpeta `CV2/` del [Drive del equipo](https://drive.google.com/drive/folders/1Lu6oGsKqbPjERQKhMCfpQjAXET4tPYzV) (ver
`GUIA_EQUIPO.md`).

## 7. Estructura

```
notebooks/              un notebook por rama y comparacion_2x2.ipynb
fisheye_utils.py        funciones compartidas: carga del test, proyección cilíndrica y mAP por anillo
fisheye_aug.py          rama 3: dataset YOLO, entrenamiento y aumento de datos con distorsión radial
finetune_utils.py       rama 4: dataset cilíndrico, fine-tuning e inferencia
entrenar_finetune.py    rama 4 desde la terminal
tools/                  partición, conversión a YOLO, inferencia, evaluación por anillo
                        y su validación contra pycocotools
config/                 mapeo de clases, cortes de anillo y calibración
splits/                 listas de la partición, sha256 de cada imagen y auditoría de duplicados
outputs/                no se versiona, salvo los resultados de la rama 2, los pesos
                        finales de las ramas 3 y 4 y las curvas de entrenamiento de la rama 3
paper_ieee/             fuente LaTeX del paper, figuras y script que las genera
paper_fisheye_ieee.pdf  paper compilado (formato de conferencia IEEE)
Presentacion.html       presentación del trabajo
vpc2_fisheye.png        diagrama del diseño 2×2
DATASET_SPLIT.md        detalle de la partición y la auditoría de fuga de información
GUIA_EQUIPO.md          guía interna del equipo (carpeta CV2/ del Drive)
pyproject.toml, uv.lock dependencias
Pautas_Trabajo_Practico_Final.pdf  consigna de la materia
```

## 8. Licencia

Este trabajo usa el Valeo WoodScape Dataset. Sus términos permiten usarlo solo en
investigación, docencia y experimentación personal, y piden citarlo en todo material
público derivado (https://woodscape.valeo.com/):

```
@inproceedings{yogamani2019woodscape,
  title={WoodScape: A multi-task, multi-camera fisheye dataset for autonomous driving},
  author={Yogamani, Senthil and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2019}
}
```
