# Detección de objetos en imágenes fisheye automotrices

Trabajo práctico final de **Visión por Computadora II** (CEIA — FIUBA), grupo 1.

Comparación de estrategias de adaptación de modelos preentrenados frente a la
distorsión radial de las cámaras de surround view.

**Integrantes**

- Franco Marcelo Morero
- Nicolás Granato
- Marcos Levi Riveros Koloszwa
- Tadeo Riveros Koloszwa

---

## 1. Problema

Los sistemas de surround view de los vehículos actuales (cámara de
estacionamiento, frontal y laterales en los espejos) usan lentes ojo de pez con
campos de visión cercanos a 190°. La proyección resultante deforma los objetos en
función de su posición en la imagen.

Eso tensiona un supuesto sobre el que se apoyan las CNN. Las convoluciones son
equivariantes a la traslación, y esa equivarianza se sostiene sobre el
*weight sharing*: el mismo filtro es válido en cualquier posición porque se asume
homogeneidad espacial de la estadística de la imagen. Bajo proyección fisheye ese
supuesto se rompe — un peatón a 20° del eje óptico y el mismo peatón a 100° tienen
geometrías marcadamente distintas.

**Pregunta de investigación:** ¿cuánto se degrada un detector preentrenado hacia
la periferia del campo de visión, y qué estrategia de adaptación recupera mejor
ese desempeño?

---

## 2. Diseño experimental

Un factorial 2×2 que cruza corrección geométrica con adaptación de dominio:

| | sin fine-tuning | con fine-tuning |
|---|---|---|
| **sin rectificar** | Rama 1 — baseline | Rama 3 — adaptación por datos |
| **rectificado** | Rama 2 — proyección cilíndrica | Rama 4 — ambas |

El diseño permite estimar por separado el efecto de rectificar `(2−1)` y `(4−3)`,
el efecto de entrenar en el dominio `(3−1)` y `(4−2)`, y la interacción entre
ambos. La hipótesis es que la corrección geométrica aporta menos cuando el modelo
ya fue expuesto a datos fisheye, porque el fine-tuning aprende la deformación de
los datos.

![diagrama](vpc2_fisheye.png)

Las cuatro ramas usan **YOLO11m**, la misma resolución de entrada, el mismo mapeo
de clases y el mismo presupuesto de épocas. Lo único que varía entre celdas es el
factor bajo estudio.

**Elección de la rectificación.** Se usa proyección cilíndrica y no perspectiva:
la proyección perspectiva diverge al aproximarse a 90° del eje óptico, de modo
que rectificar a plano obligaría a recortar el campo de visión y eliminaría
justamente la región periférica que el trabajo busca medir. Las detecciones de
las ramas 2 y 4 se remapean al espacio fisheye original antes de evaluar, para
que las cuatro ramas se midan en el mismo sistema de coordenadas.

---

## 3. Dataset

[WoodScape](https://woodscape.valeo.com/) de Valeo
([repositorio](https://github.com/valeoai/WoodScape),
[paper](https://arxiv.org/abs/1905.01489)), release público de 8234 imágenes de
cuatro cámaras fisheye (FV, MVL, MVR, RV) con anotaciones de instancia y
calibración intrínseca y extrínseca por imagen.

Descarga: Kaggle, dataset `subarnadasgupta/woodscapes`. La calibración se obtiene
del repositorio oficial de Valeo.

### Clases

Las 43 etiquetas de instancia de WoodScape se agrupan en tres clases de objeto
(`config/class_map.json`):

| id | clase | etiquetas de origen | instancias |
|---|---|---|---|
| 0 | `vehicle` | car, van, bus, truck, trailer, caravan | 54.422 |
| 1 | `person` | person, rider | 24.142 |
| 2 | `two_wheeler` | bicycle, motorcycle | 21.431 |

Las categorías de agrupación (`grouped_vehicles`,
`grouped_pedestrian_and_animals`) se excluyen por representar regiones de
multitud y no instancias individuales. `rider` se asigna a `person` siguiendo el
criterio de COCO. Se descartan cajas menores a 300 px² o con algún lado menor a
8 px.

---

## 4. Metodología

### 4.1 Partición sin fuga de información

WoodScape proviene de grabaciones continuas, por lo que un muestreo aleatorio a
nivel imagen podría repartir frames casi idénticos entre entrenamiento y prueba e
inflar las métricas por memorización. Como el efecto beneficiaría a las cuatro
ramas por igual, invalidaría la comparación.

Se verificó mediante hashes perceptuales de 256 bits:

1. **El release público no contiene secuencias contiguas.** La distancia de
   Hamming entre imágenes de índices consecutivos (mediana 60 bits) resultó mayor
   que la distancia al vecino más cercano global, indicando que el orden de los
   archivos no conserva estructura temporal.
2. **El conjunto de prueba quedó limpio.** La distancia mediana entre cada imagen
   de prueba y su vecino más próximo del conjunto de entrenamiento es de 39 bits
   sobre 256 (15,2%). Se detectó un único par por debajo del umbral de alerta y
   la imagen de entrenamiento involucrada fue excluida.

Partición resultante: 6148 imágenes de entrenamiento, 1085 de validación y 1000
de prueba, estratificadas por cámara (FV 247, MVL 251, MVR 261, RV 241). Los
parámetros y la semilla quedan registrados en `splits/split_config.json`.

### 4.2 Métrica estratificada por ángulo de incidencia

Para medir la degradación periférica se reporta el mAP estratificado según el
**ángulo de incidencia θ** respecto del eje óptico, y no según la distancia
radial en píxeles. θ se obtiene invirtiendo numéricamente el modelo de distorsión
de cuarto orden provisto en la calibración:

rho(θ) = k₁·θ + k₂·θ² + k₃·θ³ + k₄·θ⁴

Se eligió θ por dos razones: es la magnitud física que gobierna la deformación, y
permite comparar cámaras con calibraciones distintas — el conjunto incluye 23
juegos de parámetros intrínsecos, de modo que un mismo radio en píxeles
corresponde a ángulos diferentes según la cámara. Los bordes laterales de la
imagen están a unos 75° de incidencia y las esquinas a 110°, consistente con el
campo de visión nominal.

El esquema de evaluación replica el mecanismo `areaRng` de COCO: al evaluar un
anillo, los objetos de referencia externos se marcan como *ignore* en lugar de
eliminarse, y una detección que coincide con un objeto ignorado no se contabiliza
como falso positivo. **La implementación del AP se validó contra `pycocotools`,
con coincidencia a cuatro decimales** en mAP@50 y mAP@50-95.

Los límites de los anillos se fijaron en los terciles empíricos de la
distribución angular (68,3° y 80,7°), de modo que los tres contengan 3970 objetos
cada uno, y se mantienen congelados en `config/rings.json` para todas las ramas.

### 4.3 Métricas reportadas

mAP@50 y mAP@50-95, global y por anillo de incidencia; AP por clase; el cruce
anillo × tamaño de objeto como control; y latencia de inferencia medida con
warm-up sobre el mismo hardware para las cuatro ramas.

---

## 5. Resultados de la rama 1 (baseline)

YOLO11m pre-entrenado en COCO, inferencia directa sobre las 1000 imágenes de
prueba (`notebooks/rama1_baseline.ipynb`), medido con `tools/eval_radial.py`:

| anillo | θ | n_gt | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| centro | < 68,3° | 3970 | 0,5415 | 0,3523 |
| medio | 68,3–80,7° | 3970 | 0,4321 | 0,2764 |
| periferia | > 80,7° | 3970 | 0,1696 | 0,0959 |
| global | — | 11910 | 0,3776 | 0,2386 |

El detector pierde el 69% de su mAP@50 entre el anillo central y el periférico.
Cambiar de generación de detector no modifica el resultado: con este mismo
evaluador, YOLOv8m obtiene 0,374 global y 0,164 en periferia; con el de
`fisheye_utils`, YOLO26m da 0,394 y 0,171, contra 0,378 y 0,160 de YOLO11m. Tres
generaciones de YOLO quedan entre 0,16 y 0,17 en la periferia: un detector genérico
más reciente no resuelve la distorsión.

Para descartar que el fenómeno se explique por la reducción de escala aparente
que introduce la proyección, se estratificó simultáneamente por ángulo y por
tamaño de objeto según los rangos de área de COCO (mAP@50):

| anillo | chico | mediano | grande |
|---|---|---|---|
| centro | 0,301 | 0,654 | 0,841 |
| medio | 0,118 | 0,467 | 0,749 |
| periferia | 0,030 | 0,175 | 0,510 |

La degradación persiste dentro de cada rango de tamaño, lo que indica un efecto
geométrico y no un artefacto de escala. La pérdida relativa es del 90% en objetos
pequeños y del 39% en grandes: la deformación resulta más destructiva cuanto menor
es la cantidad de píxeles que describen al objeto.

Por clase, `vehicle` es la más afectada (AP@50 de 0,777 en el centro a 0,185 en la
periferia) y las cámaras laterales rinden mucho peor que la frontal y la trasera
(mAP@50 0,26 en MVL/MVR contra 0,52 en FV y 0,50 en RV). Latencia: 17,0 ms por
imagen (58 FPS) en una Mac con Apple M4 Max (MPS).

---

## 6. Resultados del diseño 2×2

Las cuatro ramas sobre las mismas 1000 imágenes de prueba, con YOLO11m, corridas en
la misma máquina (Apple M4 Max) y medidas con el mismo evaluador, `tools/eval_radial.py`
(`notebooks/comparacion_2x2.ipynb`). Anillos por ángulo de incidencia θ:

| rama | mAP@50 | mAP@50-95 | centro | medio | periferia | caída | latencia |
|---|---|---|---|---|---|---|---|
| 1 — sin rectificar ni entrenar | 0,378 | 0,239 | 0,541 | 0,432 | 0,170 | 69% | 25,0 ms |
| 2 — proyección cilíndrica | 0,418 | 0,155 | 0,528 | 0,420 | 0,316 | 40% | 26,6 ms |
| 3 — fine-tuning | **0,659** | **0,430** | **0,701** | **0,670** | **0,603** | **14%** | 22,9 ms |
| 4 — cilíndrica + fine-tuning | 0,522 | 0,172 | 0,642 | 0,522 | 0,412 | 36% | 28,6 ms |

Centro, medio y periferia son mAP@50; la caída es del centro a la periferia.
Efectos sobre el mAP@50, como diferencias entre celdas que cambian un solo factor:

| efecto | global | periferia |
|---|---|---|
| rectificar sin fine-tuning (2 − 1) | +0,041 | +0,147 |
| rectificar con fine-tuning (4 − 3) | −0,138 | −0,191 |
| fine-tuning sin rectificar (3 − 1) | +0,282 | +0,433 |
| fine-tuning con rectificar (4 − 2) | +0,104 | +0,096 |
| interacción (4 − 3) − (2 − 1) | −0,178 | −0,337 |

El fine-tuning es el factor dominante: lleva la periferia de 0,170 a 0,603 y deja la
caída del centro a la periferia en 14%. La interacción es negativa y grande:
rectificar ayuda al detector sin entrenar, pero perjudica al entrenado. La pérdida de
las ramas rectificadas se concentra en mAP@50-95, porque la caja que vuelve de la
cilíndrica a la fisheye encierra un contorno curvo y queda más grande que el objeto
(aun con una detección perfecta, la IoU mediana es 0,65 en el anillo medio y la
periferia), y porque los objetos fuera de la banda no se pueden detectar. Las
conclusiones no dependen de cómo se definan los anillos: la interacción en la
periferia es −0,339 con el filtrado por anillo de `fisheye_utils` (el que usan los
notebooks) y −0,354 con anillos por radio normalizado.

La proyección y la vuelta de las cajas suman unos 3 ms por imagen; el fine-tuning no
agrega costo, porque es el mismo modelo con otros pesos. Las latencias son de una
misma sesión: el valor absoluto del detector varía entre sesiones (17 ms en la
sección 5), pero no la diferencia entre ramas.

**Limitaciones.** Una corrida por rama, sin varianza entre semillas. La comparación
se hace en la fisheye y con cajas; con contornos o máscaras, las ramas rectificadas
podrían perder menos. Los notebooks reportan el esquema de filtrado de
`fisheye_utils`, que da valores hasta 0,04 menores por anillo que `eval_radial.py`
con las mismas conclusiones.

---

## 7. Estructura del repositorio

```
notebooks/
    rama1_baseline.ipynb          detección directa sobre fisheye
    rama2_cilindrica.ipynb        proyección cilíndrica + remapeo inverso
    rama3_finetune.ipynb          fine-tuning sobre fisheye: evaluación en val y test
    rama3_colab.ipynb             entrenamiento de la rama 3 en Google Colab (GPU)
    rama4_cilindrica_finetune.ipynb
    comparacion_2x2.ipynb         las cuatro ramas con el mismo evaluador y la misma máquina
fisheye_utils.py                  funciones compartidas: carga del test, proyección
                                  cilíndrica, detectores COCO y mAP por anillo (radio o θ)
fisheye_aug.py                    rama 3: armado del dataset YOLO, entrenamiento y
                                  augmentación de distorsión radial
finetune_utils.py                 rama 4: dataset cilíndrico (proyecta los polígonos),
                                  fine-tuning e inferencia sobre el test
entrenar_finetune.py              rama 4 de punta a punta desde la terminal
tools/
    make_test_split.py            partición con auditoría de casi-duplicados
    verify_split.py               verificación por sha256 de una copia local
    convert_to_yolo.py            polígonos de instancia a cajas YOLO
    make_drive_package.py         paquete del test para el Drive, sin recomprimir
    inspect_annotations.py        inspección del esquema de anotaciones
    train_rama3.py                entrenamiento de la rama 3 desde la terminal
    eval_radial.py                mAP global y por anillo de θ con el esquema de COCO
    validar_eval_radial.py        compara eval_radial.py contra pycocotools
    predict_baseline.py           inferencia y latencia de las ramas 1 y 2
    build_calibration.py          consolidación de la calibración de Valeo
config/
    class_map.json                WoodScape -> 3 clases
    coco_map.json                 COCO -> 3 clases
    rings.json                    cortes de anillo congelados
    calibration.json              calibración consolidada de las 8234 imágenes
splits/
    test_1000.txt, train_pool.txt, quarantine.txt
    manifest.csv                  cámara, grupo, split y sha256 por imagen
    split_config.json             parámetros y semilla
    leakage_audit.csv             pares test-train más próximos
outputs/                          predicciones, métricas y figuras (no versionado, salvo
                                  los pesos finales de las ramas 3 y 4)
vpc2_fisheye.png                  diagrama del trabajo
```

`eval_radial.py` replica `pycocotools` con θ en lugar del área: coincide con él en
todos los anillos (`validar_eval_radial.py`, diferencia máxima de 1e-16 sobre las
predicciones de las ramas 1 y 3) y con los valores reportados en el notebook de la
rama 4 (diferencias de 0,001 como máximo, el redondeo con que se reportaron).
`predict_baseline.py` reproduce las predicciones del notebook de la rama 1 (las mismas
cajas) y `build_calibration.py` regenera `config/calibration.json` byte a byte desde
las calibraciones de Valeo. Los notebooks y `fisheye_utils.py` esperan además la
carpeta `CV2/` del Drive del equipo, con el test, la calibración, el manifest y
`Scripts/convert_to_yolo.py` (ver `GUIA_EQUIPO.md`).

---

## 8. Reproducibilidad

**Requisitos:** Python `>=3.11,<3.13` y [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

### Preparar los datos

```bash
# 1. Descargar WoodScape de Kaggle (subarnadasgupta/woodscapes) y descomprimir.
#    Debe quedar <raiz>/rgb_images y <raiz>/instance_annotations

# 2. Verificar que la copia local coincide con la partición oficial
uv run python tools/verify_split.py --root <raiz> --split-dir splits/

# 3. Generar las etiquetas YOLO (no copia imágenes: data.yaml apunta al origen)
uv run python tools/convert_to_yolo.py --root <raiz> --split-dir splits/ \
    --out-dir outputs/yolo_ds --class-map config/class_map.json
```

El paso 3 debe producir 99.995 cajas sobre 8233 imágenes. Un número distinto
indica una versión diferente de `class_map.json`.

### Reproducir una rama

Las ramas 1 y 2 no entrenan: la inferencia con YOLO11m de COCO y las métricas
están dentro de `notebooks/rama1_baseline.ipynb` y `notebooks/rama2_cilindrica.ipynb`.
Los mismos pasos se pueden correr por scripts:

```bash
# Inferencia con modelo preentrenado (rama 1; con --proyeccion cilindrica, la rama 2)
uv run python tools/predict_baseline.py --yolo-dir outputs/yolo_ds \
    --modelo yolo11m.pt --out outputs/pred/rama1 --coco-map config/coco_map.json

# Evaluación estratificada (las cuatro ramas)
uv run python tools/eval_radial.py --yolo-dir outputs/yolo_ds \
    --pred-json outputs/pred/rama1/pred.json --out outputs/resultados/rama1 \
    --calibration config/calibration.json --rings-file config/rings.json \
    --cruce-tamano --etiqueta rama1

# Control del evaluador contra pycocotools (no es dependencia del proyecto)
uv run --with pycocotools python tools/validar_eval_radial.py --yolo-dir outputs/yolo_ds \
    --pred-json outputs/pred/rama1/pred.json --metricas outputs/resultados/rama1/metricas.json

# Fine-tuning sobre las fisheye sin rectificar (rama 3), 30 épocas.
# --device mps en Mac o --device 0 con GPU NVIDIA; --variante radial entrena la extensión.
uv run python tools/train_rama3.py --variante base --epochs 30 --device mps \
    --imagenes <raiz>/rgb_images --ds outputs/rama3/yolo_ds --salida outputs/rama3/mac \
    --modelo yolo11m.pt --cache ram

# Dataset cilíndrico, fine-tuning y test de la rama 4 (lee las listas de CV2/yolo_ds).
uv run python entrenar_finetune.py --modelos yolo11n yolo11s yolo11m
```

**Rama 3.** `fisheye_aug.py` arma la estructura `images/` + `labels/` que espera
Ultralytics con enlaces simbólicos a las imágenes originales (no las copia ni las
recomprime) y controla los conteos de la partición (6148 / 1085 / 1000 imágenes).
La variante `base` es el fine-tuning puro con las augmentaciones por defecto de
Ultralytics; la extensión `radial` agrega una augmentación de distorsión radial,
r_src = r·(1 + k·r²)/(1 + k) con k ~ U[−0,2; 0,3] y p = 0,5, aplicada antes del mosaico. Sin GPU
local se puede entrenar en Colab con `notebooks/rama3_colab.ipynb`. La evaluación
(val y test, por anillo, cámara y clase) y el fundamento teórico están en
`notebooks/rama3_finetune.ipynb`.

**Rama 4.** `entrenar_finetune.py` proyecta las imágenes a la cilíndrica de la rama 2
y lleva las etiquetas proyectando el polígono de cada objeto (no su caja, que se
inflaría en la periferia). Después entrena con `finetune_utils.py` y evalúa cada
modelo sobre el test apenas termina. `notebooks/rama4_cilindrica_finetune.ipynb` no
entrena: levanta lo que deja el script. Usa los mismos valores de entrenamiento que
la rama 3 (YOLO11m desde COCO, 30 épocas, `imgsz=640`, batch nominal 64, mismo
optimizador y criterio de checkpoint), salvo el batch físico (2, por la memoria de
la GPU) y la semilla.

### Evaluar sin reentrenar

El repositorio incluye los pesos finales de las ramas con fine-tuning:

| rama | pesos |
|---|---|
| 3, `base` | `outputs/rama3/mac/rama3_base/weights/best.pt` |
| 3, `radial` (extensión) | `outputs/rama3/colab/rama3_radial/weights/best.pt` |
| 4, YOLO11n, s y m | `outputs/rama4/train/yolo11{n,s,m}/weights/best.pt` |

Las ramas 1 y 2 usan YOLO11m de COCO, que se descarga en la raíz del repo con:

```bash
uv run python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
```

Con esos pesos, los datos preparados como se indica arriba y la carpeta `CV2/` del
Drive del equipo, `notebooks/comparacion_2x2.ipynb` reproduce la tabla de la sección 6
y `notebooks/rama3_finetune.ipynb` la evaluación de la rama 3, sin entrenar.

La partición se puede regenerar desde cero con `tools/make_test_split.py`
usando la semilla registrada en `splits/split_config.json`; el resultado es
idéntico.

---

## 9. Licencia y atribución

Este trabajo utiliza el **Valeo WoodScape Dataset**, cuyos términos de uso
restringen su empleo a fines de investigación, docencia y experimentación
personal, y exigen incluir una referencia al dataset en todo material público
derivado: https://woodscape.valeo.com/

```
@inproceedings{yogamani2019woodscape,
  title={WoodScape: A multi-task, multi-camera fisheye dataset for autonomous driving},
  author={Yogamani, Senthil and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2019}
}
```

El análisis completo y la discusión de resultados están en el paper (formato
conferencia IEEE).
