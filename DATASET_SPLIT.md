# TP Visión por Computadora II — Detección en imágenes fisheye

Carpeta compartida del equipo. Acá está **el split oficial** del dataset y los
scripts para reproducirlo. Todo lo que entrenemos las tres ramas sale de acá.

---

## 1. Qué hay en esta carpeta

```
README.md                    este archivo
Scripts\                     todo el código
drive_package\
    calibration.json         calibración consolidada (la usa el evaluador)
    rings.json               cortes de los anillos, CONGELADOS
    calibration_raw\         los 8234 originales de Valeo + su readme
    test_1000.txt            las 1000 imágenes del hold-out
    train_pool.txt           las 7233 imágenes para entrenar y validar
    quarantine.txt           1 imagen excluida (casi-duplicado del test)
    manifest.csv             archivo, cámara, grupo, split y sha256 de cada imagen
    split_config.json        semilla y parámetros, para citar en el paper
    leakage_audit.csv        los 200 pares test-train más parecidos
    test_set\
        rgb_images\              las 1000 imágenes de test (1,9 GB)
        instance_annotations\    sus 1000 anotaciones
```

**Las 1000 imágenes de test están acá a propósito.** No alcanza con que cada uno
las filtre de su descarga de Kaggle: si un mirror recomprime las imágenes, los
píxeles cambian y las métricas dejan de ser comparables entre nosotros. Bajando
el test de acá, los cuatro evaluamos sobre archivos idénticos bit a bit.

El `train_pool` sí sale de Kaggle: son 7233 imágenes, no tiene sentido
duplicarlas en Drive. Por eso `train_pool.txt` y `quarantine.txt` están acá aunque
las imágenes no: son las listas que necesitás para armar tu entrenamiento.

---

## 2. El dataset

WoodScape (Valeo), release público de 8234 imágenes fisheye.

Bajarlo de Kaggle: **`subarnadasgupta/woodscapes`**

Descomprimir donde quieras. La estructura tiene que quedar así:

```
<tu_carpeta>\
    rgb_images\              8234 .png
    instance_annotations\    8234 .json
```

---

## 3. Puesta en marcha (15 minutos)

### Paso 1 — Copiar los scripts

Copiá la carpeta `Scripts\` a tu máquina, al lado del dataset.

### Paso 2 — Editar las rutas

Los `.bat` tienen mis rutas. Abrí cada uno con el bloc de notas y cambiá las
dos primeras líneas:

```bat
set VENV_DIR=D:\venvs\cv2vpc      <- donde querés el entorno virtual
set BASE=D:\CV2                   <- la carpeta que contiene el dataset
```

`BASE` puede ser la carpeta padre: el script busca solo dónde está
`rgb_images`. En los scripts 06 y 07 hay además una línea `SPLIT` y otra `OUT`,
cambialas igual.

### Paso 3 — Entorno

Doble clic en `01_setup_env.bat`. Crea el venv e instala numpy y pillow.
Necesitás Python 3.9 o superior. Si te dice que no encuentra Python pero lo
tenés instalado, probá `py --version` en una consola: si eso funciona, el
script lo va a encontrar igual.

### Paso 4 — Bajar el test de acá

Copiá `drive_package\test_set\` a tu máquina. Esas son las 1000 imágenes de
evaluación, idénticas para los cuatro. No las regeneres filtrando tu descarga de
Kaggle.

Copiá también los seis archivos sueltos de `drive_package\` (los `.txt`, el
`manifest.csv`, el `split_config.json` y el `leakage_audit.csv`) a una carpeta
`split_v1\` en tu máquina. Ahí es donde apuntan los scripts.

### Paso 5 — Verificar tu copia

`04_verify_split.bat` compara tus archivos contra `manifest.csv` usando sha256.

Si dice **OK**, tu copia es idéntica a la de todos.
Si aparecen hashes distintos, tu mirror recomprimió las imágenes: **avisá en el
grupo antes de entrenar nada**, porque tus métricas no serían comparables.

### Paso 6 — Generar las etiquetas YOLO

`07_convert_to_yolo.bat`. Tarda segundos y **no copia imágenes**.

Si todavía no bajaste Kaggle y solo tenés el test, abrí el `.bat` y agregale
`--splits test` al final de la línea de `python`. Genera únicamente las etiquetas
del test, sin avisos de archivos faltantes.

Te deja en `yolo_ds\`:

```
labels\<nombre>.txt    una línea por objeto: clase cx cy ancho alto (normalizado)
train.txt              6148 rutas de imágenes
val.txt                1085 rutas
test.txt               1000 rutas
data.yaml              configuración lista para Ultralytics
reporte.txt            cajas por clase, por split y por anillo radial
reporte.csv            lo mismo, imagen por imagen
```

Los números que tiene que darte, para que compares:

| | imágenes | cajas |
|---|---|---|
| train | 6148 | 74.909 |
| val | 1085 | 13.176 |
| test | 1000 | 11.910 |
| **total** | **8233** | **99.995** |

Distribución de clases: `vehicle` 54,4%, `person` 24,1%, `two_wheeler` 21,4%.
Imágenes sin ningún objeto: 7 (se dejan, sirven como fondo).

**Si te dan números distintos, algo está mal.** Lo más probable es que tengas otra
versión de `class_map.json` o del script.

El `data.yaml` usa rutas absolutas de **tu** máquina, por eso cada uno corre
este paso en la suya en lugar de bajarlo hecho.

### Paso 7 — Calibración

Copiá `calibration.json` del Drive a `D:\CV2\` (o donde tengas tu base). Es lo
que permite estratificar por ángulo de incidencia real en lugar de por píxeles.

No hace falta que corras `08_build_calibration.bat`: eso ya está hecho y su
resultado es justamente ese archivo. El `08` está por si alguien quiere
regenerarlo desde `calibration_raw\calibration.zip`.

### Paso 8 — Copiar los cortes de anillo

Copiá `rings.json` del Drive a tu `yolo_ds\`.

**Esto no es opcional.** Si falta, el evaluador calcula los cortes sobre tus
propias predicciones, te van a dar distintos a los de los demás y las tres
tablas no se van a poder comparar. Los cortes congelados son 68,3° y 80,7°.

### Paso 9 — Generar predicciones y evaluar

Si evaluás un modelo preentrenado en COCO sin fine-tuning:

```
10_predict_baseline.bat     -> predicciones\<rama>\pred.json + latencia.json
09_eval_radial.bat          -> resultados\<rama>\metricas.txt
```

Si entrenaste tu propio modelo, saltate el `10`: generá las predicciones con
Ultralytics usando `--save-txt --save-conf` y en el `09` dejá `PRED_JSON` vacío
y completá `PRED_DIR` con la carpeta `labels` que produce.

En el `09` hay tres líneas para editar por rama: `PRED_JSON` (o `PRED_DIR`),
`OUT` y `ETIQUETA`.

---

## 4. Reglas del equipo

1. **El test se toca una sola vez.** No se mira durante el desarrollo, no se usa
   para elegir checkpoint ni para tunear hiperparámetros. Se corre al final, una
   vez por rama.
2. **`quarantine.txt` no existe para nadie.** Es una imagen casi idéntica a una
   del test; usarla sería leakage por la puerta de atrás.
3. **No modificar `class_map.json` por cuenta propia.** Si creés que un mapeo
   está mal, se discute y se cambia para los tres a la vez. Si cada uno entrena
   con clases distintas, no hay comparación posible.
4. **No recomprimir ni convertir las imágenes.** Pasarlas a JPEG cambia los
   píxeles, rompe los sha256 y arruina la comparabilidad.
5. **Mismo preprocesamiento base en las tres ramas**: resolución de entrada,
   clases y presupuesto de épocas. Lo único que cambia entre ramas es la
   estrategia bajo estudio.
6. **No tocar `rings.json` ni `coco_map.json`.** Son los que hacen que los
   números de los tres sean comparables.
7. **No subir el umbral de confianza en la inferencia.** El `10` usa
   `--conf 0.001` a propósito: el mAP necesita la cola de detecciones de baja
   confianza para armar la curva precisión-recall. Filtrar ahí baja el mAP
   artificialmente. El umbral alto es para mostrarle resultados a una persona,
   no para medir.

---

## 5. Las tres clases

| id | clase | etiquetas de WoodScape |
|---|---|---|
| 0 | `vehicle` | car, van, bus, truck, trailer, caravan |
| 1 | `person` | person, rider |
| 2 | `two_wheeler` | bicycle, motorcycle |

Decisiones tomadas, por si las preguntan en la defensa:

- **`rider` va a `person`**, criterio COCO. La persona arriba de la moto es un
  objeto distinto de la moto, y sus cajas se superponen a propósito.
- **`grouped_vehicles` y `grouped_pedestrian_and_animals` quedan excluidas.** Son
  regiones de multitud, no instancias: incluirlas mete cajas gigantes
  inconsistentes.
- **`other_wheeled_transport` queda excluida** por ser una categoría cajón de
  sastre.
- Se descartan cajas de menos de 300 px² o con algún lado menor a 8 px. Son
  26.534 cajas, el 21% del total. El `reporte.txt` incluye el desglose de ese
  descarte por anillo radial: si el porcentaje crece mucho hacia la periferia,
  el filtro estaría eliminando justo los objetos de la zona que el trabajo
  quiere medir, y habría que bajar `--min-area`.

---

## 6. Cómo se armó el split (esto va al paper)

El riesgo era el leakage por frames contiguos: si dos imágenes casi idénticas de
la misma grabación caen una en train y otra en test, el mAP sube por
memorización y las tres ramas se benefician por igual, con lo que la comparación
pierde sentido.

Se verificó con hashes perceptuales de 256 bits. Dos resultados:

1. **El release de 8.2K no tiene secuencias contiguas.** La distancia entre
   imágenes de índices consecutivos resultó mayor que la distancia al vecino más
   cercano global, o sea que el orden de los nombres de archivo no tiene
   estructura temporal. Valeo entregó un muestreo curado, no grabaciones
   continuas.
2. **El hold-out quedó limpio.** La distancia de Hamming mediana entre cada
   imagen de test y su vecino más cercano del train es de 39 bits sobre 256
   (15,2%). Hubo un único par por debajo del umbral de alerta, y esa imagen de
   train fue movida a cuarentena.

El test son 1000 imágenes estratificadas por cámara: FV 247, MVL 251, MVR 261,
RV 241, proporcional a la distribución original.

---

## 7. La métrica estratificada

La pregunta del trabajo es si el detector se degrada hacia la periferia de la
imagen. Para responderla:

**Se estratifica por ángulo de incidencia θ**, no por distancia en píxeles. θ se
obtiene invirtiendo el modelo de distorsión radial de cuarto orden de la
calibración, `rho(θ) = k₁θ + k₂θ² + k₃θ³ + k₄θ⁴`. Es la magnitud física que
causa la deformación, y permite comparar cámaras con calibraciones distintas: el
dataset tiene 23 juegos de parámetros intrínsecos. El borde lateral de la imagen
está a unos 75° y las esquinas a 110°.

**El esquema de evaluación replica `areaRng` de COCO**: para evaluar un anillo,
los objetos de referencia fuera de él se marcan como *ignore* en lugar de
borrarse, y una detección que matchea un objeto ignorado no cuenta como falso
positivo. El cálculo del AP se validó contra `pycocotools` y coincide a cuatro
decimales.

**Los anillos se cortan en los terciles empíricos** de la distribución angular,
para que los tres tengan cantidades comparables de objetos (3970 cada uno).

### Resultados de referencia (rama A: YOLOv8m COCO, sin fine-tuning)

| anillo | θ | n_gt | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| centro | < 68,3° | 3970 | 0,5318 | 0,3377 |
| medio | 68,3–80,7° | 3970 | 0,4273 | 0,2652 |
| periferia | > 80,7° | 3970 | 0,1640 | 0,0920 |
| global | — | 11910 | 0,3742 | 0,2303 |

Control cruzando anillo y tamaño de objeto (mAP@50), que descarta que la caída
sea solo por la reducción de escala aparente:

| anillo | chico | mediano | grande |
|---|---|---|---|
| centro | 0,269 | 0,645 | 0,861 |
| medio | 0,122 | 0,453 | 0,750 |
| periferia | 0,024 | 0,163 | 0,526 |

**Este es el número a batir.** Las ramas B y C tienen que mostrar recuperación en
la fila de periferia. Un modelo puede subir el mAP global sin mejorar nada ahí:
la fila que importa es la tercera.

---

## 8. Pendiente

- **Ramas B y C**: sin empezar. Es lo que falta para que haya comparación.
- **Latencia**: hay que medirla una sola vez, en una sola máquina, para las tres
  ramas. Si se mide en distintas GPU los números no van en la misma tabla.
- **`two_wheeler` rinde bajo en todos los anillos** (AP@50 de 0,19 contra 0,51 de
  `vehicle`). Parte es dificultad real, pero conviene revisar si influye que
  `rider` esté mapeado a `person`: una moto con conductor genera dos cajas
  superpuestas en la referencia.
- **Control opcional**: los objetos periféricos suelen estar truncados por el
  borde, y un objeto cortado es más difícil aunque no haya distorsión. Se
  resolvería repitiendo el análisis sin las cajas que tocan el borde.

El dataset exige citar la fuente en toda publicación: ver
`WoodScape_License_and_Terms_of_Use.pdf` y https://woodscape.valeo.com/

---

## 9. Si algo falla

- **"python no se reconoce"** → probá `py --version`. El `01` maneja ese caso.
- **"No hay espacio suficiente"** → el `06` te dice cuánto falta; apuntá `OUT` a
  otro disco.
- **"Hay más de una carpeta con rgb_images"** → tenés dos copias del dataset;
  apuntá `BASE` a la que quieras usar.
- **"Sin JSON legible"** en el `07` → tenés la versión vieja del script.
- **Cortes de anillo distintos a 68,3 / 80,7** → te falta copiar `rings.json`.
- **mAP absurdamente bajo** → casi siempre es el mapeo de clases: Ultralytics
  devuelve índices COCO (0=person, 2=car) y los nuestros son otros (0=vehicle,
  1=person, 2=two_wheeler). El `10` lo traduce; si generaste las predicciones por
  tu cuenta, hay que hacerlo a mano con `coco_map.json`.
- **Hashes distintos** en el `04` → avisá en el grupo antes de entrenar.
