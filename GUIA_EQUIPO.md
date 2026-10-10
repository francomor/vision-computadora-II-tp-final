# Guía del equipo — TP Visión por Computadora II

Carpeta compartida. Acá está el split oficial, la calibración y los scripts.
Las cuatro ramas salen de acá, y por eso los números van a ser comparables.

**Si solo leés una cosa, que sea la sección 5: las reglas.** Todo lo demás es
mecánica; esas seis reglas son lo que hace que el trabajo cierre.

---

## 1. Quién hace qué

| Rama | Qué es | Quién |
|---|---|---|
| 1 | Baseline: YOLO26m COCO directo sobre fisheye | Tadeo |
| 2 | Proyección cilíndrica + YOLO26m COCO | Franco |
| 3 | Fine-tuning sobre fisheye, sin rectificar | Marcos |
| 4 | Proyección cilíndrica + fine-tuning | Nicolás |

Es un factorial 2×2: cruza rectificar o no con hacer fine-tuning o no. Sirve
para separar cuánto aporta cada cosa por separado y si juntas aportan más o
menos que la suma.

**Las cuatro usan `yolo26m.pt`, `imgsz=640` y el mismo presupuesto de épocas.**
Si alguien cambia el modelo o la resolución, su fila no entra en la tabla.

---

## 2. Qué hay en el Drive

```
README.md                    la versión para el repo
GUIA_EQUIPO.md               este archivo
Scripts\                     todo el código
drive_package\
    calibration.json         calibración consolidada (la usa el evaluador)
    rings.json               cortes de anillo CONGELADOS
    calibration_raw\         los 8234 originales de Valeo
    test_1000.txt            las 1000 imágenes del hold-out
    train_pool.txt           las 7233 para entrenar y validar
    quarantine.txt           1 imagen excluida (casi-duplicado del test)
    manifest.csv             cámara, grupo, split y sha256 por imagen
    split_config.json        semilla y parámetros
    leakage_audit.csv        los 200 pares test-train más parecidos
    test_set\
        rgb_images\              las 1000 imágenes de test (1,9 GB)
        instance_annotations\    sus 1000 anotaciones
```

**Las 1000 de test bájenlas del Drive, no las filtren de Kaggle.** Si un mirror
recomprime las imágenes, los píxeles cambian y las métricas dejan de ser
comparables. Así los cuatro evaluamos sobre archivos idénticos bit a bit.

El `train_pool` sí sale de Kaggle: son 7233 imágenes y no tiene sentido
duplicarlas acá. Por eso están las listas aunque no las imágenes.

---

## 3. El dataset

WoodScape (Valeo), 8234 imágenes. Kaggle: **`subarnadasgupta/woodscapes`**.

Descomprimir donde quieras; tiene que quedar:

```
<tu_carpeta>\
    rgb_images\              8234 .png
    instance_annotations\    8234 .json
```

---

## 4. Puesta en marcha

### Paso 1 — Scripts y rutas

Copiá `Scripts\` a tu máquina. Abrí cada `.bat` con el bloc de notas y cambiá
las primeras líneas:

```bat
set VENV_DIR=D:\venvs\cv2vpc      <- donde querés el entorno virtual
set BASE=D:\CV2                   <- la carpeta que contiene el dataset
```

`BASE` puede ser la carpeta padre: el script busca solo dónde está `rgb_images`.
En el 06, 07, 08, 09 y 10 hay además líneas `SPLIT`, `OUT`, `PRED` o `CALIB`.

### Paso 2 — Entorno

`01_setup_env.bat`. Crea el venv e instala numpy y pillow. Necesitás Python 3.9
o superior. Si te dice que no encuentra Python pero lo tenés, probá `py --version`
en una consola: si eso anda, el script lo va a encontrar.

### Paso 3 — Bajar el test y el split

Copiá a tu máquina:

- `drive_package\test_set\` → las 1000 imágenes de evaluación
- los archivos sueltos de `drive_package\` → a una carpeta `split_v1\`
- `calibration.json` → a tu `BASE`

### Paso 4 — Verificar

`04_verify_split.bat` compara tus archivos contra el `manifest.csv` por sha256.

Si dice **OK**, tu copia es idéntica a la de todos. Si aparecen hashes
distintos, **avisá en el grupo antes de entrenar nada**.

### Paso 5 — Etiquetas YOLO

`07_convert_to_yolo.bat`. Tarda segundos y no copia imágenes.

Si todavía no bajaste Kaggle y solo tenés el test, agregale `--splits test` a la
línea de `python` del `.bat`.

Te deja en `yolo_ds\`:

```
labels\<nombre>.txt    una línea por objeto: clase cx cy ancho alto (normalizado)
train.txt              6148 rutas | val.txt 1085 | test.txt 1000
data.yaml              listo para Ultralytics
reporte.txt            cajas por clase, split y anillo
```

Los números de control:

| | imágenes | cajas |
|---|---|---|
| train | 6148 | 74.909 |
| val | 1085 | 13.176 |
| test | 1000 | 11.910 |
| **total** | **8233** | **99.995** |

Clases: `vehicle` 54,4%, `person` 24,1%, `two_wheeler` 21,4%. Siete imágenes sin
objetos (se dejan, sirven de fondo).

**Si te dan otros números, algo está desincronizado.**

### Paso 6 — Copiar los cortes de anillo

Copiá `rings.json` del Drive a tu `yolo_ds\`.

**No es opcional.** Si falta, el evaluador los calcula sobre tus predicciones, te
van a dar distintos y tu tabla no se va a poder comparar. Los congelados son
68,3° y 80,7°.

---

## 5. Las reglas

1. **El test se toca una sola vez, al final.** No se mira durante el desarrollo,
   no se usa para elegir checkpoint ni hiperparámetros.
2. **El checkpoint se elige por mAP@50-95 global sobre validación**, igual en las
   cuatro ramas. Si una rama elige por mAP de periferia y otra por global, la
   mejora deja de ser atribuible a la estrategia.
3. **`quarantine.txt` no existe para nadie.** Es una imagen casi idéntica a una
   del test; usarla sería leakage por la puerta de atrás.
4. **Nadie edita `class_map.json`, `coco_map.json` ni `rings.json` por su
   cuenta.** Son los tres archivos que hacen comparables los números. Si algo te
   hace ruido, se discute y se cambia para los cuatro a la vez.
5. **No subir el umbral de confianza.** El `10` usa `--conf 0.001` a propósito:
   el mAP necesita la cola de baja confianza para armar la curva
   precisión-recall. Filtrar ahí baja el mAP artificialmente. El umbral alto es
   para mostrarle detecciones a una persona, no para medir.
6. **Mismo modelo, misma resolución, mismas épocas**: `yolo26m.pt`, `imgsz=640`.
   Lo único que cambia entre ramas es el factor bajo estudio.

---

## 6. Cómo corre cada rama

### Ramas 1 y 2 (sin fine-tuning)

```
10_predict_baseline.bat     -> predicciones\<rama>\pred.json + latencia.json
09_eval_radial.bat          -> resultados\<rama>\metricas.txt
```

En el `10` editá `MODELO=yolo26m.pt` y `OUT`. En el `09`, `PRED_JSON`, `OUT` y
`ETIQUETA`.

### Ramas 3 y 4 (con fine-tuning)

Entrenás con Ultralytics apuntando al `data.yaml` que generó el `07`:

```
yolo detect train model=yolo26m.pt data=yolo_ds\data.yaml imgsz=640 epochs=<N>
```

Elegís checkpoint por mAP@50-95 de validación. Después generás predicciones
sobre el test con `--save-txt --save-conf` y en el `09` dejás `PRED_JSON` vacío
y completás `PRED_DIR` con la carpeta `labels` que produce.

Podés correr el `09` sobre validación todas las veces que quieras durante el
desarrollo, con `--split val`. El test queda intacto hasta el final.

---

## 7. Lo específico de las ramas 2 y 4 (rectificación)

Dos cosas que hay que resolver antes de escribir código, porque si se descubren
tarde cuestan un día entero.

**La proyección tiene que ser cilíndrica, no perspectiva.** La proyección
perspectiva diverge al acercarse a 90° del eje óptico, así que rectificar a plano
obliga a recortar el campo de visión a unos 70-80°, y eso **elimina justo el
anillo de periferia**, que es donde está el resultado del trabajo. Las métricas
globales saldrían excelentes porque habrías evaluado sobre un subconjunto fácil.

Que la rectificación a plano sea inviable para 190° es, en sí mismo, un resultado
reportable.

**Las detecciones hay que remapearlas al fisheye original antes de evaluar.** Una
caja en la imagen cilíndrica no tiene las mismas coordenadas que en la fisheye, y
el ángulo θ se calcula con la calibración de la original. Si evaluás directo, los
anillos quedan mal asignados y tu rama no se puede comparar con ninguna otra.

Concretamente: la rectificación tiene que guardar su mapeo, y por cada caja
detectada hay que proyectar sus esquinas de vuelta al fisheye y tomar el
bounding box resultante.

Los parámetros de distorsión por imagen están en `calibration.json`:
`rho(θ) = k₁θ + k₂θ² + k₃θ³ + k₄θ⁴`, con el centro óptico ya convertido a
coordenadas absolutas.

---

## 8. El número a batir

Rama 1 definitiva, YOLO26m COCO sin fine-tuning (`notebooks/rama1_baseline.ipynb`):

| anillo | θ | n_gt | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| centro | < 68,3° | 3970 | 0,5469 | 0,3605 |
| medio | 68,3–80,7° | 3970 | 0,4534 | 0,2951 |
| periferia | > 80,7° | 3970 | 0,1712 | 0,1021 |
| global | — | 11910 | 0,3944 | 0,2522 |

Latencia: 11,7 ms por imagen (85 FPS) en una RTX 4080 SUPER.

**Cómo comparar tu rama contra esta tabla.** Usá las mismas funciones que la rama 1,
desde un notebook en `notebooks/`:

```python
from fisheye_utils import cargar_test, evaluar, tabla_resultados
test = cargar_test()                     # 1000 imágenes, 11.910 cajas
# p: DataFrame con columnas stem, cajas (xyxy en la FISHEYE original), conf, clases (ids del TP)
evaluar(p, test)                         # (mAP@50, mAP@50-95) global
evaluar(p, test, anillo=2, por="theta")  # periferia, con los cortes de CV2/rings.json
tabla_resultados({"mi_rama": p}, test, por="theta")
```

`por="theta"` es la métrica del paper. El default `por="radio"` es la excentricidad
en píxeles que usó la primera versión de la rama 2; sirve solo para comparar con esa
versión. Las predicciones de la rama 1 están en `outputs/rama1/preds_yolo26m_1000.pkl`
(mismo formato que `p`), por si querés graficar las dos ramas juntas.

**La fila que importa es la de periferia.** Un modelo puede subir el mAP global
sin mejorar nada ahí. Si tu rama sube el global pero deja la periferia igual, el
resultado es que tu estrategia no ataca el problema que estudiamos.

---

## 9. Si algo falla

- **"python no se reconoce"** → probá `py --version`. El `01` maneja ese caso.
- **"No hay espacio suficiente"** → el `06` dice cuánto falta; apuntá `OUT` a otro disco.
- **"Hay más de una carpeta con rgb_images"** → tenés dos copias; apuntá `BASE` a una.
- **"Sin JSON legible"** en el `07` → tenés una versión vieja del script.
- **Cortes distintos a 68,3 / 80,7** → te falta copiar `rings.json`.
- **mAP absurdamente bajo** → casi siempre es el mapeo de clases. Ultralytics
  devuelve índices COCO (0=person, 2=car) y los nuestros son otros (0=vehicle,
  1=person, 2=two_wheeler). El `10` lo traduce; si generaste las predicciones por
  tu cuenta, hay que hacerlo con `coco_map.json`.
- **Hashes distintos** en el `04` → avisá antes de entrenar.

---

## 10. Pendientes del equipo

- **Latencia**: la mide una sola persona, en una sola máquina, para las cuatro
  ramas. Medida en GPUs distintas no va en la misma tabla.
- **`two_wheeler` rinde bajo en todos los anillos** (AP@50 0,19 contra 0,51 de
  `vehicle`). Conviene revisar si influye que `rider` esté mapeado a `person`:
  una moto con conductor genera dos cajas superpuestas en la referencia.
- **Control opcional**: los objetos periféricos suelen estar truncados por el
  borde, y un objeto cortado es más difícil aunque no haya distorsión. Se
  resolvería repitiendo el análisis sin las cajas que tocan el borde.
- **Paper, repo, video y presentación**: la exposición es el 50% de la nota y el
  video de demo (máximo 2 minutos) es lo que más se subestima.
