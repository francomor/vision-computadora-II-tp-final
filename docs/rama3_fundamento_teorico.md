### Fundamento teórico

Esta sección resume la teoría que sostiene las decisiones de la rama 3: por qué el fine-tuning sobre las fisheye (`base`, punto C del equipo) y por qué la augmentación de distorsión radial (`radial`, extensión). Las referencias están al final.

#### 1. El problema: proyección fisheye y equivarianza de la convolución

**Proyección.** En una cámara perspectiva, un rayo que llega con ángulo $\theta$ respecto del eje óptico cae a una distancia $r = f\tan\theta$ del centro de la imagen, que diverge cuando $\theta \to 90°$: no puede cubrir campos de 190°. Una lente ojo de pez usa una función que crece mucho más despacio, $r = \rho(\theta)$, cercana a $f\,\theta$. En WoodScape es un polinomio de cuarto orden calibrado por cámara, $\rho(\theta) = k_1\theta + k_2\theta^2 + k_3\theta^3 + k_4\theta^4$ [1].

**Deformación que depende de la posición.** La escala local de la imagen cambia con $\theta$ y deja de ser igual en todas las direcciones: en dirección radial vale $d\rho/d\theta$ y en dirección tangencial $\rho(\theta)/\sin\theta$. El mismo objeto en la periferia aparece con otras proporciones, comprimido y con sus líneas rectas curvadas.

**Equivarianza.** Una capa convolucional aplica el mismo filtro en todas las posiciones (*weight sharing*), y eso la hace equivariante a traslaciones: si el objeto se traslada, la respuesta se traslada con él [2]. El diseño supone que la estadística local de la imagen es la misma en todas partes. La fisheye rompe ese supuesto: llevar un peatón del centro al borde no traslada sus píxeles, le cambia la forma. Por eso el YOLO11m de COCO pierde el 74% de su mAP@50 entre el centro y la periferia (validación).

**Matiz.** Las CNN no son ciegas a la posición: el *zero-padding* y los campos receptivos grandes les permiten codificar la posición absoluta [3, 4]. Una red entrenada con imágenes fisheye puede aprender, en parte, a comportarse distinto según dónde está el objeto.

#### 2. Tres familias de soluciones

La literatura de percepción fisheye para conducción agrupa las soluciones en tres familias [5]:

| Familia | Qué cambia | Ejemplos | En el TP |
|---|---|---|---|
| Cambiar la imagen | se rectifica antes de detectar | undistort, proyección cilíndrica | ramas 2 y 4 |
| Cambiar el modelo | arquitectura o representación de los objetos | convoluciones deformables [6]; cajas rotadas, elipses o polígonos [7] | descartada por el profesor: agrega complejidad y no deja saber qué produjo la mejora |
| Cambiar los datos de entrenamiento | el modelo aprende la distorsión | fine-tuning sobre fisheye, augmentación de distorsión [8] | **rama 3** |

La rama 3 no toca la imagen en inferencia ni la arquitectura: cambia solo lo que el modelo ve al entrenar.

#### 3. Por qué funciona el fine-tuning (`base`, punto C)

**Transfer learning y cambio de dominio.** YOLO11m aprendió en COCO qué es un auto o una persona. Entre COCO y WoodScape hay un cambio de dominio: otra geometría (la proyección fisheye), otro punto de vista (cámaras bajas que ven el capó propio) y otra agrupación de clases (van, trailer y caravan como `vehicle`; bicicletas y motos como `two_wheeler`; el conductor como `person`).

Yosinski et al. [9] muestran que las primeras capas de una CNN aprenden rasgos genéricos (bordes, texturas) y las últimas rasgos específicos de la tarea. Como la fisheye deforma las formas desde los rasgos más básicos, y como hay datos suficientes (6148 imágenes, 74.909 cajas), conviene ajustar toda la red (sin congelar capas) partiendo de los pesos de COCO.

**Qué aprende la red.** Como la convolución es equivariante, no aprende un filtro para el centro y otro para el borde: amplía el repertorio de apariencias de cada clase. El peatón curvado e inclinado de la periferia pasa a ser otra forma posible de un peatón. El mosaico [10] y las traslaciones de las augmentaciones por defecto llevan objetos deformados de la periferia a otras posiciones, así la red aprende la forma sin atarla al lugar; el *padding* le da pistas de posición para ajustar el resto.

**Evidencia.** En validación la periferia pasa de 0.138 a 0.567 de mAP@50 (×4) y la caída del centro a la periferia baja del 74% al 17%.

#### 4. Por qué una augmentación de distorsión (`radial`, extensión)

**La augmentación como forma de enseñar una invariancia.** Una augmentación le muestra al modelo variaciones de lo mismo para que aprenda a ignorarlas [11]: el espejo enseña que un auto reflejado sigue siendo un auto, el cambio de color que la iluminación no importa. La augmentación radial enseña que el grado de distorsión de la lente no cambia lo que es el objeto.

**Por qué hace falta si ya se entrena con fisheye.** Las cuatro cámaras de WoodScape tienen lentes muy parecidas, así que el modelo puede quedar atado a esa distorsión en particular. Otro vehículo u otra cámara traen otra lente; simular distintos grados de distorsión prepara al modelo para eso. Es la recomendación del profesor para la tercera estrategia: "fine-tuning usando augmentaciones que simulen distintos grados de distorsión radial", manteniendo las mismas cajas y las mismas clases.

**Antecedente.** Ye et al. [8] generan imágenes fisheye sintéticas a partir de imágenes comunes variando al azar la focal y la posición de la cámara durante el entrenamiento, para simular cámaras distintas. La rama 3 aplica la misma idea partiendo de imágenes fisheye reales y variando el grado de distorsión.

**Formulación.** Para cada píxel de salida a distancia normalizada $r$ del centro óptico se lee el píxel de entrada a $r_{src} = r\,(1 + k\,r^2)/(1 + k)$, con $k \sim U[-0.2,\ 0.3]$ en la mitad de las imágenes. $k > 0$ da una lente más fisheye y $k < 0$ una rectificación parcial. Se aplica a cada imagen antes del mosaico, alrededor de su propio centro óptico, y las cajas se recalculan desde su perímetro.

**Qué predice la teoría y qué se midió.** Una augmentación de invariancia en general no mejora cuando los datos de prueba se parecen a los de entrenamiento, pero sí cuando cambian. Eso es lo que se observa: en validación (mismas cámaras) `radial` y `base` rinden igual, y con distorsión agregada `radial` pierde la mitad (0.03 contra 0.06 de mAP@50-95).

#### 5. Por qué distorsión radial y no otra deformación o corrección

| Alternativa | Qué hace | Por qué no |
|---|---|---|
| Rectificar como augmentation (cilíndrica o undistort, la corrección de la rama 2) | lleva la imagen a otra proyección | Saca la imagen del dominio fisheye que el modelo ve en inferencia, recorta campo de visión (la banda cilíndrica deja afuera el 4,3% de las cajas) y entrenar con imágenes rectificadas es la estrategia de la rama 4: mezclaría dos celdas del 2×2. |
| Augmentaciones afines (rotación, escala, traslación, shear) | aplican la misma transformación en toda la imagen | No dependen de la posición, así que no reproducen la distorsión de una lente. YOLO ya aplica escala y traslación en `base`. |
| Perspectiva (homografía) | simula rotar una cámara pinhole | Es otro modelo de cámara, no una lente ojo de pez. |
| Deformaciones elásticas o locales al azar | distorsiones sin modelo físico | No corresponden a ninguna lente real, pueden romper la forma de los objetos y no tienen un "grado" controlable. |
| Reproyección física a otra lente ($\rho'(\theta)$ con la calibración) | simula exactamente otra lente fisheye | Es la versión más fiel y queda como trabajo futuro: requiere invertir el modelo de lente de cada imagen y bastante más código. La distorsión radial aproximada varía el mismo grado de libertad con un solo parámetro. |

Razones a favor de la distorsión radial:

1. **Reproduce el grado de libertad que cambia entre lentes.** La distorsión fisheye es radial y simétrica alrededor del centro óptico: depende solo de la distancia al centro. Lo que diferencia una lente de otra es justamente esa función radial (equidistante, equisólida, estereográfica o el polinomio de cada calibración). Una deformación radial alrededor del centro óptico varía esa función y nada más.
2. **Es el modelo clásico de distorsión radial.** $r\,(1 + k\,r^2)$ es el término radial de primer orden del modelo de Brown–Conrady [12], el que usan las calibraciones de cámaras (por ejemplo OpenCV). Acá está normalizado por $(1 + k)$ para que la esquina más lejana quede fija.
3. **Tiene un "grado" controlable con un solo parámetro.** $k$ regula cuánta distorsión: $k > 0$ más fisheye, $k < 0$ más cerca de una rectificación, $k = 0$ la imagen original. Es literalmente "distintos grados de distorsión radial".
4. **La imagen sigue siendo fisheye.** El modelo entrena y se evalúa en el mismo dominio. La rectificación parcial aparece solo como uno de los grados ($k < 0$), al azar y sin recortar el campo de visión, y no como un preprocesamiento fijo.
5. **Respeta el esquema de detección.** Las cajas se recalculan desde su perímetro y las clases no cambian, como pidió el profesor.
6. **Es barata y segura.** Es monótona para $k > -1/3$ (la imagen no se pliega), no agrega bordes negros, tiene inversa numérica para llevar las cajas y cuesta un `cv2.remap` por imagen. Solo se usa al entrenar: no agrega costo en el auto.

#### 6. Diseño experimental: por qué dos variantes

Es un experimento controlado: se cambia un solo factor por vez, como pidió el profesor.

* **`base`** es la celda "con fine-tuning, sin rectificar" del diseño 2×2. Contra la rama 1 mide el efecto del fine-tuning y contra la rama 4 el de rectificar.
* **`radial` − `base`** mide solo el efecto de la augmentación: datos, modelo, hiperparámetros, semilla y criterio de checkpoint son idénticos.
* Usar la corrección de la rama 2 dentro de la rama 3 la convertiría en la rama 4: quedaría duplicada y se rompería el diseño.

#### 7. Limitaciones

1. La fórmula de distorsión es aproximada: trabaja con la distancia en píxeles $r$ y no con el ángulo $\theta$, y no usa la orientación de la cámara. Una versión física usaría el modelo de lente de la rama 2 para simular lentes reales distintas.
2. La prueba de robustez es en parte circular: usa la misma familia de distorsión con la que se entrenó `radial`. Falta validarla con un tipo de distorsión que no vio, por ejemplo una lente física simulada con el modelo de la rama 2.
3. `base` se entrenó en la Mac (MPS, precisión simple) y `radial` en Colab (A100, precisión mixta).
4. 30 épocas: en las dos variantes la mejor época fue la última, así que el modelo seguía mejorando.

#### Referencias

1. S. Yogamani et al., "WoodScape: A multi-task, multi-camera fisheye dataset for autonomous driving", ICCV 2019.
2. I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning*, MIT Press, 2016, cap. 9.
3. M. A. Islam, S. Jia y N. D. B. Bruce, "How much position information do convolutional neural networks encode?", ICLR 2020.
4. O. S. Kayhan y J. C. van Gemert, "On translation invariance in CNNs: Convolutional layers can exploit absolute spatial location", CVPR 2020.
5. V. R. Kumar, C. Eising, C. Witt y S. Yogamani, "Surround-View Fisheye Camera Perception for Automated Driving: Overview, Survey & Challenges", IEEE Transactions on Intelligent Transportation Systems, 2023. https://arxiv.org/abs/2205.13281
6. L. Deng et al., "Restricted Deformable Convolution based Road Scene Semantic Segmentation Using Surround View Cameras", arXiv:1801.00708. https://arxiv.org/abs/1801.00708
7. H. Rashed et al., "Generalized Object Detection on Fisheye Cameras for Autonomous Driving: Dataset, Representations and Baseline", WACV 2021. https://openaccess.thecvf.com/content/WACV2021/html/Rashed_Generalized_Object_Detection_on_Fisheye_Cameras_for_Autonomous_Driving_Dataset_WACV_2021_paper.html
8. Y. Ye, K. Yang, K. Xiang, J. Wang y K. Wang, "Universal Semantic Segmentation for Fisheye Urban Driving Images", arXiv:2002.03736, 2020. https://arxiv.org/abs/2002.03736
9. J. Yosinski, J. Clune, Y. Bengio y H. Lipson, "How transferable are features in deep neural networks?", NeurIPS 2014.
10. A. Bochkovskiy, C.-Y. Wang y H.-Y. M. Liao, "YOLOv4: Optimal Speed and Accuracy of Object Detection", arXiv:2004.10934, 2020.
11. C. Shorten y T. M. Khoshgoftaar, "A survey on Image Data Augmentation for Deep Learning", Journal of Big Data, 6(60), 2019.
12. D. C. Brown, "Decentering Distortion of Lenses", Photogrammetric Engineering, 32(3), 1966.

*Las referencias 5 a 8 se verificaron en línea; el resto son trabajos conocidos citados de memoria: revisar año y venue antes de pasarlos al informe.*
