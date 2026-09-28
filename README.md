# Detección de objetos en imágenes fisheye automotrices

Trabajo práctico final de **Visión por Computadora II** (CEIA - FIUBA), grupo 1: comparación de estrategias de adaptación de modelos preentrenados.

Integrantes: 
* Franco Marcelo Morero 
* Nicolás Granato
* Marcos Levi Riveros Koloszwa.
* Tadeo Rivero 

## Objetivo

Evaluar cómo detectan vehículos, personas y vehículos de dos ruedas los modelos preentrenados en COCO cuando se los usa sobre imágenes fisheye. La distorsión radial deforma los objetos según su posición en la imagen, y eso rompe la geometría en la que se apoyan las CNN: las convoluciones son equivariantes a la traslación, pero un objeto en la periferia no se ve igual que en el centro. Queremos medir cuánto se pierde y qué estrategia lo recupera mejor.

## Estrategias

- **Rama 1 - Baseline:** detectores preentrenados en COCO aplicados directamente sobre la imagen fisheye.
- **Rama 2 - Proyección cilíndrica:** se reproyecta la imagen fisheye a coordenadas cilíndricas, se detecta con los modelos preentrenados y las cajas se llevan de vuelta a la imagen fisheye.
- **Rama 3 - Fine-tuning:** se reentrenan los detectores sobre train/val con aumentación de distorsión radial.

![diagrama](vpc2_fisheye.png)

## Métricas

mAP@50, mAP@50-95, mAP estratificado por excentricidad radial (centro vs periferia) y latencia de inferencia.

## Dataset

[WoodScape](https://woodscape.valeo.com/) de Valeo ([repo](https://github.com/valeoai/WoodScape), [paper](https://arxiv.org/abs/1905.01489)), 4 cámaras fisheye (FV, MVL, MVR, RV).

## Estructura del repo

```
notebooks/              un notebook por rama (rama2_cilindrica.ipynb)
outputs/                predicciones y figuras (no versionado)
vpc2_fisheye.png        diagrama del trabajo
```

## Requisitos

- Python `>=3.11,<3.13`
- [uv](https://docs.astral.sh/uv/) para la gestión del entorno y dependencias.

## Setup

```bash
uv sync
uv run jupyter lab
```

El análisis completo y los resultados van en el paper (formato IEEE).
