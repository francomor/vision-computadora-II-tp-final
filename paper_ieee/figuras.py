"""figuras del paper desde los CSV del proyecto (eval_radial.py y la curva de robustez de la rama 3)"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

U, OUT = Path(sys.argv[1]), Path(sys.argv[2])
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7, "pdf.fonttype": 42})
TINTA, TINTA2, GRILLA, EJE = "#0b0b0b", "#52514e", "#e1e0d9", "#a9a89c"
coma = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
coma2 = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))


def estilo(ax):
    ax.grid(axis="y", color=GRILLA, linewidth=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(EJE)
        ax.spines[lado].set_linewidth(0.6)
    ax.tick_params(colors=EJE, labelcolor=TINTA2, width=0.6)


def separar(valores, minimo):
    orden = sorted(range(len(valores)), key=lambda i: valores[i])
    pos = list(valores)
    for a, b in zip(orden, orden[1:]):
        pos[b] = max(pos[b], pos[a] + minimo)
    return pos


# ---------------------------------------------------------------- mAP@50 por anillo de theta (eval_radial)
res = pd.read_csv(U / "comparacion/resultados_theta_eval_radial.csv", index_col=0)
SERIES = [("rama1", "R1 sin rectificar ni entrenar", "#eb6834", "s"),
          ("rama2", "R2 cilíndrica", "#1baf7a", "D"),
          ("rama3", "R3 fine-tuning", "#2a78d6", "o"),
          ("rama4", "R4 cilíndrica + fine-tuning", "#e87ba4", "v")]
anillos = ["centro", "medio", "periferia"]
fig, ax = plt.subplots(figsize=(3.45, 2.25))
estilo(ax)
x = [0, 1, 2]
finales = {}
for r, nombre, color, m in SERIES:
    y = [res.loc[r, f"mAP50 {a}"] for a in anillos]
    ax.plot(x, y, color=color, marker=m, markersize=4.5, linewidth=1.4, markeredgecolor="white", markeredgewidth=0.6,
            label=nombre, zorder=3)
    finales[r] = y[-1]
for (r, nombre, _, _), yy in zip(SERIES, separar([finales[s[0]] for s in SERIES], 0.05)):
    ax.text(2.1, yy, f"{nombre.split()[0]}  {finales[r]:.2f}".replace(".", ","), va="center", color=TINTA2, fontsize=7)
ax.set_xticks(x, [r"centro ($\theta<68{,}3^\circ$)", r"medio", r"periferia ($\theta>80{,}7^\circ$)"])
ax.set_xlim(-0.2, 2.55)
ax.set_ylim(0, 0.8)
ax.yaxis.set_major_formatter(coma)
ax.set_ylabel("mAP@50 en test", color=TINTA2)
ax.legend(loc="lower left", frameon=False, ncol=1, handlelength=1.6, labelcolor=TINTA2, borderaxespad=0.2)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "anillos.pdf")
plt.close(fig)

# ---------------------------------------------------------------- robustez a otra lente (val, 300 imagenes)
rob = pd.read_csv(U / "rama3/robustez_val.csv")
SER2 = [("yolo11m_coco", "COCO sin entrenar", "#eb6834", "s"),
        ("rama3_base", "R3 base", "#2a78d6", "o"),
        ("rama3_radial", "R3 + distorsión radial", "#eda100", "^")]
fig, ax = plt.subplots(figsize=(3.45, 1.95))
estilo(ax)
ax.axvline(0, color=EJE, linewidth=0.6, linestyle="--", zorder=1)
ax.text(0.01, 0.5, "lente del\ndataset", color=TINTA2, fontsize=6.5, va="center")
finales = {}
for mod, nombre, color, m in SER2:
    d = rob[rob["modelo"] == mod].sort_values("k")
    ax.plot(d["k"], d["mAP50-95"], color=color, marker=m, markersize=4.5, linewidth=1.4,
            markeredgecolor="white", markeredgewidth=0.6, label=nombre, zorder=3)
    finales[mod] = d["mAP50-95"].iloc[-1]
for (mod, nombre, _, _), yy in zip(SER2, separar([finales[s[0]] for s in SER2], 0.022)):
    ax.text(0.315, yy, f"{finales[mod]:.2f}".replace(".", ","), va="center", color=TINTA2, fontsize=7)
ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.1f}".replace(".", ",").replace("+0,0", "0")))
ax.set_xlim(-0.23, 0.36)
ax.set_ylim(0.12, 0.5)
ax.yaxis.set_major_formatter(coma2)
ax.set_xlabel(r"distorsión radial agregada $k$", color=TINTA2)
ax.set_ylabel("mAP@50-95 (val)", color=TINTA2)
ax.legend(loc="lower center", frameon=False, ncol=3, handlelength=1.4, labelcolor=TINTA2, columnspacing=0.8,
          bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "robustez.pdf")
plt.close(fig)
print("ok")
