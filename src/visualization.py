"""Figuras del proyecto (PNG en output/figures/). Estilo común y una función por figura.

Convenciones: ejes y rejilla discretos, marcas finas, valores rotulados en las barras
y ocupaciones en el eje (el color nunca es la única forma de identificar una serie).
"""

import matplotlib

matplotlib.use("Agg")  # sin ventana: funciona en Colab, en la laptop y en Streamlit
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from src.utils import FIGURES_DIR  # noqa: E402

# Paleta de referencia (categórica validada para daltonismo) y tintas de texto
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

JOB_LABELS = {
    "accountants_auditors": "Contadores/auditores",
    "construction_laborers": "Obreros de construcción",
    "elementary_middle_school_teachers": "Profesores primaria/secundaria",
    "registered_nurses": "Enfermeras/os",
    "software_developers": "Desarrolladores de software",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "font.family": "DejaVu Sans", "legend.frameon": False, "legend.fontsize": 8,
    })


def _save(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _hbars(ax, labels, values, fmt, title, xlabel, xmax=None):
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    top = xmax if xmax is not None else max(values.max() * 1.25, 1e-9)
    ax.set_xlim(0, top)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    for yi, v in zip(y, values):
        ax.text(v + top * 0.01, yi, fmt(v), va="center", fontsize=8, color=INK)


def plot_ocr_quality_by_occupation(by_job):
    """Small multiples: % de CVs con algún error, CER medio y % de rangos de fechas
    leídos exactos, por ocupación. `by_job` indexado por job."""
    _style()
    labels = [JOB_LABELS.get(j, j) for j in by_job.index]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
    _hbars(axes[0], labels, by_job["pct_cvs_with_error"].values, lambda v: f"{v:.1f}%",
           "CVs con algún error de OCR", "% de CVs")
    _hbars(axes[1], labels, by_job["cer_mean"].values * 100, lambda v: f"{v:.4f}%",
           "CER medio", "% de caracteres")
    _hbars(axes[2], labels, by_job["pct_ranges_exact"].values, lambda v: f"{v:.2f}%",
           "Rangos de fechas leídos exactos", "% de rangos", xmax=110)
    fig.suptitle("Calidad del OCR (PP-OCRv6) por ocupación, 4,000 CVs", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, "ocr_quality_by_occupation.png")


def plot_cer_wer_distribution(per_cv):
    """Distribución de CER y WER (dos paneles, un eje cada uno). Como la gran mayoría
    de CVs tiene error 0, se grafica solo la cola con error y se rotula cuántos son 0."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, col, name in [(axes[0], "cer", "CER"), (axes[1], "wer", "WER")]:
        vals = per_cv[col].values * 100
        nz = vals[vals > 0]
        ax.hist(nz, bins=20, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
        ax.set_title(f"{name} por CV (solo CVs con error)", loc="left")
        ax.set_xlabel(f"{name} (%)")
        ax.set_ylabel("Nº de CVs")
        ax.text(0.98, 0.95, f"{(vals == 0).sum():,} CVs con {name} = 0\n{len(nz):,} CVs con {name} > 0",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK_2)
    fig.tight_layout()
    return _save(fig, "ocr_cer_wer_distribution.png")


def plot_truncation_vs_line_end(job_lines):
    """¿Dónde termina la caja de la línea del trabajo? Fin de la caja relativo al ancho
    de la página, para líneas con rango de fechas completo vs truncado."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bins = np.linspace(job_lines["x1_rel"].min(), job_lines["x1_rel"].max(), 30)
    for (flag, label), color in zip([(False, "Rango completo"), (True, "Rango truncado")], SERIES[:2]):
        sub = job_lines.loc[job_lines["truncated"] == flag, "x1_rel"]
        ax.hist(sub, bins=bins, color=color, alpha=0.85 if flag else 0.6,
                label=f"{label} (n={len(sub):,})", density=True, edgecolor=SURFACE, linewidth=0.5)
    ax.set_title("Fin de la línea del trabajo vs. truncado del rango", loc="left")
    ax.set_xlabel("Borde derecho de la caja / ancho de la imagen")
    ax.set_ylabel("Densidad")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, "ocr_truncation_vs_line_end.png")
