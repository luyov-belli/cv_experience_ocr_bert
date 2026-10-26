"""Constantes, rutas y utilidades compartidas por todo el pipeline."""

import os
import random
from pathlib import Path

import numpy as np

# --- Reproducibilidad ---
SEED = 42

# --- Rutas (relativas a la raíz del repo, funcionan igual en laptop y Colab) ---
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
OCR_DIR = DATA / "ocr"
PROCESSED_DIR = DATA / "processed"
SPLITS_DIR = DATA / "splits"
SAMPLED_IDS = DATA / "sampled_ids.csv"
MODELS_DIR = ROOT / "models"
OUTPUT = ROOT / "output"
FIGURES_DIR = OUTPUT / "figures"
TABLES_DIR = OUTPUT / "tables"
METRICS_DIR = OUTPUT / "metrics"

# --- Datos ---
HF_DATASET = "sumin-yu/PopResume"
# Prefijos de los shards en data/images/ y valores de la columna `job`
OCCUPATIONS = [
    "accountants_auditors",
    "construction_laborers",
    "elementary_middle_school_teachers",
    "registered_nurses",
    "software_developers",
]
N_PER_OCCUPATION = 800

# --- Objetivo ---
TARGET = "exp_year"
TARGET_CAP = 20  # se revisa tras ver la distribución (regla del 15%)
# exp_year es continuo (p. ej. 2.7), así que las bandas del enunciado (0-2, 3-5,
# 6-10, 11-20) se interpretan como intervalos [0,3), [3,6), [6,11), [11,cap].
BAND_EDGES = [0, 3, 6, 11]
BAND_LABELS = ["0-2", "3-5", "6-10", f"11-{TARGET_CAP}"]

# Columnas que nunca pueden ser features (sex y race solo para el chequeo de equidad)
FORBIDDEN_FEATURES = ["age", "sex", "race", "name", "state", "region"]


def set_seed(seed: int = SEED) -> None:
    """Fija semillas de random, numpy y torch (si está instalado)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_hf_token() -> str:
    """Lee HF_TOKEN de la variable de entorno o, en la laptop, del archivo .env.

    En Colab, el notebook copia el Secret a la variable de entorno: los scripts
    lanzados con `!python ...` corren en otro proceso y no pueden leer los
    Secrets directamente.
    """
    if not os.environ.get("HF_TOKEN"):
        try:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "No se encontró HF_TOKEN. En la laptop: ponlo en .env. "
            "En Colab: corre la celda que lo copia desde los Secrets."
        )
    return token


def clip_target(y):
    """Recorta objetivo o predicciones al rango [0, TARGET_CAP]."""
    return np.clip(y, 0, TARGET_CAP)


def to_band(y):
    """Convierte años (continuos) a etiqueta de banda (0-2, 3-5, 6-10, 11-cap)."""
    y = clip_target(np.asarray(y, dtype=float))
    idx = np.searchsorted(BAND_EDGES, y, side="right") - 1
    return np.asarray(BAND_LABELS, dtype=object)[idx]


def read_csv_ids(path, **kwargs):
    """Lee un CSV del proyecto conservando `id` como texto ("00018", no 18)."""
    import pandas as pd

    return pd.read_csv(path, dtype={"id": str}, **kwargs)


def ensure_dirs() -> None:
    for d in [RAW_DIR, OCR_DIR, PROCESSED_DIR, SPLITS_DIR, MODELS_DIR / "baselines",
              MODELS_DIR / "bert", FIGURES_DIR, TABLES_DIR, METRICS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
