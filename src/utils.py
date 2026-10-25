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
OCCUPATIONS = [
    "registered_nurses",
    "elementary_and_middle_school_teachers",
    "software_developers",
    "accountants_and_auditors",
    "construction_laborers",
]  # TODO: confirmar los nombres exactos de las configs/particiones en Hugging Face
N_PER_OCCUPATION = 800

# --- Objetivo ---
TARGET = "exp_year"
TARGET_CAP = 20  # se revisa tras ver la distribución (regla del 15%)
BANDS = [(0, 2), (3, 5), (6, 10), (11, TARGET_CAP)]
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


def get_hf_token() -> str | None:
    """Lee HF_TOKEN de Colab Secrets, de .env o del entorno, en ese orden."""
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get("HF_TOKEN")
    except Exception:
        pass
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    return os.environ.get("HF_TOKEN")


def clip_target(y):
    """Recorta objetivo o predicciones al rango [0, TARGET_CAP]."""
    return np.clip(y, 0, TARGET_CAP)


def to_band(y):
    """Convierte años a etiqueta de banda (0-2, 3-5, 6-10, 11-cap)."""
    y = np.rint(clip_target(np.asarray(y, dtype=float))).astype(int)
    out = np.empty(len(y), dtype=object)
    for (lo, hi), label in zip(BANDS, BAND_LABELS):
        out[(y >= lo) & (y <= hi)] = label
    return out


def ensure_dirs() -> None:
    for d in [RAW_DIR, OCR_DIR, PROCESSED_DIR, SPLITS_DIR, MODELS_DIR / "baselines",
              MODELS_DIR / "bert", FIGURES_DIR, TABLES_DIR, METRICS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
