"""Métricas comunes a todos los modelos y registro de predicciones.

Todas las métricas se calculan sobre predicciones ya recortadas a [0, TARGET_CAP]:
  - MAE (métrica principal) y RMSE, en años;
  - accuracy dentro de ±1, ±2 y ±3 años;
  - accuracy y macro-F1 por banda (0-2, 3-5, 6-10, 11-20);
  - R².

Cada modelo guarda sus predicciones en output/predictions/<modelo>.csv con columnas
id, split, y_true, y_pred. Los splits son "val", "test" (texto del OCR) y
"test_ref" (texto de referencia, solo para la pregunta Q2). La evaluación final de
la etapa 6 lee esos archivos; nadie vuelve a predecir.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score

from src.utils import BAND_LABELS, METRICS_DIR, PREDICTIONS_DIR, clip_target, to_band


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = clip_target(np.asarray(y_pred, dtype=float))
    err = np.abs(y_pred - y_true)
    bt, bp = to_band(y_true), to_band(y_pred)
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "within_1": float((err <= 1).mean()),
        "within_2": float((err <= 2).mean()),
        "within_3": float((err <= 3).mean()),
        "band_acc": float((bt == bp).mean()),
        "band_macro_f1": float(f1_score(bt, bp, labels=BAND_LABELS, average="macro", zero_division=0)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_predictions(model: str, parts: dict[str, tuple]) -> pd.DataFrame:
    """parts: {"val": (df, y_pred), "test": (df, y_pred), ...} con df que tiene id y exp_year."""
    rows = []
    for split, (df, y_pred) in parts.items():
        rows.append(pd.DataFrame({"id": df["id"].values, "split": split,
                                  "y_true": df["exp_year"].values,
                                  "y_pred": clip_target(np.asarray(y_pred, dtype=float))}))
    out = pd.concat(rows, ignore_index=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PREDICTIONS_DIR / f"{model}.csv", index=False)
    return out


def save_run_info(model: str, info: dict) -> None:
    """Hiperparámetros elegidos, métricas de validación y tiempos de un modelo."""
    path = METRICS_DIR / f"{model}_run.json"
    path.write_text(json.dumps(info, indent=2, default=float), encoding="utf-8")


def fmt(m: dict) -> str:
    return (f"MAE {m['mae']:.3f} | RMSE {m['rmse']:.3f} | ±2 {m['within_2']:.1%} | "
            f"banda {m['band_acc']:.1%} | F1 {m['band_macro_f1']:.3f} | R² {m['r2']:.3f}")
