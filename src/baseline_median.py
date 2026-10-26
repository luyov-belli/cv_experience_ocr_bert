"""Baseline 0: predecir siempre la mediana de train.

Es el piso: el error que se comete sin leer el CV. Se usa la mediana y no la media
porque la mediana es el valor constante que minimiza el MAE, la métrica principal.
Todo modelo que no le gane a esto no aprendió nada útil.

Uso:
    python -m src.baseline_median
"""

import time

import numpy as np

from src.evaluation import fmt, regression_metrics, save_predictions, save_run_info
from src.utils import TARGET, ensure_dirs, load_splits

MODEL = "median"


def main() -> None:
    ensure_dirs()
    s = load_splits()
    t0 = time.perf_counter()
    median = float(np.median(s["train"][TARGET]))
    train_secs = time.perf_counter() - t0

    preds = {k: (s[k], np.full(len(s[k]), median)) for k in ("val", "test")}
    preds["test_ref"] = (s["test"], np.full(len(s["test"]), median))
    save_predictions(MODEL, preds)

    val = regression_metrics(s["val"][TARGET], preds["val"][1])
    save_run_info(MODEL, {"model": MODEL, "train_median": median, "val_metrics": val,
                          "train_secs": train_secs, "infer_secs_per_cv": 0.0})
    print(f"Mediana de train: {median:.3f} años")
    print(f"[val] {fmt(val)}")


if __name__ == "__main__":
    main()
