"""Bonus: parser de reglas que suma la duración de los trabajos (sin aprendizaje).

Hace lo que haría un reclutador:
  1. busca en el texto los rangos "AAAA.MM-AAAA.MM" de los trabajos;
  2. convierte cada rango en un intervalo de meses;
  3. fusiona los intervalos que se traslapan (dos trabajos a la vez no cuentan doble);
  4. suma los meses y divide entre 12. Los huecos entre trabajos no suman.

Rangos con el final truncado por el OCR (p. ej. "2020.04-2022." o "2022.0"): el mes
de término se estima como el punto medio de los meses compatibles con lo que se
leyó ("2022." -> 6.5; "2022.0" -> 5, uno de 01..09; "2022.1" -> 11, uno de 10..12).
Si falta también el año, el rango se descarta.

No tiene hiperparámetros ni usa train: sirve como techo de referencia de "leer las
fechas" y para juzgar si BERT las lee.

Uso:
    python -m src.baseline_rules
"""

import re
import time

import numpy as np

from src.evaluation import fmt, regression_metrics, save_predictions, save_run_info
from src.utils import TARGET, ensure_dirs, load_splits

RANGE = re.compile(r"(\d{4})\.(\d{2})\s*-\s*(\d{4})(?:\.(\d{0,2}))?")


def end_month(partial: str) -> float:
    if partial == "":
        return 6.5                       # solo el año
    if len(partial) == 2:
        return float(partial)            # mes completo
    return 5.0 if partial == "0" else 11.0  # "0" -> 01..09, "1" -> 10..12


def parse_intervals(text: str) -> list[tuple[float, float]]:
    out = []
    for y1, m1, y2, m2 in RANGE.findall(text):
        start = int(y1) * 12 + int(m1)
        end = int(y2) * 12 + end_month(m2)  # m2 == "" si el mes no se leyó
        if end > start:
            out.append((start, end))
    return out


def total_years(text: str) -> float:
    intervals = sorted(parse_intervals(text))
    months, cur_s, cur_e = 0.0, None, None
    for s, e in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                months += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)        # traslape: se extiende el intervalo
    if cur_e is not None:
        months += cur_e - cur_s
    return months / 12


def main() -> None:
    ensure_dirs()
    s = load_splits()
    t0 = time.perf_counter()
    pred_test = np.array([total_years(t) for t in s["test"]["text_clean"]])
    infer_per_cv = (time.perf_counter() - t0) / len(s["test"])
    preds = {
        "val": (s["val"], np.array([total_years(t) for t in s["val"]["text_clean"]])),
        "test": (s["test"], pred_test),
        "test_ref": (s["test"], np.array([total_years(t) for t in s["test"]["text_ref_clean"]])),
    }
    save_predictions("rules_parser", preds)

    tr_pred = np.array([total_years(t) for t in s["train"]["text_clean"]])
    bias_train = float(np.mean(tr_pred - s["train"][TARGET]))
    val = regression_metrics(s["val"][TARGET], preds["val"][1])
    save_run_info("rules_parser", {"model": "rules_parser", "val_metrics": val, "train_secs": 0.0,
                                   "infer_secs_per_cv": infer_per_cv,
                                   "mean_signed_error_train": bias_train})
    print(f"[val] {fmt(val)}")
    print(f"Error medio con signo en train (pred - real): {bias_train:+.3f} años")


if __name__ == "__main__":
    main()
