"""Baseline 1: TF-IDF + Ridge.

Representa cada CV como una "bolsa" de palabras y bigramas ponderados (TF-IDF) y
ajusta una regresión lineal regularizada (Ridge). No entiende orden ni aritmética:
para el modelo "2015" es un token más, como "audit".

Decisión H, dígitos sí o no:
  - con dígitos (por defecto): los años y meses quedan como tokens ("2009", "06").
    Ridge puede aprender, por correlación, que "2009" va con más experiencia. Es una
    forma tosca de leer fechas y da una línea base más fuerte (y una ganancia de
    BERT más creíble si la supera);
  - sin dígitos (--no-digits): se borran todos los números. El modelo solo ve
    vocabulario (p. ej. "Directed" vs "Assisted") y mide cuánto se puede predecir
    sin leer fechas. Es la variante para la pregunta Q3.

El token por defecto de scikit-learn ignora tokens de 1 carácter y separa
"2018.04-2020.06" en "2018", "04", "2020", "06".

El alpha de Ridge se elige por MAE en validación (rejilla logarítmica).

Uso:
    python -m src.baseline_tfidf
    python -m src.baseline_tfidf --no-digits
"""

import argparse
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from src.evaluation import fmt, regression_metrics, save_predictions, save_run_info
from src.utils import (
    MODELS_DIR, SEED, TARGET, ensure_dirs, load_splits, lower_text, lower_text_no_digits,
)

ALPHAS = np.logspace(-3, 3, 13)


def build_vectorizer(digits: bool) -> TfidfVectorizer:
    prep = lower_text if digits else lower_text_no_digits
    return TfidfVectorizer(preprocessor=prep, ngram_range=(1, 2), min_df=2,
                           sublinear_tf=True)


def main(digits: bool) -> None:
    ensure_dirs()
    model_name = "tfidf_ridge" if digits else "tfidf_ridge_nodigits"
    s = load_splits()
    y_tr, y_va = s["train"][TARGET].values, s["val"][TARGET].values

    t0 = time.perf_counter()
    vec = build_vectorizer(digits)
    X_tr = vec.fit_transform(s["train"]["text_clean"])
    X_va = vec.transform(s["val"]["text_clean"])

    grid = []
    for a in ALPHAS:
        m = Ridge(alpha=a, random_state=SEED).fit(X_tr, y_tr)
        grid.append((regression_metrics(y_va, m.predict(X_va))["mae"], a))
    best_mae, best_alpha = min(grid)
    model = Ridge(alpha=best_alpha, random_state=SEED).fit(X_tr, y_tr)
    train_secs = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred_test = model.predict(vec.transform(s["test"]["text_clean"]))
    infer_per_cv = (time.perf_counter() - t1) / len(s["test"])
    preds = {
        "val": (s["val"], model.predict(X_va)),
        "test": (s["test"], pred_test),
        "test_ref": (s["test"], model.predict(vec.transform(s["test"]["text_ref_clean"]))),
    }
    save_predictions(model_name, preds)
    joblib.dump({"vectorizer": vec, "model": model}, MODELS_DIR / "baselines" / f"{model_name}.joblib")

    val = regression_metrics(y_va, preds["val"][1])
    coefs = model.coef_
    vocab = np.array(vec.get_feature_names_out())
    top_pos = vocab[np.argsort(coefs)[-15:]][::-1].tolist()
    top_neg = vocab[np.argsort(coefs)[:15]].tolist()
    save_run_info(model_name, {
        "model": model_name, "digits": digits, "ngram_range": [1, 2], "min_df": 2,
        "vocab_size": len(vocab), "alpha_grid_val_mae": {f"{a:g}": m for m, a in grid},
        "best_alpha": best_alpha, "val_metrics": val, "train_secs": train_secs,
        "infer_secs_per_cv": infer_per_cv,
        "top_positive_terms": top_pos, "top_negative_terms": top_neg,
    })
    print(f"{model_name}: vocabulario {len(vocab):,} términos, alpha elegido = {best_alpha:g}")
    print(f"[val] {fmt(val)}")
    print("Términos que más suben la predicción:", top_pos[:10])
    print("Términos que más la bajan:          ", top_neg[:10])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-digits", action="store_true", help="borrar todos los dígitos del texto")
    args = parser.parse_args()
    main(digits=not args.no_digits)
