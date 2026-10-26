"""Baseline 2: embeddings de oraciones congelados + Ridge.

Un encoder ya entrenado (bge-small-en-v1.5, 33 M parámetros) convierte cada CV en un
vector de 384 dimensiones que resume su significado. El encoder NO se ajusta: solo
se entrena una Ridge encima. Mide cuánto sirve una representación semántica genérica
sin enseñarle la tarea. Estos encoders se entrenan para similitud de significado, no
para aritmética, así que se espera que capturen "seniority" mejor que fechas exactas.

Decisión I, textos más largos que el límite del encoder (512 tokens): se parten en
trozos de 512 tokens, se codifica cada uno y se promedian los vectores. En PopResume
el CV más largo tiene ~250 tokens, así que ningún CV se trocea (se reporta el conteo).

Los embeddings se guardan en models/baselines/ (no se versionan) para no recalcularlos.

Uso:
    python -m src.baseline_embeddings
"""

import argparse
import time

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge

from src.evaluation import fmt, regression_metrics, save_predictions, save_run_info
from src.utils import MODELS_DIR, SEED, TARGET, ensure_dirs, load_splits, set_seed

ENCODER = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 512
ALPHAS = np.logspace(-3, 3, 13)
CACHE = MODELS_DIR / "baselines"


def encode(model: SentenceTransformer, texts: list[str]) -> tuple[np.ndarray, int]:
    """Codifica textos; los que exceden MAX_TOKENS se trocean y se promedian."""
    tok = model.tokenizer
    chunks, owner = [], []
    for i, t in enumerate(texts):
        ids = tok(t, add_special_tokens=False)["input_ids"]
        step = MAX_TOKENS - 2  # deja lugar a [CLS] y [SEP]
        pieces = [ids[j:j + step] for j in range(0, max(len(ids), 1), step)]
        for p in pieces:
            chunks.append(tok.decode(p))
            owner.append(i)
    emb = model.encode(chunks, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    owner = np.array(owner)
    out = np.vstack([emb[owner == i].mean(axis=0) for i in range(len(texts))])
    n_chunked = int((np.bincount(owner) > 1).sum())
    return out, n_chunked


def cached_encode(model, name: str, texts: list[str], refresh: bool):
    path = CACHE / f"emb_{name}.npy"
    if path.exists() and not refresh:
        return np.load(path), None, 0.0
    t0 = time.perf_counter()
    emb, n_chunked = encode(model, texts)
    secs = time.perf_counter() - t0
    np.save(path, emb)
    return emb, n_chunked, secs


def main(refresh: bool) -> None:
    ensure_dirs()
    set_seed()
    s = load_splits()
    model = SentenceTransformer(ENCODER, device="cpu")
    model.max_seq_length = MAX_TOKENS

    t0 = time.perf_counter()
    E, chunked, enc_secs = {}, {}, {}
    for name, df, col in [("train", s["train"], "text_clean"), ("val", s["val"], "text_clean"),
                          ("test", s["test"], "text_clean"), ("test_ref", s["test"], "text_ref_clean")]:
        E[name], chunked[name], enc_secs[name] = cached_encode(model, name, df[col].tolist(), refresh)

    y_tr, y_va = s["train"][TARGET].values, s["val"][TARGET].values
    grid = []
    for a in ALPHAS:
        m = Ridge(alpha=a, random_state=SEED).fit(E["train"], y_tr)
        grid.append((regression_metrics(y_va, m.predict(E["val"]))["mae"], a))
    best_mae, best_alpha = min(grid)
    ridge = Ridge(alpha=best_alpha, random_state=SEED).fit(E["train"], y_tr)
    train_secs = time.perf_counter() - t0

    # Inferencia por CV = codificar + Ridge (se mide sobre test, sin caché)
    t1 = time.perf_counter()
    emb_test, _ = encode(model, s["test"]["text_clean"].tolist())
    ridge.predict(emb_test)
    infer_per_cv = (time.perf_counter() - t1) / len(s["test"])

    preds = {"val": (s["val"], ridge.predict(E["val"])),
             "test": (s["test"], ridge.predict(E["test"])),
             "test_ref": (s["test"], ridge.predict(E["test_ref"]))}
    save_predictions("embeddings_ridge", preds)
    joblib.dump({"encoder": ENCODER, "model": ridge}, CACHE / "embeddings_ridge.joblib")

    val = regression_metrics(y_va, preds["val"][1])
    save_run_info("embeddings_ridge", {
        "model": "embeddings_ridge", "encoder": ENCODER, "dim": int(E["train"].shape[1]),
        "max_tokens": MAX_TOKENS, "long_text_strategy": "trozos de 512 tokens + promedio",
        "cvs_chunked": chunked, "alpha_grid_val_mae": {f"{a:g}": m for m, a in grid},
        "best_alpha": best_alpha, "val_metrics": val, "train_secs": train_secs,
        "encode_secs": enc_secs, "infer_secs_per_cv": infer_per_cv, "device": "cpu",
    })
    print(f"embeddings_ridge ({ENCODER}): CVs troceados {chunked}, alpha elegido = {best_alpha:g}")
    print(f"[val] {fmt(val)}")
    print(f"Inferencia: {infer_per_cv * 1000:.1f} ms por CV en CPU")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--refresh", action="store_true", help="recalcular los embeddings")
    args = parser.parse_args()
    main(args.refresh)
