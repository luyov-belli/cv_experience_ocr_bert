"""Etapa 1: PaddleOCR sobre cada imagen de CV, con caché de un JSON por CV.

Para cada CV de data/sampled_ids.csv:
  - corre PaddleOCR (detección + reconocimiento de líneas de texto);
  - guarda en data/ocr/<id>.json cada línea con su texto, confianza y caja,
    el texto reconstruido en orden de lectura, el tiempo y el modelo usado;
  - si el JSON ya existe, se salta (el proceso se puede cortar y retomar).

Al final escribe en output/metrics/:
  - ocr_stats.csv: tiempo, confianza media y nº de líneas por CV;
  - ocr_failures.csv: imágenes que fallaron y por qué;
  - ocr_run_summary.json: hardware, modelo, tiempos y confianza agregados.

Uso:
    python -m src.ocr_extraction                  # PP-OCRv6_medium, todos los CVs
    python -m src.ocr_extraction --limit 10       # prueba rápida
    python -m src.ocr_extraction --model mobile   # otro modelo

Modelo elegido (decisión C) con --compare 10 en una Tesla T4, 50 CVs:
    modelo      seg/CV   CER medio   CER máx   WER medio
    v5 mobile    0.29     0.35%       1.16%     5.4%
    v5 server    0.63     0.70%       1.63%     8.6%
    v6 medium    0.54     0.00%       0.16%     0.03%   <- elegido
v6 es el más preciso en todas las métricas y además más rápido que v5 server.
    python -m src.ocr_extraction --compare 10     # comparar modelos (10 CVs por ocupación)
"""

import argparse
import json
import platform
import time

from src.utils import (
    METRICS_DIR, OCR_DIR, RAW_DIR, SAMPLED_IDS, configure_paddle_cache,
    ensure_dirs, read_csv_ids,
)

configure_paddle_cache()  # antes de importar paddleocr

import torch  # noqa: E402,F401  En Windows, torch debe importarse antes que paddle.
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import paddle  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402
from tqdm import tqdm  # noqa: E402

IMAGES_DIR = RAW_DIR / "images"

MODELS = {
    "server": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
    "mobile": ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"),
    "v6": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
}

# Las imágenes de PopResume son renders sintéticos: páginas planas, derechas y sin
# texto rotado. Estos tres módulos están pensados para fotos de documentos reales;
# aquí solo añadirían tiempo y riesgo de error (p. ej. rotar mal una página).
PIPELINE_FLAGS = dict(
    use_doc_orientation_classify=False,  # la página nunca viene rotada 90/180/270°
    use_doc_unwarping=False,             # la página no está curvada ni en perspectiva
    use_textline_orientation=False,      # ninguna línea de texto está al revés
)


def has_gpu() -> bool:
    return paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0


def build_ocr(model: str) -> PaddleOCR:
    det, rec = MODELS[model]
    device = "gpu:0" if has_gpu() else "cpu"
    # oneDNN (acelerador de CPU) no soporta algunas operaciones de estos modelos en
    # PaddlePaddle 3.3 ("ConvertPirAttribute2RuntimeAttribute not support"). Solo
    # aplica a CPU; en GPU no se usa.
    return PaddleOCR(lang="en", text_detection_model_name=det,
                     text_recognition_model_name=rec, device=device,
                     enable_mkldnn=False, **PIPELINE_FLAGS)


def load_image(path) -> np.ndarray:
    """Lee la imagen desde Python (np.fromfile) en vez de pasarle la ruta a OpenCV,
    que en Windows falla con rutas no ASCII."""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def reading_order(lines: list[dict], row_tol: float = 0.5) -> str:
    """Reconstruye el texto en orden de lectura a partir de las cajas.

    Agrupa en una misma fila las cajas cuyo centro vertical está a menos de
    `row_tol` alturas de línea; ordena filas de arriba abajo y, dentro de cada
    fila, de izquierda a derecha. Las plantillas de PopResume son de una sola
    columna (verificado en la muestra), así que no hace falta separar columnas.
    """
    if not lines:
        return ""
    items = sorted(lines, key=lambda l: (l["box"][1] + l["box"][3]) / 2)
    rows, current = [], [items[0]]
    for line in items[1:]:
        prev = current[-1]
        h = max(prev["box"][3] - prev["box"][1], line["box"][3] - line["box"][1], 1)
        dy = abs((line["box"][1] + line["box"][3]) / 2 - (prev["box"][1] + prev["box"][3]) / 2)
        if dy < row_tol * h:
            current.append(line)
        else:
            rows.append(current)
            current = [line]
    rows.append(current)
    return "\n".join(" ".join(l["text"] for l in sorted(r, key=lambda l: l["box"][0])) for r in rows)


def ocr_one(ocr: PaddleOCR, image_path) -> dict:
    img = load_image(image_path)
    t0 = time.perf_counter()
    res = ocr.predict(img)[0]
    secs = time.perf_counter() - t0
    lines = [
        {"text": t, "score": round(float(s), 4),
         "box": [int(v) for v in b],                     # [x0, y0, x1, y1]
         "poly": np.asarray(p).astype(int).tolist()}     # 4 vértices
        for t, s, b, p in zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"], res["rec_polys"])
    ]
    return {
        "secs": round(secs, 3),
        "image_size": [img.shape[1], img.shape[0]],  # [ancho, alto]
        "n_lines": len(lines),
        "mean_conf": round(float(np.mean([l["score"] for l in lines])), 4) if lines else None,
        "text": reading_order(lines),
        "lines": lines,
    }


def hardware() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "paddle": paddle.__version__, "cpu": platform.processor() or platform.machine()}
    if has_gpu():
        info["gpu"] = paddle.device.cuda.get_device_name(0)
    return info


def main(model: str, limit: int | None, overwrite: bool, only_available: bool) -> None:
    ensure_dirs()
    ids = read_csv_ids(SAMPLED_IDS)["id"].tolist()
    if only_available:  # p. ej. en la laptop, que solo tiene algunas imágenes
        ids = [i for i in ids if any(IMAGES_DIR.glob(f"{i}.*"))]
    ids = ids[:limit]
    ocr = build_ocr(model)
    det, rec = MODELS[model]

    stats, failures = [], []
    for cv_id in tqdm(ids, desc=f"OCR {model}"):
        out = OCR_DIR / f"{cv_id}.json"
        if out.exists() and not overwrite:
            cached = json.loads(out.read_text(encoding="utf-8"))
            stats.append({k: cached[k] for k in ("id", "secs", "mean_conf", "n_lines")})
            continue
        matches = list(IMAGES_DIR.glob(f"{cv_id}.*"))
        if not matches:
            failures.append({"id": cv_id, "error": "imagen no encontrada"})
            continue
        try:
            result = {"id": cv_id, "det_model": det, "rec_model": rec, **ocr_one(ocr, matches[0])}
        except Exception as e:  # una imagen mala no debe detener las otras 3,999
            failures.append({"id": cv_id, "error": f"{type(e).__name__}: {e}"})
            continue
        out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        stats.append({k: result[k] for k in ("id", "secs", "mean_conf", "n_lines")})

    stats = pd.DataFrame(stats)
    stats.to_csv(METRICS_DIR / "ocr_stats.csv", index=False)
    pd.DataFrame(failures, columns=["id", "error"]).to_csv(METRICS_DIR / "ocr_failures.csv", index=False)
    summary = {
        "det_model": det, "rec_model": rec, "flags": PIPELINE_FLAGS,
        "hardware": hardware(),
        "n_requested": len(ids), "n_ok": len(stats), "n_failed": len(failures),
        "secs_per_cv_mean": round(stats["secs"].mean(), 3) if len(stats) else None,
        "secs_per_cv_median": round(stats["secs"].median(), 3) if len(stats) else None,
        "mean_conf": round(stats["mean_conf"].mean(), 4) if len(stats) else None,
    }
    (METRICS_DIR / "ocr_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def compare_models(n_per_occupation: int) -> None:
    """Decisión del modelo: corre los 3 modelos sobre la MISMA submuestra
    (n por ocupación, semilla fija) y mide tiempo, CER, WER y confianza contra
    el texto de referencia. No escribe en data/ocr/ (no toca la caché)."""
    import jiwer

    from src.utils import SEED, normalize_for_cer

    ids = read_csv_ids(SAMPLED_IDS)
    sub = ids.groupby("job", group_keys=False).sample(n_per_occupation, random_state=SEED)
    ref = pd.read_parquet(RAW_DIR / "reference_text.parquet").set_index("id")["text"]

    rows = []
    for model in MODELS:
        ocr = build_ocr(model)
        ocr.predict(load_image(next(IMAGES_DIR.glob(f"{sub.id.iloc[0]}.*"))))  # calentamiento
        for cv_id, job in tqdm(zip(sub.id, sub.job), total=len(sub), desc=model):
            r = ocr_one(ocr, next(IMAGES_DIR.glob(f"{cv_id}.*")))
            gold, hyp = normalize_for_cer(ref[cv_id]), normalize_for_cer(r["text"])
            rows.append({"model": model, "id": cv_id, "job": job, "secs": r["secs"],
                         "cer": jiwer.cer(gold, hyp), "wer": jiwer.wer(gold, hyp),
                         "mean_conf": r["mean_conf"]})
        del ocr

    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / "ocr_model_comparison_detail.csv", index=False)
    summary = df.groupby("model").agg(
        secs_mean=("secs", "mean"), cer_mean=("cer", "mean"), cer_max=("cer", "max"),
        wer_mean=("wer", "mean"), conf_mean=("mean_conf", "mean"), n=("id", "size"),
    ).round(4)
    summary.to_csv(METRICS_DIR / "ocr_model_comparison.csv")
    print(f"\nHardware: {hardware()}\n")
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--compare", type=int, metavar="N_POR_OCUPACION",
                        help="comparar los modelos en N CVs por ocupación y salir")
    parser.add_argument("--model", choices=MODELS, default="v6")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only-available", action="store_true",
                        help="procesar solo los CVs cuya imagen está en data/raw/images")
    args = parser.parse_args()
    if args.compare:
        ensure_dirs()
        compare_models(args.compare)
    else:
        main(args.model, args.limit, args.overwrite, args.only_available)
