"""Etapa 2a: calidad del OCR contra el texto de referencia de PopResume.

El texto de referencia (`text`) se usa SOLO aquí, para medir el error del OCR.

Mide, por CV y por ocupación:
  - CER y WER (jiwer), tras normalizar igual ambos textos (utils.normalize_for_cer);
  - años leídos correctamente: de los años (19xx/20xx) del texto de referencia,
    cuántos aparecen en el OCR (conteo con multiplicidad);
  - rangos de fechas leídos exactos: de los rangos "AAAA.MM-AAAA.MM" de la
    referencia, cuántos aparecen idénticos en el OCR. Es más estricto que los años:
    un rango truncado ("2020.06" -> "2020.") conserva el año pero pierde el mes;
  - modos de falla: cada diferencia palabra a palabra entre referencia y OCR.

Además revisa la hipótesis de truncado: ¿los rangos incompletos están en líneas
cuya caja de detección llega más a la derecha de la página?

Salidas:
  output/metrics/ocr_quality_per_cv.csv      una fila por CV
  output/metrics/ocr_quality_summary.json    agregados
  output/tables/ocr_quality_by_occupation.csv
  output/tables/ocr_errors.csv               catálogo de diferencias (modos de falla)
  output/tables/ocr_truncation.csv           líneas de trabajo: completas vs truncadas
  output/figures/ocr_*.png

Uso:
    python -m src.ocr_quality
"""

import json
import re
from collections import Counter

import jiwer
import pandas as pd

from src.utils import (
    METRICS_DIR, OCR_DIR, RAW_DIR, TABLES_DIR, ensure_dirs, normalize_for_cer, read_csv_ids,
)
from src import visualization as viz

YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
RANGE = re.compile(r"\d{4}\.\d{2}\s*-\s*\d{4}\.\d{2}")
# Línea de trabajo: "Ocupación, Empresa, AAAA.MM-..." (el final puede venir truncado)
JOB_LINE = re.compile(r"^[^,]+,.+,\s*\d{4}\.\d{2}\s*[-\u2013\u2014]")
FULL_END = re.compile(r"\d{4}\.\d{2}\s*[-\u2013\u2014]\s*\d{4}\.\d{2}\s*$")


def year_hits(ref: str, ocr: str) -> tuple[int, int]:
    r, o = Counter(YEAR.findall(ref)), Counter(YEAR.findall(ocr))
    return sum(r.values()), sum(min(n, o[y]) for y, n in r.items())


def range_hits(ref_norm: str, ocr_norm: str) -> tuple[int, int]:
    norm = lambda s: re.sub(r"\s+", "", s)
    r = Counter(norm(m) for m in RANGE.findall(ref_norm))
    o = Counter(norm(m) for m in RANGE.findall(ocr_norm))
    return sum(r.values()), sum(min(n, o[k]) for k, n in r.items())


def word_diffs(cv_id: str, job: str, ref_norm: str, ocr_norm: str) -> list[dict]:
    out = jiwer.process_words(ref_norm, ocr_norm)
    rw, hw = ref_norm.split(), ocr_norm.split()
    return [
        {"id": cv_id, "job": job, "type": a.type,
         "ref": " ".join(rw[a.ref_start_idx:a.ref_end_idx]),
         "ocr": " ".join(hw[a.hyp_start_idx:a.hyp_end_idx])}
        for a in out.alignments[0] if a.type != "equal"
    ]


def classify(diff: dict) -> str:
    """Modo de falla de una diferencia palabra a palabra."""
    ref, ocr = diff["ref"], diff["ocr"]
    if RANGE.search(ref) and not RANGE.search(ocr):
        return "rango de fechas truncado/alterado"
    if re.search(r"\d", ref + ocr):
        return "otro error con dígitos"
    if ref.replace(" ", "") == ocr.replace(" ", ""):
        return "espaciado"
    return "letras/puntuación"


def main() -> None:
    ensure_dirs()
    meta = read_csv_ids(RAW_DIR / "metadata.csv").set_index("id")
    ref = pd.read_parquet(RAW_DIR / "reference_text.parquet").set_index("id")["text"]

    rows, diffs, job_lines = [], [], []
    for path in sorted(OCR_DIR.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        cv_id, job = d["id"], meta.loc[d["id"], "job"]
        g, h = normalize_for_cer(ref[cv_id]), normalize_for_cer(d["text"])
        n_years, years_ok = year_hits(ref[cv_id], d["text"])
        n_ranges, ranges_ok = range_hits(g, h)
        rows.append({
            "id": cv_id, "job": job, "cer": jiwer.cer(g, h), "wer": jiwer.wer(g, h),
            "n_years": n_years, "years_ok": years_ok,
            "n_ranges": n_ranges, "ranges_ok": ranges_ok, "mean_conf": d["mean_conf"],
        })
        if g != h:
            diffs += word_diffs(cv_id, job, g, h)
        width = d["image_size"][0]
        for line in d["lines"]:
            if JOB_LINE.search(line["text"]):
                job_lines.append({
                    "id": cv_id, "job": job, "n_chars": len(line["text"]),
                    "x1_rel": line["box"][2] / width, "score": line["score"],
                    "truncated": not FULL_END.search(line["text"]),
                })

    per_cv = pd.DataFrame(rows)
    per_cv.to_csv(METRICS_DIR / "ocr_quality_per_cv.csv", index=False)

    diffs = pd.DataFrame(diffs, columns=["id", "job", "type", "ref", "ocr"])
    diffs["failure_mode"] = diffs.apply(classify, axis=1) if len(diffs) else []
    # Errores dentro de emails: se conserva el modo de falla pero no el email (contiene
    # el nombre de la persona y el repo es público).
    is_email = diffs["ref"].str.contains("@") | diffs["ocr"].str.contains("@")
    diffs.loc[is_email, ["ref", "ocr"]] = "[EMAIL]"
    diffs.loc[is_email, "failure_mode"] = "letras/puntuación (en email)"
    diffs.to_csv(TABLES_DIR / "ocr_errors.csv", index=False)

    job_lines = pd.DataFrame(job_lines)
    trunc = job_lines.groupby("truncated").agg(
        n_lines=("id", "size"), x1_rel_mean=("x1_rel", "mean"), x1_rel_min=("x1_rel", "min"),
        n_chars_mean=("n_chars", "mean"), score_mean=("score", "mean"),
    ).round(4)
    trunc.to_csv(TABLES_DIR / "ocr_truncation.csv")

    by_job = per_cv.groupby("job").agg(
        n=("id", "size"), cer_mean=("cer", "mean"), wer_mean=("wer", "mean"),
        cvs_with_error=("cer", lambda s: int((s > 0).sum())),
        years=("n_years", "sum"), years_ok=("years_ok", "sum"),
        ranges=("n_ranges", "sum"), ranges_ok=("ranges_ok", "sum"), mean_conf=("mean_conf", "mean"),
    )
    by_job["pct_cvs_with_error"] = 100 * by_job["cvs_with_error"] / by_job["n"]
    by_job["pct_years_ok"] = 100 * by_job["years_ok"] / by_job["years"]
    by_job["pct_ranges_exact"] = 100 * by_job["ranges_ok"] / by_job["ranges"]
    by_job.round(5).to_csv(TABLES_DIR / "ocr_quality_by_occupation.csv")

    summary = {
        "n_cvs": len(per_cv),
        "cer_mean": per_cv.cer.mean(), "cer_median": per_cv.cer.median(), "cer_p99": per_cv.cer.quantile(.99),
        "wer_mean": per_cv.wer.mean(), "wer_median": per_cv.wer.median(), "wer_p99": per_cv.wer.quantile(.99),
        "cvs_with_any_error": int((per_cv.cer > 0).sum()),
        "pct_years_read_ok": 100 * per_cv.years_ok.sum() / per_cv.n_years.sum(),
        "pct_cvs_all_years_ok": 100 * (per_cv.years_ok == per_cv.n_years).mean(),
        "pct_ranges_exact": 100 * per_cv.ranges_ok.sum() / per_cv.n_ranges.sum(),
        "pct_cvs_all_ranges_exact": 100 * (per_cv.ranges_ok == per_cv.n_ranges).mean(),
        "failure_modes": diffs["failure_mode"].value_counts().to_dict() if len(diffs) else {},
        "truncated_job_lines": int(job_lines.truncated.sum()),
        "job_lines": len(job_lines),
    }
    summary = {k: round(v, 6) if isinstance(v, float) else v for k, v in summary.items()}
    (METRICS_DIR / "ocr_quality_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                          encoding="utf-8")

    viz.plot_ocr_quality_by_occupation(by_job)
    viz.plot_cer_wer_distribution(per_cv)
    viz.plot_truncation_vs_line_end(job_lines)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nPor ocupación:\n", by_job[["n", "cer_mean", "wer_mean", "pct_cvs_with_error",
                                         "pct_years_ok", "pct_ranges_exact"]].round(4).to_string())
    print("\nLíneas de trabajo, completas vs truncadas:\n", trunc.to_string())


if __name__ == "__main__":
    main()
