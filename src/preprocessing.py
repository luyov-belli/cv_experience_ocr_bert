"""Etapa 2b: limpieza del texto del OCR, eliminación de datos personales y enmascarado.

Pasos por CV (a partir de data/ocr/<id>.json):
  1. (Opcional) descartar líneas con confianza < --min-conf. Por defecto no se
     descarta nada: en las 57,567 líneas del OCR ninguna tiene confianza < 0.90 y
     solo 2 < 0.95, así que el filtro no cambiaría el texto (decisión F).
  2. Reconstruir el orden de lectura (utils.reading_order).
  3. Normalizar tipografía: Unicode NFKC, guiones largos -> "-", viñetas -> "-",
     espacios colapsados. No se corrigen errores del OCR (p. ej. "UI" -> "Ul"):
     medir su efecto es parte de la pregunta Q2.
  4. Unir líneas partidas: palabra cortada con guion al final de línea, o línea
     que continúa en minúscula la anterior. (En PopResume casi no ocurre; se cuenta.)
  5. Quitar datos personales: la línea del nombre y la de contacto
     ("ESTADO, USA | teléfono | email"), más cualquier email/teléfono suelto.
     El estado es una variable prohibida y el nombre es un proxy de sexo y raza.
  6. Enmascarar frases que delatan la respuesta: experiencia total ("10 years of
     experience"), edad y fecha de nacimiento -> "[MASK]". Se conservan las fechas
     de los trabajos y los años de graduación. Leyendo 50 CVs y buscando en los
     4,000 no apareció ninguna de estas frases; el paso se aplica y se cuenta igual.

El mismo pipeline (pasos 3-6) se aplica al texto de referencia, para que en Q2 los
modelos se evalúen sobre OCR y referencia procesados de forma idéntica.

Salidas:
  data/processed/<id>.txt                 texto limpio de cada CV
  data/processed/cv_text.parquet          id, job, exp_year, text_raw, text_clean, text_ref_clean
  output/metrics/preprocessing_summary.json

Uso:
    python -m src.preprocessing
    python -m src.preprocessing --min-conf 0.95   # variante con filtro de confianza
"""

import argparse
import json
import re
import unicodedata

import pandas as pd

from src.utils import (
    METRICS_DIR, OCR_DIR, PROCESSED_DIR, RAW_DIR, SAMPLED_IDS, TARGET,
    ensure_dirs, read_csv_ids, reading_order,
)

EMAIL = re.compile(r"\S+@\S+\.\S+")
PHONE = re.compile(r"\+?1?[-\s.]?\(?\d{3}\)?[-\s.]\d{3}[-\s.]\d{4}")
CONTACT_LINE = re.compile(r"\b[A-Z]{2},\s*USA\b|@|\+1-\d{3}")
NAME_LINE = re.compile(r"^[A-Z][A-Z'\-. ]+$")  # todo en mayúsculas, sin dígitos ni comas
SECTION = {"WORK HISTORY", "EDUCATION"}

MASKS = {
    "total_experience": re.compile(
        r"\b\d+(?:\.\d+)?\s*\+?\s*(?:years?|yrs?)\b[^.\n]{0,25}\bexperience\b"
        r"|\bexperience\b[^.\n]{0,15}\b\d+(?:\.\d+)?\s*\+?\s*(?:years?|yrs?)\b", re.I),
    "age": re.compile(r"\b(?:age|aged)\s*:?\s*\d{2}\b|\b\d{2}\s*(?:years?|yrs?)\s*old\b", re.I),
    "birth": re.compile(r"\b(?:date of birth|d\.?o\.?b\.?|born(?: on| in)?)\s*:?\s*[\w ,./-]{0,20}\d{4}\b", re.I),
}


def normalize(line: str) -> str:
    line = unicodedata.normalize("NFKC", line)
    line = re.sub(r"[‐-―−�]", "-", line)  # guiones largos y el carácter roto
    line = re.sub(r"^\s*[•·▪●]\s*", "- ", line)                 # viñetas al inicio
    return re.sub(r"\s+", " ", line).strip()


def join_broken(lines: list[str]) -> tuple[list[str], int]:
    out, joins = [], 0
    for line in lines:
        if out and out[-1] not in SECTION and (
            re.search(r"[A-Za-z]-$", out[-1])            # palabra cortada con guion
            or (line[:1].islower() and not out[-1].endswith(("-", ":")))
        ):
            out[-1] = (out[-1][:-1] if out[-1].endswith("-") else out[-1] + " ") + line
            joins += 1
        else:
            out.append(line)
    return out, joins


def remove_pii(lines: list[str]) -> tuple[list[str], int]:
    """Quita nombre y contacto (todo lo que va antes de la primera sección que
    sea nombre o contacto) y cualquier email/teléfono suelto."""
    kept, removed, in_header = [], 0, True
    for line in lines:
        if line in SECTION:
            in_header = False
        if in_header and (NAME_LINE.match(line) or CONTACT_LINE.search(line)):
            removed += 1
            continue
        new = PHONE.sub("[PHONE]", EMAIL.sub("[EMAIL]", line))
        removed += new != line
        kept.append(new)
    return kept, removed


def mask(text: str) -> tuple[str, dict]:
    hits = {}
    for name, pat in MASKS.items():
        text, n = pat.subn("[MASK]", text)
        hits[name] = n
    return text, hits


def clean(raw_lines: list[str]) -> tuple[str, dict]:
    lines = [normalize(l) for l in raw_lines if l.strip()]
    lines, joins = join_broken(lines)
    lines, pii = remove_pii(lines)
    text, hits = mask("\n".join(lines))
    return text, {"joins": joins, "pii_lines": pii, **{f"mask_{k}": v for k, v in hits.items()}}


def main(min_conf: float | None) -> None:
    ensure_dirs()
    ids = read_csv_ids(SAMPLED_IDS)
    ref = pd.read_parquet(RAW_DIR / "reference_text.parquet").set_index("id")["text"]

    rows, stats = [], []
    for cv in ids.itertuples():
        d = json.loads((OCR_DIR / f"{cv.id}.json").read_text(encoding="utf-8"))
        lines = d["lines"] if min_conf is None else [l for l in d["lines"] if l["score"] >= min_conf]
        raw = reading_order(lines)
        text_clean, st = clean(raw.splitlines())
        text_ref, _ = clean(ref[cv.id].splitlines())
        (PROCESSED_DIR / f"{cv.id}.txt").write_text(text_clean, encoding="utf-8")
        rows.append({"id": cv.id, "job": cv.job, TARGET: cv.exp_year,
                     "text_raw": raw, "text_clean": text_clean, "text_ref_clean": text_ref})
        stats.append({"id": cv.id, "dropped_low_conf": len(d["lines"]) - len(lines), **st})

    df = pd.DataFrame(rows)
    df.to_parquet(PROCESSED_DIR / "cv_text.parquet", index=False)
    st = pd.DataFrame(stats)
    mask_cols = [c for c in st.columns if c.startswith("mask_")]
    summary = {
        "n_cvs": len(df),
        "min_conf": min_conf,
        "lines_dropped_low_conf": int(st.dropped_low_conf.sum()),
        "broken_lines_joined": int(st.joins.sum()),
        "pii_lines_removed": int(st.pii_lines.sum()),
        "cvs_with_pii_removed": int((st.pii_lines > 0).sum()),
        "cvs_masked": int((st[mask_cols].sum(axis=1) > 0).sum()),
        "masks_by_type": {c.removeprefix("mask_"): int(st[c].sum()) for c in mask_cols},
        "residual_email_or_phone": int(df.text_clean.str.contains(r"@|\+1-\d{3}", regex=True).sum()),
        "residual_state_line": int(df.text_clean.str.contains(r"\b[A-Z]{2}, USA\b", regex=True).sum()),
        "chars_clean_mean": round(df.text_clean.str.len().mean(), 1),
    }
    (METRICS_DIR / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\n--- ejemplo ---\n" + df.text_clean.iloc[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--min-conf", type=float, default=None,
                        help="descartar líneas del OCR con confianza menor (por defecto: ninguna)")
    args = parser.parse_args()
    main(args.min_conf)
