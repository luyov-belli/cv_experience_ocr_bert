"""Etapa 0: muestreo estratificado y descarga de PopResume.

Por cada ocupación:
  1. Lee en streaming los shards de imágenes de esa ocupación.
  2. Se queda solo con CVs sin foto (has_photo == False).
  3. Baraja con semilla 42 y buffer de 5,000, y toma N (800 por defecto).
  4. Guarda la imagen original (sin recomprimir) y sus metadatos.

Luego carga la config `text` y guarda SOLO (id, text) de los CVs muestreados.

El `id` de PopResume es único en todo el dataset (60,884 ids distintos, ninguno
repetido entre ocupaciones), así que es la llave de todo el proyecto.
Ese texto se usa únicamente para medir el error del OCR, nunca como entrada de un modelo.

Uso:
    python -m src.download_data                       # 800 por ocupación
    python -m src.download_data --n-per-occupation 20 # prueba rápida

Pensado para correr en Colab: lee ~1-1.5 GB de imágenes por ocupación por el buffer
de barajado. En la laptop funciona igual, pero depende de la red.
"""

import argparse
import io

import pandas as pd
from datasets import Image, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm

from src.utils import (
    HF_DATASET, N_PER_OCCUPATION, OCCUPATIONS, RAW_DIR, SAMPLED_IDS, SEED,
    TARGET, TARGET_CAP, ensure_dirs, get_hf_token,
)

SHUFFLE_BUFFER = 5_000
IMAGES_DIR = RAW_DIR / "images"
METADATA_CSV = RAW_DIR / "metadata.csv"
REFERENCE_TEXT = RAW_DIR / "reference_text.parquet"


def sample_occupation(job: str, n: int, token: str | None,
                      buffer_size: int = SHUFFLE_BUFFER) -> list[dict]:
    ds = load_dataset(
        HF_DATASET,
        "images",
        data_files=f"data/images/{job}-*.parquet",
        split="train",
        streaming=True,
        token=token,
    )
    # decode=False: obtenemos los bytes originales en vez de un objeto PIL,
    # para guardar la imagen tal cual (sin pérdida por recompresión).
    ds = ds.cast_column("image", Image(decode=False))
    ds = ds.filter(lambda row: not row["has_photo"])
    ds = ds.shuffle(seed=SEED, buffer_size=buffer_size)

    rows = []
    for row in tqdm(ds.take(n), total=n, desc=job):
        img_bytes = row["image"]["bytes"]
        fmt = (PILImage.open(io.BytesIO(img_bytes)).format or "PNG").lower()
        ext = "jpg" if fmt == "jpeg" else fmt
        img_path = IMAGES_DIR / f"{row['id']}.{ext}"
        img_path.write_bytes(img_bytes)

        meta = {k: v for k, v in row.items() if k != "image"}
        meta["image_path"] = img_path.relative_to(RAW_DIR).as_posix()
        rows.append(meta)
    return rows


def fetch_reference_text(ids: set[str], token: str | None) -> pd.DataFrame:
    text = load_dataset(HF_DATASET, "text", split="train", token=token).to_pandas()
    # Solo id y text: el resto de columnas ya está en metadata.csv
    return text.loc[text["id"].isin(ids), ["id", "text"]].reset_index(drop=True)


def main(n_per_occupation: int, occupations: list[str], buffer_size: int) -> None:
    ensure_dirs()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    token = get_hf_token()

    rows = []
    for job in occupations:
        rows += sample_occupation(job, n_per_occupation, token, buffer_size)
    meta = pd.DataFrame(rows)
    first = ["id", "job", "image_path", TARGET]
    meta = meta[first + [c for c in meta.columns if c not in first]]

    assert meta["id"].is_unique, "id duplicado: revisar el muestreo"
    assert not meta["has_photo"].any(), "se coló un CV con foto"

    # metadata.csv (gitignored) guarda todo, incluidos los atributos sensibles,
    # que solo se usarán en el chequeo de equidad.
    meta.to_csv(METADATA_CSV, index=False)
    # sampled_ids.csv (versionado) guarda solo lo necesario para reproducir la muestra.
    meta[["id", "job", "file_name", TARGET]].to_csv(SAMPLED_IDS, index=False)

    ref = fetch_reference_text(set(meta["id"]), token)
    ref.to_parquet(REFERENCE_TEXT, index=False)

    print(f"\nCVs muestreados: {len(meta)}")
    print(meta.groupby("job").size().to_string())
    print(f"Con texto de referencia: {len(ref)} / {len(meta)}")
    at_cap = (meta[TARGET] >= TARGET_CAP).mean()
    print(f"Share con {TARGET} >= {TARGET_CAP}: {at_cap:.1%} "
          f"({'revisar tope: > 15%' if at_cap > 0.15 else 'tope OK'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-per-occupation", type=int, default=N_PER_OCCUPATION)
    # Solo para pruebas rápidas; la muestra oficial usa los valores por defecto.
    parser.add_argument("--occupations", nargs="+", default=OCCUPATIONS, choices=OCCUPATIONS)
    parser.add_argument("--shuffle-buffer", type=int, default=SHUFFLE_BUFFER)
    args = parser.parse_args()
    main(args.n_per_occupation, args.occupations, args.shuffle_buffer)
