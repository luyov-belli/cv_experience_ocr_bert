"""Etapa 3: split estratificado train / val / test (70 / 15 / 15) por id de CV.

- Estratos: ocupación × banda de experiencia (0-2, 3-5, 6-10, 11-20), es decir 20
  estratos. Así cada partición tiene la misma mezcla de ocupaciones y de juniors y
  seniors que la muestra completa; sin esto, la banda 11-20 (la más chica) podría
  quedar sub-representada en test por azar.
- Semilla fija (42): cualquiera que corra el script obtiene la misma partición.
- Todos los modelos usan exactamente estos ids. El test solo se toca en la
  evaluación final; todo ajuste (alpha de Ridge, épocas de BERT...) usa val.

Se hace en dos pasos con train_test_split: primero 70% train vs 30% resto, luego
el resto se parte 50/50 en val y test, estratificando ambas veces.

Salidas:
  data/splits/{train,val,test}.csv          id, job, exp_year, band
  output/tables/split_distribution.csv      conteos por partición × ocupación × banda
  output/metrics/split_summary.json

Uso:
    python -m src.split
"""

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import (
    METRICS_DIR, PROCESSED_DIR, SAMPLED_IDS, SEED, SPLITS_DIR, TABLES_DIR, TARGET,
    ensure_dirs, read_csv_ids, to_band,
)

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def make_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    strata = df["job"] + "|" + df["band"]
    train, rest = train_test_split(df, train_size=FRACTIONS["train"], stratify=strata,
                                   random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=strata.loc[rest.index],
                                 random_state=SEED)
    return {"train": train, "val": val, "test": test}


def main() -> None:
    ensure_dirs()
    df = read_csv_ids(SAMPLED_IDS)[["id", "job", TARGET]]
    df["band"] = to_band(df[TARGET])
    parts = make_split(df)

    for name, part in parts.items():
        part.sort_values("id").to_csv(SPLITS_DIR / f"{name}.csv", index=False)

    # Verificaciones: partición disjunta y completa
    ids = [set(p.id) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "ids repetidos entre particiones"
    assert set().union(*ids) == set(df.id), "faltan ids"

    dist = pd.concat({k: v.groupby(["job", "band"]).size() for k, v in parts.items()}, axis=1)
    dist.to_csv(TABLES_DIR / "split_distribution.csv")

    # CVs con texto limpio idéntico en particiones distintas (fuga por duplicados)
    cross_dupes = None
    text_path = PROCESSED_DIR / "cv_text.parquet"
    if text_path.exists():
        text = pd.read_parquet(text_path, columns=["id", "text_clean"])
        split_of = {i: k for k, p in parts.items() for i in p.id}
        text["split"] = text.id.map(split_of)
        g = text.groupby("text_clean")["split"].nunique()
        cross_dupes = int((g > 1).sum())

    summary = {
        "seed": SEED,
        "sizes": {k: len(v) for k, v in parts.items()},
        "target_mean": {k: round(v[TARGET].mean(), 3) for k, v in parts.items()},
        "target_median": {k: round(v[TARGET].median(), 3) for k, v in parts.items()},
        "band_share": {k: (v.band.value_counts(normalize=True).sort_index().round(3).to_dict())
                       for k, v in parts.items()},
        "smallest_test_stratum": int(dist["test"].min()),
        "identical_texts_across_splits": cross_dupes,
    }
    (METRICS_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nConteos por ocupación × banda:\n", dist.to_string())


if __name__ == "__main__":
    main()
