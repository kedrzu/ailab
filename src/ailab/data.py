"""Pobieranie i wczytywanie źródeł z korpusu Polish DynaWord.

Korpus leży na Hugging Face jako osobny plik parquet na każde źródło
(`data/<źródło>/<źródło>.parquet`), więc pobieramy tylko to, czego potrzebujemy,
zamiast całych ~9,5 GB.
"""

import random
from collections import Counter
from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download

REPO_ID = "SlayerLab/polish-dynaword"

# Przypinamy tag wydania, a nie `main`: `main` się zmienia (autorzy dodają i wycofują
# źródła), a tag gwarantuje, że za pół roku pobierzemy dokładnie te same dane.
REVISION = "v0.2.6"

# Katalog `data/` jest w .gitignore — dane są duże i odtwarzalne tym modułem.
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "polish-dynaword"


def download_source(source: str) -> Path:
    """Pobiera parquet jednego źródła (lub zwraca ścieżkę, jeśli już jest)."""
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"data/{source}/{source}.parquet",
        revision=REVISION,
        local_dir=DATA_DIR,
    )
    return Path(path)


def load_source(source: str) -> pl.DataFrame:
    """Wczytuje źródło jako tabelę (jeden wiersz = jeden dokument)."""
    return pl.read_parquet(download_source(source))


def split_by_source_text(
    ids: list[str], texts: list[str], test_fraction: float = 0.25, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Dzieli dane tak, żeby fragmenty jednego tekstu nie trafiły do obu zbiorów.

    Id ma postać `<źródło>_<tekst>_<numer fragmentu>`. Losowy podział po dokumentach dałby
    test łatwiejszy niż w rzeczywistości, bo model widziałby inne fragmenty tej samej książki.
    """
    groups = [doc_id.rsplit("_", 1)[0] for doc_id in ids]
    sizes = Counter(groups)
    names = sorted(sizes)
    random.Random(seed).shuffle(names)

    test_groups = set()
    in_test = 0
    for name in names:
        if in_test >= test_fraction * len(ids):
            break
        test_groups.add(name)
        in_test += sizes[name]

    train_texts = [t for t, g in zip(texts, groups) if g not in test_groups]
    test_texts = [t for t, g in zip(texts, groups) if g in test_groups]
    return train_texts, test_texts
