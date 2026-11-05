"""Etap 1: oględziny danych przed treningiem tokenizera.

Tokenizer nauczy się każdego znaku i każdego powtarzalnego śmiecia, który zobaczy,
więc najpierw sprawdzamy: ile jest tekstu, jak długie są dokumenty i jakie znaki
w nich siedzą.

Uruchomienie:  uv run python experiments/tokenizer/01_explore.py
"""

import random
import unicodedata
from collections import Counter

import polars as pl

from ailab.data import load_source

SOURCES = ["nkjp1m", "wolne_lektury"]

# Znaki „oczywiste” dla polskiego tekstu — wszystko spoza tej listy pokażemy osobno,
# bo to tam kryją się niespodzianki (typograficzne cudzysłowy, obce alfabety, OCR).
EXPECTED = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ0123456789 \n.,;:!?-()\"'"
)


def explore(source: str) -> None:
    df = load_source(source)
    print(f"\n{'=' * 70}\n{source}\n{'=' * 70}")
    print(f"dokumentów: {df.height:,}   kolumny: {df.columns}")

    # Długość w znakach (nie w bajtach: „ą” to 1 znak, ale 2 bajty w UTF-8).
    lengths = df["text"].str.len_chars()
    n_bytes = df["text"].str.len_bytes().sum()
    print(
        f"znaków łącznie: {lengths.sum():,}   bajtów UTF-8: {n_bytes:,}"
        f"   (bajtów/znak: {n_bytes / lengths.sum():.3f})"
    )
    print(
        f"długość dokumentu [znaki]: min {lengths.min():,} · mediana {int(lengths.median()):,}"
        f" · średnia {int(lengths.mean()):,} · p99 {int(lengths.quantile(0.99)):,}"
        f" · max {lengths.max():,}"
    )

    # Inwentarz znaków na całym tekście źródła.
    chars = Counter()
    for text in df["text"]:
        chars.update(text)
    total = sum(chars.values())
    unexpected = [(c, n) for c, n in chars.most_common() if c not in EXPECTED]
    print(f"\nunikalnych znaków: {len(chars):,}   spoza listy „oczywistych”: {len(unexpected):,}")
    print("najczęstsze nieoczywiste (znak · nazwa Unicode · ile · udział):")
    for c, n in unexpected[:20]:
        name = unicodedata.name(c, "<brak nazwy>")
        print(f"  {c!r:8} {name:40} {n:>10,}  {n / total:.4%}")
    singletons = sum(1 for _, n in chars.items() if n == 1)
    print(f"znaków występujących dokładnie raz: {singletons}")

    print("\nlosowe fragmenty (pierwsze 300 znaków):")
    random.seed(0)
    for i in random.sample(range(df.height), 2):
        row = df.row(i, named=True)
        print(f"\n--- id={row['id']} · license={row['license']}")
        print(row["text"][:300])


if __name__ == "__main__":
    pl.Config.set_tbl_rows(20)
    for source in SOURCES:
        explore(source)
