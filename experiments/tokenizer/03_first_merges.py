"""Krok 3b: pierwsze 5 scaleń w trzech wariantach BPE na nkjp1m.

Uruchomienie:  uv run python experiments/tokenizer/03_first_merges.py
"""

import time

from ailab.bpe import count_pairs, initial_sequences, merge_pair, most_frequent_pair
from ailab.data import load_source

VARIANTS = {
    "słowa + znaki": dict(split=True, unit="chars"),
    "bez podziału + znaki": dict(split=False, unit="chars"),
    "słowa + bajty": dict(split=True, unit="bytes"),
}
ROUNDS = 5


def show(symbol) -> str:
    """Bajty pokazujemy szesnastkowo: kawałek litery nie musi być poprawnym UTF-8."""
    if isinstance(symbol, bytes):
        return "[" + " ".join(f"{b:02X}" for b in symbol) + "]" + repr(symbol.decode("utf-8", "replace"))
    return repr(symbol)


def total_symbols(sequences: dict[tuple, int]) -> int:
    return sum(len(seq) * count for seq, count in sequences.items())


texts = load_source("nkjp1m")["text"].to_list()

for name, params in VARIANTS.items():
    sequences = initial_sequences(texts, **params)
    print(f"\n{name}\n{'-' * len(name)}")
    print(f"symboli na starcie: {total_symbols(sequences):,}")
    for round_number in range(1, ROUNDS + 1):
        start = time.perf_counter()
        pairs = count_pairs(sequences)
        pair = most_frequent_pair(pairs)
        before = total_symbols(sequences)
        sequences = merge_pair(sequences, pair)
        elapsed = time.perf_counter() - start
        drop = before - total_symbols(sequences)
        # Licznik pary to górna granica: w „aaa” para występuje 2 razy, a scala się raz.
        assert drop <= pairs[pair]
        print(
            f"{round_number}. {show(pair[0])} + {show(pair[1])}  licznik {pairs[pair]:,}"
            f" · spadek symboli {drop:,} · {elapsed:.2f}s"
        )
