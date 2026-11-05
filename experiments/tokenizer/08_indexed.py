"""Rust z indeksem „para → słowa”: zgodność z wersją naiwną i zysk czasu.

Uruchamiać, gdy nic innego nie obciąża procesora.

Uruchomienie:  uv run python experiments/tokenizer/08_indexed.py
"""

import json
import statistics
from collections import Counter
from pathlib import Path

import bpe_rs

from ailab.bpe import split_words
from ailab.data import load_source, split_by_source_text

ROOT = Path(__file__).resolve().parents[2]
PYTHON_MERGES = ROOT / "data" / "tokenizer" / "fertility" / "merges.json"
OUT_FILE = ROOT / "data" / "tokenizer" / "rust_indexed.json"
REPEATS = 3
SIZES = [10_000, 30_000]

df = load_source("nkjp1m")
train_texts, _ = split_by_source_text(df["id"].to_list(), df["text"].to_list())
counts = Counter()
for text in train_texts:
    counts.update(split_words(text))
chunks = list(counts.items())

# Przykład liczony wcześniej ręcznie.
tiny = [("kot", 5), ("kota", 3), ("koty", 2), ("lot", 4)]
expected = [("o", "t"), ("k", "ot"), ("l", "ot"), ("kot", "a")]
assert bpe_rs.train_indexed(tiny, 4)[1] == expected
# Nakładająca się para: „aaa” scala się raz, potem zostają dwa symbole.
assert bpe_rs.train_indexed([("aaa", 1)], 5)[1] == [("a", "a"), ("aa", "a")]
print("małe przykłady: zgodne")

reference = [tuple(pair) for pair in json.loads(PYTHON_MERGES.read_text())["merges"]]

results = []
for size in SIZES:
    runs = [bpe_rs.train_indexed(chunks, size) for _ in range(REPEATS)]
    median_run = sorted(runs, key=lambda run: run[2])[REPEATS // 2]
    merges = median_run[1]
    assert merges[:len(reference)] == reference[:size]
    row = {
        "merges": len(merges),
        "seconds": statistics.median(run[2] for run in runs),
        "all_seconds": [run[2] for run in runs],
        "phases": median_run[3],
    }
    results.append(row)
    print(row, flush=True)

# Python policzył tylko 10 000 scaleń, więc dalszą część sprawdzamy wersją naiwną w Ruście.
_, naive, naive_seconds, _ = bpe_rs.train(chunks, SIZES[-1], 6)
_, indexed, *_ = bpe_rs.train_indexed(chunks, SIZES[-1])
assert naive == indexed
print(f"{SIZES[-1]:,} scaleń: indeks zgodny z wersją naiwną (naiwna, 6 wątków: {naive_seconds:.1f}s)")

OUT_FILE.write_text(json.dumps({"runs": results, "naive_30k_6_threads_seconds": naive_seconds}, indent=2))
