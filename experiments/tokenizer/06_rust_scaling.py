"""Rust a Python: zgodność listy scaleń i skalowanie z liczbą wątków.

Uruchamiać, gdy nic innego nie obciąża procesora, inaczej czasy będą zafałszowane.

Uruchomienie:  uv run python experiments/tokenizer/06_rust_scaling.py
"""

import json
import statistics
import time
from collections import Counter
from pathlib import Path

import bpe_rs

from ailab.bpe import initial_sequences, split_words, train
from ailab.data import load_source, split_by_source_text

ROOT = Path(__file__).resolve().parents[2]
PYTHON_MERGES = ROOT / "data" / "tokenizer" / "fertility" / "merges.json"
OUT_FILE = ROOT / "data" / "tokenizer" / "rust_scaling.json"
NUM_MERGES = 10_000
BASELINE_MERGES = 1_000
THREADS = [1, 2, 4, 6, 12, 18]
REPEATS = 3

df = load_source("nkjp1m")
train_texts, _ = split_by_source_text(df["id"].to_list(), df["text"].to_list())
counts = Counter()
for text in train_texts:
    counts.update(split_words(text))
chunks = list(counts.items())
print(f"unikalnych kawałków: {len(chunks):,}")

# Najpierw zgodność: szybkość nic nie znaczy, jeśli wynik jest inny niż w Pythonie.
reference = [tuple(pair) for pair in json.loads(PYTHON_MERGES.read_text())["merges"]]
for threads in (1, max(THREADS)):
    _, merges, *_ = bpe_rs.train(chunks, NUM_MERGES, threads)
    first_difference = next((i for i, (a, b) in enumerate(zip(merges, reference)) if a != b), None)
    assert len(merges) == len(reference) and first_difference is None, (threads, first_difference)
    print(f"{threads:>2} wątków: lista {len(merges):,} scaleń identyczna jak w Pythonie")

# Sam Python, bez pomiaru testu: uczciwe porównanie na krótszym odcinku.
sequences = initial_sequences(train_texts, split=True, unit="chars")
start = time.perf_counter()
python_merges = train(sequences, BASELINE_MERGES)
python_seconds = time.perf_counter() - start
assert python_merges == reference[:BASELINE_MERGES]
rust_seconds = statistics.median(
    bpe_rs.train(chunks, BASELINE_MERGES, 1)[2] for _ in range(REPEATS)
)
print(
    f"{BASELINE_MERGES:,} scaleń: Python {python_seconds:.1f}s, Rust 1 wątek {rust_seconds:.2f}s,"
    f" {python_seconds / rust_seconds:.0f}x szybciej"
)

results = []
for threads in THREADS:
    runs = [bpe_rs.train(chunks, NUM_MERGES, threads) for _ in range(REPEATS)]
    times = [run[2] for run in runs]
    # Fazy bierzemy z biegu o medianowym czasie, żeby nie mieszać liczb z różnych biegów.
    median_run = sorted(runs, key=lambda run: run[2])[REPEATS // 2]
    results.append(
        {
            "threads": threads,
            "seconds": statistics.median(times),
            "all_seconds": times,
            "phases": median_run[3],
        }
    )
    speedup = results[0]["seconds"] / results[-1]["seconds"]
    print(f"{threads:>2} wątków: {results[-1]['seconds']:.2f}s · przyspieszenie {speedup:.2f}x", flush=True)

OUT_FILE.write_text(
    json.dumps(
        {
            "num_merges": NUM_MERGES,
            "python_baseline": {
                "merges": BASELINE_MERGES,
                "python_seconds": python_seconds,
                "rust_1_thread_seconds": rust_seconds,
            },
            "scaling": results,
        },
        indent=2,
    )
)
