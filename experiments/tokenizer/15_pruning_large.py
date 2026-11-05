"""Usuwanie tokenów na próbce zbalansowanej z 26 źródeł DynaWord: 3 warianty na 3 podziałach.

Warianty porównujemy parami, na tym samym podziale. Oprócz treningu i testu mierzymy każde źródło osobno,
żeby zobaczyć, czy zysk dotyczy wszystkich rodzajów tekstu. Wyniki zapisujemy po każdym podziale.

Uruchomienie:  uv run python experiments/tokenizer/15_pruning_large.py
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bpe_rs

from ailab.corpus import frequent_chars, is_word, load_sample, merged

CAP_TOKENS = 20_000_000
SEEDS = range(3)
TARGET_VOCAB = 30_000
MIN_CHAR_COUNT = 1_000
VARIANTS = {"bez poprawki": None, "reguła 50%": "half", "reguła L < f_best": "best"}
# Siatka zaczyna się powyżej alfabetu, a każdy rozmiar ponad nim pada w treningu.
START_VOCAB, NUM_POINTS = 500, 40
VOCAB_POINTS = sorted(
    {round(START_VOCAB * (TARGET_VOCAB / START_VOCAB) ** (i / (NUM_POINTS - 1))) for i in range(NUM_POINTS)}
)
OUT_FILE = Path(__file__).resolve().parents[2] / "data" / "tokenizer" / "pruning" / "large.json"


def tokens_on_grid(result: dict, chunks: list[tuple[str, int]]) -> list[int]:
    rows = bpe_rs.replay_curve(chunks, result["tokens"], result["parents"], result["events"], VOCAB_POINTS)
    by_vocab = {row[0]: row[1] for row in rows}
    return [by_vocab[vocab] for vocab in VOCAB_POINTS]


def run_variant(name: str, rule: str | None, chunks: list, sets: dict, by_source: dict) -> dict:
    started = time.perf_counter()
    result = bpe_rs.train_events(chunks, TARGET_VOCAB, rule)
    trained = time.perf_counter() - started
    assert result["text_preserved"] and result["vocab_size"] == TARGET_VOCAB, name
    assert not result["hit_merge_cap"], name

    started = time.perf_counter()
    tokens = {set_name: tokens_on_grid(result, set_chunks) for set_name, set_chunks in sets.items()}
    source_tokens = {source: tokens_on_grid(result, set_chunks) for source, set_chunks in by_source.items()}
    replayed = time.perf_counter() - started

    # Odtworzenie całego treningu musi trafić dokładnie w wynik treningu.
    full = bpe_rs.replay_curve(chunks, result["tokens"], result["parents"], result["events"], [])
    assert full[-1][1] == result["total_tokens"], name

    print(f"  {name}: trening {trained:.0f} s, odtworzenia {replayed:.0f} s", flush=True)
    return {
        "train_seconds": round(trained, 1),
        "replay_seconds": round(replayed, 1),
        "merges": result["merges"],
        "splits": result["splits"],
        "restores": result["restores"],
        "removed_for_good": len(result["tokens"]) - TARGET_VOCAB,
        "tokens": tokens,
        "source_tokens": source_tokens,
    }


def run_split(seed: int) -> dict:
    sample = load_sample(CAP_TOKENS, seed)
    train_all = merged(sample.train)
    alphabet = frequent_chars(train_all, MIN_CHAR_COUNT)
    assert len(alphabet) < START_VOCAB, len(alphabet)
    # Słowa z rzadkim znakiem nie biorą udziału w treningu. W teście taki znak to jeden token.
    kept = {chunk: count for chunk, count in train_all.items() if set(chunk) <= alphabet}
    chunks = list(kept.items())

    test_all = merged(sample.test)
    sets = {
        "train": [(c, n) for c, n in chunks if is_word(c)],
        "test_seen": [(c, n) for c, n in test_all.items() if is_word(c) and c in kept],
        "test_unseen": [(c, n) for c, n in test_all.items() if is_word(c) and c not in kept],
    }
    by_source = {source: [(c, n) for c, n in counter.items() if is_word(c)] for source, counter in sample.test.items()}

    started = time.perf_counter()
    with ThreadPoolExecutor(len(VARIANTS)) as pool:
        futures = {
            name: pool.submit(run_variant, name, rule, chunks, sets, by_source) for name, rule in VARIANTS.items()
        }
        variants = {name: future.result() for name, future in futures.items()}

    return {
        "seed": seed,
        "alphabet": len(alphabet),
        "train_occurrences": sum(train_all.values()),
        "train_unique": len(train_all),
        "kept_unique": len(kept),
        "words": {name: sum(n for _, n in set_chunks) for name, set_chunks in sets.items()},
        "source_words": {source: sum(n for _, n in set_chunks) for source, set_chunks in by_source.items()},
        "wall_seconds": round(time.perf_counter() - started, 1),
        "variants": variants,
    }


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    splits = []
    for seed in SEEDS:
        print(f"podział {seed}", flush=True)
        split = run_split(seed)
        splits.append(split)
        print(f"  pełny pomiar: {split['wall_seconds']} s", flush=True)
        OUT_FILE.write_text(json.dumps({"vocab": VOCAB_POINTS, "splits": splits}, ensure_ascii=False))


if __name__ == "__main__":
    main()
