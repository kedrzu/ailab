"""Czy efekt usuwania tokenów powtarza się na innych podziałach trening/test?

Każdy podział to inne losowe przypisanie tekstów źródłowych do treningu i testu. Warianty porównujemy
parami, na tym samym podziale: różnica między podziałami jest większa niż efekt, więc porównywanie
średnich z różnych podziałów by go zagłuszyło.

Podział 0 to ten z `09_pruning.py`, więc musi odtworzyć tamte liczby.

Uruchomienie:  uv run python experiments/tokenizer/11_pruning_splits.py
"""

import json
import math
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bpe_rs

from ailab.bpe import split_words
from ailab.data import load_source, split_by_source_text

SEEDS = range(10)
# Rust zwalnia blokadę Pythona, więc wątki naprawdę liczą równolegle. Każdy podział jest niezależny.
WORKERS = 6
TARGET_VOCAB = 30_000
RULES = {"bez poprawki": None, "reguła 50%": "half", "reguła L < f_best": "best"}
MEASURES = ("train", "test", "test_seen", "test_unseen")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "tokenizer" / "pruning"

# Siatka nie zależy od alfabetu, który na każdym podziale jest trochę inny. Zaczyna się powyżej
# każdego alfabetu, a każdy rozmiar ponad nim pada w treningu, więc pomiar jest wszędzie w tym samym
# miejscu. Punkty 10 154 i 30 000 pozwalają porównać wynik z poprzednim eksperymentem.
START_VOCAB, NUM_POINTS = 200, 60
VOCAB_POINTS = sorted(
    {round(START_VOCAB * (TARGET_VOCAB / START_VOCAB) ** (i / (NUM_POINTS - 1))) for i in range(NUM_POINTS)}
    | {10_154}
)


def is_word(chunk: str) -> bool:
    return any(c.isalpha() for c in chunk)


def count_chunks(texts: list[str]) -> Counter:
    counts = Counter()
    for text in texts:
        counts.update(split_words(text))
    return counts


def run_split(seed: int, ids: list[str], texts: list[str]) -> dict:
    train_texts, test_texts = split_by_source_text(ids, texts, seed=seed)
    train_counts, test_counts = count_chunks(train_texts), count_chunks(test_texts)

    all_train = list(train_counts.items())
    sets = {
        "train": [(c, n) for c, n in all_train if is_word(c)],
        "test_seen": [(c, n) for c, n in test_counts.items() if is_word(c) and c in train_counts],
        "test_unseen": [(c, n) for c, n in test_counts.items() if is_word(c) and c not in train_counts],
    }
    words = {name: sum(n for _, n in chunks) for name, chunks in sets.items()}
    alphabet_size = len({c for chunk, _ in all_train for c in chunk})
    assert alphabet_size < START_VOCAB, alphabet_size

    variants = {}
    for name, rule in RULES.items():
        result = bpe_rs.train_events(all_train, TARGET_VOCAB, rule)
        assert result["text_preserved"] and result["vocab_size"] == TARGET_VOCAB, (seed, name)
        assert not result["hit_merge_cap"], (seed, name)

        tokens = {}
        for set_name, chunks in sets.items():
            rows = bpe_rs.replay_curve(chunks, result["tokens"], result["parents"], result["events"], VOCAB_POINTS)
            tokens[set_name] = {row[0]: row[1] for row in rows}

        series = {measure: [] for measure in MEASURES}
        for vocab in VOCAB_POINTS:
            seen, unseen = tokens["test_seen"][vocab], tokens["test_unseen"][vocab]
            series["train"].append(tokens["train"][vocab] / words["train"])
            series["test"].append((seen + unseen) / (words["test_seen"] + words["test_unseen"]))
            series["test_seen"].append(seen / words["test_seen"])
            series["test_unseen"].append(unseen / words["test_unseen"])
        variants[name] = {
            "series": series,
            "merges": result["merges"],
            "splits": result["splits"],
            "restores": result["restores"],
            "removed_for_good": len(result["tokens"]) - alphabet_size - (TARGET_VOCAB - alphabet_size),
        }
    print(f"podział {seed} gotowy", flush=True)
    return {
        "seed": seed,
        "alphabet": alphabet_size,
        "words": words,
        "unseen_share": words["test_unseen"] / (words["test_seen"] + words["test_unseen"]),
        "variants": variants,
    }


def main() -> None:
    df = load_source("nkjp1m")
    ids, texts = df["id"].to_list(), df["text"].to_list()

    started = time.perf_counter()
    with ThreadPoolExecutor(WORKERS) as pool:
        splits = list(pool.map(lambda seed: run_split(seed, ids, texts), SEEDS))
    print(f"czas: {time.perf_counter() - started:.0f} s")

    # Podział 0 musi dać to samo co `09_pruning.py`.
    reference = json.loads((DATA_DIR / "metrics.json").read_text())
    for name in RULES:
        old = {row["vocab"]: row for row in reference[name]}
        new = splits[0]["variants"][name]["series"]
        for vocab in (10_154, 30_000):
            position = VOCAB_POINTS.index(vocab)
            for measure in MEASURES:
                assert abs(old[vocab][measure] - new[measure][position]) < 1e-9, (name, vocab, measure)
    print("podział 0: zgodny z poprzednim eksperymentem")

    (DATA_DIR / "splits.json").write_text(
        json.dumps({"vocab": VOCAB_POINTS, "splits": splits}, ensure_ascii=False, indent=0)
    )

    print("\nalfabet:", [s["alphabet"] for s in splits])
    print("udział słów niewidzianych w teście [%]:", [round(100 * s["unseen_share"], 1) for s in splits])
    for name in RULES:
        if RULES[name] is None:
            continue
        print(f"\n{name}: zmiana względem wersji bez poprawki [%], po podziałach")
        print("  słownik | zbiór        | mediana |     min |     max | poniżej zera")
        for vocab in (500, 1_000, 2_000, 5_630, 10_154, 20_000, 30_000):
            # Bierzemy najbliższy punkt siatki.
            position = min(range(len(VOCAB_POINTS)), key=lambda i: abs(math.log(VOCAB_POINTS[i] / vocab)))
            for measure in MEASURES:
                diffs = [
                    100 * (s["variants"][name]["series"][measure][position]
                           / s["variants"]["bez poprawki"]["series"][measure][position] - 1)
                    for s in splits
                ]
                below = sum(d < 0 for d in diffs)
                print(
                    f"  {VOCAB_POINTS[position]:>7} | {measure:<12} | {statistics.median(diffs):+7.3f} | "
                    f"{min(diffs):+7.3f} | {max(diffs):+7.3f} | {below}/{len(diffs)}"
                )
        # Ile punktów siatki ma poprawę na wszystkich podziałach naraz.
        for measure in MEASURES:
            all_below = sum(
                all(
                    s["variants"][name]["series"][measure][i] < s["variants"]["bez poprawki"]["series"][measure][i]
                    for s in splits
                )
                for i in range(len(VOCAB_POINTS))
            )
            print(f"  {measure}: punkty siatki z poprawą na wszystkich podziałach: {all_below}/{len(VOCAB_POINTS)}")


if __name__ == "__main__":
    main()
