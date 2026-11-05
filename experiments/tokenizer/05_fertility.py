"""Fertility w trakcie treningu BPE: trening vs test na nkjp1m.

Fertility = średnia liczba tokenów na jedno słowo. Liczymy tylko kawałki z literami, bo
kropka to zawsze jeden token i sztucznie zaniżałaby średnią.

Parę do scalenia wybieramy wyłącznie na zbiorze treningowym. Na teście tylko stosujemy te
same scalenia w tej samej kolejności, a to jest dokładnie kodowanie.

Uruchomienie:  uv run python experiments/tokenizer/05_fertility.py
"""

import json
import time
from pathlib import Path

from ailab.bpe import initial_sequences, merge_pair, train
from ailab.data import load_source, split_by_source_text

NUM_MERGES = 10_000
NUM_POINTS = 60
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "tokenizer" / "fertility"


def checkpoints(num_merges: int, num_points: int) -> set[int]:
    """Punkty pomiaru rozłożone logarytmicznie, bo na początku krzywa zmienia się najszybciej."""
    points = {round((num_merges + 1) ** (i / (num_points - 1))) - 1 for i in range(num_points)}
    return points | {0, num_merges}


def is_word(chunk: str) -> bool:
    return any(c.isalpha() for c in chunk)


def fertility(sequences: dict[tuple, int], keep=lambda chunk: True) -> float:
    """Tokeny na słowo, ważone liczbą wystąpień: częste słowa liczą się częściej, jak w tekście."""
    tokens = words = 0
    for seq, count in sequences.items():
        chunk = "".join(seq)
        if is_word(chunk) and keep(chunk):
            tokens += len(seq) * count
            words += count
    return tokens / words


def main() -> None:
    df = load_source("nkjp1m")
    train_texts, test_texts = split_by_source_text(df["id"].to_list(), df["text"].to_list())

    train_seqs = initial_sequences(train_texts, split=True, unit="chars")
    test_seqs = initial_sequences(test_texts, split=True, unit="chars")

    # Znaki spoza alfabetu treningu nigdy się nie scalą, więc każdy liczy się jako jeden token.
    alphabet = {symbol for seq in train_seqs for symbol in seq}
    seen_chunks = {"".join(seq) for seq in train_seqs}
    points = checkpoints(NUM_MERGES, NUM_POINTS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    state = {"test": test_seqs}
    merged_tokens = []

    with open(OUT_DIR / "metrics.jsonl", "w") as log:

        def record(step: int, current_train: dict, token: str | None) -> None:
            current_test = state["test"]
            row = {
                "merges": step,
                "vocab": len(alphabet) + step,
                "token": token,
                "train": fertility(current_train),
                "test": fertility(current_test),
                "test_seen": fertility(current_test, lambda c: c in seen_chunks),
                "test_unseen": fertility(current_test, lambda c: c not in seen_chunks),
                "seconds": time.perf_counter() - start,
            }
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()  # żeby dało się podglądać postęp w trakcie
            print(row, flush=True)

        def on_merge(step: int, pair: tuple, current_train: dict) -> None:
            state["test"] = merge_pair(state["test"], pair)
            merged_tokens.append(pair[0] + pair[1])
            if step in points:
                record(step, current_train, merged_tokens[-1])

        print(f"dokumenty: trening {len(train_texts):,}, test {len(test_texts):,}; alfabet {len(alphabet)}")
        record(0, train_seqs, None)
        merges = train(train_seqs, NUM_MERGES, on_merge)

    (OUT_DIR / "merges.json").write_text(
        json.dumps({"alphabet": sorted(alphabet), "merges": merges}, ensure_ascii=False, indent=0)
    )
    print(f"gotowe: {len(merges):,} scaleń w {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
