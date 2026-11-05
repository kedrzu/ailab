"""Fertility z usuwaniem tokenów i bez niego: trening vs test na nkjp1m.

Tokeny wybieramy wyłącznie na treningu. Na każdym tekście odtwarzamy potem tę samą historię
scaleń i usunięć, więc wszystkie warianty są mierzone tym samym sposobem. To narzędzie pomiaru,
a nie decyzja o docelowym encoderze.

Fertility liczymy tylko na kawałkach z literami, tak jak w `05_fertility.py`. Wyniki zapisujemy
do pliku, żeby zmiana wyglądu wykresu nie wymagała ponownego treningu.

Uruchomienie:  uv run python experiments/tokenizer/09_pruning.py
"""

import json
import time
from collections import Counter
from pathlib import Path

import bpe_rs

from ailab.bpe import split_words
from ailab.data import load_source, split_by_source_text

TARGET_VOCAB = 30_000
NUM_POINTS = 80
RULES = {"bez poprawki": None, "reguła 50%": "half", "reguła L < f_best": "best"}

ROOT = Path(__file__).resolve().parents[2]
OLD_METRICS = ROOT / "data" / "tokenizer" / "fertility" / "metrics.jsonl"
OUT_DIR = ROOT / "data" / "tokenizer" / "pruning"
# Rozmiar z poprzedniego eksperymentu, żeby porównać wynik z liczbami, które już znamy.
REFERENCE_VOCAB = 10_154


def count_chunks(texts: list[str]) -> Counter:
    counts = Counter()
    for text in texts:
        counts.update(split_words(text))
    return counts


def is_word(chunk: str) -> bool:
    return any(c.isalpha() for c in chunk)


def vocab_points(alphabet_size: int) -> list[int]:
    """Punkty rozłożone logarytmicznie, bo na początku krzywa zmienia się najszybciej."""
    span = TARGET_VOCAB - alphabet_size
    merges = {round((span + 1) ** (i / (NUM_POINTS - 1))) - 1 for i in range(NUM_POINTS)}
    points = {alphabet_size + m for m in merges}
    return sorted(points | {REFERENCE_VOCAB})


def curve(result: dict, chunks: list[tuple[str, int]], points: list[int]) -> dict[int, tuple]:
    rows = bpe_rs.replay_curve(chunks, result["tokens"], result["parents"], result["events"], points)
    return {row[0]: row for row in rows}


def main() -> None:
    df = load_source("nkjp1m")
    train_texts, test_texts = split_by_source_text(df["id"].to_list(), df["text"].to_list())
    train_counts = count_chunks(train_texts)
    test_counts = count_chunks(test_texts)

    all_train = list(train_counts.items())
    train_words = [(c, n) for c, n in all_train if is_word(c)]
    test_seen = [(c, n) for c, n in test_counts.items() if is_word(c) and c in train_counts]
    test_unseen = [(c, n) for c, n in test_counts.items() if is_word(c) and c not in train_counts]
    words = {
        "train": sum(n for _, n in train_words),
        "test_seen": sum(n for _, n in test_seen),
        "test_unseen": sum(n for _, n in test_unseen),
    }
    alphabet_size = len({c for chunk, _ in all_train for c in chunk})
    points = vocab_points(alphabet_size)
    print(f"alfabet {alphabet_size}, punktów pomiaru {len(points)}, słowa {words}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    metrics = {}
    tokens_by_variant = {}
    for name, rule in RULES.items():
        started = time.perf_counter()
        result = bpe_rs.train_events(all_train, TARGET_VOCAB, rule)
        trained = time.perf_counter() - started
        assert result["text_preserved"], name
        assert result["vocab_size"] == TARGET_VOCAB, (name, result["vocab_size"])
        assert not result["hit_merge_cap"], name

        started = time.perf_counter()
        train_curve = curve(result, train_words, points)
        seen_curve = curve(result, test_seen, points)
        unseen_curve = curve(result, test_unseen, points)
        replayed = time.perf_counter() - started

        # Odtworzenie na całym treningu musi trafić dokładnie w wynik treningu.
        full_replay = bpe_rs.replay_curve(all_train, result["tokens"], result["parents"], result["events"], [])
        assert full_replay[-1][1] == result["total_tokens"], name

        rows = []
        for vocab in points:
            train_row = train_curve[vocab]
            seen_row, unseen_row = seen_curve[vocab], unseen_curve[vocab]
            test_tokens = seen_row[1] + unseen_row[1]
            rows.append(
                {
                    "vocab": vocab,
                    "merges": train_row[2],
                    "splits": train_row[3],
                    "train": train_row[1] / words["train"],
                    "test": test_tokens / (words["test_seen"] + words["test_unseen"]),
                    "test_seen": seen_row[1] / words["test_seen"],
                    "test_unseen": unseen_row[1] / words["test_unseen"],
                }
            )
        metrics[name] = rows

        final_strings = [result["tokens"][t] for t in result["final_vocab"]]
        created = len(result["tokens"]) - alphabet_size
        final_learned = TARGET_VOCAB - alphabet_size
        summary[name] = {
            "scalenia": result["merges"],
            "usunięcia": result["splits"],
            "powroty": result["restores"],
            "utworzone tokeny": created,
            "usunięte na stałe": created - final_learned,
            "usunięte na stałe [%]": 100 * (created - final_learned) / created,
            "tokeny o powtórzonym napisie": len(final_strings) - len(set(final_strings)),
            "tokeny w treningu": result["total_tokens"],
            "trening [s]": round(trained, 1),
            "odtworzenia [s]": round(replayed, 1),
        }
        tokens_by_variant[name] = {
            "final": final_strings,
            "removed": [
                result["tokens"][t]
                for t in sorted(set(range(alphabet_size, len(result["tokens"]))) - set(result["final_vocab"]))
            ],
        }
        print(name, summary[name], flush=True)

        if rule is None:
            # Bez usuwania musi wyjść dokładnie to samo co dotychczasowy trening.
            _, merges, *_ = bpe_rs.train_indexed(all_train, TARGET_VOCAB - alphabet_size)
            parents = result["parents"]
            tokens = result["tokens"]
            via_events = [(tokens[parents[t][0]], tokens[parents[t][1]]) for _, t in result["events"]]
            assert via_events == merges
            print("zgodne z train_indexed", flush=True)

            # Te same liczby co w poprzednim eksperymencie, przy tym samym rozmiarze słownika.
            old = {r["vocab"]: r for r in map(json.loads, OLD_METRICS.read_text().splitlines())}[REFERENCE_VOCAB]
            new = next(r for r in rows if r["vocab"] == REFERENCE_VOCAB)
            for key in ("train", "test", "test_seen", "test_unseen"):
                print(f"  {key}: stary {old[key]:.6f}, nowy {new[key]:.6f}", flush=True)
                assert abs(old[key] - new[key]) < 1e-9, key

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    (OUT_DIR / "tokens.json").write_text(json.dumps(tokens_by_variant, ensure_ascii=False, indent=0))

    print("\nsłownik | wariant | trening | test | widziane | niewidziane")
    base = {r["vocab"]: r for r in metrics["bez poprawki"]}
    for vocab in sorted({1_000, 3_000, REFERENCE_VOCAB, TARGET_VOCAB} & set(base) | {points[len(points) // 2], points[-1]}):
        for name in RULES:
            row = next(r for r in metrics[name] if r["vocab"] == vocab)
            diff = lambda key: 100 * (row[key] / base[vocab][key] - 1)
            print(
                f"{vocab:>7} | {name:<18} | {row['train']:.4f} ({diff('train'):+.2f}%) | "
                f"{row['test']:.4f} ({diff('test'):+.2f}%) | {row['test_seen']:.4f} ({diff('test_seen'):+.2f}%) | "
                f"{row['test_unseen']:.4f} ({diff('test_unseen'):+.2f}%)"
            )


if __name__ == "__main__":
    main()
