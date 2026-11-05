"""Kalibracja czasu: jak duża próbka zmieści się w budżecie 10 minut na pełny pomiar?

Na kilku limitach tokenów na źródło mierzy czas treningu i odtworzeń, dla wersji bez poprawki i reguły
50%, która robi najwięcej pracy. Warianty liczą się równolegle, tak jak w docelowym pomiarze.

Uruchomienie:  uv run python experiments/tokenizer/14_calibrate.py
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bpe_rs

from ailab.corpus import frequent_chars, is_word, load_sample, merged

CAPS = [2_000_000, 5_000_000, 20_000_000]
SEED = 0
TARGET_VOCAB = 30_000
MIN_CHAR_COUNT = 1_000
VARIANTS = {"bez poprawki": None, "reguła 50%": "half"}
# Siatka zaczyna się powyżej alfabetu, a każdy rozmiar ponad nim pada w treningu.
START_VOCAB, NUM_POINTS = 500, 40
VOCAB_POINTS = sorted(
    {round(START_VOCAB * (TARGET_VOCAB / START_VOCAB) ** (i / (NUM_POINTS - 1))) for i in range(NUM_POINTS)}
)
OUT_FILE = Path(__file__).resolve().parents[2] / "data" / "tokenizer" / "calibration.json"


def fertility_replay(result: dict, chunks: list[tuple[str, int]]) -> float:
    rows = bpe_rs.replay_curve(chunks, result["tokens"], result["parents"], result["events"], VOCAB_POINTS)
    return rows[-1][1] / sum(n for _, n in chunks)


def run_variant(name: str, rule: str | None, chunks, sets: dict, by_source: dict) -> dict:
    started = time.perf_counter()
    result = bpe_rs.train_events(chunks, TARGET_VOCAB, rule)
    trained = time.perf_counter() - started
    assert result["text_preserved"] and result["vocab_size"] == TARGET_VOCAB, name
    assert not result["hit_merge_cap"], name

    started = time.perf_counter()
    fertility = {set_name: fertility_replay(result, chunks_) for set_name, chunks_ in sets.items()}
    replay_main = time.perf_counter() - started

    started = time.perf_counter()
    per_source = {source: fertility_replay(result, chunks_) for source, chunks_ in by_source.items()}
    replay_sources = time.perf_counter() - started

    return {
        "variant": name,
        "train_seconds": round(trained, 1),
        "replay_main_seconds": round(replay_main, 1),
        "replay_sources_seconds": round(replay_sources, 1),
        "merges": result["merges"],
        "splits": result["splits"],
        "restores": result["restores"],
        "removed_for_good": len(result["tokens"]) - TARGET_VOCAB,
        "fertility": fertility,
        "per_source": per_source,
    }


def main() -> None:
    report = []
    for cap in CAPS:
        started = time.perf_counter()
        sample = load_sample(cap, SEED)
        counted = time.perf_counter() - started

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
        by_source = {
            source: [(c, n) for c, n in counter.items() if is_word(c)] for source, counter in sample.test.items()
        }
        info = {
            "cap": cap,
            "count_seconds": round(counted, 1),
            "train_occurrences": sum(train_all.values()),
            "train_unique": len(train_all),
            "kept_unique": len(kept),
            "alphabet": len(alphabet),
            "unseen_share": sum(n for _, n in sets["test_unseen"])
            / (sum(n for _, n in sets["test_seen"]) + sum(n for _, n in sets["test_unseen"])),
        }
        print(json.dumps(info, ensure_ascii=False), flush=True)

        started = time.perf_counter()
        with ThreadPoolExecutor(len(VARIANTS)) as pool:
            futures = [
                pool.submit(run_variant, name, rule, chunks, sets, by_source) for name, rule in VARIANTS.items()
            ]
            variants = [future.result() for future in futures]
        info["wall_seconds"] = round(time.perf_counter() - started, 1)
        info["variants"] = variants
        for row in variants:
            print(json.dumps({k: v for k, v in row.items() if k != "per_source"}, ensure_ascii=False), flush=True)
        print(f"limit {cap:,}: pełny pomiar {info['wall_seconds']} s\n", flush=True)
        report.append(info)
        OUT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
