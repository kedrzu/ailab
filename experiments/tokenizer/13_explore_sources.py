"""Oględziny źródeł DynaWord przed większym treningiem: formaty ID, alfabet, liczba słów.

Niczego nie trenuje. Z każdego źródła losuje dokumenty do limitu tokenów, żeby oglądać dokładnie to,
co potem trafi do treningu. Licznik `token_count` pochodzi z datasetu i służy tylko do cięcia.

Uruchomienie:  uv run python experiments/tokenizer/13_explore_sources.py
"""

import json
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from ailab.bpe import split_words
from ailab.data import load_source

CAP_TOKENS = 20_000_000
SEED = 0
WORKERS = 6
SOURCES = """eurlex wikipedia parliamentary biblioteka_nauki dziennik_ustaw plwiki_user_talk
sejm_committee_transcripts wolne_lektury govpl plwiki_talk sejm_api sejm_api_2011_2022 eltec_pol nonsa
wikibooks wikivoyage wikinews global_voices nkjp1m studia_bas wiadomosci_statystyczne_pl rock_pollub_pl
edukacja_medialna_pl open_agh_chemistry_pl open_icm_pl wiktionary_examples""".split()

OUT_FILE = Path(__file__).resolve().parents[2] / "data" / "tokenizer" / "sources_overview.json"


def explore(source: str) -> tuple[dict, Counter, Counter]:
    started = time.perf_counter()
    df = load_source(source).select("id", "text", "token_count")
    ids = df["id"].to_list()

    # Czy końcówka ID wygląda jak numer fragmentu? Tylko wtedy grupowanie po tekście ma sens.
    groups = Counter(doc_id.rsplit("_", 1)[0] for doc_id in ids)
    numeric_suffix = sum(doc_id.rsplit("_", 1)[-1].isdigit() for doc_id in ids)

    order = list(range(len(ids)))
    random.Random(SEED).shuffle(order)
    counts = df["token_count"].to_list()
    texts = df["text"].to_list()
    total = 0
    chosen = []
    for position in order:
        if total >= CAP_TOKENS:
            break
        chosen.append(position)
        total += counts[position]

    chars = Counter()
    words = Counter()
    for position in chosen:
        chars.update(texts[position])
        words.update(split_words(texts[position]))

    lengths = sorted(len(texts[position]) for position in chosen)
    row = {
        "source": source,
        "documents": len(ids),
        "tokens_all": sum(counts),
        "ids_sample": [ids[i] for i in random.Random(1).sample(range(len(ids)), min(4, len(ids)))],
        "numeric_suffix_share": numeric_suffix / len(ids),
        "groups_by_prefix": len(groups),
        "largest_group": groups.most_common(1)[0][1],
        "chosen_documents": len(chosen),
        "chosen_tokens": total,
        "chosen_chars": sum(lengths),
        "median_doc_chars": lengths[len(lengths) // 2],
        "max_doc_chars": lengths[-1],
        "alphabet": len(chars),
        "unique_chunks": len(words),
        "occurrences": sum(words.values()),
        "seconds": round(time.perf_counter() - started, 1),
    }
    return row, chars, words


def main() -> None:
    pl.Config.set_tbl_rows(40)
    all_chars = Counter()
    all_words = Counter()
    rows = []
    with ProcessPoolExecutor(WORKERS) as pool:
        for row, chars, words in pool.map(explore, SOURCES):
            rows.append(row)
            all_chars.update(chars)
            all_words.update(words)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    frequency = sorted(all_chars.values(), reverse=True)
    thresholds = {
        threshold: {
            "alphabet": sum(n >= threshold for n in frequency),
            "dropped_char_share": sum(n for n in frequency if n < threshold) / sum(frequency),
        }
        for threshold in (1, 10, 100, 1_000, 10_000, 100_000)
    }
    summary = {
        "chars_total": sum(frequency),
        "alphabet": len(frequency),
        "alphabet_by_threshold": thresholds,
        "unique_chunks": len(all_words),
        "occurrences": sum(all_words.values()),
        "top_chars_tail": [c for c, _ in all_chars.most_common()][150:210],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    OUT_FILE.write_text(json.dumps({"sources": rows, "summary": summary}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
