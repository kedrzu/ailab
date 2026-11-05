"""Sprawdzenie usuwania tokenów na małych przykładach, zanim użyjemy prawdziwych danych.

Oczekiwane liczby policzyliśmy wcześniej ręcznie. Ten skrypt nie pobiera żadnych danych.

Uruchomienie:  uv run python experiments/tokenizer/09_pruning_check.py
"""

import random

import bpe_rs

RULES = [None, "half", "best"]


def strings(result: dict, token_ids: list[int]) -> list[str]:
    return sorted(result["tokens"][token] for token in token_ids if result["parents"][token] is not None)


def replay_tokens(result: dict, chunks: list[tuple[str, int]]) -> int:
    """Liczba tokenów na tekście po całej historii treningu: ostatni wiersz to stan końcowy."""
    rows = bpe_rs.replay_curve(chunks, result["tokens"], result["parents"], result["events"], [])
    return rows[-1][1]


# 1. Przykład z rozmowy: budżet to 3 nowe tokeny ponad 10 znaków.
corpus = [("abcd", 100), ("ef", 30), ("gh", 20), ("ij", 10)]
expected_tokens = {None: 220, "half": 170, "best": 170}
expected_vocab = {
    None: ["bcd", "abcd", "cd"],
    "half": ["abcd", "ef", "gh"],
    "best": ["abcd", "ef", "gh"],
}
for rule in RULES:
    result = bpe_rs.train_events(corpus, 13, rule)
    print(rule, result["total_tokens"], strings(result, result["final_vocab"]), result["events"])
    assert result["total_tokens"] == expected_tokens[rule]
    assert result["vocab_size"] == 13
    assert sorted(strings(result, result["final_vocab"])) == sorted(expected_vocab[rule])
    assert result["text_preserved"]
print("1. przykład abcd: 220 bez poprawki, 170 z poprawką")

# 2. Koszt na nowym słowie: bez `cd` słowo `cde` ma 3 tokeny zamiast 2.
new_word = [("cde", 1)]
plain = bpe_rs.train_events(corpus, 13, None)
pruned = bpe_rs.train_events(corpus, 13, "half")
assert replay_tokens(plain, new_word) == 2
assert replay_tokens(pruned, new_word) == 3
print("2. nowe słowo `cde`: 2 tokeny bez poprawki, 3 z poprawką")

# 3. Bez usuwania nowa funkcja musi dać dokładnie te same scalenia co `train_indexed`.
polish = [("kot", 5), ("kota", 3), ("koty", 2), ("lot", 4), ("aaa", 1), ("aaaaa", 2)]
alphabet_size = len({c for word, _ in polish for c in word})
_, merges, *_ = bpe_rs.train_indexed(polish, 6)
result = bpe_rs.train_events(polish, alphabet_size + 6, None)
via_events = [
    (result["tokens"][result["parents"][token][0]], result["tokens"][result["parents"][token][1]])
    for _, token in result["events"]
]
assert via_events == merges
print("3. bez usuwania: te same scalenia co train_indexed")

# 4. Losowe korpusy: rozkładanie martwych rodziców nie może zgubić tekstu, a odtworzenie historii
#    musi dać dokładnie ten sam wynik co trening.
rng = random.Random(0)
restores = splits = 0
for trial in range(300):
    pieces = ["".join(rng.choices("abcd", k=rng.randint(1, 4))) for _ in range(8)]
    words = {"".join(rng.choices(pieces, k=rng.randint(1, 4))) for _ in range(rng.randint(3, 30))}
    random_corpus = [(word, rng.randint(1, 50)) for word in sorted(words)]
    size = len({c for word, _ in random_corpus for c in word})
    for rule in RULES:
        result = bpe_rs.train_events(random_corpus, size + rng.randint(1, 40), rule)
        assert result["text_preserved"], (trial, rule)
        assert not result["hit_merge_cap"], (trial, rule)
        assert len(result["final_vocab"]) == result["vocab_size"], (trial, rule)
        assert replay_tokens(result, random_corpus) == result["total_tokens"], (trial, rule)
        restores += result["restores"]
        splits += result["splits"]
assert restores > 0 and splits > 0
print(f"4. 300 losowych korpusów: tekst zachowany, odtworzenie zgodne ({splits} usunięć, {restores} powrotów)")
