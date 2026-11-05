"""Próbka zbalansowana z wielu źródeł DynaWord: limit na źródło i podział 75/25 w każdym z nich.

Limit na źródło jest po to, żeby wielkie źródła prawne nie zdominowały treningu. Dzielimy osobno w każdym
źródle, żeby test miał ten sam skład co trening i żeby dało się mierzyć wynik na pojedynczym źródle.

Zliczanie słów jest najwolniejsze i nie zależy od wariantu treningu, więc zapisujemy je na dysku.
"""

import pickle
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ailab.bpe import split_words
from ailab.data import load_source

# `ted_pl` pomijamy: to szablonowe ogłoszenia, a karta datasetu podaje przy nich „no licence granted”.
SOURCES = """eurlex wikipedia parliamentary biblioteka_nauki dziennik_ustaw plwiki_user_talk
sejm_committee_transcripts wolne_lektury govpl plwiki_talk sejm_api sejm_api_2011_2022 eltec_pol nonsa
wikibooks wikivoyage wikinews global_voices nkjp1m studia_bas wiadomosci_statystyczne_pl rock_pollub_pl
edukacja_medialna_pl open_agh_chemistry_pl open_icm_pl wiktionary_examples""".split()

# Tylko tu ID wskazuje wspólny tekst lub posiedzenie (`<źródło>_<grupa>_<numer>`). W pozostałych źródłach
# przedrostek nic nie znaczy, a np. w `biblioteka_nauki` zlepiłby prawie cały zbiór w jedną grupę.
GROUPED_SOURCES = {"nkjp1m", "sejm_committee_transcripts"}

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "tokenizer" / "cache"


def split_groups(groups: list[str], test_fraction: float, seed: int) -> set[str]:
    """Losuje grupy do testu, aż obejmą żądaną część dokumentów."""
    sizes = Counter(groups)
    names = sorted(sizes)
    random.Random(seed).shuffle(names)
    test_groups: set[str] = set()
    in_test = 0
    for name in names:
        if in_test >= test_fraction * len(groups):
            break
        test_groups.add(name)
        in_test += sizes[name]
    return test_groups


def sample_positions(token_counts: list[int], cap_tokens: int, seed: int) -> list[int]:
    """Losowe dokumenty, aż ich suma tokenów przekroczy limit."""
    order = list(range(len(token_counts)))
    random.Random(seed).shuffle(order)
    chosen = []
    total = 0
    for position in order:
        if total >= cap_tokens:
            break
        chosen.append(position)
        total += token_counts[position]
    return chosen


def _count_source(job: tuple[str, int, int]) -> tuple[str, Counter, Counter]:
    source, cap_tokens, seed = job
    df = load_source(source).select("id", "text", "token_count")
    ids = df["id"].to_list()
    texts = df["text"].to_list()
    chosen = sample_positions(df["token_count"].to_list(), cap_tokens, seed)

    if source in GROUPED_SOURCES:
        groups = [ids[p].rsplit("_", 1)[0] for p in chosen]
    else:
        groups = [ids[p] for p in chosen]
    test_groups = split_groups(groups, 0.25, seed)

    train, test = Counter(), Counter()
    for position, group in zip(chosen, groups):
        (test if group in test_groups else train).update(split_words(texts[position]))
    return source, train, test


@dataclass
class Sample:
    """Liczniki kawałków tekstu: {źródło: {kawałek: liczba wystąpień}}."""

    train: dict[str, Counter]
    test: dict[str, Counter]


def load_sample(cap_tokens: int, seed: int, workers: int = 6) -> Sample:
    """Zwraca próbkę; pierwszy raz liczy ją równolegle, kolejne razy czyta z dysku."""
    path = CACHE_DIR / f"sample_cap{cap_tokens}_seed{seed}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())

    jobs = [(source, cap_tokens, seed) for source in SOURCES]
    train, test = {}, {}
    with ProcessPoolExecutor(workers) as pool:
        for source, source_train, source_test in pool.map(_count_source, jobs):
            train[source] = source_train
            test[source] = source_test
    sample = Sample(train, test)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
    return sample


def merged(counters: dict[str, Counter]) -> Counter:
    total = Counter()
    for counter in counters.values():
        total.update(counter)
    return total


def frequent_chars(train: Counter, min_count: int) -> set[str]:
    """Znaki, które zostają w alfabecie."""
    chars = Counter()
    for chunk, count in train.items():
        for char in chunk:
            chars[char] += count
    return {char for char, count in chars.items() if count >= min_count}


def is_word(chunk: str) -> bool:
    """Tak samo jak w poprzednich eksperymentach: fertility liczymy na kawałkach z literami."""
    return any(c.isalpha() for c in chunk)
