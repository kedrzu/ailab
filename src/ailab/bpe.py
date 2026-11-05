"""BPE (Byte Pair Encoding) od zera, w czystym Pythonie.

Celem jest zrozumienie algorytmu, nie szybkość. Tekst zamieniamy na ciągi symboli,
a potem w kolejnych rundach scalamy najczęstszą parę sąsiadów.

Warianty różnią się tylko punktem startowym, czyli `initial_sequences`: reszta algorytmu
nie wie, czy symbole to znaki, czy bajty, bo jedno i drugie skleja się przez `a + b`.
"""

import re
from collections import Counter
from collections.abc import Iterable

# Podział jak w GPT-2: spacja przykleja się do początku następnego kawałka, a litery, cyfry
# i interpunkcja idą osobno. Dzięki temu scalenia nie przekraczają granic słów.
# Moduł `re` nie ma \p{L}, więc „litera” to: \w bez cyfr i podkreślnika.
WORD_PATTERN = re.compile(r" ?[^\W\d_]+| ?\d+| ?(?:[^\w\s]|_)+|\s+(?!\S)|\s+")


def split_words(text: str) -> list[str]:
    """Tnie tekst na kawałki. Bezstratnie: złączenie kawałków daje oryginał."""
    return WORD_PATTERN.findall(text)


def to_symbols(chunk: str, unit: str) -> tuple:
    """Rozbija kawałek tekstu na symbole startowe: znaki (str) albo bajty UTF-8 (bytes)."""
    if unit == "chars":
        return tuple(chunk)
    if unit == "bytes":
        return tuple(bytes([b]) for b in chunk.encode("utf-8"))
    raise ValueError(f"nieznana jednostka: {unit}")


def initial_sequences(texts: Iterable[str], split: bool, unit: str) -> dict[tuple, int]:
    """Zwraca {ciąg symboli: ile razy wystąpił} — stan początkowy treningu.

    Liczniki to główna oszczędność: „ się” występuje dziesiątki tysięcy razy, ale wystarczy
    przetworzyć je raz i pomnożyć. Bez podziału (`split=False`) całym ciągiem jest dokument,
    więc prawie nic się nie powtarza i oszczędność znika.
    """
    chunks = Counter()
    for text in texts:
        chunks.update(split_words(text) if split else [text])
    return {to_symbols(chunk, unit): count for chunk, count in chunks.items()}


def count_pairs(sequences: dict[tuple, int]) -> Counter:
    """Liczy sąsiadujące pary symboli, każdą ważoną licznikiem ciągu."""
    pairs = Counter()
    for seq, count in sequences.items():
        for pair in zip(seq, seq[1:]):
            pairs[pair] += count
    return pairs


def most_frequent_pair(pairs: Counter) -> tuple:
    """Wybiera najczęstszą parę. Przy remisie decyduje sama para, żeby wynik był powtarzalny."""
    return max(pairs, key=lambda pair: (pairs[pair], pair))


def merge_pair(sequences: dict[tuple, int], pair: tuple) -> dict[tuple, int]:
    """Skleja każde wystąpienie pary w jeden symbol, idąc od lewej."""
    a, b = pair
    merged = a + b
    result = {}
    for seq, count in sequences.items():
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                new_seq.append(merged)
                i += 2  # oba symbole zużyte: w „aaa” para (a, a) scali się tylko raz
            else:
                new_seq.append(seq[i])
                i += 1
        result[tuple(new_seq)] = count
    return result


def train(sequences: dict[tuple, int], num_merges: int, on_merge=None) -> list[tuple]:
    """Uczy się listy scaleń. Kolejność jest częścią modelu: tak samo koduje się potem tekst.

    Celowo prosto: każda runda liczy wszystkie pary od nowa. To dlatego czysty Python jest
    wolny, a biblioteki aktualizują liczniki tylko tam, gdzie scalenie coś zmieniło.
    """
    merges = []
    for _ in range(num_merges):
        pairs = count_pairs(sequences)
        if not pairs:  # każdy ciąg jest już jednym symbolem, nie ma czego scalać
            break
        pair = most_frequent_pair(pairs)
        sequences = merge_pair(sequences, pair)
        merges.append(pair)
        if on_merge:
            on_merge(len(merges), pair, sequences)
    return merges
