"""Krok 3a: punkt startowy trzech wariantów BPE na nkjp1m.

Nic jeszcze nie scalamy. Sprawdzamy, ile symboli i ile ciągów ma każdy wariant, bo od
tego zależy koszt jednej rundy treningu.

Uruchomienie:  uv run python experiments/tokenizer/02_pretokenize.py
"""

import time

from ailab.bpe import initial_sequences, split_words
from ailab.data import load_source

VARIANTS = {
    "słowa + znaki": dict(split=True, unit="chars"),
    "bez podziału + znaki": dict(split=False, unit="chars"),
    "słowa + bajty": dict(split=True, unit="bytes"),
}

texts = load_source("nkjp1m")["text"].to_list()

# Podział ma być bezstratny, inaczej tokenizer nie odtworzy tekstu.
assert all("".join(split_words(t)) == t for t in texts)
print("podział na słowa jest bezstratny na wszystkich dokumentach\n")

print("przykład:", split_words("Pies gryzie kota, a w 2024 r. kot – psa!\n\nKoniec."), "\n")

header = f"{'wariant':24}{'alfabet':>9}{'ciągów':>11}{'symboli w tekście':>20}{'praca na rundę':>17}{'czas':>8}"
print(header)
print("-" * len(header))
for name, params in VARIANTS.items():
    start = time.perf_counter()
    sequences = initial_sequences(texts, **params)
    elapsed = time.perf_counter() - start
    alphabet = {symbol for seq in sequences for symbol in seq}
    total = sum(len(seq) * count for seq, count in sequences.items())
    # Praca na rundę: tyle par trzeba obejrzeć, licząc każdy unikalny ciąg raz.
    work = sum(len(seq) for seq in sequences)
    print(f"{name:24}{len(alphabet):>9,}{len(sequences):>11,}{total:>20,}{work:>17,}{elapsed:>7.2f}s")

words = initial_sequences(texts, split=True, unit="chars")
top = sorted(words.items(), key=lambda kv: -kv[1])[:15]
print("\nnajczęstsze kawałki:", [("".join(seq), count) for seq, count in top])
