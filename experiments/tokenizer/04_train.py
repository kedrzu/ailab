"""Krok 3c: pełny trening trzech wariantów BPE na nkjp1m, po 1000 scaleń.

Uruchomienie:  uv run python experiments/tokenizer/04_train.py
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ailab.bpe import initial_sequences, train
from ailab.data import load_source

VARIANTS = {
    "slowa_znaki": dict(split=True, unit="chars"),
    "bez_podzialu_znaki": dict(split=False, unit="chars"),
    "slowa_bajty": dict(split=True, unit="bytes"),
}
NUM_MERGES = 1000
CHECKPOINTS = (10, 100, 1000)
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "tokenizer" / "bpe"


def show(symbol) -> str:
    if isinstance(symbol, bytes):
        try:
            return repr(symbol.decode("utf-8"))
        except UnicodeDecodeError:
            return "[" + " ".join(f"{b:02X}" for b in symbol) + "]"
    return repr(symbol)


def to_json(symbol):
    """JSON nie ma typu bytes, więc bajty zapisujemy szesnastkowo."""
    return symbol.hex() if isinstance(symbol, bytes) else symbol


def run(name: str) -> str:
    params = VARIANTS[name]
    texts = load_source("nkjp1m")["text"].to_list()
    n_chars = sum(len(t) for t in texts)
    sequences = initial_sequences(texts, **params)

    # Pełne 256 bajtów gwarantuje, że każdy tekst da się zakodować, nawet z nieznanymi znakami.
    if params["unit"] == "bytes":
        alphabet = [bytes([b]) for b in range(256)]
    else:
        alphabet = sorted({symbol for seq in sequences for symbol in seq})

    lines = [f"\n{name}\n{'=' * len(name)}", f"alfabet: {len(alphabet)}"]
    start = time.perf_counter()

    def on_merge(step, pair, current):
        if step in CHECKPOINTS:
            tokens = sum(len(seq) * count for seq, count in current.items())
            lines.append(
                f"po {step:>4} scaleniach: słownik {len(alphabet) + step:,} · tokenów {tokens:,}"
                f" · {n_chars / tokens:.2f} znaku/token · {time.perf_counter() - start:.0f}s"
            )

    merges = train(sequences, NUM_MERGES, on_merge)
    tokens = [a + b for a, b in merges]
    longest = sorted(tokens, key=len, reverse=True)[:15]
    lines.append("scalenia 1-20:     " + " ".join(show(t) for t in tokens[:20]))
    lines.append("scalenia 491-510:  " + " ".join(show(t) for t in tokens[490:510]))
    lines.append("scalenia 981-1000: " + " ".join(show(t) for t in tokens[980:1000]))
    lines.append("najdłuższe tokeny: " + " ".join(show(t) for t in longest))
    if params["unit"] == "bytes":
        broken = sum(1 for t in tokens if not is_valid_utf8(t))
        lines.append(f"tokeny niebędące poprawnym UTF-8: {broken} z {len(tokens)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(
        json.dumps(
            {
                **params,
                "alphabet": [to_json(s) for s in alphabet],
                "merges": [[to_json(a), to_json(b)] for a, b in merges],
            },
            ensure_ascii=False,
            indent=0,
        )
    )
    return "\n".join(lines)


def is_valid_utf8(token: bytes) -> bool:
    try:
        token.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=len(VARIANTS)) as pool:
        for report in pool.map(run, VARIANTS):
            print(report, flush=True)
