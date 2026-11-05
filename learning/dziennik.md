# Dziennik nauki

Chronologiczny log sesji, najnowsze na górze. Profil (`profil.md`) to stan; dziennik to
historia, jak do niego doszliśmy.

## 2026-11-05 — eksperyment: fertility trening vs test

- Pomysł użytkownika, wstawiony przed krok 3d: fertility w trakcie treningu, osobno na 75%
  treningu i 25% testu. Wariant: słowa + znaki.
- Decyzje: podział po tekstach źródłowych, tylko kawałki z literami, 10 000 scaleń, wykresy
  w Plotly (log X). Pytanie użytkownika o zrównoleglanie: omówione.
- Oględziny: ok. 12% wystąpień i 40% unikalnych słów testu jest nowych; start fertility ok. 6,7.
- Wynik przy 10 154 tokenach: trening 1,62, test 1,65, słowa widziane 1,44, niewidziane 3,17.
  Rozjazd rośnie od ok. 600 tokenów, do +0,031; test nie rośnie w żadnym punkcie. Wykresy:
  `experiments/tokenizer/07_plots.py`, wynik w `data/tokenizer/raport.html`.
- Decyzja użytkownika: BPE w Ruście, równolegle, bez indeksu; agent napisał całość, użytkownik
  przeanalizuje (`rust/bpe/`, PyO3 + rayon, `u32`). Lista 10 000 scaleń identyczna jak w Pythonie.
- Rust na 1 wątku: 18x szybciej od Pythona. Wątki: najwyżej 1,38x (6 wątków), 1,29x (18).
  Pierwsza wersja dawała 1,19x, bo rayon tworzył do ok. 10 000 map na rundę. Zliczanie par
  nadal dominuje i prawie się nie skaluje.

## 2026-11-04 — etap 3: BPE od zera, trzy warianty

- Algorytm BPE przejrzany ręcznie na „kot / kota / koty / lot”.
- Decyzja użytkownika: zamiast jednego wariantu porównujemy trzy naraz: słowa + znaki,
  bez podziału + znaki, słowa + bajty.
- Typ użytkownika na pierwsze scalenia: częste litery z początków słów razem ze spacją.
  Do sprawdzenia w krokach 3b/3c.
- Krok 3a: `src/ailab/bpe.py`, `experiments/tokenizer/02_pretokenize.py`. Na nkjp1m:
  154 578 unikalnych kawałków; praca na rundę 1,45 mln (słowa + znaki), 6,82 mln
  (bez podziału), 1,54 mln (słowa + bajty). W danych wystąpiło 149 z 256 bajtów.
- Krok 3b: `count_pairs`, `merge_pair`, `experiments/tokenizer/03_first_merges.py`. Typ
  użytkownika w większości trafiony: w wariancie słowa + znaki po `ie` idą ` p`, ` w`, ` s`,
  ` n`. Bez podziału: `ie`, `a␣`, ` p`, `o␣`, `,␣`. Bajty: `ie`, `C5 82` (ł), ` p`, ` w`, ` s`.
  Pomiar rundy: 0,22 s (słowa) vs 0,95 s (bez podziału).
- Typ użytkownika przed krokiem 3c: średnio 3 znaki na token po 1000 scaleń.
- Krok 3c: `train`, `experiments/tokenizer/04_train.py`, 1000 scaleń w każdym wariancie.
  Wynik: 2,45 znaku/token (słowa + znaki), 2,40 (bez podziału), 2,44 (słowa + bajty), więc
  typ 3 był za wysoki. Czas: ok. 150 s dla wariantów ze słowami, 1161 s bez podziału.
  Bajty: prawie ta sama lista scaleń, przesunięta o 11-15 pozycji na odbudowę liter.
- Pytanie użytkownika: jak modele radzą sobie z wielkimi literami (rozmiar słownika, uczenie,
  ALL CAPS, literówki)? Omówione: stały budżet V, rozpad rzadkich wariantów, podobieństwo
  uczone z kontekstu, alternatywy (uncased, tokeny `xxmaj`/`xxup`). Bez pomiarów na naszym
  tokenizerze; zaproponowany eksperyment po kroku 3d.

## 2026-10-31 — plan tokenizera na Polish DynaWord

- Fork, c.d.: co jest uczone (embedding, warstwy, LM head), a co jest stałą operacją
  (lookup, mnożenie, softmax, sampling). Temperatura = tylko wybór z rozkładu (wniosek
  użytkownika). Weight tying jeszcze raz, na kształtach. Softmax bottleneck: użytkownik
  zakwestionował problem (50k punktów w 768D), wyjaśnione różnicą punkty vs rozkłady.
- Doprecyzowanie (fork rozmowy): użytkownik słusznie zauważył, że wymiar embeddingu `d`
  nie zależy od `V`. Rozdzielone: wyjście warstw `h` ma `d` liczb, logity po LM head mają `V`.
  Porównanie z embeddingami z RAG (cosinus vs iloczyn skalarny, softmax po całym słowniku,
  sampling zamiast „najbliższego”).
- Przegląd datasetu `SlayerLab/polish-dynaword` v0.2.6: 27 źródeł, ~6,65B tokenów, ~60%
  tekstów prawno-urzędowych (eurlex, parliamentary, dziennik_ustaw).
- Zaakceptowana mapa drogi (8 etapów). Decyzje: hybryda (własny BPE → biblioteka), próbka
  zbalansowana z limitem na źródło, środowisko rozpisane przy etapie 0.
- Predykcja trafiona: tokenizer z EUR-Lex podzieli „rozporządzenie” na mniej tokenów niż
  „przyjaciółeczka”.
- Własny pomysł: lematyzować wejście, bo fleksja to „głównie składnia”. Omówione: przypadki
  niosą znaczenie (kto komu), osobne słowniki wejścia/wyjścia są technicznie możliwe
  (encoder-decoder), ale w autoregresji wyjście wraca na wejście; alternatywa bezstratna —
  segmentacja temat + końcówka.
- Etap 0 zrobiony: `uv` + Python 3.13 przypięty tylko w projekcie (`.python-version`,
  `.venv/`); systemowy 3.15 bez zmian. Zostawiony układ pakietu `src/ailab/` (instalacja
  editable), żeby kod tokenizera dało się później importować z innych eksperymentów.
- Etap 1: `huggingface_hub` + `polars`, moduł `src/ailab/data.py` (tag v0.2.6),
  skrypt `experiments/tokenizer/01_explore.py`. nkjp1m: 18 381 krótkich fragmentów,
  6,8 mln znaków, 165 unikalnych znaków. wolne_lektury: 233,6 mln znaków, bardzo skośne
  długości (mediana 972, max 2,4 mln znaków), 522 unikalne znaki (greka, «», soft hyphen,
  dużo pustych linii w poezji). UTF-8: ~1,06–1,08 bajta na znak.
- Pytanie użytkownika: czy usuwa się słabo reprezentowane tokeny (pruning)? Omówione:
  udział `V×d` w parametrach, glitch tokens (SolidGoldMagikarp), profilaktyka przy
  treningu tokenizera vs vocabulary trimming po fakcie.
- Pytanie kontrolne (słownik startowy, nieznany znak): trafione w sednie. Dalsze pytania:
  tokenizery bajtowe we frontier modelach, normalizacja; agent użył „embedding / LM head”
  bez wyjaśnienia — nadrobione na przykładzie liczbowym.

## 2026-10-31 — setup i zmiana trybu pracy

- Założone repo, `CLAUDE.md`, pluginy.
- Agent zbudował polski tokenizer BPE (32k) w całości sam — z wybranymi bez dyskusji danymi,
  bibliotekami i rozmiarem słownika. Wynik był, zrozumienia nie.
- Decyzja: usunięte całe podejście (kod, dane, środowisko Pythona). Zmieniony tryb pracy
  agenta: prowadzi krok po kroku, zatrzymuje się przed każdym działaniem, decyzje
  zostawia mnie, prowadzi ten profil wiedzy.
- Na później: tokenizer od zera, z eksperymentem na rozmiarze słownika.
