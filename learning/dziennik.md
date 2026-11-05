# Dziennik nauki

Chronologiczny log sesji, najnowsze na górze. Profil (`profil.md`) to stan; dziennik to
historia, jak do niego doszliśmy.

## 2026-11-05 — pomysł: pruning tokenów pośrednich w BPE (tylko teoria)

- Pomysł użytkownika: po scaleniu odejmować jego liczność od obu rodziców, a token poniżej 50%
  liczności wejściowej usuwać ze słownika. Bez kodu i bez uruchomień.
- Omówione na papierze: koszt usunięcia = bieżąca liczność `L`, zysk = liczność najlepszej pary
  `f_best`, więc kryterium to `L < f_best`. Wyprowadzone: próg 50% jest warunkiem koniecznym,
  ale nie wystarczającym (dotyczy samych scaleń, bez nakładek typu `aaa` i rozwijania na >2 części).
- Otwarte: dzieci usuwanego tokenu, kolejność scaleń przy encode, wpływ na test.
- Literatura z wyszukiwania (czytane fragmenty): Picky BPE, Scaffold-BPE, LiteToken, Pruned BPE.
- Korekta agenta: błędne usunięcie zwykle cofa się w następnej rundzie, bo para wraca jako
  najczęstsza; koszt to głównie zmarnowana runda, a nie trwałe +`L` tokenów. Do sprawdzenia.
- Użytkownik zakwestionował problemy z encode i miał rację w sednie: wynikają z odtwarzania
  listy scaleń, a nie z samego usuwania. Omówiony encoder DP (minimum tokenów) jako alternatywa;
  decyzja DP vs historia zdarzeń jeszcze nie zapadła.
- Pytanie użytkownika: czy kolejność ma znaczenie? Wyjaśnione: liczy się kolejność scaleń przy
  encode w klasycznym BPE, a nie kolejność słownika jako zbioru; decode i DP jej nie potrzebują.

## 2026-11-05 — raport PDF dla współpracowników

- Na prośbę użytkownika: `reports/usuwanie_tokenow_bpe.pdf` (`17_report.py`), liczby czytane z wyników.
- Użytkownik wskazał brak metodyki: dodana sekcja 4 (10 i 3 podziały 75/25, jednostka podziału, liczba
  treningów, metryka, analiza parami, kontrola poprawności).
- Przy okazji symulacja agenta: próg 50% jest warunkiem koniecznym tylko bez nakładek (0 wyjątków w
  987 616 przypadkach); z nakładkami wyjątki około 0,4-0,8%. Otwarte: encoder DP vs listy usuwanych tokenów.

## 2026-11-05 — pruning na 26 źródłach: 3 podziały

- Pomiar `15_pruning_large.py`, wykresy `16_large_plots.py`: limit 20 mln tokenów na źródło, 3 warianty,
  3 podziały. Czas 48 min, a nie prognozowane 35; jeden pełny pomiar 12-19 min, ponad budżet 10 min.
- Reguła 50%, mediana parami: test -0,86% przy 4534 tokenach, -0,56% przy 10 500, -0,35% przy
  30 000. Test poprawiony na wszystkich 3 podziałach w 40 z 40 punktów siatki.
- Słowa niewidziane (ok. 2% testu): przy 30 000 -0,04%, czyli brak wyraźnej szkody widocznej na nkjp1m.
- Źródła przy 4534: od -2,09% (`plwiki_user_talk`) do -0,37% (`eltec_pol`). Przy 30 000 każde poniżej
  zera w medianie. Ostrzeżenie: w 24 źródłach podział po dokumentach, nie po tekstach.

## 2026-11-05 — większy zbiór: kalibracja czasu

- Decyzje użytkownika: próg 1000 wystąpień dla znaków, grupowanie tylko w `nkjp1m` i
  `sejm_committee_transcripts`. Kod agenta: `src/ailab/corpus.py`, `14_calibrate.py`.
- Pełny pomiar 2 wariantów, 1 podział, słownik 30 000: 99 s (limit 2 mln), 286 s (5 mln),
  607 s (20 mln). Przy 20 mln reguła 50% to 269 s treningu i 338 s odtworzeń.
- Pierwszy wynik przy 20 mln i 30 000 tokenów: trening -0,44%, słowa widziane -0,42%,
  niewidziane +0,01%; poprawa na każdym z 26 źródeł. Słów niewidzianych tylko 1,9% (12% w nkjp1m).

## 2026-11-05 — większy zbiór: oględziny 26 źródeł DynaWord

- Decyzje użytkownika: pełny pomiar 3 wariantów do 10 minut, 3 podziały, odfiltrowanie rzadkich
  znaków (próg do ustalenia), fertility osobno dla każdego źródła. `ted_pl` pominięte.
- Pobrane 5,9 GB. Próbka z limitem 20 mln tokenów na źródło (`13_explore_sources.py`): 861 mln
  znaków, 162,7 mln wystąpień, 2,71 mln unikalnych kawałków, czyli 134x i 17,5x więcej niż nkjp1m.
- Alfabet: 79 863 znaki, głównie przez `wikibooks` (76 102). Przy progu 1000 wystąpień zostaje 337
  znaków, a odpada 0,037% znaków tekstu.
- Format ID: grupowanie po tekście ma sens tylko w `nkjp1m` i `sejm_committee_transcripts`.

## 2026-11-05 — pruning w BPE: 10 podziałów trening/test

- Użytkownik odczytał z wykresu „mały, ale systematyczny” efekt; decyzja: najpierw inne podziały,
  potem większy zbiór uczący. Kod agenta: `11_pruning_splits.py`, `12_splits_plots.py`.
- Reguła 50%, mediana z 10 podziałów, parami: test -0,71% przy 5489, -0,55% przy 10 154,
  -0,20% przy 30 000 (zakres -0,24 do -0,16). Test poprawiony na 10/10 podziałów od 400 tokenów
  wzwyż. Hipoteza użytkownika potwierdzona dla treningu i testu w tym korpusie.
- Słowa niewidziane: poprawa na małych słownikach, przy 30 000 pogorszenie na 10/10 (mediana
  +0,25%). Wcześniejszy szczyt +0,67% przy 20 000 to był szum jednego podziału.

## 2026-11-05 — pruning w BPE: pierwszy pomiar na nkjp1m

- Decyzje użytkownika: test przez odtwarzanie zdarzeń, trzy warianty (bez poprawki, 50%,
  `L < f_best`), słownik do 30 000, wykres z różnicami i checkboxami
  (`10_pruning_plots.py`, wynik w `data/tokenizer/pruning/raport.html`).
- Kod agenta: `train_events` i `replay_curve` w `rust/bpe/src/lib.rs`, testy
  `09_pruning_check.py` (220 vs 170 tokenów), pomiar `09_pruning.py`.
- Wynik przy 30 000, względem wersji bez poprawki: trening -0,54%, test -0,19%, słowa widziane
  -0,31%, niewidziane +0,21%. Przy 10 154: trening -0,67%, test -0,53%, niewidziane +0,02%.
- Obie reguły dały prawie to samo. Reguła 50%: 10 296 usunięć, 8043 powroty. `L < f_best`:
  3932 usunięcia, 1894 powroty. Na stałe usunięte około 7% i 6% utworzonych tokenów.

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
- Indeks „para → słowa” w Ruście (`train_indexed`, `experiments/tokenizer/08_indexed.py`):
  10 000 scaleń w 2,18 s (naiwna na 1 wątku 64,2 s), 30 000 w 6,31 s (naiwna na 6 wątkach
  163,6 s). Listy zgodne z Pythonem i z wersją naiwną. Teraz dominuje wybór pary.

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
