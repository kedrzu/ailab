# Profil wiedzy

„Digital twin” tego, co rozumiem o ML i LLM-ach. Agent czyta go na początku sesji, żeby
dopasować poziom i nie powtarzać tego, co już wiem, i aktualizuje go po każdej większej
rundzie. Mogę go też poprawiać ręcznie.

**Poziomy:** ❓ nieznane · 👂 słyszałem · 💡 rozumiem intuicyjnie · 🛠️ zrobiłem sam / umiem wyjaśnić

Wpisujemy tylko to, co faktycznie omówiliśmy — brak pojęcia na liście = nie wiadomo, nie
zakładaj.

## Pojęcia

### Podstawy ML

| pojęcie | poziom | notatka |
|---|---|---|

### Dane

| pojęcie | poziom | notatka |
|---|---|---|
| skład korpusu a wynik tokenizera | 💡 | Trafnie przewidział, że tokenizer na EUR-Lex lepiej skompresuje „rozporządzenie” niż „przyjaciółeczka” (2026-10-31). |
| parquet | 👂 | Kolumnowy format tabeli; tak jest zapisany DynaWord. |
| znaki vs bajty (UTF-8) | 👂 | „ą” = 1 znak, 2 bajty; polski tekst ~1,07 bajta/znak. |

### Tokenizacja

| pojęcie | poziom | notatka |
|---|---|---|
| token, słownik (vocabulary) | 💡 | Tekst → kawałki ze stałego słownika → liczby. Trade-off: mały słownik = długie ciągi. |
| słownik startowy BPE, OOV | 💡 | Kontrolne 2026-10-31: trafnie — 165 znaków = słownik startowy; nieznany znak nie ma ID. Nie znał `<unk>` ani byte fallback. |
| byte-level BPE | 💡 | Sam wywnioskował, że bajty eliminują problem nieznanych znaków. Omówione: GPT-2/4, Llama 3 bajtowe; Llama 2/Gemma — znaki + byte fallback. |
| częstość a długość kodu (kompresja) | 💡 | Sam zauważył, że rzadkie = drogie się opłaca, bo rzadkie (zasada Huffmana). Uzupełnione: „rzadkie” względem danych tokenizera, nie użycia — token tax dla języków/domen spoza danych. |
| normalizacja (NFC/NFKC) vs czyszczenie danych | 👂 | Bajtowe tokenizery zwykle bez normalizacji (bezstratny round-trip); sprzątanie robi się w danych. |
| subwords / BPE | 💡 | Algorytm przejrzany ręcznie na małym przykładzie; implementacja w toku (krok 3c z 4). Typ 3 znaki/token po 1000 scaleń, wynik 2,45: kierunek dobry, wartość za wysoka. Typ pierwszych scaleń w większości trafiony (spacja + częsta litera), ale pierwsze było `ie`. |
| pre-tokenizacja | 💡 | Podział na kawałki przed BPE; liczniki powtórzeń dają ok. 4,7× mniej pracy na rundę na nkjp1m. |
| glitch / under-trained tokens | 👂 | SolidGoldMagikarp; tokenizer trenowany na innym rozkładzie niż model. Profilaktyka: te same dane, min_frequency, przycięty alfabet, bajty. |
| rozjazd danych tokenizera i modelu | 👂 | Skąd: tokenizer zamrożony wcześniej, reużywany między generacjami, trenowany na próbce, dotrenowanie na nowym języku; czasem celowo (balans języków). |
| fertility | 💡 | Sam zaproponował pomiar osobno na treningu i teście. Tokeny na słowo; nie może rosnąć z kolejnymi scaleniami. |
| podział train/test | 👂 | Omówiony podział po tekstach źródłowych, żeby fragmenty jednej książki nie trafiły do obu zbiorów. |
| generalizacja tokenizera | 💡 | Na wykresach: słowa niewidziane 3,17 tokenu wobec 1,44 dla widzianych; test nie rośnie, rośnie rozjazd. |
| aktualizacja przyrostowa (indeks) | 👂 | Algorytm dał ok. 29x, wątki 1,38x. Kod napisał agent. |
| skalowanie równoległe (Amdahl) | 👂 | Rust 1,38x na 6 wątkach; zliczanie par nie skaluje się. Kod napisał agent, użytkownik ma go przeanalizować. |
| vocabulary trimming | 👂 | Przycinanie słownika po fakcie (np. mT5 → jeden język); trzeba usunąć też scalenia BPE. |
| fleksja a tokenizacja | 💡 | Sam wyszedł z pomysłem lematyzacji wejścia dla kompresji; omówione ograniczenia (przypadek niesie znaczenie, autoregresja). |
| tokeny pośrednie, pruning w BPE | 💡 | Sam wymyślił odejmowanie liczności od rodziców i usuwanie tokenów poniżej 50% (2026-11-05). Omówione teoretycznie: kryterium `L < f_best`, 50% jako warunek konieczny; nie testowane na danych. Pokrewne: Picky BPE. Sam wyłapał, że problemy z encode wynikają z odtwarzania reguł; omówiony encoder DP. Sam odczytał z wykresu mały, systematyczny efekt; potwierdzone na 10 podziałach nkjp1m: test od -0,71% (ok. 5500 tokenów) do -0,20% (30 000), niewidziane przy 30 000 +0,25%. Na 26 źródłach (3 podziały): test od -0,86% (4534) do -0,35% (30 000), niewidziane około zera. Kod napisał agent. |
| kolejność scaleń przy encode | 👂 | Do 2026-11-05 zakładał, że słownik to lista bez znaczącej kolejności. Wyjaśnione: klasyczny encode zależy od kolejności scaleń; decode i encoder DP nie. |
| wielkość liter w tokenizacji | 👂 | Sam zapytał o koszt dla słownika i uczenia (ALL CAPS, literówki). Omówione 2026-11-04: V to stały budżet, rzadkie warianty rozpadają się na drobne tokeny; model uczy się podobieństwa wariantów z kontekstu; uncased BERT i tokeny `xxmaj`/`xxup` jako alternatywy. |

### Architektura

| pojęcie | poziom | notatka |
|---|---|---|
| rozmiar wyjścia modelu = V | 💡 | Sam wywnioskował, że wyjście modelu skaluje się z rozmiarem słownika. Tylko macierze `V×d` zależą od V; w małych modelach >50% parametrów. |
| embedding, LM head, weight tying | 💡 | Wcześniej użyte bez wyjaśnienia (błąd agenta) — wyjaśnione 2026-10-31 na przykładzie: tabela wierszy, iloczyn skalarny → logity → softmax. Sam zauważył, że `d` (wymiar embeddingu) nie zależy od `V` — i wyłapał nieprecyzyjność agenta. Rozróżnione: stan ukryty `h` [d] vs logity [V]; V wpływa na parametry przez macierz `V×d`. Zna embeddingi z RAG — różnice: tam wektor na cały tekst i cosinus, tu wektor na token i iloczyn skalarny po wszystkich tokenach. |
| parametry vs aktywacje; softmax, temperatura, sampling | 💡 | Sam wywnioskował, że temperatura/losowość to tylko wybór z rozkładu, nie zmiana działania sieci. Uzupełnione: wybrany token wraca na wejście (autoregresja), więc wpływa na dalszy ciąg; drobny niedeterminizm GPU. |
| softmax bottleneck | 👂 | Intuicja użytkownika: 50k punktów łatwo rozróżnić w 768 wymiarach — prawda, ale problem dotyczy rozkładów, nie punktów: logity leżą w podprzestrzeni wymiaru d w R^V. Przykład d=1, V=3. |

### Trening

| pojęcie | poziom | notatka |
|---|---|---|

## Chcę wrócić / pytania otwarte

- Etap 6: ile tokenów na polskie słowo daje nasz tokenizer vs cl100k (GPT-4) — „token tax”
  polskiego w angielskocentrycznym tokenizerze.

- Po treningu modelu: częstość tokenów w danych vs „jakość” ich embeddingów — czy rzadkie
  tokeny są niedouczone (pomysł z 2026-10-31, pytanie o pruning słownika).

- Czy słownik wejściowy i wyjściowy LLM-a mogą się różnić (np. lematy na wejściu, pełne formy
  na wyjściu)? Omówione teoretycznie 2026-10-31; można kiedyś sprawdzić eksperymentem
  (ile znaczenia ginie po lematyzacji, segmentacja morfologiczna vs BPE).

- Jak rozmiar słownika wpływa na tokenizację — chcę to zobaczyć w eksperymencie
  porównawczym (kilka rozmiarów obok siebie), a nie dostać jedną „optymalną” wartość.

- Pruning w BPE: dlaczego największy zysk jest przy kilku tysiącach tokenów i dlaczego `plwiki_user_talk` zyskuje najwięcej (-2,09% przy 4534)? Do sprawdzenia: encoder DP.

## Preferencje nauki

- Krok po kroku, z zatrzymaniem przed każdym działaniem; decyzje podejmuję ja.
- Eksperymenty porównawcze zamiast gotowych wartości.
- Konkretne liczby i małe przykłady.

## Projekty

| projekt | status | notatka |
|---|---|---|
| tokenizer | plan zaakceptowany | Dane: SlayerLab/polish-dynaword v0.2.6, próbka zbalansowana (limit na źródło). Podejście: najpierw własny BPE w Pythonie na małym tekście, potem biblioteka + eksperyment z rozmiarem słownika. Etap 0 (uv, Python 3.13, pakiet `src/ailab/`) zrobiony; etap 1 (oględziny nkjp1m + wolne_lektury) zrobiony; etap 3 w toku, trzy warianty BPE (słowa + znaki, bez podziału + znaki, słowa + bajty), kroki 3a-3c gotowe, eksperyment fertility i moduł Rust `rust/bpe/` gotowe; krok 3d (encode/decode) wciąż przed nami. |
