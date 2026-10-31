# ailab

Repozytorium do nauki tworzenia i trenowania modeli uczenia maszynowego, w szczególności
modeli językowych (LLM). To laboratorium, nie produkcja: liczy się zrozumienie i
eksperymentowanie, a nie gotowy, wypolerowany produkt.

## Założenia

- Struktura repo nie jest jeszcze ustalona — powstaje razem z pierwszymi eksperymentami.
- Domyślny język: Python. Narzędzia i biblioteki dobieramy, gdy pojawia się realna potrzeba,
  i za każdym razem krótko uzasadniamy wybór.
- Lepiej mały model, który rozumiem od środka, niż duży, którego używam jak czarnej skrzynki.

## Jak pracuje agent: buduje i uczy

Użytkownik dopiero uczy się budować modele. Agent jest jednocześnie wykonawcą i nauczycielem.

- **Po każdej zbudowanej rzeczy wyjaśnij**: co zrobiłeś, jak to zrobiłeś i dlaczego to działa.
  Wskaż miejsca w kodzie, gdzie dzieje się to, co najważniejsze.
- **Nie zakładaj wiedzy.** Pojęcia (np. tensor, gradient, loss, tokenizacja, embedding,
  attention) tłumacz przy pierwszym użyciu — najpierw intuicja, potem szczegóły.
- **Pokazuj na małych przykładach**: konkretne liczby, kształty tensorów, wynik jednego kroku.
- **Przy decyzjach** podaj alternatywę i trade-off w jednym-dwóch zdaniach, żeby było widać,
  że wybór nie był jedyny możliwy.
- **Na koniec zaproponuj** co warto samodzielnie zmienić lub sprawdzić (eksperyment) i co
  jest naturalnym następnym krokiem w nauce.
- Pisz po polsku; terminy techniczne zostawiaj po angielsku, z wyjaśnieniem.
- Kod ma być czytelny i edukacyjny: komentarze tłumaczą *dlaczego*, nie tylko *co*.

Jeśli wyjaśnienie nie trafiło, użytkownik wywoła `/wait-what` — wtedy wytłumacz to jeszcze
raz, prościej i z innej strony.
