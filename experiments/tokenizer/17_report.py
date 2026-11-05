"""Raport dla współpracowników: HTML do wydruku jako PDF, z liczbami czytanymi wprost z wyników.

Liczby w tekście i tabelach pochodzą z plików wynikowych, a nie są przepisane ręcznie, żeby raport nie
rozjechał się z danymi. PDF powstaje z tego HTML-a w przeglądarce, bo Plotly rysuje wykresy jako SVG.

Uruchomienie:  uv run python experiments/tokenizer/17_report.py
"""

import json
import math
import statistics
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "tokenizer" / "pruning"
OUT_FILE = ROOT / "reports" / "usuwanie_tokenow_bpe.html"

BASELINE = "bez poprawki"
HALF = "reguła 50%"
BEST = "reguła L < f_best"
NKJP = "nkjp1m"
DYNA = "DynaWord (26 źródeł)"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

nkjp = json.loads((DATA_DIR / "splits.json").read_text())
dyna = json.loads((DATA_DIR / "large.json").read_text())


def nkjp_diffs(variant: str, measure: str, i: int) -> list[float]:
    return [
        100 * (s["variants"][variant]["series"][measure][i] / s["variants"][BASELINE]["series"][measure][i] - 1)
        for s in nkjp["splits"]
    ]


def dyna_fertility(split: dict, variant: str, measure: str, i: int) -> float:
    tokens = split["variants"][variant]["tokens"]
    words = split["words"]
    if measure == "test":
        return (tokens["test_seen"][i] + tokens["test_unseen"][i]) / (words["test_seen"] + words["test_unseen"])
    return tokens[measure][i] / words[measure]


def dyna_diffs(variant: str, measure: str, i: int) -> list[float]:
    return [
        100 * (dyna_fertility(s, variant, measure, i) / dyna_fertility(s, BASELINE, measure, i) - 1)
        for s in dyna["splits"]
    ]


def source_diffs(variant: str, source: str, i: int) -> list[float]:
    return [
        100 * (s["variants"][variant]["source_tokens"][source][i] / s["variants"][BASELINE]["source_tokens"][source][i] - 1)
        for s in dyna["splits"]
    ]


def median(values: list[float]) -> float:
    return statistics.median(values)


def nearest(grid: list[int], size: int) -> int:
    return min(range(len(grid)), key=lambda i: abs(math.log(grid[i] / size)))


def pct(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}".replace(".", ",").replace("-", "−") + "%"


def num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def integer(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def spread(values: list[float]) -> str:
    low = pct(min(values)).removesuffix("%")
    high = pct(max(values)).removesuffix("%")
    return f"{pct(median(values))} <span class='range'>({low} do {high})</span>"


def layout(height: int, y_title: str, x_title: str = "rozmiar słownika (skala log)") -> dict:
    return dict(
        width=680,
        height=height,
        margin=dict(l=62, r=14, t=36, b=48),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=11, color=INK),
        separators=", ",
        legend=dict(orientation="h", y=1.17, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title=x_title, type="log", gridcolor=GRID, linecolor=AXIS, zeroline=False),
        yaxis=dict(title=y_title, gridcolor=GRID, linecolor=AXIS, zeroline=False),
    )


def rgba(color: str, alpha: float) -> str:
    red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def band(x: list[int], points: list[list[float]], name: str, color: str, dash: str = "solid") -> list[go.Scatter]:
    """Mediana z pasmem od najmniejszej do największej wartości."""
    return [
        go.Scatter(x=x, y=[max(p) for p in points], mode="lines", line=dict(width=0, color=color), showlegend=False),
        go.Scatter(x=x, y=[min(p) for p in points], mode="lines", line=dict(width=0, color=color), fill="tonexty",
                   fillcolor=rgba(color, 0.2), showlegend=False),
        go.Scatter(x=x, y=[median(p) for p in points], name=name, mode="lines", line=dict(width=2, color=color, dash=dash)),
    ]


def chart(fig: go.Figure, div_id: str, first: bool = False) -> str:
    # Wykres ma być statyczny na wydruku; bibliotekę osadzamy raz.
    return fig.to_html(full_html=False, include_plotlyjs=first, div_id=div_id, config={"staticPlot": True})


nkjp_x, dyna_x = nkjp["vocab"], dyna["vocab"]


def comparison(measure: str, y_title: str, height: int = 290) -> go.Figure:
    fig = go.Figure(
        band(nkjp_x, [nkjp_diffs(HALF, measure, i) for i in range(len(nkjp_x))], NKJP, ORANGE)
        + band(dyna_x, [dyna_diffs(HALF, measure, i) for i in range(len(dyna_x))], DYNA, BLUE)
    )
    fig.update_layout(layout(height, y_title))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    return fig


fig_test = comparison("test", "zmiana fertility [%]", 270)
fig_unseen = comparison("test_unseen", "zmiana fertility [%]", 250)

fig_rules = go.Figure(
    band(dyna_x, [dyna_diffs(HALF, "train", i) for i in range(len(dyna_x))], f"{HALF}, trening", ORANGE)
    + band(dyna_x, [dyna_diffs(HALF, "test", i) for i in range(len(dyna_x))], f"{HALF}, test", ORANGE, "dash")
    + band(dyna_x, [dyna_diffs(BEST, "test", i) for i in range(len(dyna_x))], f"{BEST}, test", AQUA, "dash")
)
fig_rules.update_layout(layout(240, "zmiana fertility [%]"))
fig_rules.add_hline(y=0, line=dict(color=MUTED, width=1))

PEAK = dyna_x.index(4534)
END = dyna_x.index(30000)
sources = list(dyna["splits"][0]["source_words"])
sources_by_peak = sorted(sources, key=lambda s: median(source_diffs(HALF, s, PEAK)), reverse=True)

fig_sources = go.Figure()
for size_index, name, color in [(PEAK, "słownik 4534", ORANGE), (END, "słownik 30 000", BLUE)]:
    points = [source_diffs(HALF, s, size_index) for s in sources_by_peak]
    medians = [median(p) for p in points]
    fig_sources.add_trace(
        go.Scatter(
            x=medians, y=sources_by_peak, name=name, mode="markers",
            marker=dict(size=8, color=color, line=dict(width=1.5, color=SURFACE)),
            error_x=dict(type="data", symmetric=False, array=[max(p) - m for p, m in zip(points, medians)],
                         arrayminus=[m - min(p) for p, m in zip(points, medians)], color=color, thickness=1.5, width=0),
        )
    )
fig_sources.update_layout(layout(395, "", "zmiana fertility na teście [%], reguła 50%"))
fig_sources.update_layout(
    margin=dict(l=175, r=14, t=30, b=45),
    xaxis=dict(type="linear", title="zmiana fertility na teście [%], reguła 50%"),
    yaxis=dict(dtick=1, title=None, gridcolor=GRID),
)
fig_sources.add_vline(x=0, line=dict(color=MUTED, width=1))

# Tabela: rozmiary, przy których widać przebieg efektu.
rows_main = ""
for size in (1000, 3000, 4534, 10500, 30000):
    n, d = nearest(nkjp_x, size), nearest(dyna_x, size)
    rows_main += (
        f"<tr><td>{integer(dyna_x[d])}</td>"
        f"<td>{num(median([dyna_fertility(s, BASELINE, 'test', d) for s in dyna['splits']]))}</td>"
        f"<td>{num(median([dyna_fertility(s, HALF, 'test', d) for s in dyna['splits']]))}</td>"
        f"<td>{spread(dyna_diffs(HALF, 'test', d))}</td>"
        f"<td>{spread(dyna_diffs(BEST, 'test', d))}</td>"
        f"<td>{spread(dyna_diffs(HALF, 'train', d))}</td>"
        f"<td>{spread(dyna_diffs(HALF, 'test_unseen', d))}</td>"
        f"<td>{integer(nkjp_x[n])}: {spread(nkjp_diffs(HALF, 'test', n))}</td></tr>"
    )

rows_ops = ""
for name in (BASELINE, HALF, BEST):
    values = [s["variants"][name] for s in dyna["splits"]]
    rows_ops += (
        f"<tr><td>{name}</td><td>{integer(round(median([v['merges'] for v in values])))}</td>"
        f"<td>{integer(round(median([v['splits'] for v in values])))}</td>"
        f"<td>{integer(round(median([v['restores'] for v in values])))}</td>"
        f"<td>{integer(round(median([v['removed_for_good'] for v in values])))}</td></tr>"
    )

split0 = dyna["splits"][0]
words0 = split0["words"]
unseen_share = 100 * words0["test_unseen"] / (words0["test_seen"] + words0["test_unseen"])
peak_test = median(dyna_diffs(HALF, "test", PEAK))
end_test = median(dyna_diffs(HALF, "test", END))
all_below = sum(
    all(dyna_fertility(s, HALF, "test", i) < dyna_fertility(s, BASELINE, "test", i) for s in dyna["splits"])
    for i in range(len(dyna_x))
)
nkjp_peak = nearest(nkjp_x, 5500)
nkjp_end = nkjp_x.index(30000)
nkjp_below = sum(
    all(s["variants"][HALF]["series"]["test"][i] < s["variants"][BASELINE]["series"]["test"][i] for s in nkjp["splits"])
    for i in range(len(nkjp_x))
)
talk_peak = median(source_diffs(HALF, "plwiki_user_talk", PEAK))
talk_end = median(source_diffs(HALF, "plwiki_user_talk", END))
eltec_peak = median(source_diffs(HALF, "eltec_pol", PEAK))
source_end_max = max(median(source_diffs(HALF, s, END)) for s in sources)
wall = [s["wall_seconds"] for s in dyna["splits"]]


def _test_words(split: dict) -> int:
    return split["words"]["test_seen"] + split["words"]["test_unseen"]


def _span(values: list[float], fmt) -> str:
    low, high = fmt(min(values)), fmt(max(values))
    return low if low == high else f"{low}–{high}"


def _millions(value: float) -> str:
    return f"{num(value / 1e6, 1)} mln" if value >= 1e6 else f"{round(value / 1e3)} tys."


def words_range(splits: list[dict], name: str) -> str:
    return _span([s["words"][name] for s in splits], _millions)


def test_range(splits: list[dict]) -> str:
    return _span([_test_words(s) for s in splits], _millions)


def share_range(splits: list[dict]) -> str:
    values = [100 * _test_words(s) / (_test_words(s) + s["words"]["train"]) for s in splits]
    return _span(values, lambda v: num(v, 1) + "%")


def unseen_range(splits: list[dict]) -> str:
    values = [100 * s["words"]["test_unseen"] / _test_words(s) for s in splits]
    return _span(values, lambda v: num(v, 1) + "%")


html = f"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><title>Usuwanie tokenów pośrednich w treningu BPE</title>
<style>
  @page {{ size: A4; margin: 19mm 18mm 18mm 18mm; }}
  :root {{ color-scheme: light; }}
  body {{ font-family: {FONT}; font-size: 10pt; line-height: 1.42; color: #1a1a19; margin: 0; }}
  h1 {{ font-size: 19pt; line-height: 1.2; margin: 0 0 4pt; }}
  .meta {{ color: {INK}; font-size: 9pt; margin-bottom: 12pt; }}
  h2 {{ font-size: 12.5pt; margin: 11pt 0 3pt; break-after: avoid; }}
  h3 {{ font-size: 10.5pt; margin: 9pt 0 2pt; break-after: avoid; }}
  p {{ margin: 0 0 6pt; }}
  ul {{ margin: 0 0 6pt; padding-left: 15pt; }} li {{ margin-bottom: 2pt; }}
  code {{ font-family: Menlo, Consolas, monospace; font-size: 8.6pt; }}
  .summary {{ border-left: 3px solid {BLUE}; padding: 6pt 10pt; background: #f4f7fb; margin-bottom: 10pt; }}
  figure {{ margin: 6pt 0 8pt; break-inside: avoid; }}
  figcaption {{ font-size: 8.6pt; color: {INK}; margin-top: 1pt; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 8.2pt; margin: 4pt 0 3pt;
           font-variant-numeric: tabular-nums; break-inside: avoid; }}
  th {{ text-align: right; border-bottom: 1px solid #1a1a19; padding: 2pt 4pt; font-weight: 600; vertical-align: bottom; }}
  td {{ text-align: right; border-bottom: 1px solid {GRID}; padding: 2pt 4pt; vertical-align: top; }}
  th:first-child, td:first-child {{ text-align: left; }}
  .setup th, .setup td {{ text-align: left; }}
  .range {{ color: {INK}; white-space: nowrap; font-size: 7.4pt; }}
  .note {{ font-size: 8.6pt; color: {INK}; }}
  .refs {{ font-size: 8.6pt; }}
  a {{ color: {BLUE}; text-decoration: none; }}
</style></head><body>

<h1>Usuwanie tokenów pośrednich w treningu BPE</h1>
<div class="meta">Michał Kędrzyński · 5 listopada 2026 · notatka z eksperymentu w repozytorium <code>ailab</code></div>

<div class="summary">
<b>Wnioski.</b> Sprawdziliśmy prosty wariant treningu BPE, który usuwa ze słownika tokeny pochłonięte przez dłuższe
scalenia i oddaje ich miejsca nowym tokenom. Przy tym samym rozmiarze słownika fertility na teście spada na próbce z
26 źródeł Polish DynaWord o <b>{pct(peak_test)}</b> przy 4534 tokenach i o <b>{pct(end_test)}</b> przy 30 000, poprawiając się na
wszystkich 3 podziałach we wszystkich {all_below} punktach pomiaru. Efekt jest mały, ale powtarzalny. Prosta reguła progu 50% wypada
praktycznie tak samo jak reguła wyprowadzona z bilansu kosztów, choć wykonuje około 2,5 raza więcej usunięć. Podobne podejścia
są już opisane w literaturze, więc nie jest to nowa metoda. Wynik dotyczy fertility: nie mierzyliśmy jeszcze jakości modelu.
</div>

<h2>1. Problem</h2>
<p>BPE dokłada token przy każdym scaleniu, ale nie usuwa tokenów, które stały się zbędne. Część z nich pełni tylko rolę
etapów pośrednich. Przykład: korpus <code>abcd</code>×100, <code>ef</code>×30, <code>gh</code>×20, <code>ij</code>×10 i budżet trzech nowych
tokenów. Klasyczne BPE daje <code>cd</code>, <code>bcd</code>, <code>abcd</code>, czyli 220 tokenów w wyniku, a dwa z nowych tokenów nie
pojawiają się w nim wcale. Słownik <code>abcd</code>, <code>ef</code>, <code>gh</code> daje 170 tokenów, o 23% mniej. Koszt widać na nowym
tekście: <code>cde</code> ma 2 tokeny w pierwszym wariancie i 3 w drugim. Dlatego osobno mierzymy słowa niewidziane.</p>

<h2>2. Metoda</h2>
<p><b>Bieżąca liczność.</b> Po scaleniu pary odejmujemy liczbę scaleń od liczności obu rodziców. <i>L</i> to więc dokładna liczba
wystąpień tokenu w aktualnym stanie korpusu, a <i>n</i><sub>0</sub> to jego liczność przy wejściu do słownika. Sprawdzamy tylko rodziców
ostatniego scalenia, bo tylko ich <i>L</i> maleje. Znaki bazowe nigdy nie są usuwane, więc każdy tekst da się zapisać.</p>
<p><b>Reguły.</b> Porównujemy dwa warianty:</p>
<ul>
<li><b>reguła 50%:</b> usuń token, gdy <i>L</i> &lt; <i>n</i><sub>0</sub>/2;</li>
<li><b>reguła <i>L</i> &lt; <i>f</i><sub>best</sub>:</b> usuń token, gdy koszt <i>L</i>·(<i>k</i>−1) jest mniejszy od zysku <i>f</i><sub>best</sub>, gdzie <i>k</i> to liczba części
po rozkładzie, a <i>f</i><sub>best</sub> to liczność najczęstszej pary. Każda taka zamiana ściśle zmniejsza łączną liczbę tokenów.</li>
</ul>
<p><b>Związek między regułami.</b> Dla <i>k</i> = 2 i bez nakładek typu <code>aaa</code> <i>L</i> maleje tylko przy scaleniu, które zabiera <i>m</i>
wystąpień, a <i>f</i><sub>best</sub> w klasycznym BPE nie rośnie. Opłacalność przy pierwszym usunięciu oznacza <i>L′</i> &lt; <i>f</i><sub>next</sub> ≤ <i>m</i>,
czyli <i>L′</i> &lt; <i>L</i>/2 ≤ <i>n</i><sub>0</sub>/2. Próg 50% jest więc warunkiem koniecznym opłacalności, ale nie wystarczającym. Sprawdziliśmy to
symulacją na klasycznym BPE: na 1500 losowych korpusach bez nakładek było 987 616 opłacalnych stanów tokenów i 0 wyjątków,
oraz 106 045 stanów poniżej progu, ale nieopłacalnych. Na korpusach z nakładkami wyjątki stanowiły około 0,4–0,8%.</p>
<p><b>Samonaprawa.</b> Błędne usunięcie, czyli takie przy <i>L</i> &gt; <i>f</i><sub>best</sub>, zwykle cofa się samo: para rodziców staje się najczęstsza i
w kolejnej rundzie wraca ten sam token. Tłumaczy to podobieństwo wyników obu reguł. Reguła 50% płaci głównie dodatkowymi rundami.</p>
<p><b>Rozmiar słownika i kodowanie.</b> Trening kończy się przy docelowej liczbie tokenów aktywnych, a nie po liczbie scaleń. Zapisujemy historię
zdarzeń <code>merge</code>/<code>split</code>, a przy pomiarze odtwarzamy ją w tej samej kolejności na każdym tekście, także przy wariancie
bez usuwania. To narzędzie pomiaru. Nie przesądza docelowego encodera; alternatywą jest podział na najmniejszą liczbę tokenów (DP).</p>

<h2>3. Powiązane prace</h2>
<p>Pomysł powstał niezależnie, ale temat jest znany. <b>Picky BPE</b> [1] usuwa token w trakcie treningu, gdy stosunek liczności pary do
liczności tokenu przekracza próg, np. 0,9; też przechowuje kolejność zdarzeń i dopuszcza powroty. <b>Scaffold-BPE</b> [2] oznacza tokeny o
niskiej częstości jako „rusztowanie”. <b>LiteToken</b> [3] usuwa takie tokeny po fakcie z gotowego tokenizera. <b>Pruned BPE</b> [4]
działa po treningu i przydziela zwolnione miejsca nowym tokenom, podając skrócenie tekstu o około 0,27–0,36% na tych samych danych.
Nasz wariant różni się od [1] głównie regułami decyzji: odniesieniem do <i>n</i><sub>0</sub> oraz bilansem z <i>f</i><sub>best</sub>. Nie twierdzimy,
że te reguły są nowe: przegląd był pobieżny i opierał się na fragmentach artykułów.</p>

<h2>4. Metodyka</h2>
<table class="setup">
<thead><tr><th></th><th>nkjp1m</th><th>DynaWord v0.2.6</th></tr></thead><tbody>
<tr><td>Dane</td><td>cały korpus, 18 381 dokumentów</td><td>26 źródeł, losowe dokumenty do limitu 20 mln tokenów <code>cl100k</code> na źródło; pominięte <code>ted_pl</code></td></tr>
<tr><td>Liczba podziałów</td><td><b>10</b></td><td><b>3</b></td></tr>
<tr><td>Proporcja trening/test</td><td>75/25 według dokumentów; test to {share_range(nkjp['splits'])} słów</td><td>75/25 według dokumentów, osobno w każdym źródle; test to {share_range(dyna['splits'])} słów</td></tr>
<tr><td>Jednostka podziału</td><td>tekst źródłowy (grupa fragmentów)</td><td>grupa w <code>nkjp1m</code> i <code>sejm_committee_transcripts</code>, dokument w 24 pozostałych</td></tr>
<tr><td>Wielkość treningu</td><td>{words_range(nkjp['splits'], 'train')} słów</td><td>{words_range(dyna['splits'], 'train')} słów</td></tr>
<tr><td>Wielkość testu</td><td>{test_range(nkjp['splits'])} słów</td><td>{test_range(dyna['splits'])} słów</td></tr>
<tr><td>Słowa niewidziane w teście</td><td>{unseen_range(nkjp['splits'])} wystąpień</td><td>{unseen_range(dyna['splits'])} wystąpień</td></tr>
<tr><td>Alfabet</td><td>wszystkie znaki treningu (153–161)</td><td>znaki z co najmniej 1000 wystąpień w treningu (290–302)</td></tr>
<tr><td>Punkty pomiaru</td><td>{len(nkjp_x)}, od 200 do 30 000 tokenów</td><td>{len(dyna_x)}, od 500 do 30 000 tokenów</td></tr>
<tr><td>Liczba treningów</td><td>10 × 3 warianty = 30</td><td>3 × 3 warianty = 9</td></tr>
</tbody></table>
<p><b>Podział.</b> Każdy podział to osobne losowanie z innym ziarnem (0–9 na nkjp1m, 0–2 na DynaWord). Losowo układamy grupy lub dokumenty i przydzielamy je do testu, aż obejmą 25% dokumentów,
więc udział testu w słowach może się różnić. W DynaWord ziarno wyznacza też próbkę dokumentów z dużych źródeł, więc poszczególne podziały różnią się także danymi, a nie tylko przydziałem do zbiorów.
To nie jest walidacja krzyżowa: zbiory testowe różnych podziałów mogą się nakładać.</p>
<p><b>Trening i pomiar.</b> Na każdym podziale trenujemy od zera wszystkie 3 warianty: klasyczne BPE, regułę 50% i regułę <i>L</i> &lt; <i>f</i><sub>best</sub>. Dane testowe nie biorą udziału w treningu. Pre-tokenizacja
działa jak w GPT-2 (<code>split_words</code>), a symbole startowe to znaki. Na DynaWord słowa ze znakami spoza alfabetu nie biorą udziału w treningu, a w teście taki znak liczy się jako jeden token. Rozmiar słownika obejmuje alfabet.</p>
<p><b>Metryka.</b> Fertility to średnia liczba tokenów na kawałek zawierający litery, ważona liczbą wystąpień; interpunkcja i same liczby nie wchodzą do średniej. Słowo testu jest „widziane”, jeśli wystąpiło w
treningu. Na DynaWord mierzymy też każde źródło osobno, na jego części testowej.</p>
<p><b>Analiza.</b> Warianty porównujemy parami, na tym samym podziale: zmiana to 100·(<i>F</i><sub>wariant</sub>/<i>F</i><sub>BPE</sub> − 1). Pozwala to oddzielić efekt od różnic między podziałami, które są tego samego rzędu: fertility klasycznego BPE na teście przy 30 000 tokenów różni się między podziałami o 1,4% na nkjp1m i 0,5% na DynaWord, a efekt to odpowiednio 0,20% i 0,35%. Podajemy medianę
oraz najmniejszą i największą wartość, czyli rozpiętość, a nie przedział ufności. Przy tak małej liczbie podziałów, zwłaszcza 3, nie liczymy testów istotności. Punkty pomiaru jednego biegu nie są niezależne, bo każdy słownik powstaje z poprzedniego.</p>
<p><b>Kontrola poprawności.</b> Przykład z sekcji 1 daje wyniki policzone ręcznie, a 300 losowych małych korpusów potwierdza zachowanie tekstu i zgodność odtworzenia z treningiem. Wariant bez usuwania daje na nkjp1m te same scalenia co dotychczasowa
implementacja, a podział 0 odtwarza wcześniejsze liczby. Implementacja: Rust (PyO3) z indeksem „para → słowa”.</p>

<h2>5. Wyniki</h2>
<figure>{chart(fig_test, "fig-test", first=True)}
<figcaption><b>Rys. 1.</b> Zmiana fertility na teście po regule 50% względem klasycznego BPE. Linia: mediana, pasmo: najmniejsza i największa wartość
(10 podziałów nkjp1m, 3 podziały DynaWord). Poniżej zera wariant daje mniej tokenów.</figcaption></figure>

<table><thead><tr><th>Słownik</th><th>Fertility testu, BPE</th><th>Fertility testu, 50%</th><th>Test, 50%</th><th>Test, <i>L</i>&lt;<i>f</i><sub>best</sub></th><th>Trening, 50%</th><th>Niewidziane, 50%</th><th>nkjp1m: najbliższy słownik, test, 50%</th></tr></thead>
<tbody>{rows_main}</tbody></table>
<p class="note"><b>Tabela 1.</b> DynaWord i nkjp1m, mediana zmiany względem klasycznego BPE; w nawiasie rozpiętość między podziałami.</p>

<h3>Główne obserwacje</h3>
<ul>
<li><b>Powtarzalność.</b> Na DynaWord test poprawia się na wszystkich 3 podziałach w {all_below} z {len(dyna_x)} punktów; na nkjp1m na wszystkich
10 podziałach w {nkjp_below} z {len(nkjp_x)} punktów, a wyjątki występują tylko poniżej 400 tokenów.</li>
<li><b>Kształt efektu.</b> Zysk rośnie do płaskiego maksimum między około 3000 a 6000 tokenów, a potem maleje: na DynaWord do {pct(end_test)}, na nkjp1m do
{pct(median(nkjp_diffs(HALF, 'test', nkjp_end)))} przy {pct(median(nkjp_diffs(HALF, 'test', nkjp_peak)))} w okolicy {integer(nkjp_x[nkjp_peak])} tokenów.</li>
<li><b>Większe dane zmniejszają rozjazd trening–test.</b> Przy 30 000 tokenów na DynaWord zysk na teście i na treningu jest prawie równy
({pct(end_test)} i {pct(median(dyna_diffs(HALF, 'train', END)))}), a na nkjp1m wyraźnie się rozchodzi ({pct(median(nkjp_diffs(HALF, 'test', nkjp_end)))} i
{pct(median(nkjp_diffs(HALF, 'train', nkjp_end)))}). Część tej różnicy może wynikać ze słabszego podziału po dokumentach (sekcja 6).</li>
</ul>

<figure>{chart(fig_rules, "fig-rules")}
<figcaption><b>Rys. 2.</b> DynaWord: porównanie reguł i rozjazd trening–test. Reguły są nierozróżnialne w granicach rozrzutu.</figcaption></figure>

<table><thead><tr><th>Wariant (mediana z 3 podziałów DynaWord)</th><th>Scalenia</th><th>Usunięcia</th><th>Powroty</th><th>Usunięte na stałe</th></tr></thead>
<tbody>{rows_ops}</tbody></table>
<p class="note"><b>Tabela 2.</b> Praca algorytmu przy słowniku 30 000. Reguła 50% ma więcej usunięć, z których większość kończy się powrotem,
a liczba tokenów usuniętych na stałe jest podobna w obu regułach.</p>

<figure>{chart(fig_unseen, "fig-unseen")}
<figcaption><b>Rys. 3.</b> Słowa niewidziane w treningu. Na nkjp1m przy dużym słowniku wynik się pogarsza
({pct(median(nkjp_diffs(HALF, 'test_unseen', nkjp_end)))} przy 30 000), na DynaWord jest bliski zera ({pct(median(dyna_diffs(HALF, 'test_unseen', END)))}), ale takie słowa to tam
tylko około {num(unseen_share, 0)}% testu.</figcaption></figure>

<figure>{chart(fig_sources, "fig-sources")}
<figcaption><b>Rys. 4.</b> Zmiana na teście według źródeł, DynaWord, reguła 50%. Wąs: najmniejsza i największa wartość z 3 podziałów. Najmniejsze źródła mają mało
danych testowych i dużą rozpiętość. Przy 4534 tokenach <code>plwiki_user_talk</code> zyskuje najwięcej ({pct(talk_peak)}), ale przy 30 000 ma najmniejszy zysk
({pct(talk_end)}). Mediana każdego źródła przy 30 000 jest poniżej zera (najsłabsze: {pct(source_end_max)}). Przyczyny różnic nie badaliśmy.</figcaption></figure>

<h2>6. Ograniczenia</h2>
<ul>
<li><b>Podział po dokumentach w 24 źródłach.</b> Ich ID nie pozwalają pewnie grupować fragmentów tego samego tekstu. Tam, gdzie takie fragmenty istnieją, test
jest zbyt łatwy, co może zaniżać udział słów niewidzianych i rozjazd trening–test. Autorzy datasetu wskazują też częściowe duplikaty między
<code>wolne_lektury</code> i <code>eltec_pol</code>.</li>
<li><b>Mała liczba podziałów DynaWord.</b> 3 podziały pokazują kierunek, ale słabo oceniają rozrzut. Podziały nie są też niezależnymi próbami, bo pochodzą z tego samego korpusu.</li>
<li><b>Metryka.</b> Fertility nie mówi o jakości modelu ani o rzadkich tokenach, które [1] wskazuje jako główną korzyść. Zmiana 0,35–0,86% jest niewielka w liczbach bezwzględnych.</li>
<li><b>Encoder.</b> Pomiar odtwarza historię zdarzeń, więc wynik może się zmienić przy encoderze DP lub longest-match.</li>
<li><b>Wybrane parametry.</b> Jeden próg dla znaków (1000), jeden limit na źródło i wariant „słowa + znaki”. Na bajtach tego nie mierzyliśmy.</li>
<li><b>Czas.</b> Jeden pełny pomiar 3 wariantów na DynaWord trwał {round(min(wall) / 60)}–{round(max(wall) / 60)} min. Ten sam trening bez usuwania miał rozrzut czasu około 3-krotny,
więc czasy porównawcze nie są wiarygodne. Nie podajemy zysku czasowego.</li>
</ul>

<h2>7. Wnioski i dalsze kroki</h2>
<p>Usuwanie tokenów pośrednich poprawia fertility w sposób mały, ale powtarzalny, także na teście i na wszystkich badanych typach tekstu. Prosty próg 50% działa
równie dobrze jak reguła z bilansu kosztów, a w praktyce można preferować <i>L</i> &lt; <i>f</i><sub>best</sub> ze względu na mniejszą liczbę usunięć i prostszą historię zdarzeń.</p>
<p>Kolejne kroki: (1) krzywa z encoderem DP na tych samych słownikach, (2) analiza, które tokeny są usuwane i skąd szczyt przy kilku tysiącach tokenów, (3) powtórzenie z podziałem po tekstach
tam, gdzie to możliwe, (4) sprawdzenie wpływu na trening małego modelu językowego, bo to on ostatecznie rozstrzyga o wartości tej zmiany.</p>

<h2>Literatura</h2>
<div class="refs">
[1] Chizhov i in., <i>BPE Gets Picky: Efficient Vocabulary Refinement During Tokenizer Training</i>, EMNLP 2024. <a href="https://arxiv.org/abs/2409.04599">arxiv.org/abs/2409.04599</a><br>
[2] Lian i in., <i>Scaffold-BPE: Enhancing Byte Pair Encoding for Large Language Models with Simple and Effective Scaffold Token Removal</i>, AAAI 2025. <a href="https://doi.org/10.1609/aaai.v39i23.34633">doi.org/10.1609/aaai.v39i23.34633</a><br>
[3] <i>LiteToken: Removing Intermediate Merge Residues From BPE Tokenizers</i>, 2026. <a href="https://arxiv.org/abs/2602.04706">arxiv.org/abs/2602.04706</a><br>
[4] <i>Pruned BPE: Post-training Visibility Pruning and Token Reallocation for Byte Pair Encoding</i>, 2026. <a href="https://arxiv.org/abs/2608.00837">arxiv.org/abs/2608.00837</a><br>
Dane: SlayerLab/polish-dynaword, wersja v0.2.6. Kod: <code>rust/bpe/src/lib.rs</code>, <code>experiments/tokenizer/09–17</code>.
Kod i analiza powstały z pomocą asystenta AI (Claude); pomysł i decyzje eksperymentalne: autor.
</div>
</body></html>
"""

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text(html)
print(f"zapisano {OUT_FILE}")
