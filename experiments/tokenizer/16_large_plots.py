"""Wykresy pomiaru na 26 źródłach: zmiana fertility po usuwaniu tokenów, ogółem i według źródeł.

Linia lub kropka to mediana z 3 podziałów, a pasmo lub wąs sięga od najmniejszej do największej wartości.
Różnice są liczone parami, na tym samym podziale. Czyta wyniki z `15_pruning_large.py`.

Uruchomienie:  uv run python experiments/tokenizer/16_large_plots.py
"""

import json
import statistics
from pathlib import Path

import plotly.graph_objects as go

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "tokenizer" / "pruning"
OUT_FILE = DATA_DIR / "raport_duze_dane.html"

BASELINE = "bez poprawki"
COLORS = {"reguła 50%": "#eb6834", "reguła L < f_best": "#1baf7a"}
DARK_COLORS = {"#eb6834": "#d95926", "#1baf7a": "#199e70"}
SECTIONS = {
    "test": "Test: wszystkie słowa",
    "train": "Trening",
    "test_seen": "Test: słowa widziane w treningu",
    "test_unseen": "Test: słowa niewidziane",
}
# Rozmiar przy największym zysku i największy mierzony.
SOURCE_VOCABS = [4_534, 30_000]

INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"

data = json.loads((DATA_DIR / "large.json").read_text())
vocab = data["vocab"]
splits = data["splits"]


def fertility(split: dict, variant: str, measure: str, i: int) -> float:
    tokens = split["variants"][variant]["tokens"]
    words = split["words"]
    if measure == "test":
        return (tokens["test_seen"][i] + tokens["test_unseen"][i]) / (words["test_seen"] + words["test_unseen"])
    return tokens[measure][i] / words[measure]


def diffs(variant: str, measure: str, i: int) -> list[float]:
    return [100 * (fertility(s, variant, measure, i) / fertility(s, BASELINE, measure, i) - 1) for s in splits]


def source_diffs(variant: str, source: str, i: int) -> list[float]:
    return [
        100 * (s["variants"][variant]["source_tokens"][source][i] / s["variants"][BASELINE]["source_tokens"][source][i] - 1)
        for s in splits
    ]


def rgba(color: str, alpha: float) -> str:
    red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def base_layout(height: int) -> dict:
    return dict(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK),
        separators=", ",  # polski zapis liczb: przecinek dziesiętny
        # Widoczność zmieniają checkboxy; drugi włącznik w legendzie mógłby się z nimi rozjechać.
        legend=dict(orientation="h", x=0, bgcolor="rgba(0,0,0,0)", itemclick=False, itemdoubleclick=False),
    )


def curve_traces(variant: str, measure: str) -> list[go.Scatter]:
    points = [diffs(variant, measure, i) for i in range(len(vocab))]
    medians = [statistics.median(p) for p in points]
    lows = [min(p) for p in points]
    highs = [max(p) for p in points]
    color = COLORS[variant]
    meta = dict(variant=variant)
    # Pasmo to dwie niewidoczne granice: dolna wypełnia obszar do górnej.
    return [
        go.Scatter(x=vocab, y=highs, mode="lines", line=dict(width=0, color=color), meta=meta,
                   showlegend=False, hoverinfo="skip"),
        go.Scatter(x=vocab, y=lows, mode="lines", line=dict(width=0, color=color), fill="tonexty",
                   fillcolor=rgba(color, 0.2), meta=meta, showlegend=False, hoverinfo="skip"),
        go.Scatter(
            x=vocab, y=medians, name=variant, mode="lines", line=dict(width=2, color=color), meta=meta,
            customdata=list(zip(lows, highs)),
            hovertemplate=f"{variant}: <b>%{{y:+.2f}}%</b> (od %{{customdata[0]:+.2f}} do %{{customdata[1]:+.2f}})<extra></extra>",
        ),
    ]


def curve_figure(measure: str) -> go.Figure:
    fig = go.Figure([trace for variant in COLORS for trace in curve_traces(variant, measure)])
    fig.update_layout(base_layout(380))
    fig.update_layout(
        margin=dict(l=70, r=30, t=20, b=60),
        hovermode="x unified",
        xaxis=dict(title="rozmiar słownika (skala log)", type="log", gridcolor=GRID, linecolor=AXIS, zeroline=False),
        yaxis=dict(title="zmiana fertility [%]", gridcolor=GRID, linecolor=AXIS, zeroline=False),
    )
    fig.update_layout(legend_y=1.1)
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    return fig


def sorted_sources(i: int) -> list[str]:
    """Na osi Y pierwsze jest to, co na górze wykresu, czyli największy zysk."""
    return sorted(
        splits[0]["source_words"],
        key=lambda source: statistics.median(source_diffs("reguła 50%", source, i)),
        reverse=True,
    )


def source_figure(size: int) -> go.Figure:
    i = vocab.index(size)
    sources_sorted = sorted_sources(i)
    traces = []
    for variant, color in COLORS.items():
        points = [source_diffs(variant, source, i) for source in sources_sorted]
        medians = [statistics.median(p) for p in points]
        traces.append(
            go.Scatter(
                x=medians,
                y=sources_sorted,
                name=variant,
                mode="markers",
                marker=dict(size=9, color=color, line=dict(width=2, color=SURFACE)),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[max(p) - m for p, m in zip(points, medians)],
                    arrayminus=[m - min(p) for p, m in zip(points, medians)],
                    color=color,
                    thickness=2,
                    width=0,
                ),
                meta=dict(variant=variant),
                customdata=[[min(p), max(p)] for p in points],
                hovertemplate=f"%{{y}}, {variant}: <b>%{{x:+.2f}}%</b> (od %{{customdata[0]:+.2f}} do %{{customdata[1]:+.2f}})<extra></extra>",
            )
        )
    fig = go.Figure(traces)
    fig.update_layout(base_layout(680))
    fig.update_layout(
        margin=dict(l=210, r=30, t=30, b=55),
        hovermode="closest",
        xaxis=dict(title="zmiana fertility [%]", gridcolor=GRID, linecolor=AXIS, zeroline=False),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, dtick=1),
    )
    fig.update_layout(legend_y=1.05)
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    return fig


def number(value: float) -> str:
    return f"{value:+.3f}".replace(".", ",").replace("-", "−")


sections = []
figures = [(f"fig-{measure}", title, curve_figure(measure)) for measure, title in SECTIONS.items()]
figures += [
    (f"fig-source-{size}", f"Według źródeł: test, słownik {size:,}".replace(",", " "), source_figure(size))
    for size in SOURCE_VOCABS
]
for index, (div_id, title, fig) in enumerate(figures):
    # Bibliotekę osadzamy raz, dzięki czemu plik działa bez internetu.
    chart = fig.to_html(full_html=False, include_plotlyjs=index == 0, div_id=div_id, config={"displaylogo": False})
    sections.append(f"<section><h2>{title}</h2>{chart}</section>")

curve_rows = "".join(
    f"<tr><td>{vocab[i]}</td><td>{variant}</td><td>{title}</td><td>{number(statistics.median(diffs(variant, measure, i)))}</td>"
    f"<td>{number(min(diffs(variant, measure, i)))}</td><td>{number(max(diffs(variant, measure, i)))}</td></tr>"
    for variant in COLORS
    for measure, title in SECTIONS.items()
    for i in range(len(vocab))
)
data_table = (
    "<table><thead><tr><th>słownik</th><th>wariant</th><th>zbiór</th><th>mediana [%]</th><th>min</th><th>max</th></tr>"
    f"</thead><tbody>{curve_rows}</tbody></table>"
)
source_rows = "".join(
    f"<tr><td>{source}</td><td>{variant}</td><td>{size}</td>"
    f"<td>{number(statistics.median(source_diffs(variant, source, vocab.index(size))))}</td>"
    f"<td>{number(min(source_diffs(variant, source, vocab.index(size))))}</td>"
    f"<td>{number(max(source_diffs(variant, source, vocab.index(size))))}</td>"
    f"<td>{splits[0]['source_words'][source]}</td></tr>"
    for source in sorted(splits[0]["source_words"])
    for variant in COLORS
    for size in SOURCE_VOCABS
)
source_table = (
    "<table><thead><tr><th>źródło</th><th>wariant</th><th>słownik</th><th>mediana [%]</th><th>min</th><th>max</th>"
    f"<th>słowa w teście (podział 0)</th></tr></thead><tbody>{source_rows}</tbody></table>"
)

boxes = "".join(f'<label><input type="checkbox" value="{name}" checked> {name}</label>' for name in COLORS)

script = f"""
<script>
const plots = document.querySelectorAll('.plotly-graph-div');
function applyCheckboxes() {{
  const variants = new Set([...document.querySelectorAll('input:checked')].map((box) => box.value));
  plots.forEach((div) => Plotly.restyle(div, {{visible: div.data.map((t) => variants.has(t.meta.variant))}}));
}}
document.querySelectorAll('input').forEach((box) => box.addEventListener('change', applyCheckboxes));

// Ciemny motyw to osobne, dobrane kolory, a nie automatyczne odwrócenie jasnych.
if (matchMedia('(prefers-color-scheme: dark)').matches) {{
  const seriesColors = {json.dumps(DARK_COLORS)};
  const ink = '#c3c2b7', surface = '#1a1a19', grid = '#2c2c2a', axis = '#383835';
  plots.forEach((div) => {{
    div.data.forEach((trace, i) => {{
      const source = trace.mode === 'markers' ? trace.marker.color : trace.line.color;
      const color = seriesColors[source];
      const [r, g, b] = [1, 3, 5].map((p) => parseInt(color.slice(p, p + 2), 16));
      const update = trace.mode === 'markers'
        ? {{'marker.color': color, 'marker.line.color': surface, 'error_x.color': color}}
        : {{'line.color': color}};
      if (trace.fill) update.fillcolor = `rgba(${{r}},${{g}},${{b}},0.25)`;
      Plotly.restyle(div, update, [i]);
    }});
    Plotly.relayout(div, {{
      paper_bgcolor: surface, plot_bgcolor: surface, 'font.color': ink,
      'xaxis.gridcolor': grid, 'yaxis.gridcolor': grid, 'xaxis.linecolor': axis, 'yaxis.linecolor': axis,
    }});
  }});
}}
</script>
"""

OUT_FILE.write_text(
    f"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><title>BPE: usuwanie tokenów na 26 źródłach</title>
<style>
  :root {{ color-scheme: light dark; --surface: {SURFACE}; --ink: #0b0b0b; --ink2: {INK}; --line: {GRID}; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --line: #2c2c2a; }}
  }}
  body {{ margin: 0 auto; max-width: 1000px; padding: 24px; background: var(--surface); color: var(--ink);
         font-family: {FONT}; }}
  h1 {{ font-size: 24px; }} h2 {{ font-size: 18px; margin: 28px 0 4px; }}
  p {{ color: var(--ink2); margin: 0 0 8px; }}
  .controls {{ position: sticky; top: 0; z-index: 10; background: var(--surface); padding: 10px 0;
              border-bottom: 1px solid var(--line); color: var(--ink2); display: flex; flex-wrap: wrap; gap: 4px 18px; }}
  .controls b {{ font-weight: 600; }}
  label {{ cursor: pointer; }}
  details {{ margin-top: 32px; color: var(--ink2); }} summary {{ cursor: pointer; }}
  table {{ border-collapse: collapse; margin-top: 8px; font-variant-numeric: tabular-nums; font-size: 13px; }}
  th, td {{ padding: 3px 12px; border-bottom: 1px solid var(--line); text-align: right; }}
</style></head><body>
<h1>BPE: usuwanie tokenów na 26 źródłach DynaWord</h1>
<p>Limit 20 mln tokenów na źródło, {len(splits)} podziały 75/25. Zmiana fertility względem wersji bez poprawki, liczona
parami na tym samym podziale. Linia lub kropka to mediana, pasmo lub wąs sięga od najmniejszej do największej wartości.
Poniżej zera wariant daje mniej tokenów na słowo. Każdy wykres ma własną skalę osi.</p>
<div class="controls"><b>Warianty</b>{boxes}</div>
{''.join(sections)}
<details><summary>Dane krzywych w tabeli</summary>{data_table}</details>
<details><summary>Dane źródeł w tabeli</summary>{source_table}</details>
{script}
</body></html>
"""
)
print(f"zapisano {OUT_FILE}")
