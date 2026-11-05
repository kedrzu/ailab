"""Wykres: zmiana fertility po usuwaniu tokenów na 10 podziałach trening/test.

Linia to mediana z podziałów, a pasmo pokazuje najmniejszą i największą wartość. Każda różnica jest
liczona parami, na tym samym podziale. Czyta wyniki z `11_pruning_splits.py`.

Uruchomienie:  uv run python experiments/tokenizer/12_splits_plots.py
"""

import json
import statistics
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "tokenizer" / "pruning"
OUT_FILE = DATA_DIR / "raport_podzialy.html"

BASELINE = "bez poprawki"
COLORS = {"reguła 50%": "#eb6834", "reguła L < f_best": "#1baf7a"}
DARK_COLORS = {"#eb6834": "#d95926", "#1baf7a": "#199e70"}
SECTIONS = {
    "test": "Test: wszystkie słowa",
    "train": "Trening",
    "test_seen": "Test: słowa widziane w treningu",
    "test_unseen": "Test: słowa niewidziane",
}

INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"

data = json.loads((DATA_DIR / "splits.json").read_text())
vocab = data["vocab"]
splits = data["splits"]


def diffs_by_point(variant: str, measure: str) -> list[list[float]]:
    """Dla każdego punktu siatki: zmiana w procentach na każdym podziale."""
    return [
        [
            100 * (s["variants"][variant]["series"][measure][i] / s["variants"][BASELINE]["series"][measure][i] - 1)
            for s in splits
        ]
        for i in range(len(vocab))
    ]


def rgba(color: str, alpha: float) -> str:
    red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def traces(variant: str, measure: str) -> list[go.Scatter]:
    points = diffs_by_point(variant, measure)
    medians = [statistics.median(p) for p in points]
    lows = [min(p) for p in points]
    highs = [max(p) for p in points]
    below = [sum(d < 0 for d in p) for p in points]
    meta = dict(variant=variant)
    color = COLORS[variant]
    # Pasmo to dwie niewidoczne granice: dolna wypełnia obszar do górnej.
    return [
        go.Scatter(x=vocab, y=highs, mode="lines", line=dict(width=0, color=color), meta=meta,
                   showlegend=False, hoverinfo="skip"),
        go.Scatter(x=vocab, y=lows, mode="lines", line=dict(width=0, color=color), fill="tonexty",
                   fillcolor=rgba(color, 0.2), meta=meta, showlegend=False, hoverinfo="skip"),
        go.Scatter(
            x=vocab, y=medians, name=variant, mode="lines", line=dict(width=2, color=color), meta=meta,
            customdata=list(zip(lows, highs, below)),
            hovertemplate=(
                f"{variant}: <b>%{{y:+.2f}}%</b> (od %{{customdata[0]:+.2f}} do %{{customdata[1]:+.2f}}), "
                f"poniżej zera: %{{customdata[2]}}/{len(splits)}<extra></extra>"
            ),
        ),
    ]


def figure(measure: str) -> go.Figure:
    fig = go.Figure([trace for variant in COLORS for trace in traces(variant, measure)])
    fig.update_layout(
        height=380,
        margin=dict(l=70, r=30, t=20, b=60),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK),
        separators=", ",  # polski zapis liczb: przecinek dziesiętny
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)", itemclick=False, itemdoubleclick=False),
        xaxis=dict(title="rozmiar słownika (skala log)", type="log", gridcolor=GRID, linecolor=AXIS, zeroline=False),
        yaxis=dict(title="zmiana fertility [%]", gridcolor=GRID, linecolor=AXIS, zeroline=False),
    )
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    return fig


def number(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}".replace(".", ",").replace("-", "−")


sections = []
for index, (measure, title) in enumerate(SECTIONS.items()):
    # Bibliotekę osadzamy raz, dzięki czemu plik działa bez internetu.
    chart = figure(measure).to_html(
        full_html=False, include_plotlyjs=index == 0, div_id=f"fig-{measure}", config={"displaylogo": False}
    )
    sections.append(f"<section><h2>{title}</h2>{chart}</section>")

rows = []
for variant in COLORS:
    for measure, title in SECTIONS.items():
        for i, points in enumerate(diffs_by_point(variant, measure)):
            rows.append(
                f"<tr><td>{vocab[i]}</td><td>{variant}</td><td>{title}</td><td>{number(statistics.median(points))}</td>"
                f"<td>{number(min(points))}</td><td>{number(max(points))}</td>"
                f"<td>{sum(d < 0 for d in points)}/{len(points)}</td></tr>"
            )
data_table = (
    "<table><thead><tr><th>słownik</th><th>wariant</th><th>zbiór</th><th>mediana [%]</th><th>min</th><th>max</th>"
    f"<th>poniżej zera</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
)

boxes = "".join(
    f'<label><input type="checkbox" value="{name}" checked> {name}</label>' for name in COLORS
)

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
      const color = seriesColors[trace.line.color];
      const [r, g, b] = [1, 3, 5].map((p) => parseInt(color.slice(p, p + 2), 16));
      const update = {{'line.color': color}};
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
<html lang="pl"><head><meta charset="utf-8"><title>BPE: usuwanie tokenów na {len(splits)} podziałach</title>
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
<h1>BPE na nkjp1m: usuwanie tokenów na {len(splits)} podziałach</h1>
<p>Zmiana fertility względem wersji bez poprawki, liczona parami na tym samym podziale 75/25. Linia to
mediana, pasmo sięga od najmniejszej do największej wartości. Poniżej zera wariant daje mniej tokenów na
słowo. Każdy wykres ma własną skalę osi Y.</p>
<div class="controls"><b>Warianty</b>{boxes}</div>
{''.join(sections)}
<details><summary>Dane w tabeli</summary>{data_table}</details>
{script}
</body></html>
"""
)
print(f"zapisano {OUT_FILE}")
