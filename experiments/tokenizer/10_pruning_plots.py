"""Wykresy eksperymentu z usuwaniem tokenów: fertility i różnica względem wersji bez poprawki.

Czyta gotowe wyniki z `09_pruning.py`, więc zmiana wyglądu nie wymaga ponownego treningu. Tworzy
jeden plik HTML, który działa bez internetu.

Uruchomienie:  uv run python experiments/tokenizer/10_pruning_plots.py
"""

import json
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "tokenizer" / "pruning"
OUT_FILE = DATA_DIR / "raport.html"

BASELINE = "bez poprawki"
# Kolor należy do wariantu, a styl linii do zbioru danych, więc każda krzywa ma oba oznaczenia.
COLORS = {BASELINE: "#2a78d6", "reguła 50%": "#eb6834", "reguła L < f_best": "#1baf7a"}
DARK_COLORS = {"#2a78d6": "#3987e5", "#eb6834": "#d95926", "#1baf7a": "#199e70"}
MEASURES = {
    "train": ("trening", "solid"),
    "test": ("test", "dash"),
    "test_seen": ("test: słowa widziane", "dot"),
    "test_unseen": ("test: słowa niewidziane", "dashdot"),
}
# Słowa widziane i niewidziane są na starcie ukryte, żeby sześć głównych krzywych było czytelnych.
HIDDEN_MEASURES = {"test_seen", "test_unseen"}

INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"

metrics = json.loads((DATA_DIR / "metrics.json").read_text())
summary = json.loads((DATA_DIR / "summary.json").read_text())
base_rows = metrics[BASELINE]


def layout(y_title: str) -> dict:
    return dict(
        height=470,
        margin=dict(l=70, r=30, t=20, b=60),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK),
        separators=", ",  # polski zapis liczb: przecinek dziesiętny
        hovermode="x unified",
        # Widoczność zmieniają checkboxy; drugi włącznik w legendzie mógłby się z nimi rozjechać.
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)", itemclick=False, itemdoubleclick=False),
        xaxis=dict(title="rozmiar słownika (skala log)", type="log", gridcolor=GRID, linecolor=AXIS, zeroline=False),
        yaxis=dict(title=y_title, gridcolor=GRID, linecolor=AXIS, zeroline=False),
    )


def trace(variant: str, measure: str, y: list[float], fmt: str) -> go.Scatter:
    label, dash = MEASURES[measure]
    name = f"{variant}, {label}"
    return go.Scatter(
        x=[row["vocab"] for row in metrics[variant]],
        y=y,
        name=name,
        mode="lines",
        line=dict(color=COLORS[variant], width=2, dash=dash),
        # Po tych polach checkboxy rozpoznają krzywą.
        meta=dict(variant=variant, measure=measure),
        visible=measure not in HIDDEN_MEASURES,
        hovertemplate=f"{name}: %{{y:{fmt}}}<extra></extra>",
    )


fig_values = go.Figure(
    [
        trace(variant, measure, [row[measure] for row in rows], ".3f")
        for variant, rows in metrics.items()
        for measure in MEASURES
    ]
)
fig_values.update_layout(layout("tokeny na słowo"))
fig_values.add_hline(y=1, line=dict(color=MUTED, width=1, dash="dot"))

fig_diff = go.Figure(
    [
        trace(
            variant,
            measure,
            [100 * (row[measure] / base[measure] - 1) for row, base in zip(rows, base_rows)],
            "+.2f",
        )
        for variant, rows in metrics.items()
        if variant != BASELINE
        for measure in MEASURES
    ]
)
fig_diff.update_layout(layout("zmiana fertility [%]"))
fig_diff.add_hline(y=0, line=dict(color=MUTED, width=1))
fig_diff.add_annotation(
    x=0.99, xref="paper", y=0, text="0 = jak bez poprawki; poniżej = mniej tokenów", showarrow=False,
    yshift=10, xanchor="right", font=dict(color=MUTED, size=12),
)


def number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def checkbox(group: str, value: str, label: str, checked: bool) -> str:
    state = " checked" if checked else ""
    return f'<label><input type="checkbox" data-group="{group}" value="{value}"{state}> {label}</label>'


variant_boxes = "".join(checkbox("variant", name, name, True) for name in metrics)
measure_boxes = "".join(
    checkbox("measure", key, label, key not in HIDDEN_MEASURES) for key, (label, _) in MEASURES.items()
)

summary_table = table(
    ["wariant", "scalenia", "usunięcia", "powroty", "usunięte na stałe", "trening [s]"],
    [
        [
            name,
            f"{row['scalenia']:,}".replace(",", " "),
            f"{row['usunięcia']:,}".replace(",", " "),
            f"{row['powroty']:,}".replace(",", " "),
            f"{row['usunięte na stałe']:,}".replace(",", " ") + f" ({number(row['usunięte na stałe [%]'], 1)}%)",
            number(row["trening [s]"], 1),
        ]
        for name, row in summary.items()
    ],
)
data_table = table(
    ["słownik", "wariant", "trening", "test", "widziane", "niewidziane"],
    [
        [str(row["vocab"]), variant, number(row["train"]), number(row["test"]),
         number(row["test_seen"]), number(row["test_unseen"])]
        for variant, rows in metrics.items()
        for row in rows
    ],
)

# Bibliotekę osadzamy raz, dzięki czemu plik działa bez internetu.
chart_values = fig_values.to_html(full_html=False, include_plotlyjs=True, div_id="fig-values", config={"displaylogo": False})
chart_diff = fig_diff.to_html(full_html=False, include_plotlyjs=False, div_id="fig-diff", config={"displaylogo": False})

script = f"""
<script>
const plots = document.querySelectorAll('.plotly-graph-div');

function applyCheckboxes() {{
  const checked = (group) => new Set(
    [...document.querySelectorAll(`input[data-group="${{group}}"]:checked`)].map((box) => box.value)
  );
  const variants = checked('variant');
  const measures = checked('measure');
  plots.forEach((div) => {{
    const visible = div.data.map((t) => variants.has(t.meta.variant) && measures.has(t.meta.measure));
    Plotly.restyle(div, {{visible}});
  }});
}}
document.querySelectorAll('input[data-group]').forEach((box) => box.addEventListener('change', applyCheckboxes));

// Ciemny motyw to osobne, dobrane kolory, a nie automatyczne odwrócenie jasnych.
if (matchMedia('(prefers-color-scheme: dark)').matches) {{
  const seriesColors = {json.dumps(DARK_COLORS)};
  const ink = '#c3c2b7', surface = '#1a1a19', grid = '#2c2c2a', axis = '#383835';
  plots.forEach((div) => {{
    div.data.forEach((trace, i) => {{
      Plotly.restyle(div, {{'line.color': seriesColors[trace.line.color]}}, [i]);
    }});
    const annotations = (div.layout.annotations || []).map((a) => ({{...a, font: {{...a.font, color: ink}}}}));
    Plotly.relayout(div, {{
      paper_bgcolor: surface, plot_bgcolor: surface, 'font.color': ink, annotations,
      'xaxis.gridcolor': grid, 'yaxis.gridcolor': grid, 'xaxis.linecolor': axis, 'yaxis.linecolor': axis,
    }});
  }});
}}
</script>
"""

OUT_FILE.write_text(
    f"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><title>BPE: usuwanie tokenów</title>
<style>
  :root {{ color-scheme: light dark; --surface: {SURFACE}; --ink: #0b0b0b; --ink2: {INK}; --line: {GRID}; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --line: #2c2c2a; }}
  }}
  body {{ margin: 0 auto; max-width: 1000px; padding: 24px; background: var(--surface); color: var(--ink);
         font-family: {FONT}; }}
  h1 {{ font-size: 24px; }} h2 {{ font-size: 18px; margin: 32px 0 4px; }}
  p {{ color: var(--ink2); margin: 0 0 8px; }}
  .controls {{ position: sticky; top: 0; z-index: 10; background: var(--surface); padding: 10px 0;
              border-bottom: 1px solid var(--line); color: var(--ink2); }}
  .controls div {{ display: flex; flex-wrap: wrap; gap: 4px 18px; margin: 3px 0; }}
  .controls b {{ min-width: 80px; font-weight: 600; }}
  label {{ cursor: pointer; }}
  details {{ margin-top: 32px; color: var(--ink2); }} summary {{ cursor: pointer; }}
  table {{ border-collapse: collapse; margin-top: 8px; font-variant-numeric: tabular-nums; font-size: 13px; }}
  th, td {{ padding: 3px 12px; border-bottom: 1px solid var(--line); text-align: right; }}
</style></head><body>
<h1>BPE na nkjp1m: z usuwaniem tokenów i bez niego</h1>
<p>Słowa + znaki, słownik do 30 000, podział 75/25 po tekstach źródłowych. Tokeny wybierane tylko na
treningu, na każdym tekście odtwarzana ta sama historia scaleń i usunięć.</p>
<div class="controls">
  <div><b>Warianty</b>{variant_boxes}</div>
  <div><b>Dane</b>{measure_boxes}</div>
</div>
<section><h2>Fertility a rozmiar słownika</h2>
<p>Kolor to wariant, styl linii to zbiór danych. Krzywe wariantów prawie się pokrywają: różnice widać na wykresie niżej.</p>
{chart_values}</section>
<section><h2>Różnica względem wersji bez poprawki</h2>
<p>Ta sama fertility, ale w procentach. Poniżej zera wariant daje mniej tokenów na słowo.</p>
{chart_diff}</section>
<section><h2>Ile pracy kosztowało usuwanie</h2>{summary_table}</section>
<details><summary>Dane krzywych w tabeli</summary>{data_table}</details>
{script}
</body></html>
"""
)
print(f"zapisano {OUT_FILE}")
