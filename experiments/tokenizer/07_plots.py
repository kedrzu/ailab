"""Wykresy eksperymentu: fertility (trening vs test) i skalowanie Rusta z liczbą wątków.

Czyta gotowe wyniki, więc zmiana wyglądu nie wymaga ponownego treningu. Tworzy jeden plik
HTML: wystarczy otworzyć go w przeglądarce.

Uruchomienie:  uv run python experiments/tokenizer/07_plots.py
"""

import html
import json
import math
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "data" / "tokenizer" / "fertility" / "metrics.jsonl"
SCALING = ROOT / "data" / "tokenizer" / "rust_scaling.json"
OUT_FILE = ROOT / "data" / "tokenizer" / "raport.html"

# Kolor należy do serii: trening i test mają ten sam kolor na każdym wykresie.
COLORS = {"train": "#2a78d6", "test": "#eb6834", "seen": "#1baf7a", "unseen": "#eda100"}
DARK_COLORS = {"#2a78d6": "#3987e5", "#eb6834": "#d95926", "#1baf7a": "#199e70", "#eda100": "#c98500"}
INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"

rows = [json.loads(line) for line in METRICS.read_text().splitlines()]
scaling = json.loads(SCALING.read_text())
vocab = [row["vocab"] for row in rows]


def show_token(token: str | None) -> str:
    """Spacja na początku tokenu jest istotna, a w HTML byłaby niewidoczna."""
    if token is None:
        return "(sam alfabet)"
    return html.escape(token).replace(" ", "␣").replace("\n", "⏎")


tokens = [show_token(row["token"]) for row in rows]


def base_layout(y_title: str, x_title: str, x_log: bool = True) -> dict:
    return dict(
        height=430,
        margin=dict(l=70, r=130, t=20, b=60),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK),
        separators=", ",  # polski zapis liczb: przecinek dziesiętny
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(
            title=x_title,
            type="log" if x_log else "linear",
            gridcolor=GRID,
            linecolor=AXIS,
            zeroline=False,
        ),
        yaxis=dict(title=y_title, gridcolor=GRID, linecolor=AXIS, zeroline=False),
    )


def line(y: list[float], name: str, color: str, fmt: str = ".3f", with_token: bool = False) -> go.Scatter:
    """Seria na osi rozmiaru słownika. Token w podpowiedzi łączy krzywą z tym, co się scaliło."""
    token_row = "ostatni token: <b>%{customdata}</b><br>" if with_token else ""
    return go.Scatter(
        x=vocab,
        y=y,
        name=name,
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=6, color=color),
        customdata=tokens,
        hovertemplate=token_row + f"{name}: %{{y:{fmt}}}<extra></extra>",
    )


def label_end(fig: go.Figure, x: float, y: float, text: str, yshift: int = 0) -> None:
    """Podpis na końcu linii, żeby kolor nie był jedynym nośnikiem informacji."""
    # Plotly podaje położenie adnotacji na osi logarytmicznej jako log10(x).
    if fig.layout.xaxis.type == "log":
        x = math.log10(x)
    fig.add_annotation(
        x=x, y=y, text=text, showarrow=False, xanchor="left", xshift=8, yshift=yshift, font=dict(color=INK)
    )


last = rows[-1]

# A. Główny: czy fertility spada i czy trening z testem się rozjeżdżają.
fig_a = go.Figure(
    [
        line([r["train"] for r in rows], "trening", COLORS["train"], with_token=True),
        line([r["test"] for r in rows], "test", COLORS["test"]),
    ]
)
fig_a.update_layout(base_layout("tokeny na słowo", "rozmiar słownika (skala log)"))
fig_a.update_yaxes(range=[0.8, 7])
fig_a.add_hline(y=1, line=dict(color=MUTED, width=1, dash="dot"))
fig_a.add_annotation(
    x=0.01, xref="paper", y=1, text="1,0 = każde słowo to jeden token", showarrow=False,
    yshift=10, xanchor="left", font=dict(color=MUTED, size=12),
)
label_end(fig_a, last["vocab"], last["train"], f"trening {last['train']:.2f}".replace(".", ","), yshift=-9)
label_end(fig_a, last["vocab"], last["test"], f"test {last['test']:.2f}".replace(".", ","), yshift=9)

# B. Rozjazd: na wykresie A różnica jest zbyt mała, żeby ją odczytać.
fig_b = go.Figure([line([r["test"] - r["train"] for r in rows], "test − trening", INK, "+.4f", with_token=True)])
fig_b.update_layout(base_layout("różnica fertility (test − trening)", "rozmiar słownika (skala log)"))
fig_b.update_layout(showlegend=False)
fig_b.add_hline(y=0, line=dict(color=AXIS, width=1))

# C. Skąd rozjazd: słowa testu znane z treningu kontra całkiem nowe.
fig_c = go.Figure(
    [
        line([r["test_seen"] for r in rows], "test: słowa widziane", COLORS["seen"], with_token=True),
        line([r["test"] for r in rows], "test: wszystkie", COLORS["test"]),
        line([r["test_unseen"] for r in rows], "test: słowa niewidziane", COLORS["unseen"]),
    ]
)
fig_c.update_layout(base_layout("tokeny na słowo", "rozmiar słownika (skala log)"))
fig_c.update_yaxes(range=[0.8, 10])
fig_c.add_hline(y=1, line=dict(color=MUTED, width=1, dash="dot"))
label_end(fig_c, last["vocab"], last["test_unseen"], f"niewidziane {last['test_unseen']:.2f}".replace(".", ","))
label_end(fig_c, last["vocab"], last["test"], f"wszystkie {last['test']:.2f}".replace(".", ","), yshift=8)
label_end(fig_c, last["vocab"], last["test_seen"], f"widziane {last['test_seen']:.2f}".replace(".", ","), yshift=-8)

# D. Skalowanie: mediana 3 biegów, z idealną linią do porównania.
threads = [r["threads"] for r in scaling["scaling"]]
base_seconds = scaling["scaling"][0]["seconds"]
speedups = [base_seconds / r["seconds"] for r in scaling["scaling"]]
fig_d = go.Figure(
    [
        go.Scatter(
            x=threads, y=threads, name="idealnie", mode="lines",
            line=dict(color=MUTED, width=1, dash="dot"),
            hovertemplate="idealnie: %{y:.0f}x<extra></extra>",
        ),
        go.Scatter(
            x=threads, y=speedups, name="zmierzone", mode="lines+markers",
            line=dict(color=COLORS["train"], width=2), marker=dict(size=8, color=COLORS["train"]),
            hovertemplate="zmierzone: %{y:.2f}x<extra></extra>",
        ),
    ]
)
fig_d.update_layout(base_layout("przyspieszenie względem 1 wątku", "liczba wątków", x_log=False))
fig_d.update_xaxes(tickvals=threads)
fig_d.update_yaxes(rangemode="tozero")
label_end(fig_d, threads[-1], threads[-1], "idealnie")
label_end(fig_d, threads[-1], speedups[-1], f"{speedups[-1]:.2f}x".replace(".", ","))

FIGURES = [
    ("fig-a", "Fertility w trakcie treningu", "Najedź na punkt, żeby zobaczyć ostatni scalony token.", fig_a),
    ("fig-b", "Rozjazd testu i treningu", "Ta sama różnica co wyżej, w skali, która pozwala ją odczytać.", fig_b),
    ("fig-c", "Słowa testu: znane i nowe", "Średnia „wszystkie” leży blisko słów widzianych, bo to one dominują w tekście.", fig_c),
    ("fig-d", "Rust: przyspieszenie z liczbą wątków", f"{scaling['num_merges']:,} scaleń, mediana z 3 biegów. Procesor ma 6 rdzeni wydajnych i 12 energooszczędnych.".replace("10,000", "10 000"), fig_d),
]


def table(headers: list[str], data: list[list[str]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in data)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


fertility_table = table(
    ["słownik", "ostatni token", "trening", "test", "widziane", "niewidziane"],
    [
        [str(r["vocab"]), f"<code>{show_token(r['token'])}</code>", number(r["train"]), number(r["test"]),
         number(r["test_seen"]), number(r["test_unseen"])]
        for r in rows
    ],
)
scaling_table = table(
    ["wątki", "czas [s]", "przyspieszenie"],
    [[str(r["threads"]), number(r["seconds"], 1), number(base_seconds / r["seconds"], 2)] for r in scaling["scaling"]],
)

sections = []
for index, (div_id, title, note, fig) in enumerate(FIGURES):
    # Bibliotekę osadzamy raz, dzięki czemu plik działa bez internetu.
    chart = fig.to_html(full_html=False, include_plotlyjs=index == 0, div_id=div_id, config={"displaylogo": False})
    sections.append(f"<section><h2>{title}</h2><p>{note}</p>{chart}</section>")

# Ciemny motyw to osobne, dobrane kolory, a nie automatyczne odwrócenie jasnych.
dark_script = f"""
<script>
if (matchMedia('(prefers-color-scheme: dark)').matches) {{
  const seriesColors = {json.dumps(DARK_COLORS)};
  const ink = '#c3c2b7', surface = '#1a1a19', grid = '#2c2c2a', axis = '#383835';
  document.querySelectorAll('.plotly-graph-div').forEach((div) => {{
    div.data.forEach((trace, i) => {{
      const color = trace.line && trace.line.color;
      const mapped = color === '{INK}' ? ink : seriesColors[color];
      if (mapped) Plotly.restyle(div, {{'line.color': mapped, 'marker.color': mapped}}, [i]);
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
<html lang="pl"><head><meta charset="utf-8"><title>BPE: fertility i skalowanie</title>
<style>
  :root {{ color-scheme: light dark; --surface: {SURFACE}; --ink: #0b0b0b; --ink2: {INK}; --line: {GRID}; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --line: #2c2c2a; }}
  }}
  body {{ margin: 0 auto; max-width: 1000px; padding: 24px; background: var(--surface); color: var(--ink);
         font-family: {FONT}; }}
  h1 {{ font-size: 24px; }} h2 {{ font-size: 18px; margin: 32px 0 4px; }}
  p {{ color: var(--ink2); margin: 0 0 8px; }}
  details {{ margin-top: 32px; color: var(--ink2); }} summary {{ cursor: pointer; }}
  table {{ border-collapse: collapse; margin-top: 8px; font-variant-numeric: tabular-nums; font-size: 13px; }}
  th, td {{ padding: 3px 12px; border-bottom: 1px solid var(--line); text-align: right; }}
  code {{ white-space: pre; }}
</style></head><body>
<h1>BPE na nkjp1m: fertility i skalowanie</h1>
<p>Wariant słowa + znaki, 10 000 scaleń, podział 75/25 po tekstach źródłowych.</p>
{''.join(sections)}
<details><summary>Dane fertility w tabeli</summary>{fertility_table}</details>
<details><summary>Dane skalowania w tabeli</summary>{scaling_table}</details>
{dark_script}
</body></html>
"""
)
print(f"zapisano {OUT_FILE}")
