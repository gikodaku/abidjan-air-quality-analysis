"""AAQA model monitoring dashboard: `streamlit run dashboard/monitoring.py --server.port 8502`."""

from __future__ import annotations

import base64
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from maps import load_grid, sampling_map

from aaqa import config as paths
from aaqa import monitoring as mon
from aaqa.calibration import Calibration
from aaqa.config import load_config
from aaqa.data.embassy import api_key, daily_means
from aaqa.models.predictor import AirQualityPredictor
from aaqa.pipeline import run
from aaqa.pollutants import POLLUTANTS

LOGO_PATH = Path(__file__).parent / "assets" / "koscon-logo.png"

# Light theme tokens; chart colours validated with the dataviz palette checker on white
SURFACE, PAGE = "#ffffff", "#f3f6f6"
INK, INK_2, MUTED = "#13262b", "#3f5358", "#5f7377"
GRID, AXIS = "#e3e9e9", "#c5d0d1"
MODEL, BASELINE = "#0b84b8", "#d95926"
# One colour per model, fixed order (CVD-safe and >= 3:1 contrast on white)
MODEL_COLORS = {"gradient_boosting": MODEL, "persistence": BASELINE, "ridge": "#6250d6",
                "climatology": "#138a60"}
BAND = "rgba(11,132,184,0.16)"
# Status badges: tinted background, dark text, icon + label (never colour alone)
CHIP = {"good": ("#e1f4e7", "#0a6b35"), "warning": ("#fff1cc", "#7a5300"),
        "critical": ("#fde0e0", "#a3102a"), "collecting": ("#eceff0", "#3f5358")}

st.set_page_config(page_title="AAQA — Model monitoring", page_icon="📡", layout="wide")


# ── Data ──────────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_inputs() -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, dict]:
    daily = pd.read_csv(paths.DAILY_PROCESSED, index_col="date",
                        parse_dates=["date"]).asfreq("D")
    metrics = json.loads(paths.METRICS_PATH.read_text())
    live = mon.score(mon.load_log(paths.MONITORING_LOG), daily)
    backtest = mon.load_backtest(paths.REPORTS_DIR, metrics)
    meta = AirQualityPredictor.load(paths.MODEL_PATH).metadata
    return daily, metrics, live, backtest, meta


@st.cache_data(ttl=60)
def load_model_predictions(source: str) -> pd.DataFrame:
    if source.startswith("Live"):
        daily = pd.read_csv(paths.DAILY_PROCESSED, index_col="date",
                            parse_dates=["date"]).asfreq("D")
        return mon.live_models(mon.score(mon.load_log(paths.MONITORING_LOG), daily))
    return mon.load_backtest_models(paths.REPORTS_DIR)


@st.cache_data
def logo_data_uri() -> str:
    return "data:image/png;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode()


# ── Rendering helpers ─────────────────────────────────────────────────────────────────────

def html(markup: str) -> None:
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def chip(status: str, text: str | None = None) -> str:
    _, icon, label = mon.STATUS[status]
    background, ink = CHIP[status]
    return (f'<span class="chip" style="background:{background};color:{ink}">'
            f"{icon} {text or label}</span>")


def tile(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="tile"><div class="k">{label}</div><div class="v">{value}</div>'
            f'<div class="s">{sub}</div></div>')


def fmt(value: float, digits: int = 2, suffix: str = "") -> str:
    return "—" if pd.isna(value) else f"{value:.{digits}f}{suffix}"


def metric_value(metric: str, value: float) -> str:
    if pd.isna(value):
        return "—"
    if metric == "aqi_category_accuracy":
        return f"{value:.0%}"
    if metric == "r2":
        return f"{value:.2f}"
    return f"{value:+.2f}" if metric == "bias" else f"{value:.2f}"


def pct(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:+.0%}" if abs(value) < 10 else "—"


def chart_layout(fig: go.Figure, title: str, height: int = 320) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 16, "color": INK}, "x": 0.01, "xanchor": "left"},
        height=height, margin={"t": 56, "b": 40, "l": 60, "r": 20},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, hovermode="x unified",
        font={"color": INK_2, "family": "Montserrat, system-ui, sans-serif", "size": 13},
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 1, "xanchor": "right",
                "font": {"color": INK, "size": 13}},
        hoverlabel={"bgcolor": SURFACE, "font": {"color": INK, "size": 13}},
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False,
                     tickfont={"color": INK_2, "size": 12})
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False,
                     tickfont={"color": INK_2, "size": 12})
    return fig


def style() -> None:
    html(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
    .stApp {{ background: {PAGE}; }}
    .stApp, .stApp p, .stApp div, .stApp label, .stApp td, .stApp th,
    .stApp span:not([data-testid="stIconMaterial"]) {{
        font-family: 'Montserrat', system-ui, sans-serif; color: {INK}; }}
    .stApp {{ font-size: 16px; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.2rem; }}
    .brandbar {{ display: flex; align-items: center; gap: 1rem; padding: 0.9rem 1.2rem;
                 border-radius: 18px; margin-bottom: 1rem;
                 background: linear-gradient(90deg, #006079, #0EA7DC); }}
    .brandbar .logo {{ background: #fff; border-radius: 12px; padding: 6px 12px;
                       display: inline-flex; }}
    .brandbar .logo img {{ height: 54px; }}
    .brandbar .title {{ font-size: 1.4rem; font-weight: 800; color: #fff; }}
    .brandbar .sub {{ color: rgba(255,255,255,0.9); font-size: 0.85rem; }}
    .brandbar .live {{ margin-left: auto; color: #fff; font-weight: 700; font-size: 0.85rem; }}
    .pulse {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%;
              background: #3FAB3D; margin-right: 6px; animation: pulse 1.6s infinite; }}
    @keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(63,171,61,0.7); }}
                        70% {{ box-shadow: 0 0 0 8px rgba(63,171,61,0); }}
                        100% {{ box-shadow: 0 0 0 0 rgba(63,171,61,0); }} }}
    .tiles {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.8rem;
              margin-bottom: 1rem; }}
    .tile {{ background: {SURFACE}; border-radius: 16px; padding: 1rem 1.1rem;
             box-shadow: 0 4px 18px rgba(16,40,44,0.07); }}
    .tile .k {{ font-size: 0.8rem; color: {INK_2}; text-transform: uppercase;
                letter-spacing: 0.06em; font-weight: 700; }}
    .tile .v {{ font-size: 2.1rem; font-weight: 800; color: {INK}; margin: 0.2rem 0; }}
    .tile .s {{ font-size: 0.92rem; color: {INK_2}; line-height: 1.4; }}
    .card {{ background: {SURFACE}; border-radius: 16px; padding: 1.1rem 1.3rem;
             margin-bottom: 1.1rem; box-shadow: 0 4px 18px rgba(16,40,44,0.07);
             font-size: 1rem; line-height: 1.55; }}
    .card h3 {{ font-size: 1.2rem; font-weight: 800; color: {INK}; margin: 0 0 0.3rem;
                padding: 0; }}
    .card .lead {{ color: {INK_2}; font-size: 0.95rem; margin-bottom: 0.8rem; }}
    .card .note {{ color: {MUTED}; font-size: 0.88rem; margin-top: 0.7rem; }}
    table.matrix {{ width: 100%; border-collapse: collapse; font-size: 1rem; }}
    table.matrix th {{ color: {INK_2}; font-size: 0.8rem; text-transform: uppercase;
                       letter-spacing: 0.06em; text-align: left; padding: 0.6rem 0.7rem;
                       border-bottom: 2px solid {AXIS}; }}
    table.matrix td {{ padding: 0.75rem 0.7rem; border-bottom: 1px solid {GRID};
                       color: {INK}; vertical-align: middle; font-variant-numeric: tabular-nums; }}
    table.matrix tr:hover td {{ background: #f7fafa; }}
    table.matrix td .m {{ color: {INK_2}; font-size: 0.9rem; margin-top: 0.35rem; }}
    table.matrix td .big {{ font-size: 1.15rem; font-weight: 800; }}
    .chip {{ display: inline-flex; align-items: center; gap: 0.35rem; border-radius: 999px;
             padding: 0.25rem 0.7rem; font-size: 0.9rem; font-weight: 700; white-space: nowrap; }}
    .dot {{ width: 0.7rem; height: 0.7rem; border-radius: 50%; display: inline-block; }}
    .stat {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.8rem;
             margin-top: 0.6rem; }}
    .stat div {{ color: {INK_2}; font-size: 0.9rem; }}
    .stat b {{ display: block; color: {INK}; font-size: 1.5rem; font-weight: 800; }}
    div[data-testid="stExpander"] details {{ background: {SURFACE}; border-radius: 14px;
        border: none; box-shadow: 0 4px 18px rgba(16,40,44,0.07); }}
    @media (max-width: 900px) {{ .tiles {{ grid-template-columns: repeat(2, 1fr); }}
                                 .stat {{ grid-template-columns: repeat(3, 1fr); }} }}
    </style>""")


# ── Page ──────────────────────────────────────────────────────────────────────────────────

style()
html(f"""<div class="brandbar"><div class="logo"><img src="{logo_data_uri()}" alt="KOSCON"></div>
<div><div class="title">Model monitoring</div>
<div class="sub">AAQA · 6 pollutants × 3 horizons · Abidjan</div></div>
<div class="live"><span class="pulse"></span>Live · refreshes every minute</div></div>""")

missing = [p for p in (paths.DAILY_PROCESSED, paths.METRICS_PATH, paths.MODEL_PATH)
           if not p.exists()]
if missing:
    st.warning("No model or data yet. Run `make all` in the project folder, then reload.")
    st.stop()

with st.container():
    action_left, action_mid, _ = st.columns([1, 1, 3])
    if action_left.button("🔄 Update data & forecast", help="Fetch the latest data and issue a "
                          "new forecast (about 1 min). No retraining."):
        with st.spinner("Fetching data and forecasting…"):
            run(["fetch", "process", "forecast"])
        st.cache_data.clear()
        st.rerun()
    if action_mid.button("🧠 Retrain all models", help="Retrain and re-evaluate the 18 models "
                         "(about 4 min)."):
        with st.spinner("Retraining…"):
            run(["train", "forecast"])
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

daily, metrics, live, backtest, meta = load_inputs()
live_scored = int(live["observed"].notna().sum()) if not live.empty else 0

filters = st.columns([1.4, 1, 1, 1, 1])
default_source = 0 if live_scored >= mon.MIN_SCORED * 6 else 1
source = filters[0].radio("Source", ["Live forecasts", "Backtest (test year)"],
                          index=default_source, horizontal=True,
                          help="Live: forecasts issued by the app, scored when the real value "
                               "arrives. Backtest: out-of-sample predictions of the test year.")
pollutant = filters[1].selectbox("Pollutant", list(POLLUTANTS),
                                 format_func=lambda k: POLLUTANTS[k].name)
horizon = filters[2].selectbox("Horizon", [1, 2, 3], format_func=lambda h: f"D+{h}")
days = filters[3].selectbox("Window", [14, 30, 90, 365], index=1,
                            format_func=lambda d: f"Last {d} days")
metric = filters[4].selectbox("Metric", list(mon.METRICS),
                              format_func=lambda m: mon.METRICS[m][0],
                              help="MAE and RMSE: lower is better (RMSE punishes big misses). "
                                   "R²: share of the day-to-day variation explained, 1 is "
                                   "perfect. Bias: average over- (+) or under- (−) forecast.")
metric_name = mon.METRICS[metric][0]
frame = live if source.startswith("Live") else backtest


@st.fragment(run_every="60s")
def live_panel() -> None:
    daily, metrics, live, backtest, meta = load_inputs()
    frame = live if source.startswith("Live") else backtest
    health = mon.health_table(frame, metrics, days)
    fresh = mon.freshness(daily)
    counts = health["status"].value_counts()
    served = health["method"].eq("gradient_boosting").sum()
    recent = mon.window(frame, days)
    overall = mon.summarise(recent)
    issued_days = live["issued_for"].nunique() if not live.empty else 0
    scored = int(live["observed"].notna().sum()) if not live.empty else 0
    last_issue = (pd.to_datetime(live["issued_at"]).max().strftime("%d %b %H:%M UTC")
                  if not live.empty else "never")
    health_line = " · ".join(f"{mon.STATUS[s][1]} {counts.get(s, 0)}"
                             for s in ("good", "warning", "critical", "collecting")
                             if counts.get(s, 0))
    html('<div class="tiles">' + "".join([
        tile("Models in service", f"{len(health)}",
             f"{served} machine-learning · {len(health) - served} baseline"),
        tile("Model health", health_line or "—", f"{source.split(' (')[0]}, last {days} days"),
        tile("Data freshness", f"{fresh['hours']:.0f} h",
             f"{chip(fresh['status'])} last observed day {fresh['last_day']}"),
        tile("Live forecasts", f"{issued_days} day{'s' if issued_days != 1 else ''}",
             f"{scored} scored · last issued {last_issue}"),
        tile("Gain vs “tomorrow = today”",
             pct(overall["skill"]) if overall["n"] >= mon.MIN_SCORED * len(health) else "—",
             f"all models, {overall['n']} scored forecasts"
             + ("" if overall["n"] >= mon.MIN_SCORED * len(health) else " (not enough yet)")),
    ]) + "</div>")

    # Health matrix: pollutants × horizons, each cell with status, live and reference error
    header = "".join(f"<th>D+{h}</th>" for h in sorted(health["horizon"].unique()))
    rows = []
    for name, group in health.groupby("name", sort=False):
        cells = []
        for row in group.sort_values("horizon").itertuples():
            served = "" if row.method == "gradient_boosting" else " · baseline served"
            if row.n < mon.MIN_SCORED:
                detail = (f'<div class="m">{int(row.n)} / {mon.MIN_SCORED} days scored'
                          f"{served}</div>")
            else:
                value = (metric_value(metric, getattr(row, metric, np.nan))
                         if metric != "aqi_category_accuracy" else "—")
                test = {"mae": row.reference_mae, "rmse": row.reference_rmse,
                        "r2": row.reference_r2}.get(metric)
                test_text = f" · test {metric_value(metric, test)}" if test is not None else ""
                detail = (f'<div class="m"><span class="big">{metric_name} {value}</span></div>'
                          f'<div class="m">gain {pct(row.skill)}{test_text}{served}</div>')
            cells.append(f'<td title="{row.reasons}">{chip(row.status)}{detail}</td>')
        rows.append(f"<tr><td><b>{name}</b><div class='m'>"
                    f"{POLLUTANTS[group['pollutant'].iloc[0]].statistic}</div></td>"
                    f"{''.join(cells)}</tr>")
    html(f'<div class="card"><h3>Health of the 18 models</h3>'
         f'<div class="lead">{metric_name} over the last {days} days (µg/m³ for errors), the '
         "gain over “tomorrow = today”, and the test-year value as reference. Hover a cell for "
         'the reason of its status.</div><table class="matrix"><tr>'
         f"<th>Pollutant</th>{header}</tr>{''.join(rows)}</table>"
         '<div class="note">⛔ Alert: more than 10% worse than “tomorrow = today”, error more '
         "than twice the test-year error, or the 80% range covering under half of the real "
         "values. ⚠️ Watch: milder versions of the same. ⏳ Collecting: fewer than "
         f"{mon.MIN_SCORED} days scored, too few to judge.</div></div>")
    with st.expander("Table view"):
        st.dataframe(health.drop(columns=["pollutant"]).rename(columns={
            "name": "Pollutant", "horizon": "Horizon", "method": "Served", "reference_mae":
            "Test MAE", "n": "Scored", "mae": "MAE", "persistence_mae": "Baseline MAE",
            "skill": "Gain", "coverage": "80% coverage", "bias": "Bias", "status": "Status",
            "reasons": "Reasons"}), hide_index=True, width="stretch")


live_panel()


@st.cache_data(ttl=60, show_spinner=False)
def service_checks() -> list[dict]:
    api_url = os.environ.get("AAQA_API_URL", "http://localhost:8000")
    return mon.check_services(load_config().location, api_url, api_key(paths.PROJECT_ROOT))


@st.fragment(run_every="60s")
def system_health() -> None:
    services = service_checks()
    rows = "".join(f"<tr><td><b>{s['service']}</b></td><td>{chip(s['status'])}</td>"
                   f"<td>{s['ms']} ms</td><td>{s['detail']}</td></tr>" for s in services)
    files = {
        "Pollution & weather download": (paths.AIR_QUALITY_RAW, 36),
        "Cleaned daily data": (paths.DAILY_PROCESSED, 36),
        "Forecast issued (monitoring log)": (paths.MONITORING_LOG, 36),
        "Calibration on the US Embassy": (paths.CALIBRATION_PATH, 36),
        "Models trained": (paths.MODEL_PATH, 24 * 30),
    }
    steps = mon.pipeline_health(files)
    if paths.EMBASSY_RAW.exists():
        last = pd.read_csv(paths.EMBASSY_RAW, parse_dates=["time"])["time"].max()
        days_since = (pd.Timestamp.now() - last).days
        steps.append({"step": "US Embassy monitor readings",
                      "status": "good" if days_since <= 2 else "critical",
                      "detail": f"last reading {last:%d %b %Y}"
                                + (" — monitor offline" if days_since > 2 else ""),
                      "expected": "hourly"})
    step_rows = "".join(f"<tr><td><b>{s['step']}</b></td><td>{chip(s['status'])}</td>"
                        f"<td>{s['detail']}</td><td>{s['expected']}</td></tr>" for s in steps)
    problems = sum(s["status"] != "good" for s in services + steps)
    summary = ("All services and data are up to date." if not problems else
               f"{problems} item{'s' if problems > 1 else ''} need attention.")
    left, right = st.columns(2)
    with left:
        html(f'<div class="card"><h3>System health · services</h3><div class="lead">{summary} '
             'Checked every minute.</div><table class="matrix"><tr><th>Service</th>'
             f"<th>Status</th><th>Response</th><th>Detail</th></tr>{rows}</table>"
             '<div class="note">The AAQA REST API must be running for the apps that use it '
             "(<code>make api</code>). Open-Meteo provides all model data; AirNow the US "
             "Embassy readings.</div></div>")
    with right:
        html('<div class="card"><h3>System health · data pipeline</h3><div class="lead">Age '
             'of each output against how often it should be refreshed.</div>'
             '<table class="matrix"><tr><th>Step</th><th>Status</th><th>Last update</th>'
             f"<th>Expected</th></tr>{step_rows}</table>"
             '<div class="note">Run <code>make update</code> daily (data, calibration, '
             "forecast) and <code>make train</code> monthly.</div></div>")


system_health()

# ── Model comparison ─────────────────────────────────────────────────────────────────────

predictions = load_model_predictions(source)
comparison = mon.compare_models(predictions, days)
available = [m for m in mon.MODEL_LABELS if not comparison.empty and m in set(comparison["model"])]
html('<div class="card"><h3>Model comparison · every pollutant, every model, every metric</h3>'
     + ("Backtest: the four models were all scored on the same test-year days."
        if source.startswith("Backtest") else
        "Live: the model in service and the “tomorrow = today” baseline. Switch the source to "
        "Backtest to compare all four models.") + "</div>")
chosen = st.multiselect("Models to compare", available, default=available,
                        format_func=lambda m: mon.MODEL_LABELS[m], key="models")

if comparison.empty or not chosen:
    st.info("No scored predictions for this source and window yet.")
else:
    table = comparison[comparison["model"].isin(chosen)]

    # Leaderboard: selected pollutant and horizon, every metric
    board = table[(table["pollutant"] == pollutant) & (table["horizon"] == horizon)]
    best = {}
    for key, (_, _, direction) in mon.METRICS.items():
        values = board.set_index("model")[key].dropna()
        if values.empty:
            continue
        best[key] = (values.abs().idxmin() if direction == "zero" else
                     values.idxmax() if direction == "high" else values.idxmin())
    head = "".join(f"<th>{name}{' (' + unit + ')' if unit and unit != '%' else ''}</th>"
                   for name, unit, _ in mon.METRICS.values())
    body = ""
    for row in board.set_index("model").reindex(chosen).dropna(how="all").itertuples():
        cells = ""
        for key in mon.METRICS:
            text = metric_value(key, getattr(row, key))
            is_best = best.get(key) == row.Index
            cells += f"<td><b>★ {text}</b></td>" if is_best else f"<td>{text}</td>"
        swatch = f'<span class="dot" style="background:{MODEL_COLORS[row.Index]}"></span>'
        body += (f"<tr><td>{swatch} <b>{mon.MODEL_LABELS[row.Index]}</b></td>{cells}"
                 f"<td>{int(row.n)}</td></tr>")
    html(f'<div class="card"><h3>Leaderboard · {POLLUTANTS[pollutant].name} · D+{horizon} · '
         f'last {days} days</h3><table class="matrix"><tr><th>Model</th>{head}<th>Days</th></tr>'
         f"{body}</table><div class=\"note\">★ best per metric. MAE: average miss. RMSE: like "
         "MAE but big misses count more. R²: share of the day-to-day variation explained (1 is "
         "perfect, below 0 is worse than always predicting the average). Bias: average over- "
         "(+) or under- (−) forecast. Right AQI category: share of days with the correct "
         "colour band.</div></div>")

    # Matrix: every pollutant × model for the chosen metric
    winners = mon.best_model(table, metric)
    head = "".join(f'<th><span class="dot" style="background:{MODEL_COLORS[m]}"></span> '
                   f"{mon.MODEL_LABELS[m]}</th>" for m in chosen)
    body = ""
    for key, spec_row in POLLUTANTS.items():
        sub = table[(table["pollutant"] == key) & (table["horizon"] == horizon)].set_index("model")
        if sub.empty:
            continue
        winner = winners.get((key, horizon))
        cells = ""
        for model in chosen:
            text = metric_value(metric, sub[metric].get(model, np.nan))
            cells += f"<td><b>★ {text}</b></td>" if model == winner else f"<td>{text}</td>"
        body += f"<tr><td><b>{spec_row.name}</b></td>{cells}</tr>"
    html(f'<div class="card"><h3>{metric_name} by pollutant and model · D+{horizon} · '
         f'last {days} days</h3><table class="matrix"><tr><th>Pollutant</th>{head}</tr>{body}'
         '</table><div class="note">★ best model for each pollutant. Errors are in µg/m³ and '
         "not comparable between pollutants (CO is measured in hundreds, SO2 below 2).</div>"
         "</div>")

    # Chart: chosen metric by horizon for the selected pollutant
    by_horizon = table[table["pollutant"] == pollutant]
    fig = go.Figure()
    for model in chosen:
        sub = by_horizon[by_horizon["model"] == model].sort_values("horizon")
        values = sub[metric] * (100 if metric == "aqi_category_accuracy" else 1)
        fig.add_bar(x=[f"D+{h}" for h in sub["horizon"]], y=values,
                    name=mon.MODEL_LABELS[model], marker={"color": MODEL_COLORS[model],
                                                         "cornerradius": 4},
                    hovertemplate="%{y:.2f}<extra>" + mon.MODEL_LABELS[model] + "</extra>")
    unit = mon.METRICS[metric][1]
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06)
    fig.update_yaxes(title_text=unit or metric_name)
    better = {"low": "lower is better", "high": "higher is better",
              "zero": "closer to 0 is better"}[mon.METRICS[metric][2]]
    st.plotly_chart(chart_layout(fig, f"{metric_name} by horizon — {POLLUTANTS[pollutant].name}"
                                 f" ({better})", 300), width="stretch")

# ── Selected model ───────────────────────────────────────────────────────────────────────

spec = POLLUTANTS[pollutant]
selected = frame[(frame["pollutant"] == pollutant) & (frame["horizon"] == horizon)]
selected = selected.sort_values("target_date")
recent = mon.window(selected, days)
summary = mon.summarise(recent)
reference = metrics[pollutant][str(horizon)]
status, reasons = mon.status_of(summary, reference["models"][reference["selected_method"]]["mae"])
html(f"""<div class="card"><h3>{spec.name} · D+{horizon} · {chip(status)}
<span style="font-weight:500;color:{INK_2};font-size:0.8rem"> {'; '.join(reasons)}</span></h3>
<div class="stat"><div><b>{summary['n']}</b>scored forecasts</div>
<div><b>{fmt(summary['mae'])}</b>MAE (µg/m³)</div>
<div><b>{fmt(summary['persistence_mae'])}</b>baseline MAE</div>
<div><b>{pct(summary['skill']) if summary['n'] >= mon.MIN_SCORED else '—'}</b>gain vs baseline</div>
<div><b>{fmt(summary['coverage'] * 100, 0, '%') if summary['n'] else '—'}</b>80% coverage</div>
<div><b>{fmt(summary['bias'], 2)}</b>bias (µg/m³)</div></div></div>""")

if selected.empty:
    st.info("No forecasts for this model in this source yet.")
else:
    view = selected[selected["target_date"] > selected["target_date"].max()
                    - pd.Timedelta(days=max(days, 30))]
    fig = go.Figure()
    fig.add_scatter(x=view["target_date"], y=view["q90"], mode="lines", line={"width": 0},
                    hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=view["target_date"], y=view["q10"], mode="lines", line={"width": 0},
                    fill="tonexty", fillcolor=BAND, name="80% interval", hoverinfo="skip")
    fig.add_scatter(x=view["target_date"], y=view["observed"], mode="lines", name="Observed",
                    line={"color": INK, "width": 2})
    overlay = predictions[(predictions["pollutant"] == pollutant)
                          & (predictions["horizon"] == horizon)
                          & (predictions["target_date"] >= view["target_date"].min())]
    for model in chosen or [reference["selected_method"]]:
        series = overlay[overlay["model"] == model].sort_values("target_date")
        if series.empty:
            continue
        dotted = model == "persistence"
        fig.add_scatter(x=series["target_date"], y=series["predicted"],
                        mode="lines+markers" if len(series) < 20 else "lines",
                        name=mon.MODEL_LABELS[model],
                        line={"color": MODEL_COLORS[model], "width": 1.5 if dotted else 2,
                              "dash": "dot" if dotted else "solid"}, marker={"size": 8})
    if spec.who_on_target:
        fig.add_hline(y=spec.who_limit, line_color=MUTED, line_width=1,
                      annotation_text=f"WHO {spec.who_label}", annotation_font_color=MUTED)
    fig.update_yaxes(title_text="µg/m³")
    st.plotly_chart(chart_layout(fig, f"Forecast vs observed — {spec.name} ({spec.statistic})"),
                    width="stretch")

    errors = mon.rolling_errors(frame, pollutant, horizon)
    errors = errors[errors.index > errors.index.max() - pd.Timedelta(days=max(days, 30))]
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_scatter(x=errors.index, y=errors["model"], mode="lines", name="Model",
                        line={"color": MODEL, "width": 2})
        fig.add_scatter(x=errors.index, y=errors["persistence"], mode="lines",
                        name="“Tomorrow = today”",
                        line={"color": BASELINE, "width": 2, "dash": "dot"})
        fig.update_yaxes(title_text="µg/m³", rangemode="tozero")
        st.plotly_chart(chart_layout(fig, "Rolling 14-day mean absolute error (lower is better)",
                                     280), width="stretch")
    with right:
        fig = go.Figure()
        fig.add_scatter(x=errors.index, y=errors["coverage"] * 100, mode="lines",
                        name="Coverage", line={"color": MODEL, "width": 2}, showlegend=False)
        fig.add_hline(y=80, line_color=MUTED, line_width=1, annotation_text="80% target",
                      annotation_font_color=MUTED)
        fig.update_yaxes(title_text="% of observations inside", range=[0, 100])
        st.plotly_chart(chart_layout(fig, "Rolling 14-day coverage of the 80% interval", 280),
                        width="stretch")

    with st.expander("Table view of this model"):
        st.dataframe(view[["issued_for", "target_date", "predicted", "q10", "q90", "observed",
                           "persistence", "abs_error"]].rename(columns={
            "issued_for": "Issued for", "target_date": "Target day", "predicted": "Forecast",
            "q10": "Low (10%)", "q90": "High (90%)", "observed": "Observed",
            "persistence": "Baseline", "abs_error": "Abs. error"}),
            hide_index=True, width="stretch")

# ── Ground truth: US Embassy monitor ─────────────────────────────────────────────────────

calibration = Calibration.load(paths.CALIBRATION_PATH)
if calibration is None or not paths.EMBASSY_RAW.exists():
    html('<div class="card"><h3>Ground truth · US Embassy monitor</h3>Not set up yet: add '
         "<code>AIRNOW_API_KEY</code> to <code>.env</code> and run <code>make update</code>."
         "</div>")
else:
    embassy = daily_means(pd.read_csv(paths.EMBASSY_RAW, parse_dates=["time"]))
    last_reading = embassy.index.max()
    stale = (f" <b>The monitor has not reported since {last_reading:%d %B %Y}</b>: later values "
             "use the last correction learned, so they are less certain."
             if (pd.Timestamp.today() - last_reading).days > 7 else "")
    best = calibration.cross_validation[calibration.choice]
    method_rows = "".join(
        f"<tr><td>{'★ ' if m == calibration.choice else ''}<b>{m}</b></td>"
        f"<td>{v['mae']:.2f}</td><td>{v['rmse']:.2f}</td><td>{v['r2']:.2f}</td>"
        f"<td>{v['bias']:+.2f}</td></tr>" for m, v in calibration.cross_validation.items())
    raw = calibration.raw
    html(f'<div class="card"><h3>Ground truth · PM2.5 calibrated on the US Embassy monitor</h3>'
         f"{calibration.n_days} days compared ({calibration.period[0]} → "
         f"{calibration.period[1]}). Raw CAMS misses by <b>{raw['mae']:.1f} µg/m³</b> on "
         f"average (bias {raw['bias']:+.1f}); calibrated ({calibration.choice}): "
         f"<b>{best['mae']:.1f} µg/m³</b> (bias {best['bias']:+.1f}), each month corrected with "
         f"what was learned before it.{stale}"
         '<table class="matrix" style="margin-top:0.6rem"><tr><th>Correction</th>'
         "<th>MAE</th><th>RMSE</th><th>R²</th><th>Bias</th></tr>"
         f"<tr><td>raw CAMS</td><td>{raw['mae']:.2f}</td><td>{raw['rmse']:.2f}</td>"
         f"<td>{raw['r2']:.2f}</td><td>{raw['bias']:+.2f}</td></tr>{method_rows}</table></div>")

    truth_scores = mon.ground_truth_scores(paths.REPORTS_DIR, calibration, embassy)
    if not truth_scores.empty and truth_scores["n"].max() > 0:
        rows = ""
        for h, group in truth_scores.groupby("horizon"):
            for row in group.itertuples():
                rows += (f"<tr><td>D+{h}</td><td>{row.forecast}</td><td>{int(row.n)}</td>"
                         f"<td>{metric_value('mae', row.mae)}</td>"
                         f"<td>{metric_value('rmse', row.rmse)}</td>"
                         f"<td>{metric_value('r2', row.r2)}</td>"
                         f"<td>{metric_value('bias', row.bias)}</td>"
                         f"<td>{metric_value('aqi_category_accuracy', row.aqi_category_accuracy)}"
                         "</td></tr>")
        html('<div class="card"><h3>Real-world accuracy · PM2.5 forecasts vs the US Embassy '
             'monitor (test year)</h3><table class="matrix"><tr><th>Horizon</th><th>Forecast'
             "</th><th>Days</th><th>MAE</th><th>RMSE</th><th>R²</th><th>Bias</th>"
             f"<th>Right AQI category</th></tr>{rows}</table>"
             '<div class="note">Only days when the monitor reported. The model forecasts the '
             "CAMS value; the calibrated forecast is what the app shows.</div></div>")

    cams = daily["pm2_5_mean"]
    joined = pd.concat({"cams": cams, "embassy": embassy}, axis=1).dropna().tail(max(days, 90))
    fig = go.Figure()
    fig.add_scatter(x=joined.index, y=joined["embassy"], mode="lines", name="US Embassy monitor",
                    line={"color": INK, "width": 2})
    fig.add_scatter(x=joined.index, y=calibration.apply(joined["cams"], joined.index),
                    mode="lines", name="CAMS calibrated", line={"color": MODEL, "width": 2})
    fig.add_scatter(x=joined.index, y=joined["cams"], mode="lines", name="CAMS raw",
                    line={"color": BASELINE, "width": 1.5, "dash": "dot"})
    fig.update_yaxes(title_text="µg/m³", rangemode="tozero")
    st.plotly_chart(chart_layout(fig, "PM2.5 daily mean: monitor vs model"), width="stretch")

# ── Data and latest forecast ─────────────────────────────────────────────────────────────

left, right = st.columns([1.1, 1])
with left:
    drift = mon.drift_table(daily)
    drift_labels = {"good": "Normal", "warning": "Unusual", "critical": "Drift"}
    rows = "".join(
        f"<tr><td><b>{r.variable}</b></td><td>{r.recent_mean:.1f}</td>"
        f"<td>{r.same_period_mean:.1f}</td><td>{r.change:+.0%}</td><td>{r.z:+.1f}</td>"
        f"<td>{chip(r.status, drift_labels[r.status])}</td></tr>" for r in drift.itertuples())
    html(f'<div class="card"><h3>Data drift · last 30 days vs the same period in past years</h3>'
         '<table class="matrix"><tr><th>Variable</th><th>Last 30 days</th><th>Past years</th>'
         f"<th>Change</th><th>z</th><th>Status</th></tr>{rows}</table>"
         '<div class="note">Seasonal comparison, so a harmattan rise is not flagged as drift. '
         "|z| above 2 is unusual, above 3 is drift: the models may need retraining.</div></div>")
    grid = load_grid(paths.GRID_POINTS, 5.36, -4.01)
    air = grid["air_quality"]
    html(f'<div class="card"><h3>Sampling location</h3>CAMS grid cell centred on '
         f'{air["latitude"]:.2f}° N, {abs(air["longitude"]):.2f}° W (~44 km); weather point '
         f'{grid["weather"]["latitude"]:.3f}° N, {abs(grid["weather"]["longitude"]):.3f}° W.'
         "</div>")
    st.plotly_chart(sampling_map(grid, MODEL, BAND, "sampled cell", height=380),
                    width="stretch", config={"scrollZoom": False})
with right:
    if live.empty:
        html('<div class="card"><h3>Latest forecast</h3>No forecast issued yet: click '
             "“Update data & forecast”.</div>")
    else:
        latest_day = live["issued_for"].max()
        latest = live[live["issued_for"] == latest_day]
        table = latest.pivot_table(index="pollutant", columns="horizon", values="predicted")
        rows = "".join(
            f"<tr><td><b>{POLLUTANTS[p].name}</b></td>"
            + "".join(f"<td>{table.loc[p, h]:.1f}</td>" for h in table.columns) + "</tr>"
            for p in POLLUTANTS if p in table.index)
        header = "".join(f"<th>D+{h}</th>" for h in table.columns)
        html(f'<div class="card"><h3>Latest forecast · issued for {latest_day:%d %b %Y}</h3>'
             f'<table class="matrix"><tr><th>Pollutant</th>{header}</tr>{rows}</table>'
             f'<div class="note">µg/m³, model scale (the app shows PM2.5 calibrated on the US '
             f'Embassy monitor). Model v{meta["version"]} trained on '
             f'{meta["trained_at"][:10]}, data up to {meta["trained_until"]}. Checked at '
             f"{datetime.now(UTC):%H:%M} UTC.</div></div>")
