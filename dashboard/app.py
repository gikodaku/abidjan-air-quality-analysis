"""AAQA by KOSCON, weather and air quality: `streamlit run dashboard/app.py`."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from illustrations import skyline
from maps import load_grid, sampling_map
from mascot import mascot

from aaqa import assistant
from aaqa import config as paths
from aaqa.aqi import CATEGORIES
from aaqa.calibration import Calibration, adjust_snapshot
from aaqa.cities import CITIES
from aaqa.config import load_config
from aaqa.data.air_now import fetch_air_now, fetch_city_air
from aaqa.data.weather_forecast import fetch_now, fetch_weather_forecast
from aaqa.features.build import HARMATTAN_MONTHS
from aaqa.gamification import BADGES, Progress, badge_info, daily_challenge, daily_quiz
from aaqa.models.predictor import AirQualityPredictor
from aaqa.pipeline import run
from aaqa.pollutants import POLLUTANTS

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LOGO_PATH = Path(__file__).parent / "assets" / "koscon-logo.png"

# KOSCON brand colours (brand guidelines)
BRAND = {"dark_green": "#074930", "green": "#3FAB3D", "petrol": "#006079", "sky": "#0EA7DC"}
INK, INK_2, MUTED = "#13262b", "#4a5f64", "#7b8d91"

# Per AQI category: readable text colour on light backgrounds, and the page tint (top, bottom)
CATEGORY_TEXT = {"good": "#0a8f45", "moderate": "#9a7600", "usg": "#c96400",
                 "unhealthy": "#c8102e", "very_unhealthy": "#7b2d8e", "hazardous": "#7e0023"}
CATEGORY_TINT = {"good": ("#e3f5ea", "#f6fbf8"), "moderate": ("#fbf1cc", "#fffcf2"),
                 "usg": ("#fde3c8", "#fff7ef"), "unhealthy": ("#fad6dc", "#fff4f5"),
                 "very_unhealthy": ("#ecdcf2", "#faf6fb"), "hazardous": ("#efd6da", "#fbf3f4")}
SCALE = [(0, 50, "Good"), (50, 100, "Moderate"), (100, 150, "Sensitive"),
         (150, 200, "Unhealthy"), (200, 300, "V. unhealthy"), (300, 500, "Hazardous")]
SUGGESTIONS = ["What's the weather tomorrow?", "Can I go running today?", "What is PM2.5?",
               "What is NO2?", "What is the harmattan?", "Best time to go out?"]

st.set_page_config(page_title="AAQA by KOSCON — Air & weather", page_icon="🐆",
                   layout="centered")


# ── Data ──────────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_daily() -> pd.DataFrame:
    return pd.read_csv(paths.DAILY_PROCESSED, index_col="date", parse_dates=["date"]).asfreq("D")


@st.cache_resource
def load_predictor() -> AirQualityPredictor:
    return AirQualityPredictor.load(paths.MODEL_PATH)


def city_location(city_key: str):
    return load_config().location if city_key == "abidjan" else CITIES[city_key].location


@st.cache_data(ttl=1800)
def load_weather(city_key: str) -> list[dict]:
    try:
        return fetch_weather_forecast(city_location(city_key))
    except Exception:  # offline: the app still works without the weather
        return []


@st.cache_data(ttl=600)
def load_now(city_key: str) -> dict | None:
    try:
        return fetch_now(city_location(city_key))
    except Exception:
        return None


@st.cache_data(ttl=600)
def load_air_now(calibrated: bool) -> dict | None:
    """Live air in Abidjan, with PM2.5 calibrated on the US Embassy monitor if asked."""
    adjust = None
    calibration = Calibration.load(paths.CALIBRATION_PATH) if calibrated else None
    if calibration:
        def adjust(value, day):
            return calibration.apply_one(value, day)
    try:
        return fetch_air_now(load_config().location, adjust)
    except Exception:
        return None


@st.cache_data(ttl=600)
def load_city_air(city_key: str) -> tuple[dict | None, list[dict]]:
    """Live air and the CAMS 3-day forecast for a city without AAQA models."""
    try:
        return fetch_city_air(CITIES[city_key].location)
    except Exception:
        return None, []


@st.cache_data(ttl=3600)
def load_embassy_latest() -> dict | None:
    """Latest hourly reading of the US Embassy monitor, if downloaded."""
    try:
        hourly = pd.read_csv(paths.EMBASSY_RAW, parse_dates=["time"]).dropna()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None
    if hourly.empty:
        return None
    last = hourly.iloc[-1]
    return {"time": last["time"], "pm2_5": float(last["pm2_5"])}


@st.cache_data
def logo_data_uri() -> str:
    return "data:image/png;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode()


def refresh() -> None:
    with st.spinner("Downloading data and training… (about 4 min)"):
        run(["fetch", "process", "report", "train"], load_config())
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()


# ── Rendering helpers ─────────────────────────────────────────────────────────────────────

def html(markup: str) -> None:
    # Indented lines would be read as code blocks by the Markdown renderer
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def logo(height: int = 54) -> str:
    # The logo is never modified and always sits on white, as in the brand guidelines; at 54 px
    # high it is about 140 px wide, above the guidelines' 3.5 cm minimum
    return f'<img class="logo" src="{logo_data_uri()}" alt="KOSCON" style="height:{height}px">'


def day_name(iso: str, today: str) -> str:
    day = pd.Timestamp(iso)
    delta = (day - pd.Timestamp(today)).days
    return {0: "Today", 1: "Tomorrow"}.get(delta, DAYS[day.day_of_week][:3])


def long_date(iso: str) -> str:
    day = pd.Timestamp(iso)
    return f"{DAYS[day.day_of_week]}, {day.day} {MONTHS[day.month - 1]}"


def short_label(label: str) -> str:
    return "Sensitive" if label == "Unhealthy for sensitive groups" else label


def pill(label: str, category: str) -> str:
    top, _ = CATEGORY_TINT[category]
    return (f'<span class="pill" style="background:{top};color:{CATEGORY_TEXT[category]}">'
            f"{label}</span>")


def greeting(hour: int) -> str:
    return "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"


def scale_position(aqi: int) -> float:
    for i, (low, high, _) in enumerate(SCALE):
        if aqi <= high:
            return 100 * (i + (max(aqi, low) - low) / (high - low)) / len(SCALE)
    return 100.0


def scale_bar(aqi: int) -> str:
    stops = ", ".join(f"{c.color} {100 * (i + 0.5) / len(SCALE):.0f}%"
                      for i, c in enumerate(CATEGORIES))
    labels = "".join(f"<span>{name}</span>" for _, _, name in SCALE)
    ticks = "".join(f"<span>{v}</span>" for v in (0, 50, 100, 150, 200, 300, "500+"))
    return (f'<div class="scale"><div class="labels">{labels}</div>'
            f'<div class="bar" style="background:linear-gradient(90deg, {stops})">'
            f'<span class="marker" style="left:{scale_position(aqi):.1f}%"></span></div>'
            f'<div class="ticks">{ticks}</div></div>')


def minutes_ago(iso: str, now: datetime) -> str:
    then = pd.Timestamp(iso).tz_localize(now.tzinfo)
    minutes = max(int((pd.Timestamp(now) - then).total_seconds() // 60), 0)
    return "just now" if minutes < 1 else f"{minutes} min ago" if minutes < 60 \
        else f"{minutes // 60} h ago"


def style(category: str) -> None:
    top, bottom = CATEGORY_TINT[category]
    html(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&display=swap');
    :root {{ --green: {BRAND["green"]}; --dark-green: {BRAND["dark_green"]};
             --petrol: {BRAND["petrol"]}; --sky: {BRAND["sky"]}; --ink: {INK}; }}
    .stApp {{ background: linear-gradient(180deg, {top} 0%, {bottom} 420px, #f7f9f8 100%); }}
    .stApp, .stApp p, .stApp div, .stApp label, .stApp li,
    .stApp span:not([data-testid="stIconMaterial"]) {{
        font-family: 'Montserrat', system-ui, sans-serif; color: {INK}; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 1rem; max-width: 560px; }}

    .splash {{ position: fixed; inset: 0; z-index: 2147483647; background: #fff; display: flex;
               align-items: center; justify-content: center; flex-direction: column;
               animation: splash-out 0.7s ease 2.6s forwards; }}
    .splash .logo {{ width: min(72vw, 420px); height: auto !important; }}
    .splash .tag {{ color: var(--petrol) !important; font-weight: 700; margin-top: 1.4rem; }}
    .splash .bar {{ width: min(60vw, 260px); height: 6px; border-radius: 3px; margin-top: 1rem;
                    background: #e6eef0; overflow: hidden; }}
    .splash .bar span {{ display: block; height: 100%; width: 40%; border-radius: 3px;
        background: linear-gradient(90deg, var(--dark-green), var(--green), var(--sky));
        animation: load 1.2s ease-in-out infinite; }}
    .splash .hint {{ color: {MUTED} !important; font-size: 0.8rem; margin-top: 0.7rem; }}
    @keyframes load {{ 0% {{ transform: translateX(-110%); }}
                       100% {{ transform: translateX(260%); }} }}
    @keyframes splash-out {{ to {{ opacity: 0; visibility: hidden; }} }}

    .topbar {{ display: flex; align-items: center; justify-content: space-between; gap: 0.8rem; }}
    .badge-white {{ background: #fff; border-radius: 14px; padding: 6px 12px;
                    display: inline-flex; box-shadow: 0 4px 14px rgba(16,40,44,0.08); }}
    .chip {{ background: #fff; border-radius: 999px; padding: 0.35rem 0.8rem; font-size: 0.8rem;
             font-weight: 700; white-space: nowrap; box-shadow: 0 2px 10px rgba(16,40,44,0.08); }}
    .place {{ display: flex; justify-content: space-between; align-items: flex-start;
              margin: 0.9rem 0 0.7rem; }}
    .place .city {{ font-size: 1.6rem; font-weight: 800; line-height: 1.1; }}
    .place .country {{ color: {INK_2} !important; font-size: 0.95rem; }}
    .place .source {{ color: {MUTED} !important; font-size: 0.78rem; margin-top: 0.15rem; }}
    .place .clock {{ text-align: right; font-weight: 800; font-size: 1.1rem; }}
    .place .clock small {{ display: block; font-weight: 500; font-size: 0.78rem;
                           color: {INK_2} !important; }}

    .card {{ background: #fff; border-radius: 22px; padding: 1rem 1.1rem; margin-bottom: 0.9rem;
             box-shadow: 0 6px 24px rgba(16,40,44,0.07); position: relative; overflow: hidden; }}
    .card h4 {{ margin: 0 0 0.7rem; padding: 0; font-size: 1.02rem; font-weight: 800; }}
    .hero {{ background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, {top} 100%);
             padding-bottom: 150px; }}
    .hero .row {{ display: flex; justify-content: space-between; align-items: flex-start; }}
    .hero .live {{ font-size: 0.85rem; font-weight: 700; display: flex; align-items: center;
                   gap: 0.4rem; }}
    .pulse {{ width: 8px; height: 8px; border-radius: 50%; background: #e8492b;
              animation: pulse 1.6s infinite; display: inline-block; }}
    @keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(232,73,43,0.6); }}
                        70% {{ box-shadow: 0 0 0 7px rgba(232,73,43,0); }}
                        100% {{ box-shadow: 0 0 0 0 rgba(232,73,43,0); }} }}
    .hero .big {{ font-size: 5.2rem; font-weight: 900; line-height: 1; margin-top: 0.2rem; }}
    .hero .is {{ text-align: center; font-size: 0.85rem; color: {INK_2} !important; }}
    .hero .label {{ margin-top: 0.3rem; padding: 0.45rem 1rem; border-radius: 10px;
                    font-weight: 800; font-size: 1.15rem; text-align: center; min-width: 9rem; }}
    .hero .pm {{ display: flex; gap: 1.4rem; margin: 0.8rem 0 0.3rem; font-size: 0.95rem; }}
    .hero .pm b {{ font-size: 1.05rem; }}
    .scale {{ margin: 0.8rem 0 0.4rem; }}
    .scale .labels, .scale .ticks {{ display: grid; grid-template-columns: repeat(6, 1fr);
        font-size: 0.66rem; color: {INK_2}; }}
    .scale .labels span {{ text-align: center; color: {INK_2} !important; }}
    .scale .ticks {{ display: flex; justify-content: space-between; }}
    .scale .ticks span {{ color: {MUTED} !important; }}
    .scale .bar {{ position: relative; height: 6px; border-radius: 3px; margin: 0.25rem 0; }}
    .scale .marker {{ position: absolute; top: 50%; width: 14px; height: 14px; border-radius: 50%;
                      background: #fff; border: 3px solid {INK}; transform: translate(-50%, -50%);
                      box-shadow: 0 1px 4px rgba(0,0,0,0.25); }}
    .weather-mini {{ background: rgba(255,255,255,0.85); border-radius: 16px;
                     padding: 0.7rem 0.9rem; margin-top: 0.8rem;
                     box-shadow: 0 2px 10px rgba(16,40,44,0.06); width: 62%; }}
    .weather-mini .top {{ display: flex; align-items: center; gap: 0.6rem; }}
    .weather-mini .icon {{ font-size: 2rem; }}
    .weather-mini .t {{ font-size: 1.6rem; font-weight: 800; }}
    .weather-mini .d {{ margin-left: auto; color: {INK_2} !important; font-weight: 600;
                        font-size: 0.85rem; text-align: right; }}
    .weather-mini .facts {{ display: flex; gap: 1rem; margin-top: 0.4rem; font-size: 0.82rem;
                            color: {INK_2} !important; }}
    .updated {{ font-size: 0.78rem; color: {MUTED} !important; margin-top: 0.6rem; }}
    .hero .kos {{ position: absolute; right: 10px; bottom: 6px; z-index: 2; line-height: 0; }}
    .skyline {{ position: absolute; left: 0; right: 0; bottom: 0; width: 100%; height: 150px; }}
    .avg {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; }}
    .avg div {{ border-radius: 14px; padding: 0.6rem 0.8rem; display: flex;
                justify-content: space-between; align-items: center; font-weight: 600; }}
    .avg b {{ font-size: 1.25rem; font-weight: 800; }}
    .say {{ display: flex; gap: 0.7rem; align-items: flex-start; }}
    .say .who {{ color: var(--dark-green) !important; font-weight: 800; font-size: 0.78rem; }}
    .hours {{ display: flex; gap: 0.45rem; overflow-x: auto; padding-bottom: 0.3rem; }}
    .hour {{ flex: 0 0 auto; min-width: 3.7rem; text-align: center; padding: 0.5rem 0.25rem;
             border-radius: 14px; background: #f3f6f6; }}
    .hour.now {{ background: {INK}; }}
    .hour.now div {{ color: #fff !important; }}
    .hour .h {{ font-size: 0.75rem; color: {INK_2} !important; }}
    .hour .icon {{ font-size: 1.5rem; margin: 0.15rem 0; }}
    .hour .t {{ font-weight: 800; }}
    .hour .rain {{ font-size: 0.7rem; color: var(--petrol) !important; min-height: 0.9rem; }}
    .days {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.45rem; }}
    .day {{ text-align: center; padding: 0.6rem 0.2rem; border-radius: 16px; background: #f3f6f6; }}
    .day .name {{ font-weight: 800; font-size: 0.85rem; }}
    .day .icon {{ font-size: 1.8rem; margin: 0.2rem 0; }}
    .day .t {{ font-weight: 700; }} .day .t .min {{ color: {MUTED} !important; }}
    .day .rain {{ font-size: 0.75rem; color: var(--petrol) !important; }}
    .day .air {{ margin-top: 0.4rem; }}
    .pill {{ display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px;
             font-size: 0.72rem; font-weight: 800; }}
    .fc {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; }}
    .fc > div {{ border-radius: 16px; padding: 0.7rem; text-align: center; }}
    .fc .n {{ font-size: 2rem; font-weight: 900; line-height: 1.1; }}
    .fc .s {{ font-size: 0.72rem; color: {INK_2} !important; }}
    .tiles {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; }}
    .tile {{ background: #f3f6f6; border-radius: 16px; padding: 0.7rem 0.8rem; }}
    .tile .k {{ font-size: 0.68rem; color: {MUTED} !important; text-transform: uppercase;
                font-weight: 700; letter-spacing: 0.05em; }}
    .tile .v {{ font-size: 1.35rem; font-weight: 800; margin: 0.1rem 0; }}
    .tile .v small {{ font-size: 0.75rem; font-weight: 500; color: {MUTED} !important; }}
    .tile .s {{ font-size: 0.75rem; color: {INK_2} !important; }}
    .tile.locked {{ opacity: 0.45; filter: grayscale(1); }}
    .ptile .bar {{ height: 7px; background: #dfe6e6; border-radius: 4px; margin-top: 0.55rem;
                   overflow: hidden; }}
    .ptile .bar span {{ display: block; height: 100%; border-radius: 4px; }}
    .ptile .tag {{ margin-left: 0.4rem; padding: 0.05rem 0.4rem; border-radius: 6px;
                   background: #e1f4e7; color: #0a6b35 !important; font-size: 0.6rem;
                   letter-spacing: 0.03em; }}
    .level {{ display: flex; gap: 1rem; align-items: center; }}
    .level .big {{ font-size: 3rem; }} .level .name {{ font-size: 1.35rem; font-weight: 800; }}
    .xp {{ height: 12px; background: #edf1f1; border-radius: 6px; overflow: hidden;
           margin: 0.5rem 0 0.3rem; }}
    .xp span {{ display: block; height: 100%;
                background: linear-gradient(90deg, var(--green), var(--sky)); }}
    .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem;
              margin-top: 0.8rem; text-align: center; }}
    .stats b {{ display: block; font-size: 1.35rem; font-weight: 800; }}
    .note {{ font-size: 0.72rem; color: {MUTED} !important; text-align: center;
             margin: 0.4rem 0 1rem; }}
    .powered {{ display: flex; flex-direction: column; align-items: center; gap: 0.5rem;
                margin: 1.6rem 0 1rem; }}
    .powered .by {{ font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase;
                    color: {INK_2} !important; font-weight: 700; }}
    .stTabs [role="tablist"] {{ gap: 0.25rem; background: #fff; border-radius: 999px;
        padding: 0.3rem; border: none; box-shadow: 0 2px 10px rgba(16,40,44,0.08); }}
    .stTabs [role="tab"] {{ border-radius: 999px; padding: 0.4rem 0.8rem; height: auto;
        background: transparent; border: none; cursor: pointer; }}
    .stTabs [role="tab"][aria-selected="true"] {{ background: {INK}; }}
    .stTabs [role="tab"][aria-selected="true"] p {{ color: #fff !important; }}
    .stTabs [role="tab"] p {{ font-weight: 700; font-size: 0.85rem; }}
    .stTabs .react-aria-SelectionIndicator {{ display: none; }}
    .stButton > button {{ border-radius: 999px; font-weight: 600; }}
    .stButton > button[kind="primary"] {{ background: var(--green); border-color: var(--green);
                                           color: #fff; }}
    .stButton > button[kind="primary"] p {{ color: #fff !important; }}
    div[data-testid="stExpander"] details {{ background: #fff; border-radius: 18px;
        border: none; box-shadow: 0 6px 24px rgba(16,40,44,0.07); }}
    div[data-testid="stChatMessage"] {{ background: #fff; border-radius: 18px;
        box-shadow: 0 2px 10px rgba(16,40,44,0.06); }}
    @media (max-width: 640px) {{
        .tiles, .stats {{ grid-template-columns: repeat(2, 1fr); }}
        .hero .big {{ font-size: 4.2rem; }}
        .weather-mini {{ width: 100%; }}
        .hero {{ padding-bottom: 170px; }}
        .day .icon {{ font-size: 1.5rem; }}
        .stTabs [role="tab"] {{ padding: 0.35rem 0.5rem; }}
        .stTabs [role="tab"] p {{ font-size: 0.74rem; }}
    }}
    </style>""")


def splash() -> None:
    """KOSCON loading screen, shown once per session; it fades out on its own."""
    if st.session_state.get("splash_done"):
        return
    st.session_state["splash_done"] = True
    html(f"""<div class="splash">{logo(160)}
    <div class="tag">Air &amp; weather, with Kos the leopard</div>
    <div class="bar"><span></span></div>
    <div class="hint">Reading the sky…</div></div>""")


def chart_layout(fig, height: int = 300):
    fig.update_layout(height=height, margin={"t": 10, "b": 10, "l": 10, "r": 10},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": INK_2, "family": "Montserrat"}, yaxis_title="µg/m³",
                      legend={"orientation": "h", "y": 1.1})
    fig.update_xaxes(gridcolor="#e6ecec")
    fig.update_yaxes(gridcolor="#e6ecec")
    return fig


def footer() -> None:
    html(f'<div class="powered"><div class="by">Powered by</div>'
         f'<div class="badge-white">{logo(60)}</div></div>')


# ── Page ──────────────────────────────────────────────────────────────────────────────────

splash()

if not (paths.DAILY_PROCESSED.exists() and paths.MODEL_PATH.exists()):
    style("good")
    html('<div class="card"><h4>Welcome to AAQA</h4>No data yet: let\'s download it.</div>')
    if st.button("Download data and train the model (about 4 min)", type="primary"):
        refresh()
    footer()
    st.stop()

if not st.session_state.get("city"):  # first visit, or the selection was cleared
    st.session_state["city"] = "abidjan"
city = CITIES[st.session_state["city"]]
location = city_location(city.key)
timezone = ZoneInfo(location.timezone)
clock = datetime.now(timezone)
daily = load_daily()
predictor = load_predictor()
weather_days = load_weather(city.key)
weather = {day["date"]: day for day in weather_days}
now = load_now(city.key)
calibration = Calibration.load(paths.CALIBRATION_PATH) if city.modelled else None
use_calibration = calibration is not None and st.session_state.get("calibrated", True)
if city.modelled:
    # Abidjan: AAQA's trained models, optionally calibrated on the US Embassy monitor
    result = predictor.forecast(daily)
    air_now = load_air_now(use_calibration)
    if use_calibration:
        result = {**result,
                  "observed": adjust_snapshot(result["observed"], calibration,
                                              result["observed"]["date"]),
                  "forecasts": [adjust_snapshot(f, calibration, f["date"])
                                for f in result["forecasts"]]}
else:
    # Other cities: live CAMS values and the CAMS forecast
    air_now, cams_forecasts = load_city_air(city.key)
    snapshot = air_now or {"overall": {"aqi": 0, "category": "good", "label": "Good",
                                       "color": "#00a84f", "advice": "", "dominant_pollutant":
                                       "pm2_5", "dominant_name": "PM2.5"}, "pollutants": []}
    result = {"observed": {"date": clock.date().isoformat(), **snapshot},
              "forecasts": cams_forecasts}

air_by_date = {result["observed"]["date"]: {**result["observed"], "observed": True}}
for forecast in result["forecasts"]:
    air_by_date[forecast["date"]] = forecast

today = weather_days[0]["date"] if weather_days else clock.date().isoformat()
today_weather = weather.get(today)
today_air = air_by_date.get(today) or (result["forecasts"][0] if result["forecasts"]
                                       else result["observed"])
# The live hourly value when available; otherwise today's AAQA forecast
live = air_now or today_air
overall = live["overall"]
style(overall["category"])
context = assistant.build_context(result, weather_days, now, today,
                                  f"{city.name}, {city.country}")

# Eco journey: daily check-in
progress = Progress.load(paths.PROGRESS_PATH)
earned = progress.check_in(clock.date())
new_badges = progress.refresh_badges()
progress.save(paths.PROGRESS_PATH)
if earned:
    st.toast(f"+{earned} points — welcome back! 🔥 {progress.streak}-day streak", icon="🌿")
for key in new_badges:
    icon, name, _ = badge_info(key)
    st.toast(f"New badge: {name}", icon=icon)

source_line = ("CAMS model · PM2.5 calibrated on the US Embassy monitor" if use_calibration
               else "Copernicus CAMS model · city-wide (~40 km grid)" if city.modelled
               else "Copernicus CAMS model and forecast · city-wide (~40 km grid)")
level = progress.level
status = f'{level["icon"]} {level["name"]} · {progress.points} pts · 🔥 {progress.streak}'
html(f"""<div class="topbar"><div class="badge-white">{logo(54)}</div>
<div class="chip">{status}</div></div>
<div class="place"><div><div class="city">{city.flag} {city.name}</div>
<div class="country">{city.country} · {greeting(clock.hour)} 👋</div>
<div class="source">{source_line}</div></div>
<div class="clock">{clock:%H:%M}<small>{long_date(clock.date().isoformat())}</small></div></div>""")

st.segmented_control("City", list(CITIES), key="city", label_visibility="collapsed",
                     format_func=lambda key: CITIES[key].label,
                     help="Abidjan uses AAQA's trained models; the other cities use the CAMS "
                          "forecast.")

if calibration is not None:
    st.toggle("Calibrated on the US Embassy monitor", value=True, key="calibrated",
              help="PM2.5 from the model is corrected with the US Embassy's reference monitor "
                   f"({calibration.n_days} days compared). Turn off to see the raw model.")

tab_today, tab_ask, tab_journey, tab_insights = st.tabs(
    ["🌤️ Today", "💬 Ask Kos", "🌱 Eco journey", "📈 Insights"], key="main_tab")

# ── Today ─────────────────────────────────────────────────────────────────────────────────

with tab_today:
    category = overall["category"]
    text_color = CATEGORY_TEXT[category]
    pollutants = {p["pollutant"]: p for p in live["pollutants"]}
    pm25, pm10 = pollutants.get("pm2_5"), pollutants.get("pm10")
    pm_row = "".join(
        f"<div>{p['name']} : <b>{p['value']:.0f}</b> µg/m³</div>" for p in (pm25, pm10) if p)
    live_label = "Live AQI" if air_now else "AQI today (forecast)"
    updated = minutes_ago(air_now["time"], clock) if air_now else "—"

    weather_card = ""
    source = now or today_weather
    if source:
        temperature = now["temperature"] if now else today_weather["temperature_max"]
        humidity = now["humidity"] if now else today_weather["humidity"]
        wind = now["wind_kmh"] if now else today_weather["wind_max_kmh"]
        uv = today_weather["uv_index"] if today_weather else None
        uv_text = f"☀️ UV {uv:.0f}" if uv is not None else ""
        weather_card = (f'<div class="weather-mini"><div class="top">'
                        f'<span class="icon">{source["icon"]}</span>'
                        f'<span class="t">{temperature:.0f}°C</span>'
                        f'<span class="d">{source["description"]}</span></div>'
                        f'<div class="facts"><span>💧 {humidity:.0f} %</span>'
                        f'<span>🧭 {wind:.0f} km/h</span><span>{uv_text}</span></div></div>')

    html(f"""<div class="card hero">
    <div class="row"><div><div class="live"><span class="pulse"></span>{live_label}</div>
    <div class="big" style="color:{text_color}">{overall["aqi"]}</div></div>
    <div><div class="is">Air quality is</div>
    <div class="label" style="color:{text_color};background:{CATEGORY_TINT[category][0]}">
    {short_label(overall["label"])}</div></div></div>
    <div class="pm">{pm_row}</div>
    {scale_bar(overall["aqi"])}
    {weather_card}
    <div class="updated">Last update: {updated}</div>
    <div class="kos">{mascot(category, 128, backdrop=False)}</div>
    {skyline(text_color, 0.22, city.key)}
    </div>""")

    if air_now and air_now["averages"]:
        chips = "".join(
            f'<div style="background:{CATEGORY_TINT[a["category"]][0]}">'
            f'<span>{window}</span><b style="color:{CATEGORY_TEXT[a["category"]]}">'
            f'{a["aqi"]}</b></div>' for window, a in air_now["averages"].items())
        html(f'<div class="card"><h4>AQI average</h4><div class="avg">{chips}</div></div>')

    html(f"""<div class="card"><div class="say"><div style="font-size:1.8rem">🐆</div><div>
    <div class="who">KOS SAYS</div><div>{assistant.daily_summary(context)}</div>
    <div style="margin-top:0.3rem;color:{INK_2};font-size:0.85rem">{overall["advice"]}</div>
    </div></div></div>""")

    # Weather forecast with an hourly / daily switch
    title, switch = st.columns([1.4, 1])
    title.markdown("#### Weather forecast")
    mode = switch.segmented_control("Forecast view", ["Hourly", "Daily"], default="Hourly",
                                    key="weather_view", label_visibility="collapsed")
    if mode == "Daily" or not (now and now["hourly"]):
        cells = []
        for date in [d["date"] for d in weather_days][:4]:
            w, air = weather[date], air_by_date.get(date)
            air_html = (f'{mascot(air["overall"]["category"], 26, backdrop=False)}<br>'
                        + pill(short_label(air["overall"]["label"]), air["overall"]["category"])
                        if air else '<span class="pill" style="background:#edf1f1">air —</span>')
            cells.append(f'<div class="day"><div class="name">{day_name(date, today)}</div>'
                         f'<div class="icon">{w["icon"]}</div>'
                         f'<div class="t">{w["temperature_max"]:.0f}° '
                         f'<span class="min">{w["temperature_min"]:.0f}°</span></div>'
                         f'<div class="rain">💧 {w["precipitation_probability"] or 0:.0f}%</div>'
                         f'<div class="air">{air_html}</div></div>')
        html(f'<div class="card"><div class="days">{"".join(cells)}</div></div>'
             if cells else '<div class="card">Weather forecast unavailable right now.</div>')
    else:
        current_hour = now["time"][:13]
        hours = []
        for hour in now["hourly"]:
            is_now = hour["time"][:13] == current_hour
            rain = hour["precipitation_probability"] or 0
            hours.append(f'<div class="hour {"now" if is_now else ""}">'
                         f'<div class="h">{"Now" if is_now else hour["hour"]}</div>'
                         f'<div class="icon">{hour["icon"]}</div>'
                         f'<div class="t">{hour["temperature"]:.0f}°</div>'
                         f'<div class="rain">{f"💧{rain:.0f}%" if rain >= 20 else ""}</div>'
                         "</div>")
        html(f'<div class="card"><div class="hours">{"".join(hours)}</div></div>')

    # Air-quality forecast from the AAQA model
    cards = []
    for day in [f for f in result["forecasts"] if f["date"] >= today]:
        o = day["overall"]
        cards.append(f'<div style="background:{CATEGORY_TINT[o["category"]][0]}">'
                     f'<div class="s"><b>{day_name(day["date"], today)}</b></div>'
                     f'{mascot(o["category"], 40, backdrop=False)}'
                     f'<div class="n" style="color:{CATEGORY_TEXT[o["category"]]}">{o["aqi"]}</div>'
                     f'<div class="s">{short_label(o["label"])} · {o["dominant_name"]}</div></div>')
    html(f'<div class="card"><h4>Air quality forecast · AAQA model</h4>'
         f'<div class="fc">{"".join(cards)}</div></div>' if cards else
         '<div class="card"><h4>Air quality forecast · AAQA model</h4>The forecast is out of '
         "date: refresh the data (Insights → Index scale and data refresh, or `make update`)."
         "</div>")

    embassy_line = ""
    latest = load_embassy_latest()
    if latest:
        age = (pd.Timestamp(clock.replace(tzinfo=None)) - latest["time"]).total_seconds() / 3600
        if age <= 7 * 24:
            when = f'{latest["time"]:%H:%M}' if age < 24 else f'{latest["time"]:%d %b %H:%M}'
            text = (f'🇺🇸 US Embassy monitor, last reading: <b>{latest["pm2_5"]:.0f} µg/m³</b> '
                    f"PM2.5 ({when}).")
        else:
            text = (f"🇺🇸 The US Embassy monitor has not reported since "
                    f'{latest["time"]:%d %B %Y}; the calibration learned until then is applied.')
        embassy_line = f'<div class="note" style="text-align:left;margin:0.5rem 0 0">{text}</div>'
    calibrated_note = (f'<div class="note" style="text-align:left;margin:0.3rem 0 0">PM2.5 '
                       f'calibrated: model value {pm25["raw_value"]:.0f} µg/m³ → '
                       f'{pm25["value"]:.0f} µg/m³.</div>'
                       if pm25 and pm25.get("raw_value") is not None else "")
    # One note card per pollutant: value, category, WHO guideline and a progress bar underneath
    tiles = []
    for value in live["pollutants"]:
        spec = POLLUTANTS[value["pollutant"]]
        reference = f"WHO {spec.who_limit:g}" if spec.who_on_target else spec.statistic
        width = 100 * min(value["aqi"] / 150, 1)
        tag = ('<span class="tag">calibrated</span>' if value.get("raw_value") is not None
               else "")
        tiles.append(f'<div class="tile ptile"><div class="k">{spec.name}{tag}</div>'
                     f'<div class="v">{value["value"]:.0f} <small>µg/m³</small></div>'
                     f'<div class="s">{short_label(value["label"])} · {reference}</div>'
                     f'<div class="bar"><span style="width:{max(width, 3):.0f}%;'
                     f'background:{value["color"]}"></span></div></div>')
    html(f'<div class="card"><h4>Pollutants</h4><div class="tiles">{"".join(tiles)}</div>'
         f"{calibrated_note}{embassy_line}"
         f'<div class="note" style="text-align:left;margin:0.4rem 0 0">The bar fills as the '
         f"index rises (full at 150, “Unhealthy”). The model data probably underestimates NO2 "
         f"near busy roads.</div></div>")

    html('<div class="card" style="padding-bottom:0.2rem"><h4>Where the data comes from</h4>'
         f'<div style="font-size:0.85rem;color:{INK_2}">The air-quality value is the average '
         "of one model grid cell about 44 km wide, coloured here by today's air quality. "
         + ("It covers every district, so it cannot tell Yopougon from Cocody."
            if city.modelled else "It covers the whole city, not one neighbourhood.")
         + "</div></div>")
    grid = (load_grid(paths.GRID_POINTS, location.latitude, location.longitude) if city.modelled
            else load_grid(Path("/nonexistent"), location.latitude, location.longitude))
    color = overall["color"]
    fill = f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.22)"
    st.plotly_chart(sampling_map(grid, color, fill, short_label(overall["label"]),
                                 districts=None if city.modelled else {}),
                    width="stretch", config={"scrollZoom": False})

    if today_weather:
        w = today_weather
        muggy = (w["feels_like_max"] or 0) - (w["temperature_max"] or 0) > 3
        details = [
            ("Feels like (max)", f'{w["feels_like_max"]:.0f}°',
             "humid heat" if muggy else "close to actual"),
            ("Humidity", f'{w["humidity"]:.0f} <small>%</small>', "daily mean"),
            ("Wind", f'{w["wind_max_kmh"]:.0f} <small>km/h</small>', f'from {w["wind_from"]}'),
            ("UV index", f'{w["uv_index"]:.0f}', w["uv_level"]),
            ("Rain", f'{w["precipitation_mm"]:.1f} <small>mm</small>',
             f'{w["precipitation_probability"] or 0:.0f}% chance'),
            ("Harmattan", "yes" if pd.Timestamp(today).month in HARMATTAN_MONTHS else "no",
             "Dec – Feb, dusty"),
        ]
        html('<div class="card"><h4>Today in detail</h4><div class="tiles">' + "".join(
            f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div>'
            f'<div class="s">{s}</div></div>' for k, v, s in details) + "</div></div>")

# ── Ask Kos ───────────────────────────────────────────────────────────────────────────────

with tab_ask:
    mode = "AI mode (Claude)" if assistant.claude_available() else "Offline mode"
    html(f'<div class="card"><div class="say"><div>{mascot("good", 56, backdrop=False)}</div><div>'
         f'<div class="who">ASK KOS · {mode.upper()}</div>Your air and weather buddy. Ask about '
         "tomorrow's weather, the air today, whether to go running, or what a pollutant is. "
         "Each question earns points (up to 5 a day)."
         + (" In AI mode, your questions and today's forecast are sent to Anthropic's Claude "
            "API to write the answer." if assistant.claude_available() else "")
         + "</div></div></div>")
    chat = st.session_state.setdefault("chat", [])

    picked = None
    columns = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        if columns[i % 2].button(suggestion, key=f"suggest_{i}", width="stretch"):
            picked = suggestion
    typed = st.chat_input("Ask Kos about the weather or the air…")
    question = typed or picked
    if question:
        reply, _ = assistant.answer(question, context, chat)
        chat += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
        if progress.asked_question(clock.date()):
            st.toast(f"+2 points for your curiosity · {progress.points} pts", icon="💬")
        for key in progress.refresh_badges():
            icon, name, _ = badge_info(key)
            st.toast(f"New badge: {name}", icon=icon)
        progress.save(paths.PROGRESS_PATH)

    for message in chat:
        with st.chat_message(message["role"], avatar="🐆" if message["role"] == "assistant"
                             else "🙂"):
            st.markdown(message["content"])

# ── Eco journey ───────────────────────────────────────────────────────────────────────────

with tab_journey:
    level = progress.level
    next_text = (f"{level['points_to_next']} points to {level['next_name']}"
                 if level["next_name"] else "Top level reached 🎉")
    html(f"""<div class="card"><h4>Your eco journey</h4><div class="level">
    <div class="big">{level["icon"]}</div><div style="flex:1">
    <div class="name">{level["name"]}</div>
    <div class="xp"><span style="width:{100 * level["progress"]:.0f}%"></span></div>
    <div style="font-size:0.85rem;color:{INK_2}">{next_text}</div></div></div>
    <div class="stats"><div><b>{progress.points}</b>points</div>
    <div><b>🔥 {progress.streak}</b>day streak</div>
    <div><b>{len(progress.badges)}/{len(BADGES)}</b>badges</div></div></div>""")

    rain_chance = (today_weather or {}).get("precipitation_probability")
    icon, challenge = daily_challenge(clock.date(), overall["aqi"], rain_chance)
    done = clock.date().isoformat() in progress.challenge_days
    html(f'<div class="card"><h4>Today\'s eco-challenge · +15 pts</h4>'
         f'<div style="font-size:1.02rem">{icon} {challenge}</div></div>')
    if done:
        st.success("Challenge done today. Thank you for the air! 🌿")
    elif st.button("I did it ✅", key="challenge", type="primary"):
        earned = progress.complete_challenge(clock.date())
        progress.refresh_badges()
        progress.save(paths.PROGRESS_PATH)
        st.toast(f"+{earned} points — eco hero!", icon="💪")
        st.rerun()

    question, options, answer_index, explanation = daily_quiz(clock.date())
    html(f'<div class="card"><h4>Daily quiz · +10 pts</h4>'
         f'<div style="font-size:1.02rem;font-weight:600">🧠 {question}</div></div>')
    answered = progress.quiz_days.get(clock.date().isoformat())
    if answered is None:
        choice = st.radio("Your answer", options, index=None, key="quiz",
                          label_visibility="collapsed")
        if st.button("Check my answer", key="quiz_submit", disabled=choice is None):
            progress.answer_quiz(clock.date(), options.index(choice))
            progress.refresh_badges()
            progress.save(paths.PROGRESS_PATH)
            st.rerun()
    else:
        verdict = "✅ Right!" if answered else f"❌ Not quite: {options[answer_index]}."
        st.info(f"{verdict} {explanation} Come back tomorrow for a new question.")

    badge_tiles = []
    for key, icon, name, description in BADGES:
        locked = "" if key in progress.badges else " locked"
        badge_tiles.append(f'<div class="tile{locked}"><div class="v">{icon}</div>'
                           f'<div class="k">{name}</div><div class="s">{description}</div></div>')
    html(f'<div class="card"><h4>Badges</h4><div class="tiles">{"".join(badge_tiles)}</div>'
         "</div>")

# ── Insights ──────────────────────────────────────────────────────────────────────────────

with tab_insights:
    if not city.modelled:
        st.info(f"The history and model accuracy below are for Abidjan, where AAQA's models are "
                f"trained. {city.name} shows live CAMS values and the CAMS forecast.")
    key = st.selectbox("Pollutant", list(POLLUTANTS),
                       format_func=lambda k: f"{POLLUTANTS[k].name} ({POLLUTANTS[k].statistic})")
    spec = POLLUTANTS[key]
    span = {"90 days": 90, "1 year": 365, "All": len(daily)}[
        st.radio("Period", ["90 days", "1 year", "All"], horizontal=True, index=1)]
    recent = daily[spec.target].dropna().tail(span)
    fig = go.Figure()
    fig.add_scatter(x=recent.index, y=recent, mode="lines", name=spec.name,
                    line={"color": "#9fb1b5", "width": 1})
    fig.add_scatter(x=recent.index, y=recent.rolling(14, center=True).mean(), mode="lines",
                    name="14-day mean", line={"color": BRAND["petrol"], "width": 2.5})
    if spec.who_on_target:
        fig.add_hline(y=spec.who_limit, line_color="#c8102e", line_width=1,
                      annotation_text=f"WHO {spec.who_label}")
    st.plotly_chart(chart_layout(fig), width="stretch")

    series = daily[spec.target].dropna()
    months = [MONTHS[m - 1] for m in series.index.month]
    box = px.box(pd.DataFrame({"month": months, spec.name: series.values}), x="month",
                 y=spec.name, category_orders={"month": MONTHS}, points=False,
                 color_discrete_sequence=[BRAND["green"]])
    if spec.who_on_target:
        box.add_hline(y=spec.who_limit, line_color="#c8102e", line_width=1)
    st.plotly_chart(chart_layout(box, 260), width="stretch")
    if spec.who_column in daily:
        share = (daily[spec.who_column].dropna().tail(span) > spec.who_limit).mean()
        st.caption(f"{share:.0%} of days above the WHO guideline ({spec.who_label}) over the "
                   f"period. Main sources: {spec.sources}.")

    with st.expander("🎯 Forecast accuracy"):
        if paths.METRICS_PATH.exists():
            metrics = json.loads(paths.METRICS_PATH.read_text())
            horizon = st.radio("Horizon", sorted(next(iter(metrics.values()))), horizontal=True,
                               format_func=lambda h: f"D+{h}")
            rows = []
            for pollutant, by_h in metrics.items():
                m = by_h[str(horizon)]
                gb, pers = m["models"]["gradient_boosting"], m["models"]["persistence"]
                rows.append({
                    "Pollutant": POLLUTANTS[pollutant].name,
                    "MAE (µg/m³)": round(gb["mae"], 2),
                    "RMSE (µg/m³)": round(gb["rmse"], 2),
                    "R²": round(gb["r2"], 2),
                    "“Same as today” MAE": round(pers["mae"], 2),
                    "Gain (cross-val.)":
                        f"{1 - m['cv_mae_mean'] / m['cv_persistence_mae_mean']:.0%}",
                    "Served": ("model" if m.get("selected_method") != "persistence"
                               else "persistence"),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Full model monitoring: run `make monitor` (http://localhost:8502).")

    with st.expander("ℹ️ Index scale, Kos the leopard's moods and data refresh"):
        html("".join(f'<div style="display:flex;gap:0.7rem;align-items:center;margin:0.35rem 0">'
                     f'{mascot(c.key, 40, backdrop=False)}<div>{pill(c.label, c.key)}<br>'
                     f'<span style="color:{INK_2};font-size:0.85rem">{c.advice}</span></div>'
                     "</div>" for c in CATEGORIES))
        st.caption("US EPA index; the overall index is that of the worst pollutant. The live "
                   "index uses this hour's values; the forecast uses daily statistics.")
        if st.button("Refresh data and retrain (about 4 min)"):
            refresh()

html('<div class="note">Weather: ECMWF, GFS and other models via Open-Meteo. Air quality: '
     "Copernicus CAMS model (~40 km grid, one value for the whole city), forecast by AAQA; "
     "PM2.5 calibrated on the US Embassy monitor (data: AirNow, US EPA / US Department of "
     "State, preliminary and not fully validated).</div>")
footer()
