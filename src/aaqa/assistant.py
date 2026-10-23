"""Kos, the app's assistant: answers questions about the weather, air quality and pollutants.

Two modes:
- offline (always available): a small intent matcher that answers from the live forecast and a
  built-in glossary;
- Claude (optional): used when ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN is set. It receives the
  same forecast as JSON and is told to answer briefly from it. Any API failure falls back to the
  offline answer, so the app never breaks.
"""

from __future__ import annotations

import json
import logging
import os
import re

import pandas as pd

from aaqa.pollutants import POLLUTANTS

log = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

GLOSSARY = {
    "pm2_5": "**PM2.5** are fine particles smaller than 2.5 micrometres, about 30 times thinner "
             "than a hair. They go deep into the lungs and even the blood, and are the "
             "pollutant with the biggest health impact. In Abidjan they come from burning "
             "(wood, charcoal, waste, traffic) and from Saharan dust during the harmattan. "
             "WHO guideline: 15 µg/m³ over 24 hours.",
    "pm10": "**PM10** are particles smaller than 10 micrometres: dust, sand, pollen and soot. "
            "They irritate the nose, throat and lungs. They peak in Abidjan when the harmattan "
            "brings Saharan dust (December to February). WHO guideline: 45 µg/m³ over 24 hours.",
    "ozone": "**Ozone (O3)** near the ground is not emitted directly: it forms in sunlight from "
             "traffic and industrial gases, so it peaks in the afternoon. It irritates the "
             "airways and worsens asthma. WHO guideline: 100 µg/m³ over 8 hours.",
    "nitrogen_dioxide": "**Nitrogen dioxide (NO2)** is a gas mostly from road traffic and diesel "
                        "generators. It inflames the airways, especially in children and people "
                        "with asthma. WHO guideline: 25 µg/m³ over 24 hours. Note: our satellite "
                        "model data probably underestimates NO2 near busy roads.",
    "sulphur_dioxide": "**Sulphur dioxide (SO2)** comes from burning high-sulphur fuels "
                       "(refineries, ships, some power plants). It irritates the lungs. "
                       "WHO guideline: 40 µg/m³ over 24 hours.",
    "carbon_monoxide": "**Carbon monoxide (CO)** is an odourless gas from incomplete burning: "
                       "traffic, charcoal and wood stoves, fires. In closed rooms it can be "
                       "deadly. Outdoors in Abidjan it stays far below the WHO guideline "
                       "(10 mg/m³ over 8 hours), but it is a good marker of combustion.",
    "aqi": "The **Air Quality Index (AQI)** turns pollutant concentrations into one score from 0 "
           "to 500 (US EPA scale): 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for sensitive "
           "groups, 151–200 Unhealthy, 201–300 Very unhealthy, above 300 Hazardous. The app's "
           "overall index is the score of the worst pollutant.",
    "who": "The **WHO guidelines** (2021) are the levels the World Health Organization "
           "recommends not exceeding to protect health, for example 15 µg/m³ of PM2.5 over 24 "
           "hours and 5 µg/m³ as an annual mean. The US Embassy monitor measured an average of "
           "25 µg/m³ in Abidjan, 5 times the annual guideline.",
    "harmattan": "The **harmattan** is a dry, dusty wind from the Sahara that reaches Abidjan "
                 "from December to February. It roughly doubles fine-particle pollution: about "
                 "two days in three exceed the WHO PM2.5 guideline during that season.",
    "dust": "**Desert dust** is sand and mineral particles carried from the Sahara, mainly by the "
            "harmattan. It counts in PM10 and PM2.5 and causes the worst pollution days in "
            "Abidjan, like 8 February 2024 (PM2.5 of 126 µg/m³).",
}

_POLLUTANT_WORDS = {
    "pm2_5": [r"pm\s?2[.,]?5", r"pm25", r"fine particle"],
    "pm10": [r"pm\s?10", r"coarse particle"],
    "ozone": [r"\bo3\b", r"ozone"],
    "nitrogen_dioxide": [r"\bno2\b", r"nitrogen"],
    "sulphur_dioxide": [r"\bso2\b", r"sulph", r"sulf"],
    "carbon_monoxide": [r"\bco\b", r"carbon monoxide"],
}
_TOPIC_WORDS = {
    "aqi": [r"\baqi\b", r"air quality index", r"\bindex\b"],
    "who": [r"\bwho\b", r"guideline"],
    "harmattan": [r"harmattan"],
    "dust": [r"\bdust\b", r"sahara"],
}
_DEFINE = re.compile(r"what\s+(is|are|'s)|what's|explain|define|meaning|mean\b|tell me about")
_WEATHER = re.compile(r"weather|temperature|\bhot\b|\bcold\b|\bwarm\b|rain|umbrella|sun|wind|"
                      r"\buv\b|humid|storm|degrees|°")
_AIR = re.compile(r"\bair\b|pollut|quality|breath|mask|\baqi\b|smog|haze")
_ACTIVITY = re.compile(r"\brun\b|running|jog|sport|exercise|walk|outside|outdoor|picnic|beach|"
                       r"football|cycle|bike|play")
_BEST_TIME = re.compile(r"best time|what time|which hour|when should|when can")
_GREETING = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|help|"
                       r"what can you do)\b")


# ── Context ─────────────────────────────────────────────────────────────────────────────────

def build_context(forecast: dict, weather_days: list[dict], now: dict | None,
                  today: str, city: str = "Abidjan, Côte d'Ivoire") -> dict:
    """Compact, JSON-ready picture of today and the next days, shared by both modes."""
    air = {forecast["observed"]["date"]: forecast["observed"]}
    air.update({f["date"]: f for f in forecast["forecasts"]})

    def air_summary(entry: dict | None) -> dict | None:
        if not entry:
            return None
        return {
            "aqi": entry["overall"]["aqi"],
            "category": entry["overall"]["label"],
            "main_pollutant": entry["overall"]["dominant_name"],
            "advice": entry["overall"]["advice"],
            "pollutants": {p["name"]: {"value_ug_m3": p["value"], "statistic": p["statistic"],
                                       "category": p["label"]} for p in entry["pollutants"]},
        }

    days = []
    for w in weather_days:
        days.append({"date": w["date"], "weekday": DAYS[pd.Timestamp(w["date"]).day_of_week],
                     "weather": {k: w[k] for k in (
                         "description", "temperature_min", "temperature_max", "feels_like_max",
                         "precipitation_mm", "precipitation_probability", "humidity",
                         "wind_max_kmh", "wind_from", "uv_index", "uv_level")},
                     "air_quality": air_summary(air.get(w["date"]))})
    if not days:  # offline: air quality only
        days = [{"date": d, "weekday": DAYS[pd.Timestamp(d).day_of_week], "weather": None,
                 "air_quality": air_summary(entry)} for d, entry in sorted(air.items())
                if d >= today]
    context = {"city": city, "today": today, "days": days}
    if now:
        context["now"] = {k: now[k] for k in ("time", "temperature", "feels_like", "humidity",
                                              "wind_kmh", "description")}
        context["next_hours"] = [{"hour": h["hour"], "temperature": h["temperature"],
                                  "rain_chance": h["precipitation_probability"]}
                                 for h in now["hourly"]]
    return context


# ── Offline mode ────────────────────────────────────────────────────────────────────────────

def _find_day(question: str, context: dict) -> dict | None:
    days = context["days"]
    if not days:
        return None
    by_date = {d["date"]: d for d in days}
    today = pd.Timestamp(context["today"])
    if "tomorrow" in question:
        return by_date.get((today + pd.Timedelta(days=1)).date().isoformat())
    if "weekend" in question:
        return next((d for d in days if d["weekday"] in ("saturday", "sunday")), None)
    for name in DAYS:
        if name in question:
            return next((d for d in days if d["weekday"] == name), None)
    return by_date.get(context["today"], days[0])


def _day_label(day: dict, context: dict) -> str:
    delta = (pd.Timestamp(day["date"]) - pd.Timestamp(context["today"])).days
    return {0: "Today", 1: "Tomorrow"}.get(delta, day["weekday"].capitalize())


def _weather_sentence(day: dict, context: dict) -> str:
    w = day["weather"]
    if not w:
        return "I can't reach the weather service right now, so I have no weather forecast."
    rain = w["precipitation_probability"] or 0
    city = context["city"].split(",")[0]
    text = (f"{_day_label(day, context)} in {city}: **{w['description'].lower()}**, "
            f"{w['temperature_min']:.0f}–{w['temperature_max']:.0f} °C "
            f"(feels like up to {w['feels_like_max']:.0f} °C), {rain:.0f}% chance of rain, "
            f"wind up to {w['wind_max_kmh']:.0f} km/h from the {w['wind_from']}.")
    if rain >= 60:
        text += " ☔ Take an umbrella."
    if (w["uv_index"] or 0) >= 8:
        text += " 🧴 UV is very high: sunscreen and a hat around midday."
    return text


def _air_sentence(day: dict, context: dict) -> str:
    air = day.get("air_quality")
    if not air:
        return (f"I only forecast air quality three days ahead, so I have nothing for "
                f"{_day_label(day, context).lower()} yet.")
    return (f"Air quality {_day_label(day, context).lower()}: **{air['category']}** "
            f"(AQI {air['aqi']}, main pollutant {air['main_pollutant']}). {air['advice']}")


def _activity_sentence(day: dict, context: dict) -> str:
    air, w = day.get("air_quality"), day.get("weather")
    parts = []
    if air:
        if air["aqi"] <= 50:
            parts.append("✅ The air is clean, so outdoor sport is fine.")
        elif air["aqi"] <= 100:
            parts.append("🙂 The air is acceptable; if you are very sensitive, keep the "
                         "effort moderate.")
        else:
            parts.append(f"⚠️ The air is {air['category'].lower()}: prefer indoor exercise or "
                         "a short, gentle session.")
    if w:
        if (w["precipitation_probability"] or 0) >= 60:
            parts.append("Rain is likely, so plan a covered spot or a dry window.")
        if (w["feels_like_max"] or 0) >= 35:
            parts.append("It will feel very hot: go early morning or after sunset and drink "
                         "plenty of water.")
    return " ".join(parts) or "I don't have enough data to advise for that day."


def _best_time_sentence(context: dict) -> str:
    hours = context.get("next_hours") or []
    if not hours:
        return "I can't see the hourly forecast right now."
    pleasant = [h for h in hours if (h["rain_chance"] or 0) < 30]
    pool = pleasant or hours
    best = min(pool, key=lambda h: h["temperature"])
    return (f"The coolest dry moment in the next 24 hours is around **{best['hour']}** "
            f"({best['temperature']:.0f} °C, {best['rain_chance'] or 0:.0f}% chance of rain). "
            "Early morning is usually the coolest time for sport.")


def _pollutant_value(key: str, day: dict, context: dict) -> str | None:
    air = day.get("air_quality")
    name = POLLUTANTS[key].name
    if not air or name not in air["pollutants"]:
        return None
    p = air["pollutants"][name]
    return (f"{name} {_day_label(day, context).lower()}: **{p['value_ug_m3']:.1f} µg/m³** "
            f"({p['statistic']}), {p['category'].lower()}.")


def answer_offline(question: str, context: dict) -> str:
    q = question.lower().strip()
    pollutants = [k for k, pats in _POLLUTANT_WORDS.items() if any(re.search(p, q) for p in pats)]
    topics = [k for k, pats in _TOPIC_WORDS.items() if any(re.search(p, q) for p in pats)]
    day = _find_day(q, context)

    if _GREETING.search(q) and not (pollutants or topics or _WEATHER.search(q)):
        return ("Hi, I'm **Kos** 🐆, your air and weather buddy. Ask me things like "
                "*What's the weather tomorrow?*, *Can I go running today?*, *What is PM2.5?* or "
                "*What is the harmattan?*")
    if _BEST_TIME.search(q):
        return _best_time_sentence(context)
    if pollutants or topics:
        wants_value = day is not None and not _DEFINE.search(q) and pollutants
        answers = []
        for key in pollutants:
            value = _pollutant_value(key, day, context) if day else None
            answers.append(value if wants_value and value else GLOSSARY[key])
            if not wants_value and value:
                answers.append(f"Right now: {value}")
        answers += [GLOSSARY[k] for k in topics if not (k == "aqi" and pollutants)]
        return "\n\n".join(answers)
    if day is None:
        return "I don't have forecast data loaded yet."
    if _ACTIVITY.search(q):
        return _activity_sentence(day, context)
    wants_weather, wants_air = bool(_WEATHER.search(q)), bool(_AIR.search(q))
    if wants_weather and not wants_air:
        return _weather_sentence(day, context)
    if wants_air and not wants_weather:
        return _air_sentence(day, context)
    if wants_weather or wants_air or re.search(r"today|tomorrow|tonight|forecast|"
                                               + "|".join(DAYS), q):
        return f"{_weather_sentence(day, context)}\n\n{_air_sentence(day, context)}"
    return ("I'm not sure I understood 🤔. I can tell you the weather and air quality for today "
            "and the next days, whether it's a good time to go outside, or explain PM2.5, PM10, "
            "ozone, NO2, SO2, CO, the AQI, the WHO guidelines and the harmattan.")


# ── Claude mode ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Kos, the friendly assistant of the AAQA app by KOSCON ("together for \
the better"), which shows the weather and air quality forecast for Abidjan, Côte d'Ivoire, and \
other cities. The city is given in the forecast JSON; answer for that city.

Answer questions about the weather, air quality, pollutants (PM2.5, PM10, O3, NO2, SO2, CO), the \
Air Quality Index, WHO guidelines, the harmattan, and practical advice such as whether to exercise \
outdoors. Each user message starts with a <forecast> JSON block: base every figure you give on it \
and never invent numbers. If the data does not cover the question, say so. Pollution values come \
from the Copernicus CAMS model (about 40 km resolution, one value for the whole city), which \
probably underestimates NO2 near busy roads; mention it only when relevant.

Reply in English, in 1 to 4 short sentences, warm and plain, with at most one emoji. Use **bold** \
for the key figure. You are not a doctor: for health symptoms, advise seeing a health \
professional. Politely decline topics unrelated to weather, air quality or the environment."""


def claude_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def answer_with_claude(question: str, context: dict, history: list[dict], client=None) -> str:
    import anthropic

    client = client or anthropic.Anthropic(max_retries=1, timeout=30.0)
    payload = json.dumps(context, ensure_ascii=False, sort_keys=True)
    messages = [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    messages.append({"role": "user",
                     "content": f"<forecast>\n{payload}\n</forecast>\n\n{question}"})
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        output_config={"effort": "low"},
        # Server-side refusal fallback: if the model declines, the API retries on a fallback
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=messages,
    )
    if response.stop_reason == "refusal":
        return "Sorry, I can't help with that one. Ask me about the weather or the air."
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise RuntimeError(f"empty answer (stop_reason={response.stop_reason})")
    return text


def answer(question: str, context: dict, history: list[dict] | None = None,
           client=None) -> tuple[str, str]:
    """Return (answer, mode) where mode is "claude" or "offline"."""
    if client is not None or claude_available():
        try:
            return answer_with_claude(question, context, history or [], client), "claude"
        except Exception as exc:  # network, credentials, rate limit: fall back silently
            log.warning("Claude unavailable, answering offline: %s", exc)
    return answer_offline(question, context), "offline"


def daily_summary(context: dict) -> str:
    """One friendly line for the top of the app: now, today's air, tomorrow's weather."""
    parts = []
    now = context.get("now")
    if now:
        parts.append(f"{now['description']} and {now['temperature']:.0f} °C right now")
    today = next((d for d in context["days"] if d["date"] == context["today"]), None)
    if today and today.get("air_quality"):
        category = today["air_quality"]["category"].lower()
        parts.append(f"the air is {category}" if parts else f"The air is {category}")
    sentence = (", ".join(parts) + ".") if parts else ""
    sentence = sentence[:1].upper() + sentence[1:]
    tomorrow = _find_day("tomorrow", context)
    if tomorrow and tomorrow.get("weather"):
        w = tomorrow["weather"]
        sentence += (f" Tomorrow: {w['description'].lower()}, up to "
                     f"{w['temperature_max']:.0f} °C")
        rain = w["precipitation_probability"] or 0
        sentence += " — keep an umbrella handy ☔." if rain >= 60 else "."
    return sentence.strip() or "Ask me anything about the weather or the air."
