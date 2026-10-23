"""Official weather forecast for the coming days (Open-Meteo: ECMWF, GFS, ICON… models).

Our model does not forecast the weather: numerical models run by weather services are far
more accurate than anything that could be learned from four years of local data. We fetch
their forecast and show it next to the pollution forecast.
"""

from __future__ import annotations

from aaqa.config import Location
from aaqa.data.fetch import WEATHER_FORECAST_URL, _get

DAILY_VARS = [
    "weather_code", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max",
    "precipitation_sum", "precipitation_probability_max", "relative_humidity_2m_mean",
    "wind_speed_10m_max", "wind_direction_10m_dominant", "uv_index_max",
]

# WMO weather codes (WMO 4677, simplified set used by Open-Meteo): (description, icon, sky)
WEATHER_CODES = {
    0: ("Clear sky", "☀️", "clear"), 1: ("Mainly clear", "🌤️", "clear"),
    2: ("Partly cloudy", "⛅", "cloudy"), 3: ("Overcast", "☁️", "cloudy"),
    45: ("Fog", "🌫️", "cloudy"), 48: ("Freezing fog", "🌫️", "cloudy"),
    51: ("Light drizzle", "🌦️", "rain"), 53: ("Drizzle", "🌦️", "rain"),
    55: ("Dense drizzle", "🌧️", "rain"),
    61: ("Light rain", "🌧️", "rain"), 63: ("Rain", "🌧️", "rain"), 65: ("Heavy rain", "🌧️", "rain"),
    80: ("Showers", "🌦️", "rain"), 81: ("Heavy showers", "🌧️", "rain"),
    82: ("Violent showers", "⛈️", "storm"),
    95: ("Thunderstorm", "⛈️", "storm"), 96: ("Thunderstorm with hail", "⛈️", "storm"),
    99: ("Severe thunderstorm with hail", "⛈️", "storm"),
}
UNKNOWN_CODE = ("—", "🌡️", "clear")
# After sunset: night icons (no sun) and a night sky
NIGHT_ICONS = {0: "🌙", 1: "🌙", 2: "☁️", 51: "🌧️", 53: "🌧️", 80: "🌧️"}
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

CURRENT_VARS = ["temperature_2m", "apparent_temperature", "relative_humidity_2m", "weather_code",
                "wind_speed_10m", "wind_direction_10m", "is_day"]
HOURLY_VARS = ["temperature_2m", "weather_code", "precipitation_probability", "is_day"]


def describe_code(code: int | None, is_day: bool = True) -> tuple[str, str, str]:
    """(description, icon, sky) for a WMO weather code, by day or by night."""
    description, icon, sky = WEATHER_CODES.get(code, UNKNOWN_CODE)
    if not is_day:
        icon = NIGHT_ICONS.get(code, icon)
        sky = "storm" if sky == "storm" else "night"
    return description, icon, sky


def wind_sector(degrees: float | None) -> str | None:
    if degrees is None:
        return None
    return COMPASS[round(degrees / 45) % 8]


def uv_level(uv: float | None) -> str | None:
    if uv is None:
        return None
    for limit, label in ((3, "low"), (6, "moderate"), (8, "high"), (11, "very high")):
        if uv < limit:
            return label
    return "extreme"


def parse_daily(payload: dict) -> list[dict]:
    daily = payload["daily"]
    days = []
    for i, day in enumerate(daily["time"]):
        value = {var: daily.get(var, [None] * len(daily["time"]))[i] for var in DAILY_VARS}
        description, icon, sky = describe_code(value["weather_code"])
        days.append({
            "date": day,
            "description": description,
            "icon": icon,
            "sky": sky,
            "temperature_min": value["temperature_2m_min"],
            "temperature_max": value["temperature_2m_max"],
            "feels_like_max": value["apparent_temperature_max"],
            "precipitation_mm": value["precipitation_sum"],
            "precipitation_probability": value["precipitation_probability_max"],
            "humidity": value["relative_humidity_2m_mean"],
            "wind_max_kmh": value["wind_speed_10m_max"],
            "wind_from": wind_sector(value["wind_direction_10m_dominant"]),
            "uv_index": value["uv_index_max"],
            "uv_level": uv_level(value["uv_index_max"]),
        })
    return days


def fetch_weather_forecast(loc: Location, days: int = 4) -> list[dict]:
    """Daily forecast: today, then the following `days - 1` days."""
    payload = _get(WEATHER_FORECAST_URL, {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "daily": ",".join(DAILY_VARS),
        "timezone": loc.timezone,
        "forecast_days": days,
    }, retries=2, timeout=15)
    return parse_daily(payload)


def parse_now(payload: dict) -> dict:
    """Current conditions and the hourly forecast that follows."""
    current = payload["current"]
    is_day = bool(current.get("is_day", 1))
    description, icon, sky = describe_code(current.get("weather_code"), is_day)
    hourly = payload.get("hourly", {})
    hours = []
    for i, time in enumerate(hourly.get("time", [])):
        value = {var: hourly.get(var, [None] * len(hourly["time"]))[i] for var in HOURLY_VARS}
        _, hour_icon, _ = describe_code(value["weather_code"], bool(value["is_day"]))
        hours.append({
            "time": time,
            "hour": time[11:16],
            "temperature": value["temperature_2m"],
            "icon": hour_icon,
            "precipitation_probability": value["precipitation_probability"],
        })
    return {
        "time": current["time"],
        "temperature": current.get("temperature_2m"),
        "feels_like": current.get("apparent_temperature"),
        "humidity": current.get("relative_humidity_2m"),
        "wind_kmh": current.get("wind_speed_10m"),
        "wind_from": wind_sector(current.get("wind_direction_10m")),
        "description": description,
        "icon": icon,
        "sky": sky,
        "is_day": is_day,
        "hourly": hours,
    }


def fetch_now(loc: Location, hours: int = 24) -> dict:
    """Current conditions plus the next `hours` hours."""
    payload = _get(WEATHER_FORECAST_URL, {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "current": ",".join(CURRENT_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "forecast_hours": hours,
        "timezone": loc.timezone,
    }, retries=2, timeout=15)
    return parse_now(payload)
