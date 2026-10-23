"""Live air quality: the current hour and the last 24 hours from the CAMS model (Open-Meteo).

The live index applies the AQI breakpoints to the current hourly values, as consumer air-quality
apps do; the official index uses 24-hour (particles) or 8-hour (O3, CO) means, which the
averages over 1, 8 and 24 hours approximate.
"""

from __future__ import annotations

import pandas as pd

from aaqa.aqi import describe, overall
from aaqa.config import Location
from aaqa.data.fetch import AIR_QUALITY_URL, _get

LIVE_VARS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide"]


def parse_air_now(payload: dict, adjust_pm25=None) -> dict:
    """`adjust_pm25(value, day)` optionally corrects PM2.5 (calibration on the US Embassy)."""
    current = payload["current"]
    hourly = pd.DataFrame({var: payload["hourly"].get(var) for var in LIVE_VARS},
                          index=pd.to_datetime(payload["hourly"]["time"]))
    now = pd.Timestamp(current["time"])
    past = hourly[hourly.index <= now].dropna(how="all")

    def snapshot(values: dict) -> dict:
        values, raw = dict(values), None
        if adjust_pm25 and values.get("pm2_5") is not None and pd.notna(values["pm2_5"]):
            raw = float(values["pm2_5"])
            values["pm2_5"] = adjust_pm25(raw, now)
        described = [describe(key, float(values[key])) for key in LIVE_VARS
                     if values.get(key) is not None and pd.notna(values[key])]
        if raw is not None:
            for value in described:
                if value["pollutant"] == "pm2_5":
                    value["raw_value"] = round(raw, 1)
        return {"overall": overall(described), "pollutants": described}

    averages = {}
    for hours in (1, 8, 24):
        window = past[past.index > now - pd.Timedelta(hours=hours)]
        if not window.empty:
            averages[f"{hours}h"] = snapshot(window.mean().to_dict())["overall"]
    return {"time": current["time"], **snapshot(current), "averages": averages}


def fetch_air_now(loc: Location, adjust_pm25=None) -> dict:
    payload = _get(AIR_QUALITY_URL, {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "current": ",".join(LIVE_VARS),
        "hourly": ",".join(LIVE_VARS),
        "past_days": 1,
        "forecast_days": 1,
        "timezone": loc.timezone,
    }, retries=2, timeout=15)  # live display: fail fast rather than freeze the app
    return parse_air_now(payload, adjust_pm25)


def daily_forecast(payload: dict, days: int = 3) -> list[dict]:
    """CAMS hourly forecast → one snapshot per day (same statistics as the AAQA models)."""
    hourly = pd.DataFrame({var: payload["hourly"].get(var) for var in LIVE_VARS},
                          index=pd.to_datetime(payload["hourly"]["time"]))
    eight_hour = hourly[["ozone", "carbon_monoxide"]].rolling(8, min_periods=6).mean()
    grouped, grouped_8h = hourly.groupby(hourly.index.normalize()), eight_hour.groupby(
        eight_hour.index.normalize())
    daily = pd.DataFrame({
        "pm2_5": grouped["pm2_5"].mean(), "pm10": grouped["pm10"].mean(),
        "ozone": grouped_8h["ozone"].max(), "nitrogen_dioxide": grouped["nitrogen_dioxide"].max(),
        "sulphur_dioxide": grouped["sulphur_dioxide"].max(),
        "carbon_monoxide": grouped_8h["carbon_monoxide"].max(),
    })[grouped["pm2_5"].count() >= 18]
    first = pd.Timestamp(payload["current"]["time"]).normalize() + pd.Timedelta(days=1)
    forecasts = []
    for horizon in range(1, days + 1):
        day = first + pd.Timedelta(days=horizon - 1)
        if day not in daily.index:
            continue
        values = [describe(key, float(v)) for key, v in daily.loc[day].items() if pd.notna(v)]
        forecasts.append({"date": day.date().isoformat(), "horizon_days": horizon,
                          "overall": overall(values), "pollutants": values})
    return forecasts


def fetch_city_air(loc: Location, days: int = 3) -> tuple[dict, list[dict]]:
    """Live snapshot and the CAMS daily forecast for any city (one request)."""
    payload = _get(AIR_QUALITY_URL, {
        "latitude": loc.latitude, "longitude": loc.longitude,
        "current": ",".join(LIVE_VARS), "hourly": ",".join(LIVE_VARS),
        "past_days": 1, "forecast_days": days + 2, "timezone": loc.timezone,
    }, retries=2, timeout=15)
    return parse_air_now(payload), daily_forecast(payload, days)
