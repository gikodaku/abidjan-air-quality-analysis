"""Download hourly pollution (CAMS) and weather (ERA5) data from Open-Meteo.

Free sources, no API key needed:
- https://open-meteo.com/en/docs/air-quality-api  (Copernicus CAMS model, ~40 km grid)
- https://open-meteo.com/en/docs/historical-weather-api  (ERA5 reanalysis)
- https://open-meteo.com/en/docs  (forecasts, used for the last days ERA5 does not cover yet)
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd
import requests

from aaqa.config import Location

log = logging.getLogger(__name__)

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

AIR_QUALITY_VARS = [
    "pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide", "sulphur_dioxide", "dust",
]
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m",
    "wind_direction_10m", "surface_pressure",
]

# ERA5 is published with a delay of about 5 days
ERA5_DELAY_DAYS = 6
CHUNK_DAYS = 365


def _get(url: str, params: dict, retries: int = 3, timeout: float = 60) -> dict:
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(payload.get("reason", "Open-Meteo error"))
            return payload
        except (requests.RequestException, RuntimeError) as exc:
            if attempt == retries:
                raise
            log.warning("%s failed (%s), retry %d/%d", url, exc, attempt, retries)
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def _to_frame(payload: dict, variables: list[str]) -> pd.DataFrame:
    hourly = payload["hourly"]
    frame = pd.DataFrame({var: hourly.get(var) for var in variables})
    frame.insert(0, "time", pd.to_datetime(hourly["time"]))
    return frame


def _date_chunks(start: date, end: date, days: int = CHUNK_DAYS):
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def _fetch_range(url: str, variables: list[str], loc: Location, start: date, end: date):
    frames = []
    for chunk_start, chunk_end in _date_chunks(start, end):
        log.info("%s : %s → %s", url.split("//")[1].split("/")[0], chunk_start, chunk_end)
        payload = _get(url, {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "hourly": ",".join(variables),
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "timezone": loc.timezone,
        })
        frames.append(_to_frame(payload, variables))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["time"])


def fetch_air_quality(loc: Location, start: date, end: date) -> pd.DataFrame:
    """Hourly pollution: PM2.5, PM10, NO2, O3, CO, SO2 and desert dust (µg/m³)."""
    return _fetch_range(AIR_QUALITY_URL, AIR_QUALITY_VARS, loc, start, end)


def fetch_weather(loc: Location, start: date, end: date) -> pd.DataFrame:
    """Hourly weather: ERA5 up to D-6, then the forecast API for recent days."""
    archive_end = min(end, date.today() - timedelta(days=ERA5_DELAY_DAYS))
    archive = _fetch_range(WEATHER_ARCHIVE_URL, WEATHER_VARS, loc, start, archive_end)
    if end <= archive_end:
        return archive

    recent_start = archive_end + timedelta(days=1)
    payload = _get(WEATHER_FORECAST_URL, {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "hourly": ",".join(WEATHER_VARS),
        "start_date": recent_start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": loc.timezone,
    })
    recent = _to_frame(payload, WEATHER_VARS)
    return pd.concat([archive, recent], ignore_index=True)


def resolve_dates(start_date: str, end_date: str | None) -> tuple[date, date]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date) if end_date else date.today() - timedelta(days=1)
    if end < start:
        raise ValueError(f"end_date ({end}) is before start_date ({start})")
    return start, end


def fetch_grid_points(loc: Location) -> dict:
    """Model grid points that the downloads actually come from (the API snaps to them)."""
    day = (date.today() - timedelta(days=ERA5_DELAY_DAYS + 1)).isoformat()
    params = {"latitude": loc.latitude, "longitude": loc.longitude, "start_date": day,
              "end_date": day, "timezone": loc.timezone}
    air = _get(AIR_QUALITY_URL, {**params, "hourly": "pm2_5"}, retries=2, timeout=30)
    weather = _get(WEATHER_ARCHIVE_URL, {**params, "hourly": "temperature_2m"},
                   retries=2, timeout=30)
    return {
        "requested": {"latitude": loc.latitude, "longitude": loc.longitude},
        "air_quality": {"latitude": air["latitude"], "longitude": air["longitude"],
                        "model": "Copernicus CAMS global", "resolution_deg": 0.4},
        "weather": {"latitude": weather["latitude"], "longitude": weather["longitude"],
                    "model": "ERA5 reanalysis (Open-Meteo)"},
    }
