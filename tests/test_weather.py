import pytest

from aaqa.config import Location
from aaqa.data import weather_forecast as wf

PAYLOAD = {"daily": {
    "time": ["2026-10-22", "2026-10-23"],
    "weather_code": [80, 0],
    "temperature_2m_max": [31.5, 33.0],
    "temperature_2m_min": [24.9, 25.1],
    "apparent_temperature_max": [37.0, 38.2],
    "precipitation_sum": [6.6, 0.0],
    "precipitation_probability_max": [98, 3],
    "relative_humidity_2m_mean": [86, 72],
    "wind_speed_10m_max": [8.5, 14.0],
    "wind_direction_10m_dominant": [225, 10],
    "uv_index_max": [8.5, 2.0],
}}


def test_parse_daily():
    days = wf.parse_daily(PAYLOAD)
    assert [d["date"] for d in days] == ["2026-10-22", "2026-10-23"]
    rainy, sunny = days
    assert rainy["description"] == "Showers" and rainy["wind_from"] == "SW"
    assert rainy["sky"] == "rain" and sunny["sky"] == "clear"
    assert rainy["temperature_max"] == 31.5 and rainy["precipitation_probability"] == 98
    assert sunny["description"] == "Clear sky" and sunny["wind_from"] == "N"
    assert rainy["uv_level"] == "very high" and sunny["uv_level"] == "low"


def test_unknown_code_and_missing_values():
    payload = {"daily": {"time": ["2026-10-22"], "weather_code": [7]}}
    day = wf.parse_daily(payload)[0]
    assert day["description"] == "—"
    assert day["temperature_max"] is None and day["wind_from"] is None


@pytest.mark.parametrize("degrees, sector", [(0, "N"), (44, "NE"), (180, "S"), (350, "N")])
def test_wind_sector(degrees, sector):
    assert wf.wind_sector(degrees) == sector


def test_fetch_requests_daily_variables(monkeypatch):
    calls = []
    monkeypatch.setattr(wf, "_get", lambda url, params, **kw: calls.append(params) or PAYLOAD)
    assert len(wf.fetch_weather_forecast(Location(), days=2)) == 2
    assert "temperature_2m_max" in calls[0]["daily"] and calls[0]["forecast_days"] == 2


NOW_PAYLOAD = {
    "current": {"time": "2026-10-22T21:30", "temperature_2m": 27.0, "apparent_temperature": 31.9,
                "relative_humidity_2m": 82, "weather_code": 0, "wind_speed_10m": 6.8,
                "wind_direction_10m": 207, "is_day": 0},
    "hourly": {"time": ["2026-10-22T21:00", "2026-10-23T07:00"], "temperature_2m": [27.1, 25.4],
               "weather_code": [0, 80], "precipitation_probability": [4, 60], "is_day": [0, 1]},
}
NOW = wf.parse_now(NOW_PAYLOAD)


def test_parse_now():
    assert NOW["temperature"] == 27.0 and NOW["feels_like"] == 31.9
    assert NOW["description"] == "Clear sky" and NOW["wind_from"] == "SW"
    assert NOW["icon"] == "🌙" and NOW["sky"] == "night" and NOW["is_day"] is False
    night, morning = NOW["hourly"]
    assert night["hour"] == "21:00" and night["icon"] == "🌙"
    assert morning["icon"] == "🌦️" and morning["precipitation_probability"] == 60
    assert wf.describe_code(80, is_day=False)[1] == "🌧️"  # no sun in night showers


@pytest.mark.parametrize("code, is_day, icon, sky", [
    (0, True, "☀️", "clear"), (0, False, "🌙", "night"), (63, False, "🌧️", "night"),
    (95, False, "⛈️", "storm"),
])
def test_describe_code_by_day_and_night(code, is_day, icon, sky):
    _, got_icon, got_sky = wf.describe_code(code, is_day)
    assert (got_icon, got_sky) == (icon, sky)


def test_fetch_now_requests_current_and_hourly(monkeypatch):
    calls = []
    monkeypatch.setattr(wf, "_get", lambda url, params, **kw: calls.append(params) or NOW_PAYLOAD)
    assert wf.fetch_now(Location(), hours=2)["temperature"] == 27.0
    assert "is_day" in calls[0]["current"] and calls[0]["forecast_hours"] == 2
