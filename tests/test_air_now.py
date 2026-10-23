import pandas as pd
import pytest

from aaqa.config import Location
from aaqa.data import air_now

TIMES = pd.date_range("2026-10-22 00:00", periods=48, freq="h")
NOW_TIME = "2026-10-23T00:00"


def _series(base: float) -> list[float]:
    return [base + (i % 24) for i in range(len(TIMES))]


PAYLOAD = {
    "current": {"time": NOW_TIME, "pm2_5": 40.0, "pm10": 60.0, "nitrogen_dioxide": 20.0,
                "ozone": 50.0, "sulphur_dioxide": 2.0, "carbon_monoxide": 300.0},
    "hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in TIMES],
               "pm2_5": _series(5), "pm10": _series(10), "nitrogen_dioxide": _series(1),
               "ozone": _series(30), "sulphur_dioxide": _series(0.5),
               "carbon_monoxide": _series(200)},
}
LIVE = air_now.parse_air_now(PAYLOAD)


def test_live_index_from_current_hour():
    assert LIVE["time"] == NOW_TIME
    assert LIVE["overall"]["dominant_pollutant"] == "pm2_5"
    assert LIVE["overall"]["category"] == "usg"  # 40 µg/m³ PM2.5
    assert len(LIVE["pollutants"]) == 6


def test_averages_use_past_hours_only():
    assert list(LIVE["averages"]) == ["1h", "8h", "24h"]
    # 1h = the 00:00 value (5 µg/m³ PM2.5) — future hours of the payload are ignored
    assert LIVE["averages"]["1h"]["aqi"] < LIVE["overall"]["aqi"]
    assert LIVE["averages"]["24h"]["aqi"] > LIVE["averages"]["1h"]["aqi"]


def test_fetch_uses_short_timeout(monkeypatch):
    calls = []

    def fake_get(url, params, retries=3, timeout=60):
        calls.append((params, retries, timeout))
        return PAYLOAD

    monkeypatch.setattr(air_now, "_get", fake_get)
    assert air_now.fetch_air_now(Location())["overall"]["aqi"] == LIVE["overall"]["aqi"]
    params, retries, timeout = calls[0]
    assert "pm2_5" in params["current"] and params["past_days"] == 1
    assert timeout <= 15 and retries <= 2


@pytest.mark.parametrize("aqi, position", [(0, 0), (50, 100 / 6), (75, 25), (500, 100)])
def test_scale_position(aqi, position):
    scale = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 500)]
    for i, (low, high) in enumerate(scale):
        if aqi <= high:
            expected = 100 * (i + (max(aqi, low) - low) / (high - low)) / 6
            break
    assert expected == pytest.approx(position)


def test_daily_cams_forecast_for_any_city():
    times = pd.date_range("2026-10-23 00:00", periods=5 * 24, freq="h")
    payload = {"current": {"time": "2026-10-23T09:00"},
               "hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                          **{var: [10.0 + (i // 24) * 30 for i in range(len(times))]
                             for var in air_now.LIVE_VARS}}}
    days = air_now.daily_forecast(payload, days=3)
    assert [d["date"] for d in days] == ["2026-10-24", "2026-10-25", "2026-10-26"]
    assert [d["horizon_days"] for d in days] == [1, 2, 3]
    pm = [next(p for p in d["pollutants"] if p["pollutant"] == "pm2_5")["value"] for d in days]
    assert pm == [40.0, 70.0, 100.0]  # daily means
    assert days[2]["overall"]["aqi"] > days[0]["overall"]["aqi"]


def test_cities():
    from aaqa.cities import CITIES

    assert list(CITIES) == ["abidjan", "paris", "new_york", "new_delhi"]
    assert CITIES["abidjan"].modelled and not CITIES["new_delhi"].modelled
    assert CITIES["paris"].location.timezone == "Europe/Paris"
    assert CITIES["new_york"].label.endswith("New York")
