from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from aaqa.config import Location
from aaqa.data import fetch
from aaqa.data.clean import clean_hourly


def test_clean_produces_complete_hourly_grid(raw, hourly):
    air, _ = raw
    assert hourly.index.is_monotonic_increasing
    assert hourly.index.freq is None or hourly.index.freqstr == "h"
    assert len(hourly) == len(air)
    assert hourly["pm2_5"].notna().all()


def test_clean_removes_outliers_and_interpolates_short_gaps(raw):
    air, weather = (frame.copy() for frame in raw)
    air.loc[10, "pm2_5"] = -5           # aberrant
    air.loc[20:22, "pm2_5"] = np.nan    # trou de 3 h
    air.loc[100:120, "pm2_5"] = np.nan  # trou de 21 h, trop long
    cleaned = clean_hourly(air, weather, max_gap_hours=6)
    times = air["time"]
    assert cleaned.loc[times[10], "pm2_5"] >= 0
    assert cleaned.loc[times[21], "pm2_5"] == pytest.approx(
        (air.loc[19, "pm2_5"] + air.loc[23, "pm2_5"]) / 2, rel=0.5)
    assert times[110] not in cleaned.index


def test_clean_drops_duplicate_timestamps(raw):
    air, weather = raw
    doubled = pd.concat([air, air.iloc[:5]])
    assert len(clean_hourly(doubled, weather)) == len(air)


def test_date_chunks_cover_range_without_overlap():
    start, end = date(2022, 8, 1), date(2024, 3, 15)
    chunks = list(fetch._date_chunks(start, end, days=200))
    assert chunks[0][0] == start and chunks[-1][1] == end
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
        assert next_start == prev_end + timedelta(days=1)


def test_resolve_dates():
    assert fetch.resolve_dates("2023-01-01", "2023-02-01") == (date(2023, 1, 1), date(2023, 2, 1))
    assert fetch.resolve_dates("2023-01-01", None)[1] == date.today() - timedelta(days=1)
    with pytest.raises(ValueError):
        fetch.resolve_dates("2023-02-01", "2023-01-01")


def test_fetch_air_quality_parses_payload(monkeypatch):
    calls = []

    def fake_get(url, params, retries=3, timeout=60):
        calls.append(params)
        times = pd.date_range(params["start_date"], params["end_date"] + " 23:00", freq="h")
        hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
        hourly |= {var: [1.0] * len(times) for var in fetch.AIR_QUALITY_VARS}
        return {"hourly": hourly}

    monkeypatch.setattr(fetch, "_get", fake_get)
    frame = fetch.fetch_air_quality(Location(), date(2023, 1, 1), date(2023, 1, 3))
    assert len(frame) == 72
    assert list(frame.columns) == ["time", *fetch.AIR_QUALITY_VARS]
    assert calls[0]["latitude"] == Location().latitude
