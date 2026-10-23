"""Cleaning and merging of the hourly series."""

from __future__ import annotations

import pandas as pd

# Physically plausible ranges; values outside are treated as errors
PLAUSIBLE_RANGES = {
    "pm2_5": (0, 1000),
    "pm10": (0, 2000),
    "nitrogen_dioxide": (0, 1000),
    "ozone": (0, 1000),
    "carbon_monoxide": (0, 50000),
    "sulphur_dioxide": (0, 2000),
    "dust": (0, 5000),
    "temperature_2m": (5, 50),
    "relative_humidity_2m": (0, 100),
    "precipitation": (0, 300),
    "wind_speed_10m": (0, 200),
    "wind_direction_10m": (0, 360),
    "surface_pressure": (900, 1100),
}


def clean_hourly(air_quality: pd.DataFrame, weather: pd.DataFrame,
                 max_gap_hours: int = 6) -> pd.DataFrame:
    """Merge pollution and weather on a complete hourly grid.

    - drops duplicate timestamps,
    - replaces values outside plausible ranges with NaN,
    - interpolates short gaps (≤ max_gap_hours); longer gaps stay NaN,
    - removes hours where PM2.5 is still missing.
    """
    frames = []
    for frame in (air_quality, weather):
        frame = frame.copy()
        frame["time"] = pd.to_datetime(frame["time"])
        frames.append(frame.drop_duplicates("time").set_index("time"))
    merged = frames[0].join(frames[1], how="outer").sort_index()

    full_index = pd.date_range(merged.index.min(), merged.index.max(), freq="h", name="time")
    merged = merged.reindex(full_index)

    for column, (low, high) in PLAUSIBLE_RANGES.items():
        if column in merged:
            merged[column] = merged[column].where(merged[column].between(low, high))

    # Wind direction is circular, so it is not interpolated linearly
    linear = [c for c in merged.columns if c != "wind_direction_10m"]
    merged[linear] = merged[linear].interpolate(
        method="time", limit=max_gap_hours, limit_area="inside")
    if "wind_direction_10m" in merged:
        merged["wind_direction_10m"] = merged["wind_direction_10m"].ffill(limit=max_gap_hours)

    return merged.dropna(subset=["pm2_5"])
