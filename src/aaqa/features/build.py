"""Daily aggregation and explanatory features for the D+h forecast.

Every feature on a row dated D uses only information available at the end of day D: no
information leaks from the future into the past.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aaqa.pollutants import POLLUTANTS

TARGET = POLLUTANTS["pm2_5"].target

# Abidjan's climate seasons (transitional equatorial climate, four seasons)
SEASONS = {
    12: "long_dry_season", 1: "long_dry_season", 2: "long_dry_season", 3: "long_dry_season",
    4: "long_rainy_season", 5: "long_rainy_season", 6: "long_rainy_season",
    7: "long_rainy_season",
    8: "short_dry_season", 9: "short_dry_season",
    10: "short_rainy_season", 11: "short_rainy_season",
}
# The harmattan (a dry wind carrying Saharan dust) blows mostly from December to February
HARMATTAN_MONTHS = {12, 1, 2}

_DAILY_AGG = {
    "pm2_5": ["mean", "max"],
    "pm10": ["mean"],
    "nitrogen_dioxide": ["mean", "max"],
    "ozone": ["mean", "max"],
    "carbon_monoxide": ["mean"],
    "sulphur_dioxide": ["mean", "max"],
    "dust": ["mean", "max"],
    "temperature_2m": ["mean", "min", "max"],
    "relative_humidity_2m": ["mean"],
    "precipitation": ["sum"],
    "wind_speed_10m": ["mean"],
    "surface_pressure": ["mean"],
}


def aggregate_daily(hourly: pd.DataFrame, min_hours: int = 18) -> pd.DataFrame:
    """Aggregate hourly data into days; a day with fewer than min_hours hours of PM2.5 is
    dropped so that its mean stays representative."""
    hourly = hourly.copy()
    if "wind_direction_10m" in hourly:
        # Vector mean of the wind (a plain mean of 350° and 10° would give 180°)
        radians = np.deg2rad(hourly["wind_direction_10m"])
        speed = hourly.get("wind_speed_10m", 1.0)
        hourly["wind_u"] = -speed * np.sin(radians)
        hourly["wind_v"] = -speed * np.cos(radians)

    # Rolling 8-hour means (the index statistic for O3 and CO); the daily maximum is taken over
    # the windows ending on that day
    for gas in ("ozone", "carbon_monoxide"):
        if gas in hourly:
            hourly[f"{gas}_8h"] = hourly[gas].rolling(8, min_periods=6).mean()

    agg = {k: v for k, v in _DAILY_AGG.items() if k in hourly}
    for component in ("wind_u", "wind_v"):
        if component in hourly:
            agg[component] = ["mean"]
    for gas in ("ozone", "carbon_monoxide"):
        if f"{gas}_8h" in hourly:
            agg[f"{gas}_8h"] = ["max"]

    grouped = hourly.groupby(hourly.index.normalize())
    daily = grouped.agg(agg)
    daily.columns = [f"{col}_{stat}".replace("_8h_max", "_max8h") for col, stat in daily.columns]
    daily = daily[grouped["pm2_5"].count() >= min_hours]
    daily.index.name = "date"
    return daily.asfreq("D")


def add_calendar_features(frame: pd.DataFrame, index: pd.DatetimeIndex | None = None,
                          prefix: str = "") -> pd.DataFrame:
    idx = frame.index if index is None else index
    out = pd.DataFrame(index=frame.index)
    doy = idx.day_of_year
    out[f"{prefix}doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out[f"{prefix}doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out[f"{prefix}dow"] = idx.day_of_week
    out[f"{prefix}is_weekend"] = (idx.day_of_week >= 5).astype(int)
    out[f"{prefix}is_harmattan"] = idx.month.isin(HARMATTAN_MONTHS).astype(int)
    out[f"{prefix}is_dry_season"] = pd.Index(idx.month).map(SEASONS).str.contains(
        "dry").astype(int)
    return out


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Build the features of day D (past and present only).

    Full history for PM2.5 (the main pollutant) and NO2; a reduced history (previous day, 3- and
    7-day means) for the other pollutants, to keep the number of features reasonable against
    ~1,200 training days.
    """
    features = daily.copy()
    pm = daily[TARGET]

    for lag in (1, 2, 3, 7):
        features[f"pm2_5_lag{lag}"] = pm.shift(lag)
    for window in (3, 7, 14, 30):
        features[f"pm2_5_roll{window}"] = pm.rolling(window, min_periods=max(2, window // 2)).mean()
    features["pm2_5_std7"] = pm.rolling(7, min_periods=4).std()
    features["pm2_5_diff1"] = pm.diff()

    # NO2 also gets a full history: its daily peak is noisy, and the longer history raises its
    # D+1 gain over "tomorrow = today" from 1.8% to 3.1% in cross-validation
    for column, short in (("nitrogen_dioxide_max", "no2max"), ("nitrogen_dioxide_mean", "no2mean")):
        if column not in daily:
            continue
        series = daily[column]
        for lag in (1, 2, 3, 7):
            features[f"{short}_lag{lag}"] = series.shift(lag)
        for window in (3, 7, 14, 30):
            features[f"{short}_roll{window}"] = series.rolling(
                window, min_periods=max(2, window // 2)).mean()
        features[f"{short}_std7"] = series.rolling(7, min_periods=4).std()
        features[f"{short}_diff1"] = series.diff()

    for pollutant in POLLUTANTS.values():
        if pollutant.key in ("pm2_5", "nitrogen_dioxide") or pollutant.target not in daily:
            continue
        series = daily[pollutant.target]
        features[f"{pollutant.key}_lag1"] = series.shift(1)
        features[f"{pollutant.key}_roll3"] = series.rolling(3, min_periods=2).mean()
        features[f"{pollutant.key}_roll7"] = series.rolling(7, min_periods=4).mean()

    if "dust_mean" in daily:
        features["dust_lag1"] = daily["dust_mean"].shift(1)
        features["dust_roll3"] = daily["dust_mean"].rolling(3, min_periods=2).mean()
    if "precipitation_sum" in daily:
        features["rain_roll3"] = daily["precipitation_sum"].rolling(3, min_periods=1).sum()
        features["dry_days"] = _consecutive_dry_days(daily["precipitation_sum"])
    if "temperature_2m_max" in daily and "temperature_2m_min" in daily:
        features["temp_range"] = daily["temperature_2m_max"] - daily["temperature_2m_min"]

    return features.join(add_calendar_features(daily))


def _consecutive_dry_days(rain: pd.Series, threshold: float = 1.0) -> pd.Series:
    dry = (rain.fillna(0) < threshold).astype(int)
    groups = (dry == 0).cumsum()
    return dry.groupby(groups).cumsum()


def make_supervised(features: pd.DataFrame, horizon: int,
                    target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    """Pair the features of day D with the value of `target` on day D+horizon.

    The calendar of the target day (day of week in particular) is known in advance and added.
    """
    y = features[target].shift(-horizon).rename(f"target_h{horizon}")
    target_index = features.index + pd.Timedelta(days=horizon)
    X = features.join(add_calendar_features(features, target_index, prefix="target_"))
    mask = y.notna() & X[target].notna()
    return X[mask], y[mask]
