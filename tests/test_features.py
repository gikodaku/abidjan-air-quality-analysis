import numpy as np
import pandas as pd
import pytest

from aaqa.features.build import TARGET, aggregate_daily, build_features, make_supervised


def test_daily_aggregation(hourly, daily):
    first_day = hourly.loc[hourly.index[:24]]
    assert daily[TARGET].iloc[0] == pytest.approx(first_day["pm2_5"].mean())
    assert daily.index.freqstr == "D"
    assert {"wind_u_mean", "wind_v_mean", "precipitation_sum"} <= set(daily.columns)


def test_incomplete_days_are_dropped(hourly):
    partial = hourly.drop(hourly.index[:10])  # le premier jour n'a plus que 14 h
    assert pd.Timestamp(hourly.index[0].date()) not in aggregate_daily(partial).dropna().index


def test_vector_wind_mean():
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    hourly = pd.DataFrame({"pm2_5": 10.0, "wind_speed_10m": 10.0,
                           "wind_direction_10m": [350, 10] * 12}, index=index)
    daily = aggregate_daily(hourly)
    # northerly wind: negative v component, u ≈ 0
    assert daily["wind_u_mean"].iloc[0] == pytest.approx(0, abs=1e-9)
    assert daily["wind_v_mean"].iloc[0] < -9


def test_pollutant_statistics(hourly, daily):
    day = hourly.index[0].normalize() + pd.Timedelta(days=1)
    hours = hourly.loc[day: day + pd.Timedelta(hours=23)]
    assert daily.loc[day, "nitrogen_dioxide_max"] == pytest.approx(hours["nitrogen_dioxide"].max())
    window = hourly["ozone"].rolling(8, min_periods=6).mean()
    assert daily.loc[day, "ozone_max8h"] == pytest.approx(window.loc[hours.index].max())
    assert {"carbon_monoxide_max8h", "sulphur_dioxide_max", "pm10_mean"} <= set(daily.columns)


def test_other_pollutant_history(daily, features):
    assert features["pm10_lag1"].iloc[10] == daily["pm10_mean"].iloc[9]
    assert features["no2max_lag2"].iloc[10] == daily["nitrogen_dioxide_max"].iloc[8]
    assert "no2mean_roll30" in features and "nitrogen_dioxide_lag1" not in features
    assert features["ozone_roll3"].iloc[10] == pytest.approx(
        daily["ozone_max8h"].iloc[8:11].mean())


def test_lags_use_only_the_past(daily, features):
    pm = daily[TARGET]
    assert features["pm2_5_lag1"].iloc[10] == pm.iloc[9]
    assert features["pm2_5_lag7"].iloc[10] == pm.iloc[3]
    assert features["pm2_5_roll3"].iloc[10] == pytest.approx(pm.iloc[8:11].mean())


def test_no_future_leakage(daily):
    """Changing the future must not change any feature of the past."""
    base = build_features(daily)
    altered = daily.copy()
    altered.iloc[200:, :] = altered.iloc[200:, :] * 10
    changed = build_features(altered)
    pd.testing.assert_frame_equal(base.iloc[:200], changed.iloc[:200])


def test_calendar_flags(features):
    january = features[features.index.month == 1]
    july = features[features.index.month == 7]
    assert january["is_harmattan"].eq(1).all() and july["is_harmattan"].eq(0).all()
    assert january["is_dry_season"].eq(1).all() and july["is_dry_season"].eq(0).all()
    assert features["doy_sin"].between(-1, 1).all()


@pytest.mark.parametrize("horizon, target", [(1, TARGET), (3, TARGET), (2, "ozone_max8h")])
def test_make_supervised_aligns_target(features, horizon, target):
    X, y = make_supervised(features, horizon, target)
    day = X.index[50]
    assert y.loc[day] == features.loc[day + pd.Timedelta(days=horizon), target]
    assert (X["target_dow"] == (X.index + pd.Timedelta(days=horizon)).day_of_week).all()
    assert not y.isna().any()
    assert np.isfinite(X[TARGET]).all()
