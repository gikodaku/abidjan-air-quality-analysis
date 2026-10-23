"""Reproducible synthetic data: the tests never depend on the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aaqa.config import ModelConfig
from aaqa.data.clean import clean_hourly
from aaqa.features.build import aggregate_daily, build_features
from aaqa.models.train import train


def make_raw(days: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    time = pd.date_range("2023-01-01", periods=days * 24, freq="h")
    doy, hour = time.day_of_year.to_numpy(), time.hour.to_numpy()
    harmattan = np.exp(-((np.minimum(doy, 366 - doy)) / 30) ** 2)  # peaks in early January
    noise = pd.Series(rng.normal(0, 1, len(time))).rolling(48, min_periods=1).mean().to_numpy()
    dust = 5 + 60 * harmattan + rng.gamma(2, 2, len(time))
    pm25 = 8 + 0.25 * dust + 3 * np.sin(2 * np.pi * (hour - 7) / 24) + 4 * noise
    air = pd.DataFrame({
        "time": time,
        "pm2_5": np.clip(pm25, 1, None),
        "pm10": np.clip(pm25 * 1.8, 1, None),
        "nitrogen_dioxide": rng.uniform(5, 30, len(time)),
        "ozone": rng.uniform(20, 80, len(time)),
        "carbon_monoxide": rng.uniform(150, 400, len(time)),
        "sulphur_dioxide": rng.uniform(1, 10, len(time)),
        "dust": dust,
    })
    weather = pd.DataFrame({
        "time": time,
        "temperature_2m": 27 + 3 * np.sin(2 * np.pi * (hour - 9) / 24),
        "relative_humidity_2m": np.clip(80 - 25 * harmattan + rng.normal(0, 5, len(time)), 0, 100),
        "precipitation": rng.exponential(0.2, len(time)) * (1 - harmattan),
        "wind_speed_10m": rng.uniform(3, 20, len(time)),
        "wind_direction_10m": rng.uniform(0, 360, len(time)),
        "surface_pressure": 1010 + rng.normal(0, 1, len(time)),
    })
    return air, weather


@pytest.fixture(scope="session")
def raw():
    return make_raw()


@pytest.fixture(scope="session")
def hourly(raw):
    return clean_hourly(*raw)


@pytest.fixture(scope="session")
def daily(hourly):
    return aggregate_daily(hourly)


@pytest.fixture(scope="session")
def features(daily):
    return build_features(daily)


@pytest.fixture(scope="session")
def model_config():
    return ModelConfig(horizons=[1, 2], test_days=60, cv_splits=2)


@pytest.fixture(scope="session")
def trained(features, model_config):
    return train(features, model_config)
