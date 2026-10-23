"""Configuration loading and project paths."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(os.environ.get("AAQA_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

AIR_QUALITY_RAW = RAW_DIR / "air_quality.csv"
WEATHER_RAW = RAW_DIR / "weather.csv"
GRID_POINTS = RAW_DIR / "grid_points.json"
EMBASSY_RAW = RAW_DIR / "embassy_pm25.csv"
HOURLY_PROCESSED = PROCESSED_DIR / "hourly.csv"
DAILY_PROCESSED = PROCESSED_DIR / "daily.csv"
MODEL_PATH = MODELS_DIR / "forecaster.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
PROGRESS_PATH = DATA_DIR / "user" / "progress.json"
MONITORING_LOG = DATA_DIR / "monitoring" / "forecast_log.csv"
CALIBRATION_PATH = REPORTS_DIR / "calibration.json"


@dataclass
class Location:
    name: str = "Abidjan"
    latitude: float = 5.36
    longitude: float = -4.01
    timezone: str = "Africa/Abidjan"


@dataclass
class DataConfig:
    start_date: str = "2022-08-01"
    end_date: str | None = None
    max_gap_hours: int = 6


@dataclass
class ModelConfig:
    pollutants: list[str] = field(default_factory=lambda: [
        "pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide"])
    horizons: list[int] = field(default_factory=lambda: [1, 2, 3])
    test_days: int = 365
    cv_splits: int = 5
    random_state: int = 42


@dataclass
class Config:
    location: Location = field(default_factory=Location)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Read config.yaml; missing values keep their defaults."""
    if not path.exists():
        return Config()
    raw = yaml.safe_load(path.read_text()) or {}
    return Config(
        location=Location(**raw.get("location", {})),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
    )
