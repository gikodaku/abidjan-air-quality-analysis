"""Forecast REST API: `uvicorn aaqa.api.main:app`, then http://localhost:8000/docs."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from aaqa import __version__
from aaqa import config as paths
from aaqa.aqi import describe
from aaqa.config import load_config
from aaqa.data.weather_forecast import fetch_weather_forecast
from aaqa.models.predictor import AirQualityPredictor
from aaqa.pollutants import POLLUTANTS

log = logging.getLogger(__name__)


class PollutantValue(BaseModel):
    pollutant: str
    name: str
    statistic: str
    value: float
    unit: str
    aqi: int
    category: str
    label: str
    color: str
    exceeds_who: bool | None = Field(
        description="WHO guideline exceeded; null when the guideline uses another statistic")


class PollutantForecast(PollutantValue):
    interval_80: list[float]
    method: str = Field(description="gradient_boosting, or persistence when the model does not "
                                    "beat that baseline in cross-validation")


class Overall(BaseModel):
    aqi: int
    category: str
    label: str
    color: str
    advice: str
    dominant_pollutant: str
    dominant_name: str


class Observed(BaseModel):
    date: str
    overall: Overall
    pollutants: list[PollutantValue]


class DayForecast(BaseModel):
    date: str
    horizon_days: int
    overall: Overall
    pollutants: list[PollutantForecast]


class Weather(BaseModel):
    date: str
    description: str
    icon: str
    sky: str
    temperature_min: float | None
    temperature_max: float | None
    feels_like_max: float | None
    precipitation_mm: float | None
    precipitation_probability: float | None
    humidity: float | None
    wind_max_kmh: float | None
    wind_from: str | None
    uv_index: float | None
    uv_level: str | None


class ForecastResponse(BaseModel):
    issued_for: str
    observed: Observed
    forecasts: list[DayForecast]
    weather: list[Weather] = Field(description="Official weather forecast (Open-Meteo)")


class PredictRequest(BaseModel):
    day: date = Field(description="Day D whose features are supplied")
    horizon: int = Field(1, ge=1, description="Number of days after D")
    pollutant: str = Field("pm2_5", description="Pollutant key (see /pollutants)")
    features: dict[str, float | None] = Field(
        description="Day-D features; must include at least the pollutant's statistic "
                    "(e.g. pm2_5_mean). Absent features are treated as missing.")


class PredictResponse(PollutantForecast):
    date: str
    horizon_days: int


class HistoryPoint(BaseModel):
    date: str
    value: float
    aqi: int
    category: str


def create_app(model_path: Path = paths.MODEL_PATH,
               daily_path: Path = paths.DAILY_PROCESSED,
               weather_provider=None) -> FastAPI:
    app = FastAPI(
        title="AAQA — Abidjan air quality",
        version=__version__,
        description="D+1, D+2 and D+3 forecasts of six pollutants (PM2.5, PM10, O3, NO2, SO2, "
                    "CO), overall US AQI, health advice and the weather for the coming days.",
    )
    state: dict = {}
    if weather_provider is None:
        def weather_provider():
            return fetch_weather_forecast(load_config().location)

    def predictor() -> AirQualityPredictor:
        if not Path(model_path).exists():
            raise HTTPException(503, "No model: run `aaqa train`.")
        # Reload when the model file changes, so a retraining is picked up without a restart
        modified = Path(model_path).stat().st_mtime
        if state.get("modified") != modified:
            state["predictor"] = AirQualityPredictor.load(model_path)
            state["modified"] = modified
        return state["predictor"]

    def daily() -> pd.DataFrame:
        if not Path(daily_path).exists():
            raise HTTPException(503, "No data: run `aaqa fetch process`.")
        return pd.read_csv(daily_path, index_col="date", parse_dates=["date"]).asfreq("D")

    def weather() -> list[dict]:
        try:
            return weather_provider()
        except Exception as exc:  # the pollution forecast is still useful without the weather
            log.warning("Weather forecast unavailable: %s", exc)
            return []

    def check_pollutant(key: str) -> None:
        if key not in POLLUTANTS:
            raise HTTPException(422, f"unknown pollutant {key!r} ({', '.join(POLLUTANTS)})")

    @app.get("/health")
    def health():
        model_ready = Path(model_path).exists()
        return {
            "status": "ok" if model_ready else "degraded",
            "version": __version__,
            "model": predictor().metadata if model_ready else None,
        }

    @app.get("/forecast", response_model=ForecastResponse)
    def forecast():
        """Forecast of every pollutant from the last observed day, with the weather."""
        return {**predictor().forecast(daily()), "weather": weather()}

    @app.get("/weather", response_model=list[Weather])
    def weather_forecast():
        """Weather for today and the next three days."""
        days = weather()
        if not days:
            raise HTTPException(503, "Weather forecast unavailable")
        return days

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        """Forecast one pollutant from features supplied by the caller."""
        check_pollutant(request.pollutant)
        target = POLLUTANTS[request.pollutant].target
        if request.features.get(target) is None:
            raise HTTPException(422, f"features.{target} is required")
        try:
            return predictor().predict_features(
                request.features, request.horizon, request.day.isoformat(), request.pollutant)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/history", response_model=list[HistoryPoint])
    def history(days: int = Query(30, ge=1, le=3650), pollutant: str = "pm2_5"):
        """Observed daily values for the last `days` days."""
        check_pollutant(pollutant)
        series = daily()[POLLUTANTS[pollutant].target].dropna().tail(days)
        result = []
        for day, value in series.items():
            info = describe(pollutant, value)
            result.append({"date": day.date().isoformat(), "value": info["value"],
                           "aqi": info["aqi"], "category": info["category"]})
        return result

    @app.get("/aqi", response_model=PollutantValue)
    def aqi(value: float = Query(..., ge=0, description="Concentration in µg/m³"),
            pollutant: str = "pm2_5"):
        """Convert a concentration (on the pollutant's statistic) to the AQI."""
        check_pollutant(pollutant)
        return describe(pollutant, value)

    @app.get("/pollutants")
    def pollutants():
        """Tracked pollutants, forecast statistic and WHO guideline."""
        return [{"key": p.key, "name": p.name, "statistic": p.statistic, "column": p.target,
                 "who_limit": p.who_limit, "who_label": p.who_label, "sources": p.sources}
                for p in POLLUTANTS.values()]

    return app


app = create_app()
