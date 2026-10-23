"""Loading of trained models and production of forecasts."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from aaqa.aqi import describe, overall
from aaqa.features.build import TARGET, add_calendar_features, build_features
from aaqa.pollutants import get_pollutant


class AirQualityPredictor:
    """D+1…D+n forecast of every pollutant, with an 80% interval and an overall index."""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.models: dict = bundle["pollutants"]

    @classmethod
    def load(cls, path: Path | str) -> AirQualityPredictor:
        return cls(joblib.load(path))

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.bundle, path)

    @property
    def pollutants(self) -> list[str]:
        return list(self.models)

    @property
    def horizons(self) -> list[int]:
        return sorted(next(iter(self.models.values())))

    @property
    def metadata(self) -> dict:
        return {
            "version": self.bundle.get("version"),
            "trained_at": self.bundle.get("trained_at"),
            "trained_until": self.bundle.get("trained_until"),
            "pollutants": self.pollutants,
            "horizons": self.horizons,
        }

    def predict_row(self, row: pd.DataFrame, horizon: int, pollutant: str = "pm2_5") -> dict:
        """Forecast one pollutant from a row of day-D features."""
        if pollutant not in self.models:
            raise ValueError(f"pollutant {pollutant!r} not available ({', '.join(self.models)})")
        if horizon not in self.models[pollutant]:
            raise ValueError(f"horizon {horizon} not available ({self.horizons})")
        models = self.models[pollutant][horizon]
        target_index = row.index + pd.Timedelta(days=horizon)
        X = row.join(add_calendar_features(row, target_index, prefix="target_"))
        X = X.reindex(columns=models["features"])

        point = max(float(models["point"].predict(X)[0]), 0.0)
        low, high = float(models["q10"].predict(X)[0]), float(models["q90"].predict(X)[0])
        # The quantile models are independent, so enforce 0 ≤ low ≤ point ≤ high
        low, high = max(0.0, min(low, point)), max(high, point)
        return {**describe(pollutant, point), "interval_80": [round(low, 1), round(high, 1)],
                "method": models.get("method", "gradient_boosting")}

    def forecast(self, daily: pd.DataFrame) -> dict:
        """Forecasts of every pollutant and horizon from the last complete day."""
        features = build_features(daily)
        last = features[features[TARGET].notna()].iloc[[-1]]
        day = last.index[0]

        observed = [describe(p, float(last[get_pollutant(p).target].iloc[0]))
                    for p in self.pollutants
                    if pd.notna(last[get_pollutant(p).target].iloc[0])]
        forecasts = []
        for h in self.horizons:
            values = [self.predict_row(last, h, p) for p in self.pollutants]
            forecasts.append({
                "date": (day + pd.Timedelta(days=h)).date().isoformat(),
                "horizon_days": h,
                "overall": overall(values),
                "pollutants": values,
            })
        return {
            "issued_for": day.date().isoformat(),
            "observed": {"date": day.date().isoformat(), "overall": overall(observed),
                         "pollutants": observed},
            "forecasts": forecasts,
        }

    def predict_features(self, features: dict, horizon: int, day: str,
                         pollutant: str = "pm2_5") -> dict:
        """Forecast from explicitly supplied features (API /predict)."""
        row = pd.DataFrame([features], index=pd.DatetimeIndex([pd.Timestamp(day)], name="date"))
        row = row.apply(pd.to_numeric, errors="coerce").astype(float)
        result = self.predict_row(row, horizon, pollutant)
        return {"date": (pd.Timestamp(day) + pd.Timedelta(days=horizon)).date().isoformat(),
                "horizon_days": horizon, **result}
