from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from aaqa import monitoring as mon
from aaqa.models.predictor import AirQualityPredictor


@pytest.fixture
def forecast(trained, daily):
    predictor = AirQualityPredictor(trained[0])
    return predictor.forecast(daily.iloc[:-5]), predictor.metadata


def test_log_forecast_appends_and_replaces_same_day(forecast, tmp_path):
    result, meta = forecast
    path = tmp_path / "log.csv"
    log = mon.log_forecast(result, meta, path, datetime(2024, 1, 1, tzinfo=UTC))
    assert len(log) == len(result["forecasts"]) * len(result["forecasts"][0]["pollutants"])
    assert set(log.columns) == set(mon.LOG_COLUMNS)
    again = mon.log_forecast(result, meta, path, datetime(2024, 1, 1, 6, tzinfo=UTC))
    assert len(again) == len(log)  # same issue day: replaced, not duplicated
    assert again["issued_at"].str.contains("06:00").all()


def test_score_uses_observations_once_available(forecast, daily, tmp_path):
    result, meta = forecast
    log = mon.log_forecast(result, meta, tmp_path / "log.csv")
    scored = mon.score(log, daily)
    row = scored[(scored["pollutant"] == "pm2_5") & (scored["horizon"] == 1)].iloc[0]
    assert row["observed"] == pytest.approx(daily.loc[row["target_date"], "pm2_5_mean"])
    assert row["persistence"] == pytest.approx(daily.loc[row["issued_for"], "pm2_5_mean"])
    assert row["abs_error"] == pytest.approx(abs(row["predicted"] - row["observed"]))
    future = mon.score(log, daily.iloc[:-5])  # target days not observed yet
    assert future["observed"].isna().all() and future["in_interval"].isna().all()


def test_empty_log_scores_to_empty_frame(daily, tmp_path):
    scored = mon.score(mon.load_log(tmp_path / "missing.csv"), daily)
    assert scored.empty and "abs_error" in scored


def test_backtest_and_health(trained, tmp_path):
    bundle, evaluations = trained
    for pollutant, by_h in evaluations.items():
        for h, ev in by_h.items():
            ev["predictions"].to_csv(tmp_path / f"test_predictions_{pollutant}_h{h}.csv")
    metrics = {p: {str(h): m for h, m in by_h.items()} for p, by_h in bundle["metrics"].items()}
    backtest = mon.load_backtest(tmp_path, metrics)
    assert set(backtest["horizon"]) == {1, 2} and backtest["pollutant"].nunique() == 6
    assert (backtest["target_date"] - backtest["issued_for"]).dt.days.isin([1, 2]).all()
    health = mon.health_table(backtest, metrics, days=30)
    assert len(health) == 12 and health["status"].isin(mon.STATUS).all()
    assert (health["n"] > 0).all()
    rolling = mon.rolling_errors(backtest, "pm2_5", 1)
    assert {"model", "persistence", "coverage"} <= set(rolling.columns)


@pytest.mark.parametrize("summary, reference, expected", [
    ({"n": 3, "mae": 1, "persistence_mae": 1, "skill": 0.1, "coverage": 0.8, "bias": 0},
     1.0, "collecting"),
    ({"n": 30, "mae": 1, "persistence_mae": 1.2, "skill": 0.17, "coverage": 0.8, "bias": 0},
     1.0, "good"),
    ({"n": 30, "mae": 1.1, "persistence_mae": 1.05, "skill": -0.05, "coverage": 0.8, "bias": 0},
     1.0, "warning"),
    ({"n": 30, "mae": 1.4, "persistence_mae": 1.5, "skill": 0.07, "coverage": 0.8, "bias": 0},
     1.0, "warning"),
    ({"n": 30, "mae": 2.5, "persistence_mae": 3, "skill": 0.17, "coverage": 0.8, "bias": 0},
     1.0, "critical"),
    ({"n": 30, "mae": 1, "persistence_mae": 0.8, "skill": -0.25, "coverage": 0.8, "bias": 0},
     1.0, "critical"),
    ({"n": 30, "mae": 1, "persistence_mae": 1.2, "skill": 0.17, "coverage": 0.4, "bias": 0},
     1.0, "critical"),
])
def test_status_rules(summary, reference, expected):
    status, reasons = mon.status_of(summary, reference)
    assert status == expected and reasons


def test_freshness():
    daily = pd.DataFrame({"pm2_5_mean": [10.0, np.nan]},
                         index=pd.to_datetime(["2026-10-21", "2026-10-22"]))
    fresh = mon.freshness(daily, datetime(2026, 10, 22, 12, tzinfo=UTC))
    assert fresh["last_day"] == "2026-10-21" and fresh["hours"] == pytest.approx(12)
    assert fresh["status"] == "good"
    assert mon.freshness(daily, datetime(2026, 10, 25, tzinfo=UTC))["status"] == "critical"


def test_drift_is_seasonal():
    index = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    seasonal = 20 + 10 * np.cos(2 * np.pi * index.day_of_year / 365.25)
    rng = np.random.default_rng(0)
    daily = pd.DataFrame({"pm2_5_mean": seasonal + rng.normal(0, 1, len(index))}, index=index)
    normal = mon.drift_table(daily)
    assert normal.loc[0, "status"] == "good"  # a seasonal peak is not drift
    shifted = daily.copy()
    shifted.iloc[-30:] += 15
    assert mon.drift_table(shifted).loc[0, "status"] == "critical"


def test_scores_match_definitions():
    observed, predicted = [10.0, 20.0, 30.0, 40.0], [12.0, 18.0, 33.0, 37.0]
    result = mon.scores(observed, predicted, "pm2_5")
    assert result["mae"] == pytest.approx(2.5)
    assert result["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9 + 9) / 4))
    assert result["r2"] == pytest.approx(1 - 26 / 500)
    assert result["bias"] == pytest.approx(0.0)
    assert 0 <= result["aqi_category_accuracy"] <= 1


def test_compare_models_on_backtest(trained, tmp_path):
    _, evaluations = trained
    for pollutant, by_h in evaluations.items():
        for h, ev in by_h.items():
            ev["predictions"].to_csv(tmp_path / f"test_predictions_{pollutant}_h{h}.csv")
    long = mon.load_backtest_models(tmp_path)
    assert set(long["model"]) == set(mon.MODEL_LABELS)
    table = mon.compare_models(long, days=60)
    assert len(table) == 6 * 2 * 4  # pollutants × horizons × models
    assert {"mae", "rmse", "r2", "bias", "aqi_category_accuracy"} <= set(table.columns)
    winners = mon.best_model(table, "mae")
    pm = table[(table["pollutant"] == "pm2_5") & (table["horizon"] == 1)]
    assert winners[("pm2_5", 1)] == pm.loc[pm["mae"].idxmin(), "model"]
    assert mon.best_model(table, "r2")[("pm2_5", 1)] == pm.loc[pm["r2"].idxmax(), "model"]


def test_live_models_layout(forecast, daily, tmp_path):
    result, meta = forecast
    scored = mon.score(mon.log_forecast(result, meta, tmp_path / "log.csv"), daily)
    long = mon.live_models(scored)
    assert set(long["model"]) <= set(mon.MODEL_LABELS) and "persistence" in set(long["model"])


class _Response:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


def test_check_service_statuses(monkeypatch):
    monkeypatch.setattr(mon.requests, "get",
                        lambda *a, **k: _Response(200, {"version": "1.2.0", "status": "ok"}))
    ok = mon.check_service("api", "http://x/health")
    assert ok["status"] == "good" and "v1.2.0" in ok["detail"]
    monkeypatch.setattr(mon.requests, "get", lambda *a, **k: _Response(503, {}))
    assert mon.check_service("api", "http://x/health")["status"] == "critical"

    def refuse(*a, **k):
        raise ConnectionError("refused")

    monkeypatch.setattr(mon.requests, "get", refuse)
    down = mon.check_service("api", "http://x/health")
    assert down["status"] == "critical" and down["detail"] == "ConnectionError"


def test_check_services_hides_the_key(monkeypatch):
    from aaqa.config import Location

    monkeypatch.setattr(mon, "check_service", lambda name, url, params=None, **k: {
        "service": name, "status": "critical", "ms": 1,
        "detail": f"error with {params.get('API_KEY') if params else ''}"})
    results = mon.check_services(Location(), airnow_key="secret-123")
    assert len(results) == 5 and not any("secret-123" in r["detail"] for r in results)


def test_pipeline_health(tmp_path):
    fresh, old = tmp_path / "fresh.csv", tmp_path / "old.csv"
    fresh.write_text("x")
    old.write_text("x")
    import os
    import time as _time
    os.utime(old, (_time.time() - 80 * 3600, _time.time() - 80 * 3600))
    rows = {r["step"]: r for r in mon.pipeline_health(
        {"fresh": (fresh, 36), "old": (old, 36), "missing": (tmp_path / "none", 36)})}
    assert rows["fresh"]["status"] == "good"
    assert rows["old"]["status"] == "critical" and "days ago" in rows["old"]["detail"]
    assert rows["missing"]["status"] == "critical" and rows["missing"]["detail"] == "missing"
