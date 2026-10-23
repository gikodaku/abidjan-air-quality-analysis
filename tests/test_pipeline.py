"""Full pipeline (CLI → files) with simulated downloads and temporary folders."""

import contextlib
import io
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from aaqa import cli, pipeline
from aaqa import config as paths
from aaqa import monitoring as mon
from aaqa.config import Config, DataConfig, ModelConfig
from aaqa.data import air_now, weather_forecast
from tests.test_air_now import LIVE
from tests.test_api import WEATHER
from tests.test_weather import NOW


def _paths(root: Path) -> dict:
    return {
        "RAW_DIR": root / "raw",
        "PROCESSED_DIR": root / "processed",
        "REPORTS_DIR": root / "reports",
        "FIGURES_DIR": root / "reports" / "figures",
        "AIR_QUALITY_RAW": root / "raw" / "air.csv",
        "WEATHER_RAW": root / "raw" / "weather.csv",
        "HOURLY_PROCESSED": root / "processed" / "hourly.csv",
        "DAILY_PROCESSED": root / "processed" / "daily.csv",
        "MODEL_PATH": root / "models" / "forecaster.joblib",
        "METRICS_PATH": root / "reports" / "metrics.json",
        "PROGRESS_PATH": root / "user" / "progress.json",
        "MONITORING_LOG": root / "monitoring" / "forecast_log.csv",
        "GRID_POINTS": root / "raw" / "grid_points.json",
        "EMBASSY_RAW": root / "raw" / "embassy.csv",
        "CALIBRATION_PATH": root / "reports" / "calibration.json",
    }


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory, raw):
    """Run the full pipeline once for the module (training is the slow part)."""
    air, weather = raw
    root = tmp_path_factory.mktemp("pipeline")
    with pytest.MonkeyPatch.context() as mp:
        for name, value in _paths(root).items():
            mp.setattr(paths, name, value)
        mp.setattr(pipeline, "fetch_air_quality", lambda *a: air)
        mp.setattr(pipeline, "fetch_weather", lambda *a: weather)
        mp.setattr(pipeline, "fetch_grid_points", lambda loc: {
            "requested": {"latitude": 5.36, "longitude": -4.01},
            "air_quality": {"latitude": 5.4, "longitude": -4.0, "model": "CAMS",
                            "resolution_deg": 0.4},
            "weather": {"latitude": 5.38, "longitude": -4.02, "model": "ERA5"}})
        mp.setattr(pipeline, "fetch_weather_forecast", lambda *a: WEATHER)
        # Simulated US Embassy monitor: twice the model value, as measured in Abidjan
        embassy = pd.DataFrame({"time": air["time"], "pm2_5": air["pm2_5"] * 2})
        mp.setattr(pipeline, "api_key", lambda root: "test-key")
        mp.setattr(mon, "check_services", lambda *a, **k: [
            {"service": "AAQA REST API", "status": "good", "ms": 12, "detail": "HTTP 200"},
            {"service": "Open-Meteo air quality (CAMS)", "status": "critical", "ms": 8000,
             "detail": "ReadTimeout"}])

        def fake_update(path, key):
            embassy.to_csv(path, index=False)
            return embassy

        mp.setattr(pipeline, "update_embassy", fake_update)
        mp.setattr(weather_forecast, "fetch_weather_forecast", lambda *a: WEATHER)
        mp.setattr(weather_forecast, "fetch_now", lambda *a: NOW)
        mp.setattr(air_now, "fetch_air_now", lambda *a: LIVE)
        city_forecast = [{"date": f"2026-10-2{4 + i}", "horizon_days": i + 1,
                          "overall": LIVE["overall"], "pollutants": LIVE["pollutants"]}
                         for i in range(3)]
        mp.setattr(air_now, "fetch_city_air", lambda *a, **k: (LIVE, city_forecast))
        cfg = Config(data=DataConfig(start_date="2023-01-01", end_date="2024-02-04"),
                     model=ModelConfig(horizons=[1, 2], test_days=60, cv_splits=2))
        mp.setattr(pipeline, "load_config", lambda: cfg)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            assert cli.main(["all"]) == 0
        yield root, output.getvalue()


def test_cli_runs_full_pipeline(sandbox):
    sandbox, output = sandbox
    forecast = json.loads(output)
    assert [f["horizon_days"] for f in forecast["forecasts"]] == [1, 2]
    assert forecast["weather"] == WEATHER

    reports = sandbox / "reports"
    for name in ("pollutants_timeseries", "pollutants_monthly", "pollutants_diurnal",
                 "pm25_weekday", "correlations", "forecast_test", "feature_importance"):
        assert (reports / "figures" / f"{name}.png").stat().st_size > 0
    metrics = json.loads((reports / "metrics.json").read_text())
    assert set(metrics["pm2_5"]) == {"1", "2"} and "ozone" in metrics
    assert (reports / "test_predictions_pm10_h1.csv").exists()
    assert (reports / "test_predictions_ozone_h2.csv").exists()
    assert json.loads((sandbox / "raw" / "grid_points.json").read_text())["air_quality"][
        "latitude"] == 5.4
    assert (sandbox / "monitoring" / "forecast_log.csv").exists()  # forecast logged
    calibration = json.loads((reports / "calibration.json").read_text())
    best = calibration["cross_validation"][calibration["choice"]]
    assert best["mae"] < calibration["raw"]["mae"] / 3  # the 2× bias is corrected
    assert (reports / "figures" / "calibration.png").stat().st_size > 0
    stats = json.loads((reports / "statistics.json").read_text())
    pm = stats["pollutants"]["pm2_5"]
    assert pm["harmattan_mean"] > pm["rainy_season_mean"]
    assert 0 <= stats["pollutants"]["ozone"]["share_days_above_who"] <= 1


def test_missing_inputs_give_clear_error(tmp_path, monkeypatch):
    for name, value in _paths(tmp_path).items():
        monkeypatch.setattr(paths, name, value)
    with pytest.raises(FileNotFoundError, match="previous step"):
        pipeline.load_daily()


def test_dashboard_renders(sandbox):
    sandbox, _ = sandbox
    script = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    app = AppTest.from_file(str(script), default_timeout=60).run()
    assert not app.exception
    page = " ".join(block.value for block in app.markdown)
    for text in ("splash", "KOSCON", "Powered by", "Abidjan", "Live AQI", "Air quality is",
                 "Sensitive", "AQI average", "KOS SAYS", "Weather forecast", "Now", "27°",
                 "Air quality forecast", "Pollutants", "calibrated", "Today in detail",
                 "Kos the leopard, unhealthy for sensitive groups",
                 "Abidjan skyline", "Where the data comes from",
                 "calibrated on the US Embassy monitor", "US Embassy monitor has not reported",
                 "eco journey", "eco-challenge",
                 "Daily quiz", "Badges"):
        assert text in page, text
    assert [tab.label for tab in app.tabs] == ["🌤️ Today", "💬 Ask Kos", "🌱 Eco journey",
                                               "📈 Insights"]
    assert (sandbox / "user" / "progress.json").exists()  # daily check-in saved

    app.toggle[0].set_value(False).run()  # raw model view
    assert not app.exception
    assert "Copernicus CAMS model · city-wide" in " ".join(b.value for b in app.markdown)
    app.toggle[0].set_value(True).run()

    app.segmented_control[1].set_value("Daily").run()
    assert not app.exception
    assert "Tomorrow" in " ".join(block.value for block in app.markdown)

    app.segmented_control[0].set_value("new_delhi").run()  # another city
    assert not app.exception
    page = " ".join(block.value for block in app.markdown)
    assert "New Delhi" in page and "CAMS model and forecast" in page
    assert "New Delhi skyline" in page and not app.toggle  # no calibration outside Abidjan
    app.segmented_control[0].set_value("abidjan").run()

    app.chat_input[0].set_value("What is NO2?").run()
    assert not app.exception
    assert "Nitrogen dioxide" in app.chat_message[-1].markdown[0].value

    app.selectbox[0].select("ozone").run()
    assert not app.exception
    assert len(app.dataframe) == 1  # accuracy table


def test_monitoring_dashboard_renders(sandbox):
    script = Path(__file__).resolve().parents[1] / "dashboard" / "monitoring.py"
    app = AppTest.from_file(str(script), default_timeout=60).run()
    assert not app.exception
    page = " ".join(block.value for block in app.markdown)
    for text in ("Model monitoring", "Models in service", "Health of the", "Model comparison",
                 "Leaderboard", "by pollutant and model", "Ground truth", "Real-world accuracy",
                 "System health · services", "System health · data pipeline", "ReadTimeout",
                 "monitor offline",
                 "Data drift", "Latest forecast"):
        assert text in page, text
    # metric by horizon, forecast vs observed, rolling error, coverage, monitor vs model, map
    assert len(app.get("plotly_chart")) == 6
    app.selectbox[0].select("pm10").run()
    app.selectbox[1].select(2).run()
    app.selectbox[3].select("r2").run()
    assert not app.exception
    page = " ".join(block.value for block in app.markdown)
    assert "R² by pollutant and model" in page and "Ridge regression" in page
