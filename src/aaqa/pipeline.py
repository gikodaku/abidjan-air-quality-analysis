"""Pipeline steps, each reading and writing files in data/, models/ and reports/."""

from __future__ import annotations

import json
import logging

import pandas as pd

from aaqa import config as paths
from aaqa.aqi import WHO_PM10_ANNUAL, WHO_PM25_ANNUAL
from aaqa.calibration import calibrate
from aaqa.config import Config, load_config
from aaqa.data.clean import clean_hourly
from aaqa.data.embassy import api_key, daily_means, update_embassy
from aaqa.data.fetch import fetch_air_quality, fetch_grid_points, fetch_weather, resolve_dates
from aaqa.data.weather_forecast import fetch_weather_forecast
from aaqa.features.build import HARMATTAN_MONTHS, aggregate_daily, build_features
from aaqa.models.predictor import AirQualityPredictor
from aaqa.models.train import train
from aaqa.monitoring import log_forecast
from aaqa.pollutants import POLLUTANTS
from aaqa.visualization import plots

log = logging.getLogger(__name__)

RAINY_MONTHS = [4, 5, 6, 7]
ANNUAL_GUIDELINES = {"pm2_5": WHO_PM25_ANNUAL, "pm10": WHO_PM10_ANNUAL}


def _read_timeseries(path, index_col) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the previous step first")
    return pd.read_csv(path, index_col=index_col, parse_dates=[index_col])


def load_hourly() -> pd.DataFrame:
    return _read_timeseries(paths.HOURLY_PROCESSED, "time")


def load_daily() -> pd.DataFrame:
    return _read_timeseries(paths.DAILY_PROCESSED, "date").asfreq("D")


def step_fetch(cfg: Config) -> None:
    start, end = resolve_dates(cfg.data.start_date, cfg.data.end_date)
    paths.RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetch_air_quality(cfg.location, start, end).to_csv(paths.AIR_QUALITY_RAW, index=False)
    fetch_weather(cfg.location, start, end).to_csv(paths.WEATHER_RAW, index=False)
    try:  # only used by the map in the app
        paths.GRID_POINTS.write_text(json.dumps(fetch_grid_points(cfg.location), indent=2))
    except Exception as exc:
        log.warning("Grid points unavailable: %s", exc)
    log.info("Raw data written to %s", paths.RAW_DIR)


def step_process(cfg: Config) -> pd.DataFrame:
    air = pd.read_csv(paths.AIR_QUALITY_RAW)
    weather = pd.read_csv(paths.WEATHER_RAW)
    hourly = clean_hourly(air, weather, cfg.data.max_gap_hours)
    daily = aggregate_daily(hourly)
    paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(paths.HOURLY_PROCESSED)
    daily.to_csv(paths.DAILY_PROCESSED)
    log.info("%d hours and %d days cleaned (%s → %s)", len(hourly), daily["pm2_5_mean"].count(),
             daily.index.min().date(), daily.index.max().date())
    return daily


def step_calibrate(cfg: Config) -> dict | None:
    """Download the US Embassy PM2.5 and fit the CAMS correction (needs an AirNow API key)."""
    key = api_key(paths.PROJECT_ROOT)
    if key is None:
        log.warning("No AIRNOW_API_KEY: skipping the calibration on the US Embassy monitor")
        return None
    embassy = daily_means(update_embassy(paths.EMBASSY_RAW, key))
    result = calibrate(load_daily()["pm2_5_mean"], embassy)
    result.save(paths.CALIBRATION_PATH)
    plots.plot_calibration(load_daily()["pm2_5_mean"], embassy, result,
                           paths.FIGURES_DIR / "calibration.png")
    best = result.cross_validation[result.choice]
    log.info("Calibration (%s, %d days): MAE %.1f → %.1f µg/m³, bias %+.1f → %+.1f",
             result.choice, result.n_days, result.raw["mae"], best["mae"], result.raw["bias"],
             best["bias"])
    return {"method": result.method, "n_days": result.n_days}


def step_train(cfg: Config) -> dict:
    features = build_features(load_daily())
    bundle, evaluations = train(features, cfg.model)
    AirQualityPredictor(bundle).save(paths.MODEL_PATH)

    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    paths.METRICS_PATH.write_text(json.dumps(
        {p: {str(h): m for h, m in by_h.items()} for p, by_h in bundle["metrics"].items()},
        indent=2, ensure_ascii=False))

    first = min(cfg.model.horizons)
    predictions, importances = {}, {}
    for pollutant, by_h in evaluations.items():
        predictions[pollutant] = by_h[first]["predictions"]
        importances[pollutant] = by_h[first]["importances"]
        # Every horizon's test-year predictions feed the monitoring backtest
        for h, ev in by_h.items():
            ev["predictions"].to_csv(paths.REPORTS_DIR / f"test_predictions_{pollutant}_h{h}.csv")
    pd.DataFrame(importances).to_csv(paths.REPORTS_DIR / "feature_importance.csv")
    plots.plot_forecast_vs_actual(predictions, paths.FIGURES_DIR / "forecast_test.png", first)
    plots.plot_importances(importances, paths.FIGURES_DIR / "feature_importance.png", first)

    for pollutant, by_h in evaluations.items():
        for h, ev in by_h.items():
            gb, pers = ev["models"]["gradient_boosting"], ev["models"]["persistence"]
            log.info("%-4s D+%d: MAE %6.2f µg/m³ (persistence %6.2f, gain %+4.0f%%), "
                     "AQI category %3.0f%%", POLLUTANTS[pollutant].name, h, gb["mae"],
                     pers["mae"], 100 * gb["skill_vs_persistence"],
                     100 * gb["aqi_category_accuracy"])
    return bundle["metrics"]


def _pollutant_statistics(daily: pd.DataFrame, key: str) -> dict:
    spec = POLLUTANTS[key]
    series = daily[spec.target].dropna()
    months = series.index.month
    harmattan, rainy = series[months.isin(HARMATTAN_MONTHS)], series[months.isin(RAINY_MONTHS)]
    yearly = series.groupby(series.index.year).agg(["mean", "count"])
    stats = {
        "name": spec.name,
        "statistic": spec.statistic,
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
        "max_date": series.idxmax().date().isoformat(),
        "harmattan_mean": float(harmattan.mean()),
        "rainy_season_mean": float(rainy.mean()),
        "annual_means": {str(y): float(m)
                         for y, m in yearly.loc[yearly["count"] >= 330, "mean"].items()},
        "monthly_means": {int(m): float(v) for m, v in series.groupby(months).mean().items()},
        "who_limit": spec.who_limit,
        "who_label": spec.who_label,
    }
    if spec.who_column in daily:
        who = daily[spec.who_column].dropna()
        who_months = who.index.month
        stats["share_days_above_who"] = float((who > spec.who_limit).mean())
        stats["harmattan_share_days_above_who"] = float(
            (who[who_months.isin(HARMATTAN_MONTHS)] > spec.who_limit).mean())
        stats["rainy_share_days_above_who"] = float(
            (who[who_months.isin(RAINY_MONTHS)] > spec.who_limit).mean())
    if key in ANNUAL_GUIDELINES:
        stats["ratio_annual_to_who"] = float(series.mean() / ANNUAL_GUIDELINES[key])
    return stats


def compute_statistics(daily: pd.DataFrame) -> dict:
    pm = daily["pm2_5_mean"].dropna()
    return {
        "period": [pm.index.min().date().isoformat(), pm.index.max().date().isoformat()],
        "n_days": int(pm.size),
        "pollutants": {key: _pollutant_statistics(daily, key)
                       for key, spec in POLLUTANTS.items() if spec.target in daily},
        "weekday_means_pm2_5": {int(d): float(v)
                                for d, v in pm.groupby(pm.index.day_of_week).mean().items()},
        "pm2_5_spearman": {
            column: float(daily[column].corr(pm, method="spearman"))
            for column in ("pm10_mean", "dust_mean", "carbon_monoxide_mean",
                           "nitrogen_dioxide_mean", "relative_humidity_2m_mean",
                           "precipitation_sum", "wind_speed_10m_mean")
            if column in daily
        },
    }


def step_report(cfg: Config) -> dict:
    daily, hourly = load_daily(), load_hourly()
    fig = paths.FIGURES_DIR
    plots.plot_timeseries(daily, fig / "pollutants_timeseries.png")
    plots.plot_monthly(daily, fig / "pollutants_monthly.png")
    plots.plot_diurnal(hourly, fig / "pollutants_diurnal.png")
    plots.plot_weekday(daily, fig / "pm25_weekday.png")
    plots.plot_correlations(daily, fig / "correlations.png")
    stats = compute_statistics(daily)
    (paths.REPORTS_DIR / "statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False))
    log.info("Figures written to %s", fig)
    return stats


def step_forecast(cfg: Config) -> dict:
    predictor = AirQualityPredictor.load(paths.MODEL_PATH)
    result = predictor.forecast(load_daily())
    log_forecast(result, predictor.metadata, paths.MONITORING_LOG)
    try:
        result["weather"] = fetch_weather_forecast(cfg.location)
    except Exception as exc:  # the pollution forecast is still useful without the weather
        log.warning("Weather forecast unavailable: %s", exc)
        result["weather"] = []
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.REPORTS_DIR / "latest_forecast.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))
    return result


STEPS = {
    "fetch": step_fetch,
    "process": step_process,
    "calibrate": step_calibrate,
    "report": step_report,
    "train": step_train,
    "forecast": step_forecast,
}


def run(steps: list[str], cfg: Config | None = None):
    cfg = cfg or load_config()
    result = None
    for name in steps:
        log.info("── step %s", name)
        result = STEPS[name](cfg)
    return result
