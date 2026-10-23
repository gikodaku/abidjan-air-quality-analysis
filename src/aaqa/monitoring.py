"""Model monitoring: log every issued forecast, score it when the observation arrives, and
summarise the health of each model (pollutant × horizon).

Two sources share one frame layout:
- live: forecasts written by `aaqa forecast` to data/monitoring/forecast_log.csv, scored against
  the observed values once the target day is in the data;
- backtest: the out-of-sample predictions of the test year saved by `aaqa train`.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from aaqa.aqi import aqi_category, concentration_to_aqi
from aaqa.data.fetch import AIR_QUALITY_URL, WEATHER_ARCHIVE_URL, WEATHER_FORECAST_URL
from aaqa.pollutants import POLLUTANTS

LOG_COLUMNS = ["issued_at", "issued_for", "target_date", "horizon", "pollutant", "method",
               "predicted", "q10", "q90", "model_version", "trained_until"]
KEY = ["issued_for", "horizon", "pollutant"]

MIN_SCORED = 7          # fewer scored forecasts than this: not enough to judge a model
STALE_HOURS = 48        # no new observation for this long: data alert
STATUS = {
    "good": ("#0ca30c", "✅", "Healthy"),
    "warning": ("#fab219", "⚠️", "Watch"),
    "critical": ("#d03b3b", "⛔", "Alert"),
    "collecting": ("#898781", "⏳", "Collecting"),
}


# ── Live log ────────────────────────────────────────────────────────────────────────────────

def log_forecast(result: dict, metadata: dict, path: Path,
                 issued_at: datetime | None = None) -> pd.DataFrame:
    """Append the forecasts of `result` to the log; re-issuing the same day replaces it."""
    issued_at = (issued_at or datetime.now(UTC)).isoformat(timespec="seconds")
    rows = []
    for day in result["forecasts"]:
        for value in day["pollutants"]:
            rows.append({
                "issued_at": issued_at,
                "issued_for": result["issued_for"],
                "target_date": day["date"],
                "horizon": day["horizon_days"],
                "pollutant": value["pollutant"],
                "method": value.get("method", "gradient_boosting"),
                "predicted": value["value"],
                "q10": value["interval_80"][0],
                "q90": value["interval_80"][1],
                "model_version": metadata.get("version"),
                "trained_until": metadata.get("trained_until"),
            })
    log = pd.concat([load_log(path), pd.DataFrame(rows, columns=LOG_COLUMNS)], ignore_index=True)
    log = log.drop_duplicates(KEY, keep="last").sort_values(KEY).reset_index(drop=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    return log


def load_log(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(path)


def score(frame: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Add observed values, persistence baseline and errors; unscored rows keep NaN."""
    frame = frame.copy()
    if frame.empty:
        for column in ("observed", "persistence", "error", "abs_error", "persistence_abs_error",
                       "in_interval"):
            frame[column] = pd.Series(dtype=float)
        return frame
    frame["issued_for"] = pd.to_datetime(frame["issued_for"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    observed, persistence = [], []
    for row in frame.itertuples():
        series = daily[POLLUTANTS[row.pollutant].target]
        observed.append(series.get(row.target_date, np.nan))
        persistence.append(series.get(row.issued_for, np.nan))
    frame["observed"] = observed
    if "persistence" not in frame or frame["persistence"].isna().all():
        frame["persistence"] = persistence
    frame["error"] = frame["predicted"] - frame["observed"]
    frame["abs_error"] = frame["error"].abs()
    frame["persistence_abs_error"] = (frame["persistence"] - frame["observed"]).abs()
    inside = (frame["observed"] >= frame["q10"]) & (frame["observed"] <= frame["q90"])
    frame["in_interval"] = inside.where(frame["observed"].notna())
    return frame


# ── Backtest ────────────────────────────────────────────────────────────────────────────────

def load_backtest(reports_dir: Path, metrics: dict) -> pd.DataFrame:
    """Test-year predictions of every pollutant and horizon, in the live-log layout."""
    frames = []
    for path in sorted(Path(reports_dir).glob("test_predictions_*_h*.csv")):
        pollutant, horizon = path.stem.removeprefix("test_predictions_").rsplit("_h", 1)
        if pollutant not in POLLUTANTS:
            continue
        data = pd.read_csv(path, index_col=0, parse_dates=True)
        method = metrics.get(pollutant, {}).get(horizon, {}).get(
            "selected_method", "gradient_boosting")
        predicted = data["persistence"] if method == "persistence" else data["gradient_boosting"]
        frames.append(pd.DataFrame({
            "issued_for": data.index,
            "target_date": data.index + pd.Timedelta(days=int(horizon)),
            "horizon": int(horizon),
            "pollutant": pollutant,
            "method": method,
            "predicted": predicted.to_numpy(),
            "q10": data["q10"].to_numpy(),
            "q90": data["q90"].to_numpy(),
            "observed": data["y_true"].to_numpy(),
            "persistence": data["persistence"].to_numpy(),
        }))
    if not frames:
        return score(pd.DataFrame(columns=LOG_COLUMNS), pd.DataFrame())
    frame = pd.concat(frames, ignore_index=True)
    frame["error"] = frame["predicted"] - frame["observed"]
    frame["abs_error"] = frame["error"].abs()
    frame["persistence_abs_error"] = (frame["persistence"] - frame["observed"]).abs()
    frame["in_interval"] = ((frame["observed"] >= frame["q10"])
                            & (frame["observed"] <= frame["q90"])).astype(float)
    return frame


# ── Health ──────────────────────────────────────────────────────────────────────────────────

def window(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    """Scored rows whose target date falls in the last `days` days of the frame."""
    scored = frame.dropna(subset=["observed"])
    if scored.empty:
        return scored
    end = scored["target_date"].max()
    return scored[scored["target_date"] > end - pd.Timedelta(days=days)]


def summarise(scored: pd.DataFrame) -> dict:
    n = len(scored)
    if n == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "persistence_mae": np.nan,
                "skill": np.nan, "coverage": np.nan, "bias": np.nan}
    mae, base = scored["abs_error"].mean(), scored["persistence_abs_error"].mean()
    observed = scored["observed"].astype(float)
    variance = float(((observed - observed.mean()) ** 2).sum())
    sse = float((scored["error"].astype(float) ** 2).sum())
    return {"n": n, "mae": float(mae), "rmse": float(np.sqrt(sse / n)),
            "r2": float(1 - sse / variance) if n >= 3 and variance > 0 else np.nan,
            "persistence_mae": float(base),
            "skill": float(1 - mae / base) if base > 0 else np.nan,
            "coverage": float(scored["in_interval"].mean()),
            "bias": float(scored["error"].mean())}


def status_of(summary: dict, reference_mae: float | None) -> tuple[str, list[str]]:
    """Health rule, with the reasons that triggered it."""
    if summary["n"] < MIN_SCORED:
        return "collecting", [f"{summary['n']} scored forecasts (need {MIN_SCORED})"]
    critical, warning = [], []
    if summary["skill"] < -0.10:
        critical.append(f"{-summary['skill']:.0%} worse than “tomorrow = today”")
    elif summary["skill"] < 0:
        warning.append("not beating “tomorrow = today”")
    if reference_mae:
        ratio = summary["mae"] / reference_mae
        if ratio > 2:
            critical.append(f"error {ratio:.1f}× the test-year error")
        elif ratio > 1.3:
            warning.append(f"error {ratio:.1f}× the test-year error")
    if summary["coverage"] < 0.5:
        critical.append(f"80% interval covers only {summary['coverage']:.0%}")
    elif summary["coverage"] < 0.6:
        warning.append(f"80% interval covers {summary['coverage']:.0%}")
    if critical:
        return "critical", critical + warning
    if warning:
        return "warning", warning
    return "good", ["within expected range"]


def health_table(frame: pd.DataFrame, metrics: dict, days: int) -> pd.DataFrame:
    """One row per pollutant × horizon with live metrics, reference metrics and status."""
    recent = window(frame, days)
    rows = []
    for pollutant in POLLUTANTS:
        for horizon_key, reference in sorted(metrics.get(pollutant, {}).items()):
            horizon = int(horizon_key)
            subset = recent[(recent["pollutant"] == pollutant) & (recent["horizon"] == horizon)]
            summary = summarise(subset)
            method = reference.get("selected_method", "gradient_boosting")
            reference_mae = reference["models"][method]["mae"]
            status, reasons = status_of(summary, reference_mae)
            test = reference["models"][method]
            rows.append({"pollutant": pollutant, "name": POLLUTANTS[pollutant].name,
                         "horizon": horizon, "method": method, "reference_mae": reference_mae,
                         "reference_rmse": test.get("rmse"), "reference_r2": test.get("r2"),
                         **summary, "status": status, "reasons": "; ".join(reasons)})
    return pd.DataFrame(rows)


def rolling_errors(frame: pd.DataFrame, pollutant: str, horizon: int,
                   window_days: int = 14) -> pd.DataFrame:
    subset = frame[(frame["pollutant"] == pollutant) & (frame["horizon"] == horizon)]
    subset = subset.dropna(subset=["observed"]).set_index("target_date").sort_index()
    return pd.DataFrame({
        "model": subset["abs_error"].rolling(f"{window_days}D").mean(),
        "persistence": subset["persistence_abs_error"].rolling(f"{window_days}D").mean(),
        "coverage": subset["in_interval"].astype(float).rolling(f"{window_days}D").mean(),
    })


# ── Data health ─────────────────────────────────────────────────────────────────────────────

def freshness(daily: pd.DataFrame, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    last_day = daily["pm2_5_mean"].dropna().index.max()
    # A day is complete at its end (local time = UTC in Abidjan)
    complete_at = (last_day + pd.Timedelta(days=1)).tz_localize("UTC")
    hours = (pd.Timestamp(now) - complete_at).total_seconds() / 3600
    status = "critical" if hours > STALE_HOURS else "warning" if hours > 30 else "good"
    return {"last_day": last_day.date().isoformat(), "hours": max(hours, 0.0), "status": status}


def drift_table(daily: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """Compare the last `days` days with the same calendar period in earlier years.

    z = (recent mean − historical mean of the same period) / spread of those historical means.
    Seasonal comparison matters here: a harmattan rise is expected, not drift. The spread has a
    floor of 10% of the historical mean, so that very regular years don't turn a small change
    into an alert.
    """
    end = daily.index.max()
    start = end - pd.Timedelta(days=days - 1)
    columns = {spec.name: spec.target for spec in POLLUTANTS.values()} | {
        "Temperature": "temperature_2m_mean", "Humidity": "relative_humidity_2m_mean",
        "Rain": "precipitation_sum", "Wind": "wind_speed_10m_mean", "Dust": "dust_mean"}
    rows = []
    for name, column in columns.items():
        if column not in daily:
            continue
        recent = daily.loc[start:end, column].mean()
        past = []
        for years_back in range(1, 10):
            offset = pd.DateOffset(years=years_back)
            values = daily.loc[start - offset:end - offset, column].dropna()
            if len(values) >= days * 0.8:
                past.append(values.mean())
        if len(past) < 2:
            continue
        mean = float(np.mean(past))
        spread = max(float(np.std(past, ddof=1)), 0.1 * abs(mean))
        z = (recent - mean) / spread if spread > 0 else 0.0
        status = "critical" if abs(z) > 3 else "warning" if abs(z) > 2 else "good"
        rows.append({"variable": name, "recent_mean": float(recent), "same_period_mean": mean,
                     "change": float(recent / mean - 1) if mean else np.nan, "z": float(z),
                     "years": len(past), "status": status})
    return pd.DataFrame(rows)



# ── Model comparison ────────────────────────────────────────────────────────────────────────

MODEL_LABELS = {
    "gradient_boosting": "Gradient boosting",
    "persistence": "“Tomorrow = today”",
    "ridge": "Ridge regression",
    "climatology": "Monthly average",
}
# name, unit, which direction is better ("low", "high", or "zero" for bias)
METRICS = {
    "mae": ("MAE", "µg/m³", "low"),
    "rmse": ("RMSE", "µg/m³", "low"),
    "r2": ("R²", "", "high"),
    "bias": ("Bias", "µg/m³", "zero"),
    "aqi_category_accuracy": ("Right AQI category", "%", "high"),
}


def load_backtest_models(reports_dir: Path) -> pd.DataFrame:
    """Test-year predictions of every model, one row per (day, pollutant, horizon, model)."""
    frames = []
    for path in sorted(Path(reports_dir).glob("test_predictions_*_h*.csv")):
        pollutant, horizon = path.stem.removeprefix("test_predictions_").rsplit("_h", 1)
        if pollutant not in POLLUTANTS:
            continue
        data = pd.read_csv(path, index_col=0, parse_dates=True)
        for model in MODEL_LABELS:
            if model in data:
                frames.append(pd.DataFrame({
                    "target_date": data.index + pd.Timedelta(days=int(horizon)),
                    "horizon": int(horizon), "pollutant": pollutant, "model": model,
                    "predicted": data[model].to_numpy(), "observed": data["y_true"].to_numpy(),
                }))
    columns = ["target_date", "horizon", "pollutant", "model", "predicted", "observed"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def live_models(scored: pd.DataFrame) -> pd.DataFrame:
    """Live log in the same long layout: the served model and the persistence baseline."""
    scored = scored.dropna(subset=["observed"])
    served = scored.assign(model=scored["method"])[
        ["target_date", "horizon", "pollutant", "model", "predicted", "observed"]]
    baseline = scored.assign(model="persistence", predicted=scored["persistence"])[
        ["target_date", "horizon", "pollutant", "model", "predicted", "observed"]]
    long = pd.concat([served, baseline], ignore_index=True)
    return long.drop_duplicates(["target_date", "horizon", "pollutant", "model"])


def scores(observed, predicted, pollutant: str) -> dict:
    """MAE, RMSE, R², bias and AQI-category accuracy of one model on one sample."""
    y, p = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    keep = ~(np.isnan(y) | np.isnan(p))
    y, p = y[keep], p[keep]
    n = len(y)
    if n == 0:
        return {"n": 0, **{key: np.nan for key in METRICS}}
    error = p - y
    variance = float(((y - y.mean()) ** 2).sum())

    def category(v):
        return aqi_category(concentration_to_aqi(pollutant, max(v, 0.0))).key

    return {
        "n": n,
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "r2": float(1 - (error ** 2).sum() / variance) if n >= 3 and variance > 0 else np.nan,
        "bias": float(error.mean()),
        "aqi_category_accuracy": float(np.mean([category(a) == category(b)
                                                for a, b in zip(y, p, strict=True)])),
    }


def compare_models(long: pd.DataFrame, days: int) -> pd.DataFrame:
    """Scores of every model for every pollutant and horizon over the last `days` days."""
    rows = []
    for (pollutant, horizon), group in long.groupby(["pollutant", "horizon"]):
        group = group.dropna(subset=["observed"])
        if group.empty:
            continue
        recent = group[group["target_date"] > group["target_date"].max()
                       - pd.Timedelta(days=days)]
        for model, sample in recent.groupby("model"):
            rows.append({"pollutant": pollutant, "horizon": int(horizon), "model": model,
                         **scores(sample["observed"], sample["predicted"], pollutant)})
    return pd.DataFrame(rows)


def best_model(table: pd.DataFrame, metric: str) -> pd.Series:
    """Name of the best model per (pollutant, horizon) for a metric."""
    direction = METRICS[metric][2]
    values = table.assign(_key=table[metric].abs() if direction == "zero" else table[metric])
    values = values.dropna(subset=["_key"])
    pick = (values.groupby(["pollutant", "horizon"])["_key"].idxmax() if direction == "high"
            else values.groupby(["pollutant", "horizon"])["_key"].idxmin())
    return values.loc[pick].set_index(["pollutant", "horizon"])["model"]


# ── Ground truth (US Embassy monitor) ───────────────────────────────────────────────────────

def ground_truth_scores(reports_dir: Path, calibration, embassy_daily: pd.Series) -> pd.DataFrame:
    """PM2.5 forecasts of the test year scored against the US Embassy monitor.

    For each horizon: the raw model forecast, the calibrated forecast, and the "tomorrow = today"
    baseline built from the embassy itself (the best a monitor-only forecast could do).
    """
    rows = []
    for path in sorted(Path(reports_dir).glob("test_predictions_pm2_5_h*.csv")):
        horizon = int(path.stem.rsplit("_h", 1)[1])
        data = pd.read_csv(path, index_col=0, parse_dates=True)
        target = data.index + pd.Timedelta(days=horizon)
        truth = embassy_daily.reindex(target).to_numpy()
        candidates = {
            "Model, raw (CAMS scale)": data["gradient_boosting"].to_numpy(),
            "Model, calibrated": calibration.apply(data["gradient_boosting"], target),
            "“Tomorrow = today” on the monitor": embassy_daily.reindex(data.index).to_numpy(),
        }
        for name, prediction in candidates.items():
            rows.append({"horizon": horizon, "forecast": name,
                         **scores(truth, prediction, "pm2_5")})
    return pd.DataFrame(rows)


# ── System health: services and data pipeline ──────────────────────────────────────────────

def check_service(name: str, url: str, params: dict | None = None, timeout: float = 8.0,
                  expect_json: bool = True) -> dict:
    """One request: status, response time and a short detail. Never raises."""
    started = time.perf_counter()
    try:
        response = requests.get(url, params=params, timeout=timeout)
        elapsed = (time.perf_counter() - started) * 1000
        ok = response.status_code == 200
        detail = f"HTTP {response.status_code}"
        if ok and expect_json:
            body = response.json()
            if isinstance(body, dict) and body.get("version"):
                detail += f" · v{body['version']} · {body.get('status', '')}"
        status = "good" if ok and elapsed < 3000 else "warning" if ok else "critical"
    except Exception as exc:  # connection refused, timeout, invalid JSON
        elapsed = (time.perf_counter() - started) * 1000
        status, detail = "critical", type(exc).__name__
    return {"service": name, "status": status, "ms": round(elapsed), "detail": detail}


def check_services(location, api_url: str = "http://localhost:8000",
                   airnow_key: str | None = None) -> list[dict]:
    """The AAQA API and every external data source, checked in parallel."""
    point = {"latitude": location.latitude, "longitude": location.longitude}
    checks = [
        ("AAQA REST API", f"{api_url}/health", None),
        ("Open-Meteo air quality (CAMS)", AIR_QUALITY_URL, {**point, "current": "pm2_5"}),
        ("Open-Meteo forecast", WEATHER_FORECAST_URL, {**point, "current": "temperature_2m"}),
        ("Open-Meteo archive (ERA5)", WEATHER_ARCHIVE_URL,
         {**point, "hourly": "temperature_2m", "start_date": "2024-01-01",
          "end_date": "2024-01-01"}),
    ]
    if airnow_key:
        day = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
        checks.append(("AirNow (US Embassy)", "https://www.airnowapi.org/aq/data/",
                       {"startDate": day.strftime("%Y-%m-%dT00"),
                        "endDate": day.strftime("%Y-%m-%dT01"), "parameters": "PM25",
                        "BBOX": "-4.3,5.1,-3.7,5.7", "dataType": "B", "format":
                        "application/json", "API_KEY": airnow_key}))
    with ThreadPoolExecutor(max_workers=len(checks)) as pool:
        results = list(pool.map(lambda c: check_service(*c), checks))
    for result in results:  # never show the key, even in an error detail
        if airnow_key:
            result["detail"] = result["detail"].replace(airnow_key, "***")
    return results


def file_age(path: Path, now: datetime | None = None) -> float | None:
    """Hours since `path` was last written, or None if it does not exist."""
    if not Path(path).exists():
        return None
    now = now or datetime.now(UTC)
    modified = datetime.fromtimestamp(Path(path).stat().st_mtime, UTC)
    return (now - modified).total_seconds() / 3600


def pipeline_health(files: dict[str, tuple[Path, float]], now: datetime | None = None
                    ) -> list[dict]:
    """Age of each pipeline output against the age it should not exceed (hours)."""
    rows = []
    for name, (path, max_hours) in files.items():
        age = file_age(path, now)
        if age is None:
            status, detail = "critical", "missing"
        else:
            status = "good" if age <= max_hours else "warning" if age <= 2 * max_hours \
                else "critical"
            detail = f"{age:.0f} h ago" if age < 48 else f"{age / 24:.0f} days ago"
        rows.append({"step": name, "status": status, "detail": detail,
                     "expected": f"every {max_hours:.0f} h" if max_hours < 48
                     else f"every {max_hours / 24:.0f} days"})
    return rows
