"""PM2.5 measured by the US Embassy monitor in Abidjan (Met One BAM-1022), via the AirNow API.

The monitor is reference-grade ground truth, used to calibrate the CAMS model values. Access
needs a free AirNow API key (https://docs.airnowapi.org/) in the `AIRNOW_API_KEY` environment
variable or in a `.env` file at the project root.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

AIRNOW_DATA_URL = "https://www.airnowapi.org/aq/data/"
SITE = "Abidjan"
BBOX = "-4.3,5.1,-3.7,5.7"   # around the embassy (Riviera, Cocody), excludes Accra and Lomé
FIRST_MONTH = "2020-02-01"   # monitor installed in early February 2020
MIN_HOURS = 18               # hours needed for a representative daily mean
WORKERS = 8                  # parallel monthly requests


def api_key(project_root: Path | None = None) -> str | None:
    key = os.environ.get("AIRNOW_API_KEY")
    if not key and project_root is not None and (project_root / ".env").exists():
        from dotenv import dotenv_values

        key = dotenv_values(project_root / ".env").get("AIRNOW_API_KEY")
    key = (key or "").strip().strip("\"'")
    return key or None


def parse_records(records: list[dict]) -> pd.DataFrame:
    """Hourly PM2.5 (µg/m³) of the Abidjan site; −999 and other invalid values are dropped."""
    rows = []
    for record in records:
        if SITE.lower() not in (record.get("SiteName") or "").lower():
            continue
        value = record.get("RawConcentration", record.get("Value"))
        if value is None or value < 0 or value > 1000:
            continue
        rows.append({"time": pd.Timestamp(record["UTC"]), "pm2_5": float(value)})
    frame = pd.DataFrame(rows, columns=["time", "pm2_5"])
    # Abidjan is on UTC all year, so UTC hours are local hours
    return frame.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def fetch_month(key: str, month_start: pd.Timestamp) -> pd.DataFrame:
    month_end = month_start + pd.offsets.MonthEnd(0)
    response = requests.get(AIRNOW_DATA_URL, params={
        "startDate": month_start.strftime("%Y-%m-%dT00"),
        "endDate": month_end.strftime("%Y-%m-%dT23"),
        "parameters": "PM25", "BBOX": BBOX, "dataType": "B", "verbose": 1,
        "monitorType": 0, "includerawconcentrations": 1, "format": "application/json",
        "API_KEY": key,
    }, timeout=180)
    if response.status_code != 200:
        # requests' own message contains the full URL, API key included: never log it
        raise requests.HTTPError(f"AirNow returned HTTP {response.status_code} for "
                                 f"{month_start:%Y-%m}")
    return parse_records(response.json())


def update_embassy(path: Path, key: str, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Download the months missing from `path` (plus the latest saved one, which may be partial)."""
    today = (today or pd.Timestamp.today()).normalize()
    existing = (pd.read_csv(path, parse_dates=["time"]) if Path(path).exists()
                else pd.DataFrame(columns=["time", "pm2_5"]))
    start = (existing["time"].max().to_period("M").to_timestamp() if not existing.empty
             else pd.Timestamp(FIRST_MONTH))
    data = existing[existing["time"] < start] if not existing.empty else existing
    months = list(pd.date_range(start, today, freq="MS"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def get(month):
        try:
            return fetch_month(key, month)
        except requests.RequestException as exc:
            log.warning("Embassy data for %s unavailable: %s", month.strftime("%Y-%m"),
                        str(exc).replace(key, "***"))
            return None

    # Each monthly request takes about a minute: run them in parallel, well under AirNow's
    # limit of 500 requests per hour, and save after every batch so progress is never lost
    for batch_start in range(0, len(months), WORKERS):
        batch = months[batch_start:batch_start + WORKERS]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            frames = [f for f in pool.map(get, batch) if f is not None and not f.empty]
        data = (pd.concat([data, *frames], ignore_index=True).drop_duplicates("time")
                .sort_values("time").reset_index(drop=True))
        data.to_csv(path, index=False)
        log.info("Embassy PM2.5: %s → %s downloaded", batch[0].strftime("%Y-%m"),
                 batch[-1].strftime("%Y-%m"))
    log.info("Embassy PM2.5: %d hours (%s → %s)", len(data),
             data["time"].min() if len(data) else "—", data["time"].max() if len(data) else "—")
    return data


def daily_means(hourly: pd.DataFrame, min_hours: int = MIN_HOURS) -> pd.Series:
    """Daily mean PM2.5; days with fewer than `min_hours` valid hours are dropped."""
    if hourly.empty:
        return pd.Series(dtype=float, name="embassy_pm2_5")
    series = hourly.set_index(pd.to_datetime(hourly["time"]))["pm2_5"]
    grouped = series.groupby(series.index.normalize())
    daily = grouped.mean()[grouped.count() >= min_hours]
    daily.index.name = "date"
    return daily.rename("embassy_pm2_5")
