"""Calibration of CAMS PM2.5 on the US Embassy monitor (ground truth).

CAMS describes a ~44 km cell and reads less PM2.5 than the city's monitor, and the gap changed
over time (CAMS dropped from 2024 while the monitor did not). Three corrections, each learned on
all past days or only the most recent 365 or 180 days, are compared walk-forward: every month is
corrected with what was learned before it. The combination with the lowest error is kept and
refitted on the most recent window.

- ratio:     embassy = k · CAMS
- linear:    embassy = a + b · CAMS
- seasonal:  log(1 + embassy) = a + b · log(1 + CAMS) + c · harmattan + d · rainy season
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from aaqa.aqi import describe, overall
from aaqa.features.build import HARMATTAN_MONTHS

RAINY_MONTHS = {4, 5, 6, 7}
METHODS = ("ratio", "linear", "seasonal")
WINDOWS = (None, 365, 180)    # training window in days (None = all past days)


def _season(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    months = pd.DatetimeIndex(dates).month
    return (np.isin(months, list(HARMATTAN_MONTHS)).astype(float),
            np.isin(months, list(RAINY_MONTHS)).astype(float))


def fit(method: str, cams: np.ndarray, embassy: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    cams, embassy = np.asarray(cams, float), np.asarray(embassy, float)
    if method == "ratio":
        return {"k": float(np.median(embassy / np.maximum(cams, 0.1)))}
    if method == "linear":
        b, a = np.polyfit(cams, embassy, 1)
        return {"a": float(a), "b": float(b)}
    if method == "seasonal":
        harmattan, rainy = _season(dates)
        design = np.column_stack([np.ones_like(cams), np.log1p(cams), harmattan, rainy])
        coef, *_ = np.linalg.lstsq(design, np.log1p(embassy), rcond=None)
        residuals = np.log1p(embassy) - design @ coef
        # Smearing factor: corrects the bias of transforming a log-scale mean back
        smear = float(np.mean(np.exp(residuals)))
        return {"a": float(coef[0]), "b": float(coef[1]), "harmattan": float(coef[2]),
                "rainy": float(coef[3]), "smear": smear}
    raise ValueError(f"unknown method {method!r}")


def predict(method: str, params: dict, cams, dates) -> np.ndarray:
    cams = np.maximum(np.asarray(cams, float), 0.0)
    if method == "ratio":
        out = params["k"] * cams
    elif method == "linear":
        out = params["a"] + params["b"] * cams
    else:
        harmattan, rainy = _season(pd.DatetimeIndex(dates))
        log_value = (params["a"] + params["b"] * np.log1p(cams) + params["harmattan"] * harmattan
                     + params["rainy"] * rainy)
        out = np.expm1(log_value) * params["smear"]
    return np.maximum(out, 0.0)


def _scores(truth: np.ndarray, estimate: np.ndarray) -> dict:
    error = estimate - truth
    variance = float(((truth - truth.mean()) ** 2).sum())
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt((error ** 2).mean())),
            "bias": float(error.mean()),
            "r2": float(1 - (error ** 2).sum() / variance) if variance > 0 else float("nan")}


@dataclass
class Calibration:
    method: str
    params: dict
    n_days: int
    period: list[str]
    raw: dict                       # CAMS vs embassy, uncorrected (evaluated days)
    cross_validation: dict          # "method" or "method, last N days" → walk-forward scores
    window_days: int | None = None  # training window of the chosen correction
    choice: str = ""                # key of the chosen correction in `cross_validation`
    fitted_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(
        timespec="seconds"))

    def apply(self, cams, dates) -> np.ndarray:
        return predict(self.method, self.params, cams, dates)

    def apply_one(self, cams: float, day) -> float:
        return float(self.apply([cams], pd.DatetimeIndex([pd.Timestamp(day)]))[0])

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> Calibration | None:
        try:
            return cls(**json.loads(Path(path).read_text()))
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return None


def paired(cams_daily: pd.Series, embassy_daily: pd.Series) -> pd.DataFrame:
    """Days with both a CAMS and an embassy daily mean."""
    frame = pd.concat({"cams": cams_daily, "embassy": embassy_daily}, axis=1).dropna()
    return frame[(frame["cams"] >= 0) & (frame["embassy"] > 0)]


def _label(method: str, window: int | None) -> str:
    return method if window is None else f"{method}, last {window} days"


def calibrate(cams_daily: pd.Series, embassy_daily: pd.Series, min_days: int = 60,
              warmup_months: int = 6) -> Calibration:
    """Walk-forward evaluation of every correction × training window, then refit the best."""
    data = paired(cams_daily, embassy_daily)
    if len(data) < min_days:
        raise ValueError(f"only {len(data)} days with both CAMS and embassy data "
                         f"(need {min_days})")
    cams, embassy, dates = data["cams"].to_numpy(), data["embassy"].to_numpy(), data.index
    months = pd.period_range(dates.min(), dates.max(), freq="M")[warmup_months:]
    by_month = dates.to_period("M")
    errors = {(m, w): [] for m in METHODS for w in WINDOWS}
    truth, raw = [], []
    for month in months:
        test = by_month == month
        past = dates < month.start_time
        if not test.any() or past.sum() < 30:
            continue
        truth.append(embassy[test])
        raw.append(cams[test])
        for window in WINDOWS:
            train = past if window is None else past & (
                dates >= month.start_time - pd.Timedelta(days=window))
            if train.sum() < 30:
                train = past
            for method in METHODS:
                params = fit(method, cams[train], embassy[train], dates[train])
                errors[(method, window)].append(predict(method, params, cams[test], dates[test]))
    if not truth:
        raise ValueError("not enough history to evaluate the calibration")
    truth_all = np.concatenate(truth)
    scores = {_label(m, w): _scores(truth_all, np.concatenate(p)) for (m, w), p in errors.items()}
    method, window = min(errors, key=lambda k: scores[_label(*k)]["mae"])
    recent = (dates >= dates.max() - pd.Timedelta(days=window)) if window else np.ones(
        len(dates), bool)
    return Calibration(
        method=method,
        params=fit(method, cams[recent], embassy[recent], dates[recent]),
        n_days=len(data),
        period=[dates.min().date().isoformat(), dates.max().date().isoformat()],
        raw=_scores(truth_all, np.concatenate(raw)),
        cross_validation=scores,
        window_days=window,
        choice=_label(method, window),
    )


def adjust_snapshot(snapshot: dict, calibration: Calibration, day) -> dict:
    """Replace PM2.5 in a forecast or live snapshot by its calibrated value and recompute the
    overall index. The raw model value is kept as `raw_value`."""
    pollutants = []
    for value in snapshot["pollutants"]:
        if value["pollutant"] == "pm2_5":
            corrected = calibration.apply_one(value["value"], day)
            new = {**value, **describe("pm2_5", corrected), "raw_value": value["value"],
                   "calibrated": True}
            if "interval_80" in value:
                low, high = value["interval_80"]
                new["interval_80"] = [round(calibration.apply_one(low, day), 1),
                                      round(calibration.apply_one(high, day), 1)]
            value = new
        pollutants.append(value)
    return {**snapshot, "pollutants": pollutants, "overall": overall(pollutants)}
