"""Report figures (matplotlib, PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aaqa.pollutants import POLLUTANTS

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ACCENT = "#1f6f8b"
WHO_COLOR = "#c0392b"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _grid(keys: list[str], width: float = 13, row_height: float = 3.2):
    rows = (len(keys) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(width, row_height * rows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(keys):]:
        ax.set_visible(False)
    return fig, dict(zip(keys, flat, strict=False))


def _title(key: str) -> str:
    spec = POLLUTANTS[key]
    return f"{spec.name} ({spec.statistic})"


def _who_line(ax, key: str) -> None:
    spec = POLLUTANTS[key]
    if spec.who_on_target:
        ax.axhline(spec.who_limit, color=WHO_COLOR, ls="--", lw=1, label=f"WHO {spec.who_label}")


def _available(frame: pd.DataFrame, column_of=lambda spec: spec.target) -> list[str]:
    return [key for key, spec in POLLUTANTS.items() if column_of(spec) in frame]


def plot_timeseries(daily: pd.DataFrame, path: Path) -> Path:
    keys = _available(daily)
    fig, axes = _grid(keys)
    for key, ax in axes.items():
        series = daily[POLLUTANTS[key].target]
        ax.plot(series.index, series, color=ACCENT, lw=0.6)
        ax.plot(series.index, series.rolling(30, center=True).mean(), color="black", lw=1.4)
        _who_line(ax, key)
        ax.set(title=_title(key), ylabel="µg/m³")
        if POLLUTANTS[key].who_on_target:
            ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Pollution in Abidjan — daily value and 30-day rolling mean")
    return _save(fig, path)


def plot_monthly(daily: pd.DataFrame, path: Path) -> Path:
    keys = _available(daily)
    fig, axes = _grid(keys)
    for key, ax in axes.items():
        column = POLLUTANTS[key].target
        groups = [daily.loc[daily.index.month == m, column].dropna() for m in range(1, 13)]
        ax.boxplot(groups, tick_labels=MONTHS, showfliers=False,
                   medianprops={"color": ACCENT, "lw": 2})
        _who_line(ax, key)
        ax.set(title=_title(key), ylabel="µg/m³")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Seasonality by pollutant (the harmattan peaks from December to February)")
    return _save(fig, path)


def plot_diurnal(hourly: pd.DataFrame, path: Path) -> Path:
    keys = _available(hourly, lambda spec: spec.key)
    fig, axes = _grid(keys)
    for key, ax in axes.items():
        for label, months in (("Harmattan (Dec–Feb)", [12, 1, 2]),
                              ("Rainy season (Apr–Jul)", [4, 5, 6, 7])):
            subset = hourly[hourly.index.month.isin(months)]
            ax.plot(subset.groupby(subset.index.hour)[key].mean(), marker="o", ms=2, label=label)
        ax.set(title=f"{POLLUTANTS[key].name} — mean hourly profile", xlabel="Local time",
               ylabel="µg/m³", xticks=range(0, 24, 3))
    next(iter(axes.values())).legend(frameon=False, fontsize=8)
    return _save(fig, path)


def plot_weekday(daily: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    means = daily.groupby(daily.index.day_of_week)["pm2_5_mean"].mean()
    ax.bar([DAYS[d] for d in means.index], means.values, color=ACCENT)
    ax.set(title="Mean PM2.5 by day of week", ylabel="µg/m³")
    return _save(fig, path)


def plot_correlations(daily: pd.DataFrame, path: Path) -> Path:
    columns = {
        "pm2_5_mean": "PM2.5", "pm10_mean": "PM10", "dust_mean": "Dust",
        "nitrogen_dioxide_mean": "NO2", "sulphur_dioxide_mean": "SO2",
        "carbon_monoxide_mean": "CO", "ozone_max8h": "O3 (8-h)",
        "temperature_2m_mean": "Temperature", "relative_humidity_2m_mean": "Humidity",
        "precipitation_sum": "Rain", "wind_speed_10m_mean": "Wind",
    }
    available = {k: v for k, v in columns.items() if k in daily}
    corr = daily[list(available)].rename(columns=available).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title("Spearman correlations (daily)")
    return _save(fig, path)


def plot_forecast_vs_actual(predictions: dict[str, pd.DataFrame], path: Path,
                            horizon: int = 1) -> Path:
    fig, axes = _grid(list(predictions))
    for key, ax in axes.items():
        frame = predictions[key]
        ax.fill_between(frame.index, frame["q10"], frame["q90"], color=ACCENT, alpha=0.2,
                        label="80% interval")
        ax.plot(frame.index, frame["y_true"], color="black", lw=0.9, label="Observed")
        ax.plot(frame.index, frame["gradient_boosting"], color=ACCENT, lw=1.1,
                label=f"D+{horizon} forecast")
        _who_line(ax, key)
        ax.set(title=_title(key), ylabel="µg/m³")
        ax.tick_params(axis="x", labelsize=8)
    next(iter(axes.values())).legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle(f"D+{horizon} forecast over the test year")
    return _save(fig, path)


def plot_importances(importances: dict[str, pd.Series], path: Path, horizon: int = 1,
                     top: int = 8) -> Path:
    fig, axes = _grid(list(importances), row_height=3.0)
    for key, ax in axes.items():
        # The day-D value dominates by construction (the model predicts the change relative to
        # it), so it is left out to show what the other features contribute
        data = importances[key].drop(POLLUTANTS[key].target, errors="ignore").head(top)[::-1]
        ax.barh(data.index, data.values, color=ACCENT)
        ax.set_title(POLLUTANTS[key].name)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(f"Feature importance (permutation, D+{horizon}, excluding the day's value) — "
                 "increase in mean absolute error (µg/m³)")
    return _save(fig, path)


def plot_calibration(cams: pd.Series, embassy: pd.Series, calibration, path: Path) -> Path:
    """CAMS vs the US Embassy monitor, before and after calibration."""
    data = pd.concat({"cams": cams, "embassy": embassy}, axis=1).dropna()
    corrected = calibration.apply(data["cams"], data.index)
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.2),
                                      gridspec_kw={"width_ratios": [1, 2]})
    top = max(data["embassy"].max(), data["cams"].max()) * 1.05
    left.scatter(data["cams"], data["embassy"], s=8, alpha=0.5, color=ACCENT, label="Days")
    left.plot([0, top], [0, top], color="black", lw=1, label="1:1")
    xs = np.linspace(0, data["cams"].max(), 50)
    left.plot(xs, calibration.apply(xs, pd.DatetimeIndex([data.index[len(data) // 2]] * 50)),
              color=WHO_COLOR, lw=2, label=f"Correction ({calibration.method})")
    left.set(xlabel="CAMS PM2.5 (µg/m³)", ylabel="US Embassy PM2.5 (µg/m³)",
             title="Daily mean: model vs monitor")
    left.legend(frameon=False, fontsize=8)
    right.plot(data.index, data["embassy"], color="black", lw=1, label="US Embassy monitor")
    right.plot(data.index, data["cams"], color=ACCENT, lw=1, alpha=0.6, label="CAMS (raw)")
    right.plot(data.index, corrected, color=WHO_COLOR, lw=1, label="CAMS calibrated")
    right.set(ylabel="µg/m³", title="PM2.5 daily mean")
    right.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, path)
