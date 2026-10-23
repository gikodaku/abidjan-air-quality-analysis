"""Air Quality Index (US EPA, 2024 revision) and health advice."""

from __future__ import annotations

import math
from dataclasses import dataclass

from aaqa.pollutants import POLLUTANTS, Pollutant, get_pollutant

# WHO 2021 guidelines (µg/m³), kept for direct access
WHO_PM25_24H = POLLUTANTS["pm2_5"].who_limit
WHO_PM25_ANNUAL = 5.0
WHO_PM10_ANNUAL = 15.0


@dataclass(frozen=True)
class AqiCategory:
    key: str
    label: str
    color: str
    advice: str
    upper: int  # upper bound of the index


CATEGORIES = [
    AqiCategory("good", "Good", "#00a84f",
                "Air quality is satisfactory. No restrictions on outdoor activities.", 50),
    AqiCategory("moderate", "Moderate", "#e6c700",
                "Acceptable. Unusually sensitive people should limit prolonged outdoor "
                "exertion.", 100),
    AqiCategory("usg", "Unhealthy for sensitive groups", "#ff7e00",
                "Children, older adults, people with asthma and pregnant women: reduce intense "
                "outdoor exertion.", 150),
    AqiCategory("unhealthy", "Unhealthy", "#e8003f",
                "Everyone may feel effects. Avoid outdoor sport; wear an FFP2 mask during "
                "prolonged exposure.", 200),
    AqiCategory("very_unhealthy", "Very unhealthy", "#8f3f97",
                "Health alert: stay indoors with windows closed. Sensitive groups: no outdoor "
                "activity.", 300),
    AqiCategory("hazardous", "Hazardous", "#7e0023",
                "Health emergency: everyone should avoid outdoor exposure.", 500),
]


def concentration_to_aqi(pollutant: Pollutant | str, ug_m3: float) -> int:
    """Convert a concentration (µg/m³, on the pollutant's statistic) to the US AQI."""
    if isinstance(pollutant, str):
        pollutant = get_pollutant(pollutant)
    if ug_m3 is None or math.isnan(ug_m3):
        raise ValueError("missing concentration")
    step = pollutant.truncate
    # EPA truncation; the small epsilon keeps 9.0 from becoming 8.9 through rounding error
    c = max(0.0, math.floor(pollutant.to_aqi_unit(ug_m3) / step + 1e-9) * step)
    for c_lo, c_hi, i_lo, i_hi in pollutant.breakpoints:
        if c <= c_hi + 1e-12:
            return _round_half_up((i_hi - i_lo) / (c_hi - c_lo) * (c - c_lo) + i_lo)
    # Beyond the table (rare): extend the last band, capped at 500
    c_lo, c_hi, i_lo, i_hi = pollutant.breakpoints[-1]
    return min(500, _round_half_up((i_hi - i_lo) / (c_hi - c_lo) * (c - c_lo) + i_lo))


def _round_half_up(value: float) -> int:
    # Python's round() sends 118.5 to 118 (banker's rounding); the EPA rounds half up
    return math.floor(value + 0.5)


def pm25_to_aqi(concentration: float) -> int:
    """US AQI of a 24-hour mean PM2.5 concentration (µg/m³)."""
    return concentration_to_aqi("pm2_5", concentration)


def aqi_category(aqi: int) -> AqiCategory:
    for category in CATEGORIES:
        if aqi <= category.upper:
            return category
    return CATEGORIES[-1]


def pm25_category(concentration: float) -> AqiCategory:
    return aqi_category(pm25_to_aqi(concentration))


def describe(pollutant: Pollutant | str, ug_m3: float) -> dict:
    """JSON-ready summary of a value: index, category, WHO exceedance.

    `exceeds_who` is None when the WHO guideline is defined on a different statistic from the
    one forecast (NO2 and SO2: 24-hour guideline, hourly maximum forecast).
    """
    if isinstance(pollutant, str):
        pollutant = get_pollutant(pollutant)
    aqi = concentration_to_aqi(pollutant, ug_m3)
    category = aqi_category(aqi)
    return {
        "pollutant": pollutant.key,
        "name": pollutant.name,
        "statistic": pollutant.statistic,
        "value": round(float(ug_m3), 1),
        "unit": "µg/m³",
        "aqi": aqi,
        "category": category.key,
        "label": category.label,
        "color": category.color,
        "exceeds_who": bool(ug_m3 > pollutant.who_limit) if pollutant.who_on_target else None,
    }


def describe_pm25(concentration: float) -> dict:
    return describe("pm2_5", concentration)


def overall(descriptions: list[dict]) -> dict:
    """Overall index = index of the worst pollutant (EPA method)."""
    worst = max(descriptions, key=lambda d: d["aqi"])
    category = aqi_category(worst["aqi"])
    return {
        "aqi": worst["aqi"],
        "category": category.key,
        "label": category.label,
        "color": category.color,
        "advice": category.advice,
        "dominant_pollutant": worst["pollutant"],
        "dominant_name": worst["name"],
    }
