import math

import pytest

from aaqa.aqi import (
    concentration_to_aqi,
    describe,
    describe_pm25,
    overall,
    pm25_category,
    pm25_to_aqi,
)
from aaqa.pollutants import POLLUTANTS


@pytest.mark.parametrize("concentration, expected", [
    (0.0, 0),
    (9.0, 50),
    (9.1, 51),
    (35.4, 100),
    (35.5, 101),
    (55.4, 150),
    (125.4, 200),
    (225.4, 300),
    (325.4, 500),
    (1000.0, 500),
])
def test_breakpoints(concentration, expected):
    assert pm25_to_aqi(concentration) == expected


def test_truncates_to_one_decimal():
    # 9.08 is truncated to 9.0 (EPA), so it stays "Good"
    assert pm25_to_aqi(9.08) == 50


def test_negative_is_clamped():
    assert pm25_to_aqi(-3) == 0


def test_nan_rejected():
    with pytest.raises(ValueError):
        pm25_to_aqi(math.nan)


@pytest.mark.parametrize("concentration, key", [
    (5, "good"), (20, "moderate"), (40, "usg"), (80, "unhealthy"),
    (150, "very_unhealthy"), (300, "hazardous"),
])
def test_categories(concentration, key):
    assert pm25_category(concentration).key == key


def test_describe_flags_who_exceedance():
    assert describe_pm25(14.9)["exceeds_who"] is False
    info = describe_pm25(15.1)
    assert info["exceeds_who"] is True
    assert info["label"] == "Moderate"
    assert info["name"] == "PM2.5" and info["unit"] == "µg/m³"


@pytest.mark.parametrize("pollutant, ug_m3, expected", [
    ("pm10", 54, 50),
    ("pm10", 55, 51),
    ("pm10", 154, 100),
    ("pm10", 200, 123),
    ("ozone", 100, 46),             # 100 µg/m³ = 0.0509 ppm, truncated to 0.050
    ("ozone", 150, 119),            # 0.076 ppm → 118.5, rounded half up
    ("nitrogen_dioxide", 200, 102),  # 106 ppb
    ("sulphur_dioxide", 40, 21),    # 15 ppb
    ("carbon_monoxide", 10000, 93),  # 8,7 ppm
])
def test_other_pollutants(pollutant, ug_m3, expected):
    assert concentration_to_aqi(pollutant, ug_m3) == expected


def test_unit_conversion():
    assert POLLUTANTS["ozone"].to_aqi_unit(1000) == pytest.approx(1000 * 24.45 / 48.00 / 1000)
    assert POLLUTANTS["nitrogen_dioxide"].to_aqi_unit(46.01) == pytest.approx(24.45)
    assert POLLUTANTS["pm10"].to_aqi_unit(30) == 30


def test_beyond_scale_is_capped():
    assert concentration_to_aqi("sulphur_dioxide", 5000) == 500


def test_who_flag_only_on_matching_statistic():
    # SO2: WHO guideline on the 24-hour mean, forecast of the hourly maximum
    assert describe("sulphur_dioxide", 100)["exceeds_who"] is None
    assert describe("ozone", 120)["exceeds_who"] is True


def test_overall_uses_worst_pollutant():
    values = [describe("pm2_5", 10), describe("pm10", 200), describe("ozone", 60)]
    result = overall(values)
    assert result["dominant_pollutant"] == "pm10"
    assert result["aqi"] == 123
    assert result["category"] == "usg" and result["advice"]


def test_unknown_pollutant():
    with pytest.raises(ValueError, match="unknown pollutant"):
        describe("radon", 1)
