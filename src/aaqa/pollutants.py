"""Pollutants covered: forecast daily statistic, WHO guideline and AQI breakpoints.

Each pollutant is forecast on the statistic used by the US EPA index (24-hour mean for
particles, maximum 8-hour mean for ozone and CO, maximum hourly value for NO2 and SO2).
The WHO 2021 guideline is compared with the statistic it is defined on (`who_column`).
"""

from __future__ import annotations

from dataclasses import dataclass

# Molar volume at 25 °C and 1 atm (L/mol), to convert µg/m³ to ppb
MOLAR_VOLUME = 24.45


@dataclass(frozen=True)
class Pollutant:
    key: str
    name: str
    target: str               # forecast daily column
    statistic: str            # human-readable statistic
    who_column: str | None    # column compared with the WHO guideline
    who_limit: float | None   # µg/m³
    who_label: str
    # EPA breakpoints: (C_low, C_high, I_low, I_high) in `aqi_unit`
    breakpoints: tuple[tuple[float, float, int, int], ...]
    aqi_unit: str
    truncate: float           # EPA truncation step, in `aqi_unit`
    molar_mass: float | None = None  # g/mol, for gases
    sources: str = ""

    def to_aqi_unit(self, ug_m3: float) -> float:
        """Convert a concentration in µg/m³ to the unit of the EPA breakpoints."""
        if self.aqi_unit == "µg/m³":
            return ug_m3
        ppb = ug_m3 * MOLAR_VOLUME / self.molar_mass
        return ppb / 1000 if self.aqi_unit == "ppm" else ppb

    @property
    def who_on_target(self) -> bool:
        return self.who_column == self.target and self.who_limit is not None


POLLUTANTS: dict[str, Pollutant] = {p.key: p for p in [
    Pollutant(
        key="pm2_5", name="PM2.5", target="pm2_5_mean", statistic="24-h mean",
        who_column="pm2_5_mean", who_limit=15.0, who_label="15 µg/m³ (24-h)",
        breakpoints=((0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
                     (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500)),
        aqi_unit="µg/m³", truncate=0.1,
        sources="combustion (wood, charcoal, waste, traffic), Saharan dust",
    ),
    Pollutant(
        key="pm10", name="PM10", target="pm10_mean", statistic="24-h mean",
        who_column="pm10_mean", who_limit=45.0, who_label="45 µg/m³ (24-h)",
        breakpoints=((0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
                     (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)),
        aqi_unit="µg/m³", truncate=1,
        sources="Saharan and road dust, construction, combustion",
    ),
    Pollutant(
        key="ozone", name="O3", target="ozone_max8h", statistic="max 8-h mean",
        who_column="ozone_max8h", who_limit=100.0, who_label="100 µg/m³ (8-h peak)",
        breakpoints=((0.0, 0.054, 0, 50), (0.055, 0.070, 51, 100), (0.071, 0.085, 101, 150),
                     (0.086, 0.105, 151, 200), (0.106, 0.200, 201, 300)),
        aqi_unit="ppm", truncate=0.001, molar_mass=48.00,
        sources="secondary pollutant formed in sunlight from NO2 and organic compounds",
    ),
    Pollutant(
        key="nitrogen_dioxide", name="NO2", target="nitrogen_dioxide_max",
        statistic="max hourly",
        who_column="nitrogen_dioxide_mean", who_limit=25.0, who_label="25 µg/m³ (24-h)",
        breakpoints=((0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
                     (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500)),
        aqi_unit="ppb", truncate=1, molar_mass=46.01,
        sources="road traffic, diesel generators, industry",
    ),
    Pollutant(
        key="sulphur_dioxide", name="SO2", target="sulphur_dioxide_max",
        statistic="max hourly",
        who_column="sulphur_dioxide_mean", who_limit=40.0, who_label="40 µg/m³ (24-h)",
        breakpoints=((0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
                     (186, 304, 151, 200)),
        aqi_unit="ppb", truncate=1, molar_mass=64.07,
        sources="refinery, port, high-sulphur fuels, thermal power plants",
    ),
    Pollutant(
        key="carbon_monoxide", name="CO", target="carbon_monoxide_max8h",
        statistic="max 8-h mean",
        who_column="carbon_monoxide_max8h", who_limit=10000.0, who_label="10 mg/m³ (8-h)",
        breakpoints=((0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
                     (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 50.4, 301, 500)),
        aqi_unit="ppm", truncate=0.1, molar_mass=28.01,
        sources="incomplete combustion: traffic, wood and charcoal cooking, fires",
    ),
]}


def get_pollutant(key: str) -> Pollutant:
    try:
        return POLLUTANTS[key]
    except KeyError:
        raise ValueError(f"unknown pollutant {key!r} ({', '.join(POLLUTANTS)})") from None
