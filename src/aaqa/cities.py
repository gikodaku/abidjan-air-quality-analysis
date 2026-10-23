"""Cities the app can show. Abidjan has the full AAQA pipeline (trained models, history,
calibration on the US Embassy monitor); the others use live CAMS values and the CAMS forecast."""

from __future__ import annotations

from dataclasses import dataclass

from aaqa.config import Location


@dataclass(frozen=True)
class City:
    key: str
    name: str
    country: str
    flag: str
    latitude: float
    longitude: float
    timezone: str
    modelled: bool = False  # AAQA models, history and calibration exist for this city

    @property
    def location(self) -> Location:
        return Location(self.name, self.latitude, self.longitude, self.timezone)

    @property
    def label(self) -> str:
        return f"{self.flag} {self.name}"


CITIES = {c.key: c for c in [
    City("abidjan", "Abidjan", "Côte d'Ivoire", "🇨🇮", 5.36, -4.01, "Africa/Abidjan", True),
    City("paris", "Paris", "France", "🇫🇷", 48.8566, 2.3522, "Europe/Paris"),
    City("new_york", "New York", "United States", "🇺🇸", 40.7128, -74.0060, "America/New_York"),
    City("new_delhi", "New Delhi", "India", "🇮🇳", 28.6139, 77.2090, "Asia/Kolkata"),
]}
