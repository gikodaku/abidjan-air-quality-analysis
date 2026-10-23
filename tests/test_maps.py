import json
import sys
from pathlib import Path

import pytest

from aaqa.config import Location
from aaqa.data import fetch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
import maps  # noqa: E402

GRID = {"requested": {"latitude": 5.36, "longitude": -4.01},
        "air_quality": {"latitude": 5.4, "longitude": -4.0, "model": "CAMS",
                        "resolution_deg": 0.4},
        "weather": {"latitude": 5.378, "longitude": -4.023, "model": "ERA5"}}


def test_sampling_map_draws_cell_points_and_districts():
    fig = maps.sampling_map(GRID, "#00a84f", "rgba(0,168,79,0.2)", "Good")
    cell, districts, weather, centre, requested = fig.data
    assert min(cell.lat) == pytest.approx(5.2) and max(cell.lat) == pytest.approx(5.6)
    assert min(cell.lon) == pytest.approx(-4.2) and max(cell.lon) == pytest.approx(-3.8)
    assert "~44 km" in cell.name and "Good" in cell.name
    assert "Yopougon" in districts.text and len(districts.text) == len(maps.DISTRICTS)
    assert (centre.lat[0], centre.lon[0]) == (5.4, -4.0)
    assert fig.layout.map.style == "carto-positron"
    assert maps.sampling_map(GRID, "#1ea0d8", "rgba(0,0,0,0)", "x", dark=True).layout.map.style \
        == "carto-darkmatter"


def test_every_district_lies_in_the_cams_cell():
    half = GRID["air_quality"]["resolution_deg"] / 2
    for lat, lon in maps.DISTRICTS.values():
        assert abs(lat - 5.4) <= half and abs(lon + 4.0) <= half


def test_load_grid_falls_back_to_config(tmp_path):
    fallback = maps.load_grid(tmp_path / "missing.json", 5.36, -4.01)
    assert fallback["air_quality"]["latitude"] == 5.36
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(GRID))
    assert maps.load_grid(path, 0, 0) == GRID


def test_fetch_grid_points_reads_snapped_coordinates(monkeypatch):
    responses = iter([{"latitude": 5.4000015, "longitude": -4.0},
                      {"latitude": 5.377856, "longitude": -4.0231934}])
    monkeypatch.setattr(fetch, "_get", lambda url, params, **kw: next(responses))
    grid = fetch.fetch_grid_points(Location())
    assert grid["air_quality"]["latitude"] == pytest.approx(5.4)
    assert grid["weather"]["longitude"] == pytest.approx(-4.023, abs=1e-3)
