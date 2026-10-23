import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
import illustrations  # noqa: E402
import maps  # noqa: E402
import mascot  # noqa: E402

LEVELS = list(mascot.LEVELS)


def test_six_distinct_leopards():
    assert LEVELS == ["good", "moderate", "usg", "unhealthy", "very_unhealthy", "hazardous"]
    assert len({level[0] for level in mascot.LEVELS.values()}) == 6  # one fur colour each
    animations = {mascot.MOTION[level][0].split()[0] for level in LEVELS}
    assert len(animations) == 6  # hop, bob, fidget, cough, shiver, lurch


@pytest.mark.parametrize("level, expected", [
    ("good", ["enjoying the outdoors", "kos-sun", "kos-sparkle", "kos-hop"]),
    ("usg", ["runny nose", "kos-drip"]),
    ("unhealthy", ["wearing a mask", "kos-puff", "kos-cough"]),
    ("very_unhealthy", ["staying at home", "kos-thermo", "kos-shiver"]),
    ("hazardous", ["zombie", "kos-fly", "kos-lurch"]),
])
def test_scene_matches_the_advice(level, expected):
    svg = mascot.mascot(level)
    for text in expected:
        assert text in svg, (level, text)


def test_particles_grow_with_pollution():
    counts = [mascot.mascot(level).count('class="kos-dust"') for level in LEVELS]
    assert counts[0] == 0 and counts[-1] == 26 and counts[3] > counts[1]
    assert 'class="kos-dust"' not in mascot.mascot("hazardous", 40)  # small icon


def test_svg_details():
    svg = mascot.mascot("good", standalone=True)
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "prefers-reduced-motion" in svg and 'id="kos-frame-good"' in svg
    assert "Kos the leopard" in svg


def test_gallery_builds(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_mascot_gallery as gallery

    monkeypatch.setattr(gallery, "OUT", tmp_path)
    gallery.main()
    assert {p.name for p in tmp_path.iterdir()} == {"index.html"} | {
        f"kos-{level}.svg" for level in LEVELS}
    page = (tmp_path / "index.html").read_text()
    assert "zombie" in page and "FFP2" in page


@pytest.mark.parametrize("city, landmark", [("abidjan", "Abidjan"), ("paris", "Paris"),
                                            ("new_york", "New York"),
                                            ("new_delhi", "New Delhi")])
def test_city_skylines(city, landmark):
    assert f"{landmark} skyline" in illustrations.skyline("#000", 0.2, city)


def test_map_without_districts():
    grid = maps.load_grid(Path("/nonexistent"), 28.6, 77.2)
    names = [trace.name for trace in maps.sampling_map(grid, "#c8102e", "rgba(0,0,0,0)",
                                                       "x", districts={}).data]
    assert "Districts" not in names and len(names) == 4
