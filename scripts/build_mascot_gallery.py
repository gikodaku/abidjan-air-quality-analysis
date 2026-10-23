"""Build docs/mascot/: one animated SVG per AQI level and a gallery page (`make mascot`)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))

from mascot import LEVELS, mascot  # noqa: E402

from aaqa.aqi import CATEGORIES  # noqa: E402

OUT = ROOT / "docs" / "mascot"
RANGES = {"good": "0–50", "moderate": "51–100", "usg": "101–150", "unhealthy": "151–200",
          "very_unhealthy": "201–300", "hazardous": "301–500"}
SCENES = {
    "good": "Outdoors in the sun, jumping for joy: enjoy the fresh air.",
    "moderate": "Outdoors, calm, a light haze: fine for almost everyone.",
    "usg": "Worried, runny nose, sweating: sensitive people should take it easy.",
    "unhealthy": "Sick and coughing behind an FFP2 mask: wear one if you stay out long.",
    "very_unhealthy": "At home behind the window, mask, blanket and thermometer: stay indoors.",
    "hazardous": "A zombie in dark smog with flies around: health emergency.",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cards = []
    for category in CATEGORIES:
        svg = mascot(category.key, 220, standalone=True)
        (OUT / f"kos-{category.key}.svg").write_text(svg)
        cards.append(f"""
        <article>
          {mascot(category.key, 220)}
          <h2><span class="dot" style="background:{category.color}"></span>{category.label}</h2>
          <p class="range">AQI {RANGES[category.key]}</p>
          <p>{SCENES[category.key]}</p>
          <p class="advice">Advice: {category.advice}</p>
          <p class="file"><a href="kos-{category.key}.svg">kos-{category.key}.svg</a></p>
        </article>""")
    icons = "".join(mascot(key, 48) for key in LEVELS)
    (OUT / "index.html").write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kos the leopard</title>
<style>
  body {{ margin: 0; padding: 24px; background: #f3f6f6; color: #13262b;
         font-family: Montserrat, system-ui, sans-serif; }}
  h1 {{ margin: 0 0 4px; }} .lead {{ color: #3f5358; margin: 0 0 20px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
           gap: 18px; }}
  article {{ background: #fff; border-radius: 18px; padding: 18px; text-align: center;
             box-shadow: 0 4px 18px rgba(16,40,44,0.07); }}
  h2 {{ font-size: 1.15rem; margin: 10px 0 2px; display: flex; gap: 8px;
        justify-content: center; align-items: center; }}
  .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
  .range {{ color: #5f7377; margin: 0 0 8px; font-weight: 600; }}
  p {{ margin: 6px 0; line-height: 1.45; }} .advice {{ color: #3f5358; font-size: 0.9rem; }}
  .file {{ font-size: 0.8rem; }} .icons {{ display: flex; gap: 18px; margin: 24px 0 8px; }}
</style></head><body>
<h1>Kos the leopard</h1>
<p class="lead">The AAQA mascot at each AQI level. Its fur, health, scene and animation follow
the air quality and the health advice. Animations stop when the system's "reduce motion" setting
is on.</p>
<div class="grid">{''.join(cards)}</div>
<h2 style="justify-content:flex-start">Small icons (forecast cards)</h2>
<div class="icons">{icons}</div>
</body></html>
""")
    print(f"Gallery written to {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
