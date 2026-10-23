"""Kos the leopard: the app's mascot. Its fur, health, scene and animation follow the AQI.

| Level          | Advice                    | Kos                                             |
|----------------|---------------------------|-------------------------------------------------|
| good           | enjoy the outdoors        | golden, outside in the sun, jumping for joy      |
| moderate       | fine for most people      | warm beige, outside, calm, light haze            |
| usg            | sensitive groups: go easy | duller fur, worried, runny nose, sweating        |
| unhealthy      | wear an FFP2 mask         | greyish, sick, FFP2 mask, coughing               |
| very_unhealthy | stay indoors              | pale, at home behind a window, mask, blanket     |
| hazardous      | health emergency          | zombie: green-grey, blank eyes, stitches, flies  |
"""

from __future__ import annotations

INK = "#2b1d14"

# fur, spots, muzzle, scene background, eye colour, description (for screen readers)
LEVELS = {
    "good": ("#f4b23e", "#7a4a12", "#fff4dc", "#bfe8ff", "#9ccc3a",
             "good air: enjoying the outdoors"),
    "moderate": ("#e9c06e", "#86602a", "#fff6e2", "#e4ecd2", "#b6c84a",
                 "moderate air: calm outside"),
    "usg": ("#d8a370", "#8b5e36", "#fbecd8", "#f3dfc6", "#c9b04a",
            "unhealthy for sensitive groups: worried, runny nose"),
    "unhealthy": ("#bfa58c", "#6f5a46", "#f2e7da", "#e6d7d3", "#b5a24c",
                  "unhealthy air: sick, wearing a mask"),
    "very_unhealthy": ("#b3a3b8", "#6c5a73", "#efe7f0", "#d9cfe0", "#a99bb0",
                       "very unhealthy air: staying at home with a mask"),
    "hazardous": ("#8fa585", "#4d5e47", "#c9d6c0", "#3b3640", "#e8f0d8",
                  "hazardous air: zombie"),
}

# Body animation, and the particles in the air: (count, colour, seconds per drift)
MOTION = {
    "good": ("kos-hop 1.3s ease-in-out infinite", (0, "#000", 0)),
    "moderate": ("kos-bob 4s ease-in-out infinite", (5, "#c9b46a", 9)),
    "usg": ("kos-fidget 1.4s ease-in-out infinite", (9, "#c98a3c", 7)),
    "unhealthy": ("kos-cough 2.6s ease-in-out infinite", (14, "#8a8f94", 6)),
    "very_unhealthy": ("kos-shiver 0.25s linear infinite", (0, "#000", 0)),
    "hazardous": ("kos-lurch 3.2s ease-in-out infinite", (26, "#5b6b4f", 4)),
}

_CSS = """
.kos .kos-body { transform-box: view-box; transform-origin: 80px 170px; }
.kos .kos-fill { transform-box: fill-box; transform-origin: center; }
.kos .kos-eyes { animation: kos-blink 4.5s infinite; }
.kos .kos-sun { animation: kos-spin 12s linear infinite; }
.kos .kos-sparkle { animation: kos-twinkle 1.6s ease-in-out infinite; }
.kos .kos-late { animation-delay: -0.8s; }
.kos .kos-drop { animation: kos-drop 2s ease-in infinite; }
.kos .kos-drip { animation: kos-drip 2.4s ease-in infinite; }
.kos .kos-puff { animation: kos-puff 2.6s ease-out infinite; }
.kos .kos-thermo { animation: kos-wag 1.2s ease-in-out infinite; }
.kos .kos-fly { animation: kos-orbit 2.2s linear infinite; }
.kos .kos-dust { animation: kos-drift linear infinite; }
.kos .kos-smog { animation: kos-smog 6s ease-in-out infinite; }
@keyframes kos-hop { 0%, 100% { transform: translateY(0) scale(1, 1); }
  15% { transform: translateY(0) scale(1.05, 0.94); }
  45% { transform: translateY(-14px) scale(0.97, 1.04); }
  75% { transform: translateY(0) scale(1.04, 0.96); } }
@keyframes kos-bob { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-4px); } }
@keyframes kos-fidget { 0%, 100% { transform: rotate(0); } 25% { transform: rotate(-2.5deg); }
  50% { transform: rotate(2deg); } 75% { transform: rotate(-1deg); } }
@keyframes kos-cough { 0%, 55%, 100% { transform: translateY(0) scale(1); }
  60%, 72% { transform: translateY(4px) scale(1.04, 0.94); } 66%, 78% { transform: none; } }
@keyframes kos-shiver { 0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-1.2px); }
  75% { transform: translateX(1.2px); } }
@keyframes kos-lurch { 0%, 100% { transform: rotate(-6deg) translateX(-3px); }
  50% { transform: rotate(7deg) translateX(3px); } }
@keyframes kos-blink { 0%, 92%, 100% { transform: scaleY(1); } 95% { transform: scaleY(0.1); } }
@keyframes kos-spin { to { transform: rotate(360deg); } }
@keyframes kos-twinkle { 0%, 100% { opacity: 0.2; transform: scale(0.6); }
  50% { opacity: 1; transform: scale(1); } }
@keyframes kos-drop { 0% { transform: translateY(0); opacity: 0; } 15% { opacity: 1; }
  100% { transform: translateY(16px); opacity: 0; } }
@keyframes kos-drip { 0%, 30% { transform: translateY(0) scaleY(0.4); opacity: 0.9; }
  80% { transform: translateY(6px) scaleY(1.2); opacity: 0.9; }
  100% { transform: translateY(14px); opacity: 0; } }
@keyframes kos-puff { 0% { transform: translate(0, 0) scale(0.5); opacity: 0; }
  30% { opacity: 0.85; } 100% { transform: translate(-18px, -6px) scale(1.6); opacity: 0; } }
@keyframes kos-wag { 0%, 100% { transform: rotate(-6deg); } 50% { transform: rotate(6deg); } }
@keyframes kos-orbit { from { transform: rotate(0deg) translateX(14px) rotate(0deg); }
  to { transform: rotate(360deg) translateX(14px) rotate(-360deg); } }
@keyframes kos-drift { 0% { transform: translate(0, 0); opacity: 0; }
  20% { opacity: var(--o); } 80% { opacity: var(--o); }
  100% { transform: translate(var(--dx), var(--dy)); opacity: 0; } }
@keyframes kos-smog { 0%, 100% { transform: translateX(-6px); }
  50% { transform: translateX(6px); } }
@media (prefers-reduced-motion: reduce) { .kos *, .kos { animation: none !important; } }
"""


# ── Scenes ──────────────────────────────────────────────────────────────────────────────────

def _outdoor(sky: str, sunny: bool, haze: bool, backdrop: bool = True) -> str:
    sun = ('<g class="kos-sun kos-fill"><circle cx="128" cy="34" r="13" fill="#ffd34d"/>'
           + "".join(f'<rect x="126.5" y="12" width="3" height="7" rx="1.5" fill="#ffd34d" '
                     f'transform="rotate({a} 128 34)"/>' for a in range(0, 360, 45))
           + "</g>") if sunny else ('<circle cx="128" cy="34" r="12" fill="#f6e7a8"/>'
                                    '<ellipse cx="118" cy="38" rx="16" ry="7" fill="#fff" '
                                    'opacity="0.9"/>')
    hills = ('<path d="M0 150 Q40 128 80 140 T160 136 V170 H0z" fill="#7cc35a"/>'
             '<path d="M0 158 Q50 144 100 154 T160 150 V170 H0z" fill="#5aa845"/>')
    mist = ('<g class="kos-smog"><rect x="-10" y="60" width="180" height="40" rx="20" '
            'fill="#fff" opacity="0.35"/></g>' if haze else "")
    if not backdrop:  # the card behind provides the background: keep the sun and a soft haze
        soft = ('<g class="kos-smog"><ellipse cx="80" cy="78" rx="62" ry="16" fill="#fff" '
                'opacity="0.3"/></g>' if haze else "")
        return f"{sun}{soft}"
    return (f'<rect x="0" y="0" width="160" height="170" rx="26" fill="{sky}"/>'
            f"{sun}{hills}{mist}")


def _home(sky: str, backdrop: bool = True) -> str:
    """Kos seen through its window: frame, curtains, smog outside."""
    frame = ('<path d="M80 14 V126 M14 70 H146" stroke="#8b6b4a" stroke-width="5"/>'
             '<rect x="14" y="14" width="132" height="112" rx="6" fill="none" stroke="#8b6b4a" '
             'stroke-width="7"/>'
             '<path d="M14 14 Q30 70 18 126 H8 V14z M146 14 Q130 70 142 126 H152 V14z" '
             'fill="#d9776f"/>')
    if not backdrop:  # just the window around Kos: it is at home
        return frame
    return (f'<rect x="0" y="0" width="160" height="170" rx="26" fill="#f4ece2"/>'
            f'<rect x="14" y="14" width="132" height="112" rx="6" fill="{sky}"/>'
            '<g class="kos-smog"><ellipse cx="50" cy="50" rx="40" ry="14" fill="#9c8fa6" '
            'opacity="0.55"/><ellipse cx="112" cy="70" rx="36" ry="12" fill="#9c8fa6" '
            'opacity="0.45"/></g>'
            '<path d="M80 14 V126 M14 70 H146" stroke="#8b6b4a" stroke-width="5"/>'
            '<rect x="14" y="14" width="132" height="112" rx="6" fill="none" stroke="#8b6b4a" '
            'stroke-width="7"/>'
            '<path d="M14 14 Q30 70 18 126 H8 V14z M146 14 Q130 70 142 126 H152 V14z" '
            'fill="#d9776f"/>'
            '<rect x="0" y="126" width="160" height="44" fill="#c9a27a"/>')


def _zombie_scene(sky: str, backdrop: bool = True) -> str:
    moon_and_smog = ('<circle cx="124" cy="32" r="14" fill="#c9d8a8" opacity="0.8"/>'
                     '<g class="kos-smog"><ellipse cx="40" cy="60" rx="50" ry="16" '
                     'fill="#5b6b4f" opacity="0.45"/><ellipse cx="120" cy="90" rx="50" ry="16" '
                     'fill="#5b6b4f" opacity="0.35"/></g>')
    if not backdrop:
        return moon_and_smog
    return (f'<rect x="0" y="0" width="160" height="170" rx="26" fill="{sky}"/>'
            + moon_and_smog
            + '<path d="M0 150 Q40 140 80 146 T160 142 V170 H0z" fill="#2a2630"/>'
            '<path d="M18 150 v-14 h8 v14 M22 140 h-6 M22 140 h6" stroke="#4a4552" '
            'stroke-width="3"/>')


# ── The leopard ─────────────────────────────────────────────────────────────────────────────

def _spots(color: str, fur: str) -> str:
    """Leopard rosettes (four dark blotches around a tan centre) and small forehead spots."""
    rosettes = [(48, 70, 1.0), (112, 70, 1.0), (40, 96, 0.9), (120, 96, 0.9), (60, 128, 0.8),
                (100, 128, 0.8), (36, 156, 1.1), (124, 156, 1.1), (80, 160, 1.0),
                (58, 148, 0.8), (102, 148, 0.8)]
    out = []
    for x, y, k in rosettes:
        out.append(f'<circle cx="{x}" cy="{y}" r="{4 * k:.1f}" fill="{color}" opacity="0.35"/>')
        for dx, dy, rot in ((-5, -2, -30), (2, -5, 20), (5, 3, 60), (-2, 5, -60)):
            out.append(f'<ellipse cx="{x + dx * k:.1f}" cy="{y + dy * k:.1f}" '
                       f'rx="{3.2 * k:.1f}" ry="{1.9 * k:.1f}" fill="{color}" '
                       f'transform="rotate({rot} {x + dx * k:.1f} {y + dy * k:.1f})"/>')
    dots = [(70, 54), (80, 50), (90, 54), (64, 64), (96, 64), (75, 62), (85, 62), (80, 70)]
    out += [f'<circle cx="{x}" cy="{y}" r="2.1" fill="{color}"/>' for x, y in dots]
    return "".join(out)


def _eyes(level: str, iris: str) -> str:
    if level == "good":  # happy, closed eyes
        return (f'<path d="M54 86 q8 -9 16 0 M90 86 q8 -9 16 0" stroke="{INK}" '
                'stroke-width="4" fill="none" stroke-linecap="round"/>')
    if level == "hazardous":  # zombie: blank eyes, dark rings, one droopy
        return ('<ellipse cx="62" cy="86" rx="11" ry="10" fill="#3f4a38"/>'
                '<ellipse cx="98" cy="88" rx="11" ry="8" fill="#3f4a38"/>'
                '<circle cx="62" cy="86" r="7.5" fill="#f2f6e6"/>'
                '<ellipse cx="98" cy="89" rx="7.5" ry="5" fill="#f2f6e6"/>'
                '<circle cx="64" cy="87" r="1.6" fill="#8a9a7a"/>')
    lids = {"unhealthy": 3.5, "very_unhealthy": 4.5}.get(level, 0)
    eyes = (f'<ellipse cx="62" cy="86" rx="9" ry="8" fill="{iris}"/>'
            f'<ellipse cx="98" cy="86" rx="9" ry="8" fill="{iris}"/>'
            f'<ellipse cx="62" cy="86" rx="2.4" ry="6.5" fill="{INK}"/>'
            f'<ellipse cx="98" cy="86" rx="2.4" ry="6.5" fill="{INK}"/>'
            '<circle cx="65" cy="83" r="1.8" fill="#fff"/><circle cx="101" cy="83" r="1.8" '
            'fill="#fff"/>')
    if lids:  # heavy, sick eyelids
        eyes += (f'<path d="M52 {86 - lids} Q62 {78 - lids / 2} 72 {86 - lids} V78 H52z '
                 f'M88 {86 - lids} Q98 {78 - lids / 2} 108 {86 - lids} V78 H88z" fill="{{fur}}"/>')
    if level in ("usg", "unhealthy", "very_unhealthy"):  # worried eyebrows, inner ends raised
        eyes += (f'<path d="M52 74 l14 -5 M108 74 l-14 -5" stroke="{INK}" stroke-width="3" '
                 'stroke-linecap="round"/>')
    return eyes if level in ("unhealthy", "very_unhealthy", "usg") else \
        f'<g class="kos-eyes kos-fill">{eyes}</g>'


def _mouth(level: str) -> str:
    if level == "good":  # big open smile with tongue
        return (f'<path d="M66 110 Q80 128 94 110 Z" fill="#7a2a1e" stroke="{INK}" '
                'stroke-width="2.5" stroke-linejoin="round"/><path d="M73 118 Q80 126 87 118 '
                'Q80 121 73 118z" fill="#ff8a8a"/>')
    if level == "moderate":
        return (f'<path d="M70 112 Q75 117 80 112 Q85 117 90 112" stroke="{INK}" '
                'stroke-width="2.6" fill="none" stroke-linecap="round"/>')
    if level == "usg":
        return (f'<path d="M70 116 Q75 112 80 115 Q85 118 90 114" stroke="{INK}" '
                'stroke-width="2.6" fill="none" stroke-linecap="round"/>')
    if level == "hazardous":  # jagged zombie mouth with green drool
        return (f'<path d="M64 112 L70 118 L76 112 L82 119 L88 112 L95 117" stroke="{INK}" '
                'stroke-width="2.8" fill="none" stroke-linejoin="round"/>'
                '<g class="kos-drip kos-fill"><path d="M88 116 q3 8 0 12 q-3 -4 0 -12z" '
                'fill="#9be36b"/></g>')
    return ""  # covered by the mask


def _nose(level: str) -> str:
    nose = (f'<path d="M73 100 Q80 96 87 100 Q84 106 80 107 Q76 106 73 100z" fill="{INK}"/>'
            f'<path d="M80 107 v4" stroke="{INK}" stroke-width="2.4"/>')
    if level == "usg":  # runny nose
        nose += ('<g class="kos-drip kos-fill"><path d="M84 106 q3 6 0 9 q-3 -3 0 -9z" '
                 'fill="#9fd8f5"/></g>')
    return nose


def _whiskers(level: str) -> str:
    droop = 6 if level in ("unhealthy", "very_unhealthy", "hazardous") else 0
    return (f'<path d="M58 106 l-22 {-4 + droop} M58 110 l-22 {3 + droop} M102 106 l22 '
            f'{-4 + droop} M102 110 l22 {3 + droop}" stroke="{INK}" stroke-width="1.6" '
            'stroke-linecap="round" opacity="0.7"/>')


def _accessories(level: str) -> str:
    parts = []
    if level == "good":  # sparkles of fresh air
        parts.append('<g class="kos-sparkle kos-fill"><path d="M26 40 l3 7 l7 3 l-7 3 l-3 7 '
                     'l-3 -7 l-7 -3 l7 -3z" fill="#fff"/></g><g class="kos-sparkle kos-fill '
                     'kos-late"><path d="M138 70 l2 5 l5 2 l-5 2 l-2 5 l-2 -5 l-5 -2 l5 -2z" '
                     'fill="#fff"/></g>')
    if level == "good" or level == "moderate":  # rosy cheeks
        parts.append('<circle cx="50" cy="102" r="6" fill="#ff9a8a" opacity="0.6"/>'
                     '<circle cx="110" cy="102" r="6" fill="#ff9a8a" opacity="0.6"/>')
    if level in ("usg", "unhealthy"):  # sweat drop
        parts.append('<g class="kos-drop kos-fill"><path d="M118 62 q6 9 0 13 q-6 -4 0 -13z" '
                     'fill="#7fd3f0"/></g>')
    if level in ("unhealthy", "very_unhealthy"):  # FFP2 mask and sick cheeks
        parts.append('<circle cx="46" cy="98" r="7" fill="#a6c76a" opacity="0.55"/>'
                     '<circle cx="114" cy="98" r="7" fill="#a6c76a" opacity="0.55"/>'
                     '<path d="M44 62 Q40 92 54 104 M116 62 Q120 92 106 104" stroke="#e9eef0" '
                     'stroke-width="2.4" fill="none"/>'
                     '<path d="M52 100 Q80 90 108 100 L102 122 Q80 132 58 122 Z" fill="#fff" '
                     'stroke="#cfd8dc" stroke-width="1.6"/>'
                     '<path d="M80 94 V128 M60 108 Q80 104 100 108" stroke="#d6dee2" '
                     'stroke-width="1.6" fill="none"/>')
    if level == "unhealthy":  # cough puffs
        parts.append('<g fill="#dfe4e6"><circle class="kos-puff kos-fill" cx="44" cy="118" '
                     'r="6"/><circle class="kos-puff kos-fill" style="animation-delay:-0.4s" '
                     'cx="36" cy="110" r="4.5"/></g>')
    if level == "very_unhealthy":  # blanket and thermometer
        parts.append('<path d="M20 150 Q40 128 80 134 Q120 128 140 150 V170 H20z" '
                     'fill="#6fa8dc"/><path d="M30 152 h100 M40 162 h80" stroke="#9ec6ea" '
                     'stroke-width="3"/>'
                     '<g class="kos-thermo kos-fill"><rect x="100" y="112" width="26" height="5" '
                     'rx="2.5" fill="#fff" stroke="#9aa5ab" stroke-width="1" '
                     'transform="rotate(-20 100 114)"/><circle cx="125" cy="105" r="3.5" '
                     'fill="#e53935"/></g>')
    if level == "hazardous":  # stitches, torn ear, flies
        parts.append(f'<path d="M66 56 L94 62 M70 52 l-2 8 M78 54 l-2 8 M86 56 l-2 8" '
                     f'stroke="{INK}" stroke-width="2" stroke-linecap="round"/>'
                     '<path d="M104 26 l6 8 l-8 2z" fill="#3b3640"/>'
                     + "".join(f'<g transform="translate({x} {y})"><g class="kos-fly kos-fill" '
                               f'style="animation-delay:-{d}s"><circle r="2.2" fill="#111"/>'
                               '<ellipse cx="-2" cy="-2.5" rx="2" ry="1.2" fill="#ccd" '
                               'opacity="0.8"/><ellipse cx="2" cy="-2.5" rx="2" ry="1.2" '
                               'fill="#ccd" opacity="0.8"/></g></g>'
                               for x, y, d in ((34, 46, 0), (128, 60, 0.7), (122, 120, 1.4))))
    return "".join(parts)


def _particles(count: int, color: str, duration: float, seed: int) -> str:
    """Floating pollution particles, placed pseudo-randomly but reproducibly."""
    state, dots = seed * 7919 + 13, []
    for _ in range(count):
        values = []
        for _ in range(5):
            state = (state * 1103515245 + 12345) % 2**31
            values.append(state / 2**31)
        x, y = 8 + 144 * values[0], 12 + 130 * values[1]
        if 34 < x < 126 and 30 < y < 140:  # keep the leopard clear
            x = 6 + 22 * values[0] if values[2] < 0.5 else 132 + 22 * values[0]
        radius = 1.6 + 2.4 * values[3] + (0.8 if count > 15 else 0)
        dx, dy = (values[2] - 0.5) * 30, -10 - 25 * values[4]
        dots.append(f'<circle class="kos-dust" cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
                    f'fill="{color}" style="--dx:{dx:.0f}px;--dy:{dy:.0f}px;'
                    f'--o:{0.45 + 0.4 * values[3]:.2f};animation-duration:'
                    f'{duration * (0.7 + 0.6 * values[4]):.1f}s;'
                    f'animation-delay:-{duration * values[1]:.1f}s"/>')
    return "".join(dots)


def mascot(category: str, size: int = 120, standalone: bool = False,
           backdrop: bool = True) -> str:
    """Kos the leopard for one AQI level, as an animated inline SVG (or a standalone .svg).

    With `backdrop=False` (inside the app's cards) the scene's background is left out, so Kos
    blends into the card behind it, and the shoulders fade out instead of being cut off.
    """
    category = category if category in LEVELS else "moderate"
    fur, spots, muzzle, sky, iris, description = LEVELS[category]
    body_animation, (count, dust, speed) = MOTION[category]
    detailed = size >= 60  # small icons keep the leopard and its motion, not the scene details

    if category in ("good", "moderate", "usg", "unhealthy"):
        scene = _outdoor(sky, sunny=category == "good", haze=category != "good",
                         backdrop=backdrop)
    elif category == "very_unhealthy":
        scene = _home(sky, backdrop)
    else:
        scene = _zombie_scene(sky, backdrop)
    if not detailed:
        scene = f'<rect width="160" height="170" rx="26" fill="{sky}"/>' if backdrop else ""
    air = _particles(count, dust, speed, list(LEVELS).index(category) + 1) if detailed else ""

    ear_inner = "#e8a0a0" if category != "hazardous" else "#7d8f74"
    torn = category == "hazardous"
    leopard = (
        # tail curling up on the side, with a dark ringed tip
        f'<path d="M126 168 Q150 150 146 126 Q143 112 152 104" stroke="{fur}" '
        'stroke-width="9" fill="none" stroke-linecap="round"/>'
        f'<path d="M146 124 l6 -3 M144 138 l6 -1" stroke="{spots}" stroke-width="3" '
        'stroke-linecap="round"/><circle cx="152" cy="104" r="5" fill="' + spots + '"/>'
        f'<ellipse cx="80" cy="168" rx="48" ry="30" fill="{fur}"/>'           # shoulders
        # ears: small and set wide, like a leopard's
        f'<path d="M36 52 Q38 30 56 36 Q50 46 44 58z" fill="{fur}"/>'
        f'<path d="M124 52 Q122 30 104 36 Q110 46 116 58z" fill="{fur}"/>'
        f'<path d="M41 50 Q43 38 52 40 Q48 46 45 53z" fill="{ear_inner}"/>'
        + ('' if torn else f'<path d="M119 50 Q117 38 108 40 Q112 46 115 53z" '
           f'fill="{ear_inner}"/>')
        + f'<ellipse cx="80" cy="90" rx="54" ry="45" fill="{fur}"/>'          # head
        + f'<path d="M30 98 Q34 118 52 124 M130 98 Q126 118 108 124" stroke="{fur}" '
        'stroke-width="10" fill="none" stroke-linecap="round"/>'              # cheek ruff
        + _spots(spots, fur)
        + f'<ellipse cx="80" cy="110" rx="26" ry="18" fill="{muzzle}"/>'
        + _eyes(category, iris).replace("{fur}", fur)
        + _nose(category) + _mouth(category) + _whiskers(category) + _accessories(category)
    )
    namespace = ' xmlns="http://www.w3.org/2000/svg"' if standalone else ""
    return f"""<svg{namespace} class="kos kos-{category}" width="{size}"
      height="{size * 170 / 160:.0f}"
      viewBox="0 0 160 170" role="img" aria-label="Kos the leopard, {description}">
      <style>{_CSS}.kos-{category} .kos-body {{ animation: {body_animation}; }}</style>
      <defs><clipPath id="kos-frame-{category}"><rect width="160" height="170" rx="26"/>
      </clipPath>
      <linearGradient id="kos-fade-{category}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0.82" stop-color="#fff"/><stop offset="1" stop-color="#000"/>
      </linearGradient>
      <mask id="kos-mask-{category}" maskUnits="userSpaceOnUse" x="0" y="0" width="160"
            height="170"><rect width="160" height="170" fill="url(#kos-fade-{category})"/></mask>
      </defs>
      <g clip-path="url(#kos-frame-{category})">
        {scene}
        {air}
        <g {'' if backdrop else f'mask="url(#kos-mask-{category})"'}>
          <g class="kos-body">{leopard}</g>
        </g>
      </g>
    </svg>"""
