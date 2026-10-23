"""Inline SVG illustration for the app: the skyline of the selected city."""

from __future__ import annotations

# Landmarks of the other cities, drawn over a row of low buildings
_LANDMARKS = {
    "paris": (  # Eiffel Tower, Sacré-Cœur
        '<path d="M290 150 L304 96 L298 96 L310 40 L313 18 L316 40 L328 96 L322 96 L336 150 '
        'L326 150 Q313 118 300 150z"/><rect x="300" y="92" width="26" height="5"/>'
        '<rect x="306" y="66" width="14" height="4"/>'
        '<path d="M430 150 V112 h60 V150z M440 112 q20 -34 40 0z M456 84 q4 -12 8 0z"/>'),
    "new_york": (  # Empire State Building, Chrysler Building
        '<path d="M300 150 V70 h10 V50 h8 V30 h4 V12 h2 V30 h4 V50 h8 V70 h10 V150z"/>'
        '<path d="M380 150 V64 h28 V150z M380 64 L394 22 L408 64z"/>'
        '<rect x="240" y="80" width="40" height="70"/><rect x="430" y="58" width="34" '
        'height="92"/><rect x="200" y="96" width="34" height="54"/>'),
    "new_delhi": (  # India Gate, Lotus Temple
        '<path d="M282 150 V60 h66 V150 h-18 V96 q-15 -20 -30 0 V150z M276 60 h78 v-10 h-78z '
        'M300 50 v-8 h30 v8z"/>'
        '<path d="M420 150 q18 -50 36 0z M440 150 q18 -62 36 0z M460 150 q18 -50 36 0z"/>'),
}


def skyline(color: str, opacity: float = 0.35, city: str = "abidjan") -> str:
    """The city seen from afar: Abidjan's Plateau across the lagoon, or a landmark for the
    other cities."""
    if city in _LANDMARKS:
        return f"""<svg class="skyline" viewBox="0 0 600 150" preserveAspectRatio="xMidYMax slice"
      role="img" aria-label="{city.replace('_', ' ').title()} skyline">
      <g fill="{color}" opacity="{opacity}">
        <rect x="0" y="118" width="190" height="32"/><rect x="40" y="104" width="50" height="46"/>
        <rect x="120" y="110" width="44" height="40"/><rect x="520" y="108" width="80"
        height="42"/><rect x="540" y="96" width="30" height="54"/>{_LANDMARKS[city]}
      </g>
    </svg>"""
    return f"""<svg class="skyline" viewBox="0 0 600 150" preserveAspectRatio="xMidYMax slice"
      role="img" aria-label="Abidjan skyline">
      <g fill="{color}" opacity="{opacity}">
        <path d="M20 150 V112 q8 -30 14 -40 q2 10 0 40 V150z"/>
        <path d="M34 72 q-18 -2 -26 8 q14 -6 26 -4 q-6 -14 -20 -16 q16 2 22 10 q4 -14 18 -16
                 q-12 8 -14 18 q14 -4 24 4 q-14 -2 -24 0z"/>
        <rect x="70" y="92" width="34" height="58"/>
        <rect x="110" y="60" width="26" height="90"/>
        <rect x="142" y="78" width="30" height="72"/>
        <path d="M180 150 L196 70 L228 70 L244 150z"/>
        <rect x="252" y="40" width="30" height="110"/>
        <rect x="288" y="68" width="22" height="82"/>
        <rect x="316" y="52" width="34" height="98"/>
        <path d="M360 150 V118 h36 V150z"/>
        <path d="M398 150 L404 104 L470 18 L462 104 L468 150z"/>
        <rect x="478" y="84" width="28" height="66"/>
        <rect x="512" y="100" width="40" height="50"/>
        <path d="M560 150 V118 q6 -26 10 -34 q2 10 0 34 V150z"/>
        <path d="M570 84 q-14 -2 -20 6 q10 -4 20 -2 q-4 -12 -16 -14 q14 2 18 8 q4 -12 16 -12
                 q-10 6 -12 14 q12 -2 20 4 q-12 -2 -20 0z"/>
      </g>
      <path d="M0 138 q25 -8 50 0 t50 0 t50 0 t50 0 t50 0 t50 0 t50 0 t50 0 t50 0 t50 0 t50 0
               t50 0 V150 H0z" fill="{color}" opacity="{opacity * 0.8:.2f}"/>
    </svg>"""
