"""Eco journey: points, levels, streaks, badges, a daily quiz and a daily eco-challenge.

Progress is stored locally as JSON (one user per installation). Every action is idempotent per
day, so refreshing the page never awards points twice.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

POINTS = {"check_in": 5, "quiz": 10, "challenge": 15, "question": 2}
MAX_QUESTION_POINTS_PER_DAY = 5  # questions to Kos that earn points each day

# Levels grow like the KOSCON leaf: (name, icon, points needed)
LEVELS = [
    ("Seed", "🌰", 0),
    ("Sprout", "🌱", 50),
    ("Sapling", "🌿", 150),
    ("Tree", "🌳", 350),
    ("Forest", "🌲", 700),
]

QUIZ = [
    ("What does PM2.5 measure?",
     ["Particles smaller than 2.5 micrometres", "Pollution at 2:5 pm", "A gas from cars"], 0,
     "PM2.5 are fine particles under 2.5 µm, about 30 times thinner than a hair."),
    ("What is the WHO 24-hour guideline for PM2.5?",
     ["5 µg/m³", "15 µg/m³", "50 µg/m³"], 1,
     "15 µg/m³ over 24 hours; the annual guideline is 5 µg/m³."),
    ("When does the harmattan bring Saharan dust to Abidjan?",
     ["June to August", "December to February", "All year round"], 1,
     "From December to February, fine-particle pollution roughly doubles."),
    ("Which pollutant forms in sunlight and peaks in the afternoon?",
     ["Ozone (O3)", "Carbon monoxide (CO)", "PM10"], 0,
     "Ground-level ozone forms from traffic gases in sunlight."),
    ("What is the main source of NO2 in cities?",
     ["Trees", "Road traffic", "The ocean"], 1,
     "NO2 mostly comes from engines, especially diesel."),
    ("An AQI of 120 means the air is…",
     ["Good", "Moderate", "Unhealthy for sensitive groups"], 2,
     "101–150 is unhealthy for children, older adults and people with asthma."),
    ("Which mask actually filters fine particles?",
     ["A cloth mask", "An FFP2 / N95 mask", "A scarf"], 1,
     "FFP2 / N95 masks filter most fine particles; cloth filters very little."),
    ("Which wind brings the cleanest air to Abidjan?",
     ["Northerly wind from the Sahara", "Southerly wind from the ocean", "No wind"], 1,
     "Ocean winds bring about half as many particles as northerly winds."),
    ("Burning waste in the open releases…",
     ["Only water vapour", "Fine particles and toxic gases", "Oxygen"], 1,
     "Open burning is a major and avoidable source of PM2.5."),
    ("How much higher than the WHO annual guideline is Abidjan's average PM2.5?",
     ["About the same", "About 3 times higher", "10 times lower"], 1,
     "About 14 µg/m³ on average, against a guideline of 5."),
    ("Which is better for the air on a short trip?",
     ["Driving alone", "Walking or a shared bus", "Idling the engine"], 1,
     "Every car trip avoided cuts NO2 and particles."),
    ("What does CO (carbon monoxide) come from?",
     ["Incomplete burning", "Plants", "Rain"], 0,
     "Charcoal stoves, traffic and fires; deadly in closed rooms."),
    ("Rain usually makes particle pollution…",
     ["Go up", "Go down", "Stay exactly the same"], 1,
     "Rain washes particles out of the air."),
    ("What does the overall AQI in this app show?",
     ["The average of all pollutants", "The worst pollutant's index", "Only PM10"], 1,
     "Like the US EPA, the overall index is that of the most penalising pollutant."),
    ("Which group is most sensitive to air pollution?",
     ["Children and people with asthma", "Only athletes", "Nobody"], 0,
     "Children, older adults, pregnant women and people with asthma or heart disease."),
]

CHALLENGES = {
    "clean_air": [
        ("🏃", "Enjoy the clean air: take a 20-minute walk or run outside."),
        ("🚌", "Leave the car: walk, cycle or take a shared bus for one trip today."),
        ("🌳", "Spend a moment near the lagoon or a green space today."),
    ],
    "polluted": [
        ("🏠", "Keep windows closed during the most polluted hours."),
        ("😷", "Wear an FFP2 mask if you must spend a long time outside."),
        ("👵", "Check on an older or asthmatic neighbour and share today's air advice."),
    ],
    "rain": [
        ("☔", "Rainy day: plan indoor activities and avoid flooded roads."),
        ("💧", "Collect rainwater for plants instead of using tap water."),
    ],
    "any": [
        ("🔥", "Don't burn waste: sort it and use a collection point."),
        ("💡", "Switch off lights and devices you are not using."),
        ("🍃", "Cook with gas or an improved stove instead of open charcoal."),
        ("📣", "Share the app's air forecast with a friend or family group."),
    ],
}

BADGES = [
    ("first_breath", "🌬️", "First breath", "Opened the app for the first time"),
    ("streak_3", "🔥", "On fire", "3-day streak"),
    ("streak_7", "🏅", "Weekly guardian", "7-day streak"),
    ("quiz_5", "🧠", "Air scholar", "5 quiz answers right"),
    ("challenge_5", "💪", "Eco hero", "5 eco-challenges done"),
    ("curious", "💬", "Curious mind", "Asked Kos 10 questions"),
    ("harmattan", "🏜️", "Harmattan ready", "Checked in during the harmattan season"),
    ("sapling", "🌿", "Growing", "Reached the Sapling level"),
]


@dataclass
class Progress:
    points: int = 0
    streak: int = 0
    best_streak: int = 0
    last_check_in: str | None = None
    quiz_correct: int = 0
    quiz_days: dict[str, bool] = field(default_factory=dict)       # date → answered right?
    challenge_days: list[str] = field(default_factory=list)
    questions: int = 0
    question_points: dict[str, int] = field(default_factory=dict)  # date → questions rewarded
    badges: list[str] = field(default_factory=list)

    # ── Storage ──

    @classmethod
    def load(cls, path: Path) -> Progress:
        try:
            return cls(**json.loads(Path(path).read_text()))
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return cls()

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    # ── Actions (each returns the points earned) ──

    def check_in(self, today: date) -> int:
        day = today.isoformat()
        if self.last_check_in == day:
            return 0
        yesterday = (today - timedelta(days=1)).isoformat()
        self.streak = self.streak + 1 if self.last_check_in == yesterday else 1
        self.best_streak = max(self.best_streak, self.streak)
        self.last_check_in = day
        earned = POINTS["check_in"] + min(self.streak - 1, 5)  # small streak bonus
        self.points += earned
        if today.month in (12, 1, 2):
            self._award("harmattan")
        return earned

    def answer_quiz(self, today: date, choice: int) -> tuple[bool, int]:
        day = today.isoformat()
        if day in self.quiz_days:
            return self.quiz_days[day], 0
        correct = choice == daily_quiz(today)[2]
        self.quiz_days[day] = correct
        earned = POINTS["quiz"] if correct else 0
        self.quiz_correct += int(correct)
        self.points += earned
        return correct, earned

    def complete_challenge(self, today: date) -> int:
        day = today.isoformat()
        if day in self.challenge_days:
            return 0
        self.challenge_days.append(day)
        self.points += POINTS["challenge"]
        return POINTS["challenge"]

    def asked_question(self, today: date) -> int:
        day = today.isoformat()
        self.questions += 1
        if self.question_points.get(day, 0) >= MAX_QUESTION_POINTS_PER_DAY:
            return 0
        self.question_points[day] = self.question_points.get(day, 0) + 1
        self.points += POINTS["question"]
        return POINTS["question"]

    # ── Status ──

    def refresh_badges(self) -> list[str]:
        """Award every badge whose condition now holds; return the newly earned ones."""
        conditions = {
            "first_breath": self.last_check_in is not None,
            "streak_3": self.best_streak >= 3,
            "streak_7": self.best_streak >= 7,
            "quiz_5": self.quiz_correct >= 5,
            "challenge_5": len(self.challenge_days) >= 5,
            "curious": self.questions >= 10,
            "sapling": self.points >= LEVELS[2][2],
        }
        new = [key for key, ok in conditions.items() if ok and key not in self.badges]
        self.badges.extend(new)
        return new

    def _award(self, key: str) -> None:
        if key not in self.badges:
            self.badges.append(key)

    @property
    def level(self) -> dict:
        index = max(i for i, (_, _, need) in enumerate(LEVELS) if self.points >= need)
        name, icon, need = LEVELS[index]
        nxt = LEVELS[index + 1] if index + 1 < len(LEVELS) else None
        progress = 1.0 if nxt is None else (self.points - need) / (nxt[2] - need)
        return {"name": name, "icon": icon, "index": index, "progress": progress,
                "next_name": nxt[0] if nxt else None,
                "points_to_next": nxt[2] - self.points if nxt else 0}


def daily_quiz(today: date) -> tuple[str, list[str], int, str]:
    """Same question for everyone on a given day; cycles through the bank."""
    return QUIZ[today.toordinal() % len(QUIZ)]


def daily_challenge(today: date, aqi: int | None, rain_chance: float | None) -> tuple[str, str]:
    """A challenge that fits today's conditions two days out of three, a general one otherwise."""
    day = today.toordinal()
    if day % 3 == 0:
        pool = CHALLENGES["any"]
    elif rain_chance is not None and rain_chance >= 70:
        pool = CHALLENGES["rain"]
    elif aqi is not None and aqi > 100:
        pool = CHALLENGES["polluted"]
    else:
        pool = CHALLENGES["clean_air"]
    return pool[(day // 3) % len(pool)]


def badge_info(key: str) -> tuple[str, str, str]:
    for badge_key, icon, name, description in BADGES:
        if badge_key == key:
            return icon, name, description
    raise KeyError(key)
