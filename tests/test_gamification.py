from datetime import date

from aaqa.gamification import (
    BADGES,
    LEVELS,
    QUIZ,
    Progress,
    badge_info,
    daily_challenge,
    daily_quiz,
)

DAY = date(2026, 10, 22)


def test_check_in_streak_and_idempotence():
    p = Progress()
    assert p.check_in(DAY) == 5
    assert p.check_in(DAY) == 0  # same day: nothing more
    assert p.check_in(date(2026, 10, 23)) == 6 and p.streak == 2  # streak bonus
    assert p.check_in(date(2026, 10, 26)) == 5 and p.streak == 1  # gap resets the streak
    assert p.best_streak == 2


def test_harmattan_badge():
    p = Progress()
    p.check_in(date(2027, 1, 10))
    assert "harmattan" in p.badges


def test_quiz_once_per_day():
    p = Progress()
    right = daily_quiz(DAY)[2]
    assert p.answer_quiz(DAY, right) == (True, 10)
    assert p.answer_quiz(DAY, right) == (True, 0)
    wrong = (daily_quiz(date(2026, 10, 23))[2] + 1) % 3
    assert p.answer_quiz(date(2026, 10, 23), wrong) == (False, 0)
    assert p.quiz_correct == 1 and p.points == 10


def test_challenge_and_question_points():
    p = Progress()
    assert p.complete_challenge(DAY) == 15 and p.complete_challenge(DAY) == 0
    earned = [p.asked_question(DAY) for _ in range(7)]
    assert sum(earned) == 10 and p.questions == 7  # capped at 5 rewarded questions a day


def test_levels_and_badges():
    p = Progress(points=160, best_streak=3, quiz_correct=5, questions=10,
                 challenge_days=[f"2026-10-{d:02d}" for d in range(1, 6)], last_check_in="x")
    assert p.level["name"] == "Sapling" and p.level["next_name"] == "Tree"
    assert set(p.refresh_badges()) == {"first_breath", "streak_3", "quiz_5", "challenge_5",
                                      "curious", "sapling"}
    assert p.refresh_badges() == []
    top = Progress(points=LEVELS[-1][2] + 50).level
    assert top["name"] == "Forest" and top["next_name"] is None and top["progress"] == 1.0


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "user" / "progress.json"
    p = Progress()
    p.check_in(DAY)
    p.save(path)
    assert Progress.load(path) == p
    path.write_text("{broken")
    assert Progress.load(path) == Progress()


def test_daily_content_is_stable_and_contextual():
    assert daily_quiz(DAY) == daily_quiz(DAY)
    assert len({daily_quiz(date(2026, 1, d))[0] for d in range(1, len(QUIZ) + 1)}) == len(QUIZ)
    assert all(len(options) == 3 and 0 <= answer < 3 for _, options, answer, _ in QUIZ)
    specific_day = DAY if DAY.toordinal() % 3 else date(2026, 10, 23)
    rainy = daily_challenge(specific_day, 40, 90)[1]
    polluted = daily_challenge(specific_day, 130, 10)[1]
    clean = daily_challenge(specific_day, 30, 10)[1]
    assert len({rainy, polluted, clean}) == 3  # each condition has its own challenge
    assert all(badge_info(key) for key, *_ in BADGES)
