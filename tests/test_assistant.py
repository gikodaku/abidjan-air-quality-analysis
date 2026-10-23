from types import SimpleNamespace

import pytest

from aaqa import assistant
from tests.test_weather import NOW

WEATHER_DAYS = [
    {"date": "2026-10-22", "description": "Light drizzle", "temperature_min": 24.9,
     "temperature_max": 32.1, "feels_like_max": 37.6, "precipitation_mm": 0.4,
     "precipitation_probability": 30, "humidity": 79, "wind_max_kmh": 9.9, "wind_from": "S",
     "uv_index": 8.5, "uv_level": "very high"},
    {"date": "2026-10-23", "description": "Dense drizzle", "temperature_min": 25.3,
     "temperature_max": 30.8, "feels_like_max": 33.0, "precipitation_mm": 2.9,
     "precipitation_probability": 100, "humidity": 80, "wind_max_kmh": 10.9, "wind_from": "S",
     "uv_index": 6.6, "uv_level": "high"},
]


def air_day(date: str, aqi: int, label: str) -> dict:
    return {"date": date,
            "overall": {"aqi": aqi, "label": label, "dominant_name": "PM2.5",
                        "advice": "Some advice."},
            "pollutants": [{"name": "PM2.5", "value": 7.0, "statistic": "24-h mean",
                            "label": label},
                           {"name": "NO2", "value": 3.0, "statistic": "max hourly",
                            "label": "Good"}]}


FORECAST = {"observed": air_day("2026-10-21", 39, "Good"),
            "forecasts": [air_day("2026-10-22", 39, "Good"),
                          air_day("2026-10-23", 120, "Unhealthy for sensitive groups")]}


@pytest.fixture
def context():
    return assistant.build_context(FORECAST, WEATHER_DAYS, NOW, "2026-10-22")


def test_context_is_json_ready(context):
    import json
    json.dumps(context)
    assert [d["weekday"] for d in context["days"]] == ["thursday", "friday"]
    assert context["days"][1]["air_quality"]["aqi"] == 120
    assert context["now"]["temperature"] == 27.0 and len(context["next_hours"]) == 2


@pytest.mark.parametrize("question, expected", [
    ("What's the weather tomorrow?", ["Tomorrow", "dense drizzle", "25–31", "umbrella"]),
    ("Will it rain today?", ["Today", "light drizzle", "30% chance"]),
    ("How is the air tomorrow?", ["unhealthy for sensitive groups", "AQI 120"]),
    ("What is PM2.5?", ["fine particles", "15 µg/m³"]),
    ("what is no2", ["Nitrogen dioxide", "traffic"]),
    ("Explain the harmattan", ["Sahara", "December to February"]),
    ("What does AQI mean?", ["0 to 500"]),
    ("PM2.5 tomorrow?", ["PM2.5 tomorrow", "7.0 µg/m³"]),
    ("Can I go running today?", ["clean", "outdoor sport"]),
    ("Can I go running tomorrow?", ["indoor", "Rain is likely"]),
    ("Best time to go out?", ["21:00", "4% chance"]),  # 07:00 is rainy
    ("Hello", ["Kos"]),
    ("Tell me a joke about cats", ["not sure I understood"]),
])
def test_offline_answers(context, question, expected):
    reply = assistant.answer_offline(question, context)
    for text in expected:
        assert text.lower() in reply.lower(), (question, reply)


def test_offline_without_weather():
    context = assistant.build_context(FORECAST, [], None, "2026-10-22")
    assert "can't reach the weather" in assistant.answer_offline("weather today?", context)
    assert "AQI 39" in assistant.answer_offline("air today?", context)


def test_daily_summary(context):
    summary = assistant.daily_summary(context)
    assert summary.startswith("Clear sky and 27 °C right now")
    assert "the air is good" in summary and "umbrella" in summary


class FakeClient:
    def __init__(self, text="**31 °C** tomorrow ☔", stop_reason="end_turn", fail=False):
        self.calls, self.text, self.stop_reason, self.fail = [], text, stop_reason, fail
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise ConnectionError("offline")
        block = SimpleNamespace(type="text", text=self.text)
        return SimpleNamespace(content=[block], stop_reason=self.stop_reason)


def test_claude_mode_sends_context_and_history(context):
    client = FakeClient()
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}]
    reply, mode = assistant.answer("Weather tomorrow?", context, history, client=client)
    assert (reply, mode) == ("**31 °C** tomorrow ☔", "claude")
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["fallbacks"] == "default"
    assert call["messages"][:2] == history
    assert "<forecast>" in call["messages"][-1]["content"]
    assert '"temperature_max": 30.8' in call["messages"][-1]["content"]


def test_claude_refusal_and_failure(context):
    reply, mode = assistant.answer("x", context, client=FakeClient(stop_reason="refusal"))
    assert mode == "claude" and "can't help" in reply
    reply, mode = assistant.answer("What is PM2.5?", context, client=FakeClient(fail=True))
    assert mode == "offline" and "fine particles" in reply


def test_offline_when_no_credentials(context, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert assistant.answer("What is ozone?", context)[1] == "offline"
