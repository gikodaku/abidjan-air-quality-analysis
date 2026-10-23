import pytest
from fastapi.testclient import TestClient

from aaqa.api.main import create_app
from aaqa.models.predictor import AirQualityPredictor
from aaqa.pollutants import POLLUTANTS

WEATHER = [{
    "date": "2024-02-05", "description": "Clear sky", "icon": "☀️", "sky": "clear",
    "temperature_min": 24.0,
    "temperature_max": 33.0, "feels_like_max": 38.0, "precipitation_mm": 0.0,
    "precipitation_probability": 5, "humidity": 70, "wind_max_kmh": 12.0, "wind_from": "NE",
    "uv_index": 9.0, "uv_level": "very high",
}]


def failing_weather():
    raise ConnectionError("hors ligne")


@pytest.fixture(scope="module")
def files(trained, daily, tmp_path_factory):
    folder = tmp_path_factory.mktemp("api")
    AirQualityPredictor(trained[0]).save(folder / "model.joblib")
    daily.to_csv(folder / "daily.csv")
    return folder / "model.joblib", folder / "daily.csv"


@pytest.fixture(scope="module")
def client(files):
    return TestClient(create_app(*files, weather_provider=lambda: WEATHER))


@pytest.fixture
def empty_client(tmp_path):
    return TestClient(create_app(tmp_path / "absent.joblib", tmp_path / "absent.csv",
                                 weather_provider=lambda: WEATHER))


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"]["horizons"] == [1, 2]
    assert body["model"]["pollutants"] == list(POLLUTANTS)


def test_health_without_model(empty_client):
    assert empty_client.get("/health").json()["status"] == "degraded"
    assert empty_client.get("/forecast").status_code == 503


def test_forecast(client):
    response = client.get("/forecast")
    assert response.status_code == 200
    body = response.json()
    assert body["observed"]["date"] == body["issued_for"]
    assert len(body["forecasts"]) == 2
    day = body["forecasts"][0]
    assert {"aqi", "label", "advice", "dominant_name"} <= set(day["overall"])
    assert len(day["pollutants"]) == len(POLLUTANTS)
    assert {"value", "aqi", "interval_80", "method"} <= set(day["pollutants"][0])
    assert body["weather"][0]["temperature_max"] == 33.0


def test_forecast_survives_weather_outage(files):
    client = TestClient(create_app(*files, weather_provider=failing_weather))
    response = client.get("/forecast")
    assert response.status_code == 200 and response.json()["weather"] == []
    assert client.get("/weather").status_code == 503


def test_weather(client):
    assert client.get("/weather").json()[0]["uv_level"] == "very high"


def test_predict(client):
    response = client.post("/predict", json={
        "day": "2024-01-10", "horizon": 2, "features": {"pm2_5_mean": 25.0, "dust_mean": 40.0}})
    assert response.status_code == 200
    assert response.json()["date"] == "2024-01-12"
    ozone = client.post("/predict", json={
        "day": "2024-01-10", "pollutant": "ozone", "features": {"ozone_max8h": 90.0}})
    assert ozone.status_code == 200 and ozone.json()["name"] == "O3"


@pytest.mark.parametrize("payload", [
    {"day": "2024-01-10", "horizon": 1, "features": {"dust_mean": 40.0}},
    {"day": "2024-01-10", "horizon": 9, "features": {"pm2_5_mean": 25.0}},
    {"day": "pas-une-date", "horizon": 1, "features": {"pm2_5_mean": 25.0}},
    {"day": "2024-01-10", "pollutant": "radon", "features": {"pm2_5_mean": 25.0}},
    {"day": "2024-01-10", "pollutant": "ozone", "features": {"pm2_5_mean": 25.0}},
])
def test_predict_rejects_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_history(client):
    body = client.get("/history", params={"days": 7}).json()
    assert len(body) == 7
    assert body[0]["date"] < body[-1]["date"]
    pm10 = client.get("/history", params={"days": 3, "pollutant": "pm10"}).json()
    assert len(pm10) == 3 and pm10[0]["value"] != body[-3]["value"]


def test_aqi_endpoint(client):
    assert client.get("/aqi", params={"value": 40}).json()["category"] == "usg"
    assert client.get("/aqi", params={"value": 200, "pollutant": "pm10"}).json()["aqi"] == 123
    assert client.get("/aqi", params={"value": -1}).status_code == 422
    assert client.get("/aqi", params={"value": 1, "pollutant": "radon"}).status_code == 422


def test_pollutants(client):
    body = client.get("/pollutants").json()
    assert [p["key"] for p in body] == list(POLLUTANTS)
    assert body[0]["who_limit"] == 15.0


def test_model_is_reloaded_after_retraining(files):
    import os

    model_path, daily_path = files
    client = TestClient(create_app(model_path, daily_path, weather_provider=lambda: WEATHER))
    first = client.get("/health").json()["model"]["trained_at"]
    predictor = AirQualityPredictor.load(model_path)
    predictor.bundle["trained_at"] = "2099-01-01T00:00:00+00:00"
    predictor.save(model_path)
    stat = model_path.stat()
    os.utime(model_path, (stat.st_atime, stat.st_mtime + 5))
    assert client.get("/health").json()["model"]["trained_at"] != first
