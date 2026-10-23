import numpy as np
import pandas as pd
import pytest

from aaqa import calibration as cal
from aaqa.data import embassy


def test_parse_records_keeps_abidjan_valid_values():
    records = [
        {"UTC": "2024-02-08T10:00", "SiteName": "Abidjan", "RawConcentration": 123.4},
        {"UTC": "2024-02-08T11:00", "SiteName": "Abidjan", "RawConcentration": -999},
        {"UTC": "2024-02-08T12:00", "SiteName": "Accra", "RawConcentration": 50.0},
        {"UTC": "2024-02-08T13:00", "SiteName": "ABIDJAN", "Value": 60.0},
        {"UTC": "2024-02-08T13:00", "SiteName": "Abidjan", "Value": 60.0},  # duplicate hour
    ]
    frame = embassy.parse_records(records)
    assert frame["pm2_5"].tolist() == [123.4, 60.0]


def test_daily_means_need_enough_hours():
    full = pd.date_range("2024-01-01", periods=24, freq="h")
    partial = pd.date_range("2024-01-02", periods=10, freq="h")
    hourly = pd.DataFrame({"time": full.append(partial), "pm2_5": 10.0})
    daily = embassy.daily_means(hourly)
    assert daily.index.tolist() == [pd.Timestamp("2024-01-01")] and daily.iloc[0] == 10


def test_api_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("AIRNOW_API_KEY", raising=False)
    assert embassy.api_key(tmp_path) is None
    (tmp_path / ".env").write_text('AIRNOW_API_KEY="abc-123"\n')
    assert embassy.api_key(tmp_path) == "abc-123"
    monkeypatch.setenv("AIRNOW_API_KEY", "from-env")
    assert embassy.api_key(tmp_path) == "from-env"


def test_update_embassy_appends_months(tmp_path, monkeypatch):
    calls = []

    def fake_month(key, month):
        calls.append(month)
        return pd.DataFrame({"time": [month + pd.Timedelta(hours=1)], "pm2_5": [5.0]})

    monkeypatch.setattr(embassy, "fetch_month", fake_month)
    path = tmp_path / "embassy.csv"
    first = embassy.update_embassy(path, "k", today=pd.Timestamp("2020-04-15"))
    assert len(first) == 3 and len(calls) == 3  # Feb, Mar, Apr 2020
    calls.clear()
    again = embassy.update_embassy(path, "k", today=pd.Timestamp("2020-05-02"))
    assert [c.strftime("%Y-%m") for c in calls] == ["2020-04", "2020-05"]  # refresh + new
    assert len(again) == 4


@pytest.fixture
def synthetic():
    """Embassy ≈ 2 × CAMS + noise, with a stronger gap during the harmattan."""
    rng = np.random.default_rng(1)
    dates = pd.date_range("2021-01-01", "2024-12-31", freq="D")
    cams = pd.Series(8 + 10 * np.isin(dates.month, [12, 1, 2]) + rng.gamma(2, 2, len(dates)),
                     index=dates)
    factor = np.where(np.isin(dates.month, [12, 1, 2]), 2.4, 1.9)
    truth = pd.Series(cams * factor * rng.lognormal(0, 0.1, len(dates)), index=dates)
    return cams, truth


def test_calibration_corrects_the_bias(synthetic):
    cams, truth = synthetic
    result = cal.calibrate(cams, truth)
    assert result.method in cal.METHODS and result.n_days == len(cams)
    best = result.cross_validation[result.choice]
    assert result.raw["bias"] < -10            # CAMS reads far too low
    assert abs(best["bias"]) < 2 and best["mae"] < result.raw["mae"] / 3
    assert result.apply_one(10.0, "2023-07-01") == pytest.approx(19, rel=0.15)


def test_seasonal_method_wins_when_bias_depends_on_season(synthetic):
    cams, truth = synthetic
    result = cal.calibrate(cams, truth)
    assert result.cross_validation["seasonal"]["mae"] <= result.cross_validation["ratio"]["mae"]
    assert set(result.cross_validation) >= {"ratio", "linear, last 365 days",
                                            "seasonal, last 180 days"}


def test_save_load_roundtrip(synthetic, tmp_path):
    result = cal.calibrate(*synthetic)
    result.save(tmp_path / "cal.json")
    loaded = cal.Calibration.load(tmp_path / "cal.json")
    assert loaded.method == result.method
    assert loaded.apply_one(12, "2024-01-10") == pytest.approx(result.apply_one(12, "2024-01-10"))
    assert cal.Calibration.load(tmp_path / "missing.json") is None


def test_too_few_days_is_an_error():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    with pytest.raises(ValueError, match="only 10 days"):
        cal.calibrate(pd.Series(5.0, index=dates), pd.Series(10.0, index=dates))


@pytest.mark.parametrize("method", cal.METHODS)
def test_predictions_are_never_negative(method, synthetic):
    cams, truth = synthetic
    params = cal.fit(method, cams.to_numpy(), truth.to_numpy(), cams.index)
    assert (cal.predict(method, params, [0, 0.5, 100], cams.index[:3]) >= 0).all()


def test_adjust_snapshot_recomputes_overall(synthetic):
    from aaqa.aqi import describe, overall

    result = cal.calibrate(*synthetic)
    values = [describe("pm2_5", 10.0) | {"interval_80": [8.0, 12.0]}, describe("ozone", 40.0)]
    snapshot = {"date": "2024-07-01", "overall": overall(values), "pollutants": values}
    adjusted = cal.adjust_snapshot(snapshot, result, "2024-07-01")
    pm = adjusted["pollutants"][0]
    assert pm["raw_value"] == 10.0 and pm["value"] > 15 and pm["calibrated"]
    assert pm["interval_80"][0] < pm["value"] < pm["interval_80"][1]
    assert adjusted["overall"]["aqi"] > snapshot["overall"]["aqi"]
    assert adjusted["pollutants"][1] == values[1]  # other pollutants untouched


def test_live_air_can_be_calibrated():
    from aaqa.data import air_now
    from tests.test_air_now import PAYLOAD

    live = air_now.parse_air_now(PAYLOAD, lambda value, day: value * 2)
    pm = next(p for p in live["pollutants"] if p["pollutant"] == "pm2_5")
    assert pm["value"] == 80.0 and pm["raw_value"] == 40.0


def test_ground_truth_scores(trained, synthetic, tmp_path):
    from aaqa import monitoring as mon

    _, evaluations = trained
    for h, ev in evaluations["pm2_5"].items():
        ev["predictions"].to_csv(tmp_path / f"test_predictions_pm2_5_h{h}.csv")
    predictions = evaluations["pm2_5"][1]["predictions"]
    target = predictions.index + pd.Timedelta(days=1)
    embassy_daily = pd.Series(predictions["y_true"].to_numpy() * 2, index=target)
    calibration = cal.calibrate(pd.Series(predictions["y_true"].to_numpy(), index=target),
                                embassy_daily, min_days=30, warmup_months=1)
    table = mon.ground_truth_scores(tmp_path, calibration, embassy_daily)
    assert set(table["horizon"]) == {1, 2} and len(table) == 6
    d1 = table[table["horizon"] == 1].set_index("forecast")
    assert d1.loc["Model, calibrated", "mae"] < d1.loc["Model, raw (CAMS scale)", "mae"]


def test_airnow_errors_never_contain_the_key(monkeypatch, tmp_path, caplog):
    class Rejected:
        status_code = 401

    monkeypatch.setattr(embassy.requests, "get", lambda *a, **k: Rejected())
    with pytest.raises(embassy.requests.HTTPError) as error:
        embassy.fetch_month("SECRET-KEY", pd.Timestamp("2024-02-01"))
    assert "SECRET-KEY" not in str(error.value) and "401" in str(error.value)

    def leaky(key, month):
        raise embassy.requests.ConnectionError(f"failed for url ...?API_KEY={key}")

    monkeypatch.setattr(embassy, "fetch_month", leaky)
    with caplog.at_level("WARNING"):
        embassy.update_embassy(tmp_path / "e.csv", "SECRET-KEY", today=pd.Timestamp("2020-02-10"))
    assert "SECRET-KEY" not in caplog.text and "***" in caplog.text
