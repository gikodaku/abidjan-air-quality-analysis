import pandas as pd
import pytest

from aaqa.features.build import TARGET, make_supervised
from aaqa.models.baselines import ClimatologyBaseline, PersistenceBaseline
from aaqa.models.predictor import AirQualityPredictor
from aaqa.models.train import (
    chronological_split,
    gradient_boosting,
    regression_metrics,
    select_method,
)
from aaqa.pollutants import POLLUTANTS


def test_chronological_split_has_no_overlap(features):
    X, y = make_supervised(features, 1)
    X_train, X_test, _, _ = chronological_split(X, y, test_days=60)
    assert X_train.index.max() < X_test.index.min()
    assert len(X_test) == 60


def test_persistence_returns_today(features):
    X, y = make_supervised(features, 1, "pm10_mean")
    pred = PersistenceBaseline("pm10_mean").fit(X, y).predict(X)
    assert (pred == X["pm10_mean"].to_numpy()).all()


def test_climatology_uses_target_month(features):
    X, y = make_supervised(features, 1)
    model = ClimatologyBaseline(horizon=1).fit(X, y)
    dec31 = X[(X.index.month == 12) & (X.index.day == 31)].iloc[[0]]
    assert model.predict(dec31)[0] == pytest.approx(model.monthly_[1])


@pytest.mark.parametrize("target", [TARGET, "ozone_max8h", "carbon_monoxide_max8h"])
def test_log_ratio_model_is_positive_and_fits(features, target):
    X, y = make_supervised(features, 1, target)
    pred = gradient_boosting(target).fit(X, y).predict(X)
    assert (pred > 0).all()
    assert regression_metrics(y, pred)["mae"] < regression_metrics(y, X[target])["mae"] * 1.5


def test_metrics_perfect_prediction():
    y = pd.Series([5.0, 20.0, 40.0])
    metrics = regression_metrics(y, y)
    assert metrics["mae"] == 0 and metrics["aqi_category_accuracy"] == 1
    assert metrics["who_exceedance_recall"] == 1


def test_metrics_without_exceedances_or_matching_threshold():
    y = pd.Series([1.0, 2.0, 3.0])
    assert regression_metrics(y, y, "pm2_5")["who_exceedance_recall"] is None
    assert regression_metrics(y, y, "sulphur_dioxide")["who_exceedance_precision"] is None


def test_select_method():
    assert select_method({"cv_mae_mean": 1.0, "cv_persistence_mae_mean": 1.2}) == \
        "gradient_boosting"
    assert select_method({"cv_mae_mean": 1.3, "cv_persistence_mae_mean": 1.2}) == "persistence"


def test_training_report(trained, model_config):
    bundle, evaluations = trained
    assert set(bundle["pollutants"]) == set(model_config.pollutants)
    for pollutant in model_config.pollutants:
        for h in model_config.horizons:
            report = bundle["metrics"][pollutant][h]
            assert set(report["models"]) == {"persistence", "climatology", "ridge",
                                             "gradient_boosting"}
            assert report["n_test"] == model_config.test_days
            assert report["selected_method"] in {"gradient_boosting", "persistence"}
            assert bundle["pollutants"][pollutant][h]["method"] == report["selected_method"]
            assert "predictions" not in report
            assert len(evaluations[pollutant][h]["predictions"]) == model_config.test_days
        # importance is computed for the first horizon only
        assert evaluations[pollutant][1]["importances"] is not None
        assert evaluations[pollutant][2]["importances"] is None
    # On strongly seasonal data, the model must beat climatology
    gb = bundle["metrics"]["pm2_5"][1]["models"]
    assert gb["gradient_boosting"]["mae"] < gb["climatology"]["mae"]


def test_predictor_forecast_and_roundtrip(trained, daily, tmp_path):
    predictor = AirQualityPredictor(trained[0])
    path = tmp_path / "model.joblib"
    predictor.save(path)
    result = AirQualityPredictor.load(path).forecast(daily)

    last_day = daily[TARGET].dropna().index.max()
    assert result["issued_for"] == last_day.date().isoformat()
    assert len(result["observed"]["pollutants"]) == len(POLLUTANTS)
    assert [f["horizon_days"] for f in result["forecasts"]] == [1, 2]
    for forecast in result["forecasts"]:
        assert forecast["date"] == (last_day + pd.Timedelta(days=forecast["horizon_days"])
                                    ).date().isoformat()
        values = forecast["pollutants"]
        assert [v["pollutant"] for v in values] == list(POLLUTANTS)
        for value in values:
            low, high = value["interval_80"]
            assert 0 <= low <= value["value"] <= high
        assert forecast["overall"]["aqi"] == max(v["aqi"] for v in values)


def test_persistence_is_served_when_selected(trained, daily):
    bundle = dict(trained[0])
    bundle["pollutants"] = {**bundle["pollutants"]}
    ozone = {h: dict(models) for h, models in bundle["pollutants"]["ozone"].items()}
    ozone[1]["method"], ozone[1]["point"] = "persistence", PersistenceBaseline("ozone_max8h")
    bundle["pollutants"]["ozone"] = ozone
    result = AirQualityPredictor(bundle).forecast(daily)
    observed = next(v for v in result["observed"]["pollutants"] if v["pollutant"] == "ozone")
    forecast = next(v for v in result["forecasts"][0]["pollutants"] if v["pollutant"] == "ozone")
    assert forecast["method"] == "persistence"
    assert forecast["value"] == observed["value"]


def test_predict_features_with_partial_inputs(trained):
    predictor = AirQualityPredictor(trained[0])
    result = predictor.predict_features({"pm2_5_mean": 30.0, "dust_mean": None}, 1, "2024-01-10")
    assert result["date"] == "2024-01-11"
    assert result["value"] > 0
    ozone = predictor.predict_features({"ozone_max8h": 80.0}, 2, "2024-01-10", "ozone")
    assert ozone["name"] == "O3" and ozone["date"] == "2024-01-12"
    with pytest.raises(ValueError):
        predictor.predict_features({"pm2_5_mean": 30.0}, 7, "2024-01-10")
    with pytest.raises(ValueError):
        predictor.predict_features({"pm2_5_mean": 30.0}, 1, "2024-01-10", "radon")


def test_no2_features_are_reserved_to_no2_models(features):
    from aaqa.models.train import select_features

    X, _ = make_supervised(features, 1)
    assert any(c.startswith("no2max_") for c in X.columns)
    assert not any(c.startswith(("no2max_", "no2mean_")) for c in select_features(X, "pm2_5"))
    assert any(c.startswith("no2max_") for c in select_features(X, "nitrogen_dioxide"))
