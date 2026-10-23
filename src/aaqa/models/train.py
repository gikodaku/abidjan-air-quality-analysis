"""Training and evaluation of the D+1, D+2, D+3 forecast models for every pollutant.

Protocol:
1. For each pollutant and horizon h, the target is the pollutant's daily statistic (see
   aaqa.pollutants) on day D+h.
2. The last `test_days` days form the test set (never seen in training); cross-validation on
   the rest keeps chronological order (TimeSeriesSplit).
3. The main model (gradient boosting on the D+h / D log-ratio, see LogRatioRegressor) is
   compared with three baselines: persistence, monthly climatology and Ridge regression.
4. Two quantile models (10% and 90%) give an 80% prediction interval.
5. The shipped models are retrained on all available data. If the model does not beat
   persistence in cross-validation, persistence is served for that pollutant and horizon
   (a choice made without looking at the test set).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from aaqa import __version__
from aaqa.aqi import aqi_category, concentration_to_aqi
from aaqa.config import ModelConfig
from aaqa.features.build import make_supervised
from aaqa.models.baselines import ClimatologyBaseline, PersistenceBaseline, ridge_baseline
from aaqa.pollutants import Pollutant, get_pollutant

log = logging.getLogger(__name__)

QUANTILES = (0.1, 0.9)


class LogRatioRegressor(BaseEstimator, RegressorMixin):
    """Predict log(1 + C on D+h) − log(1 + C on day D), then convert back to µg/m³.

    Trees cannot extrapolate: when the mean pollution level drops from one year to the next
    (17.8 µg/m³ in 2023, 10.0 in 2025), a model predicting the absolute level stays anchored
    to the past. Predicting the change relative to today follows the current level
    automatically. Since the transform is monotonic, quantiles of the log-ratio map directly
    to quantiles of the concentration.
    """

    def __init__(self, estimator, base_column: str):
        self.estimator = estimator
        self.base_column = base_column

    def fit(self, X: pd.DataFrame, y: pd.Series):
        ratio = np.log1p(y) - np.log1p(X[self.base_column])
        self.estimator_ = clone(self.estimator).fit(X, ratio)
        return self

    def predict(self, X: pd.DataFrame):
        base = np.log1p(X[self.base_column].to_numpy())
        return np.expm1(base + self.estimator_.predict(X))


def gradient_boosting(target: str, loss: str = "absolute_error", quantile: float | None = None,
                      random_state: int = 42) -> LogRatioRegressor:
    # Strong regularisation: only ~1,200 training days
    return LogRatioRegressor(HistGradientBoostingRegressor(
        loss=loss,
        quantile=quantile,
        learning_rate=0.03,
        max_iter=250,
        max_leaf_nodes=8,
        min_samples_leaf=40,
        l2_regularization=2.0,
        random_state=random_state,
    ), base_column=target)


def regression_metrics(y_true, y_pred, pollutant: Pollutant | str = "pm2_5") -> dict:
    if isinstance(pollutant, str):
        pollutant = get_pollutant(pollutant)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    def categories(values):
        return np.array([aqi_category(concentration_to_aqi(pollutant, v)).key for v in values])

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "aqi_category_accuracy": float(np.mean(categories(y_true) == categories(y_pred))),
        "who_exceedance_precision": None,
        "who_exceedance_recall": None,
    }
    if pollutant.who_on_target:
        exceed_true, exceed_pred = y_true > pollutant.who_limit, y_pred > pollutant.who_limit
        tp = int((exceed_true & exceed_pred).sum())
        # None when there is no exceedance: the measure is meaningless
        if exceed_pred.any():
            metrics["who_exceedance_precision"] = tp / int(exceed_pred.sum())
        if exceed_true.any():
            metrics["who_exceedance_recall"] = tp / int(exceed_true.sum())
    return metrics


# Feature families reserved to one pollutant's models (they add noise to the others)
OWN_FEATURES = {"nitrogen_dioxide": ("no2max_", "no2mean_")}


def select_features(X: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    """Drop the feature families that belong to another pollutant's models."""
    foreign = tuple(prefix for owner, prefixes in OWN_FEATURES.items() if owner != pollutant
                    for prefix in prefixes)
    return X.loc[:, [c for c in X.columns if not c.startswith(foreign)]] if foreign else X


def chronological_split(X: pd.DataFrame, y: pd.Series, test_days: int):
    cutoff = X.index.max() - pd.Timedelta(days=test_days)
    train = X.index <= cutoff
    return X[train], X[~train], y[train], y[~train]


def evaluate_horizon(features: pd.DataFrame, horizon: int, cfg: ModelConfig,
                     pollutant: str = "pm2_5", importance: bool = True) -> dict:
    spec = get_pollutant(pollutant)
    target = spec.target
    X, y = make_supervised(features, horizon, target)
    X = select_features(X, pollutant)
    X_train, X_test, y_train, y_test = chronological_split(X, y, cfg.test_days)
    log.info("%s D+%d: %d training days, %d test days", spec.name, horizon,
             len(X_train), len(X_test))

    def model(loss="absolute_error", quantile=None):
        return gradient_boosting(target, loss, quantile, cfg.random_state)

    candidates = {
        "persistence": PersistenceBaseline(target),
        "climatology": ClimatologyBaseline(horizon),
        "ridge": ridge_baseline(),
        "gradient_boosting": model(),
    }
    results, predictions = {}, pd.DataFrame({"y_true": y_test}, index=X_test.index)
    for name, candidate in candidates.items():
        candidate.fit(X_train, y_train)
        pred = np.clip(candidate.predict(X_test), 0, None)
        predictions[name] = pred
        results[name] = regression_metrics(y_test, pred, spec)

    reference_mae = results["persistence"]["mae"]
    for scores in results.values():
        scores["skill_vs_persistence"] = 1 - scores["mae"] / reference_mae

    low, high = (model("quantile", q).fit(X_train, y_train) for q in QUANTILES)
    predictions["q10"], predictions["q90"] = low.predict(X_test), high.predict(X_test)
    coverage = float(((y_test >= predictions["q10"]) & (y_test <= predictions["q90"])).mean())

    cv_mae, cv_persistence_mae = [], []
    for fold_train, fold_val in TimeSeriesSplit(n_splits=cfg.cv_splits).split(X_train):
        fold_model = model().fit(X_train.iloc[fold_train], y_train.iloc[fold_train])
        X_val, y_val = X_train.iloc[fold_val], y_train.iloc[fold_val]
        cv_mae.append(mean_absolute_error(y_val, fold_model.predict(X_val)))
        cv_persistence_mae.append(mean_absolute_error(y_val, X_val[target]))

    importances = None
    if importance:
        result = permutation_importance(
            candidates["gradient_boosting"], X_test, y_test, n_repeats=5,
            random_state=cfg.random_state, scoring="neg_mean_absolute_error")
        importances = (pd.Series(result.importances_mean, index=X_test.columns)
                       .sort_values(ascending=False))

    return {
        "pollutant": pollutant,
        "target": target,
        "horizon": horizon,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_start": X_test.index.min().date().isoformat(),
        "test_end": X_test.index.max().date().isoformat(),
        "models": results,
        "interval_80_coverage": coverage,
        "cv_mae_mean": float(np.mean(cv_mae)),
        "cv_mae_std": float(np.std(cv_mae)),
        "cv_persistence_mae_mean": float(np.mean(cv_persistence_mae)),
        "predictions": predictions,
        "importances": importances,
    }


def select_method(evaluation: dict) -> str:
    """Served model: gradient boosting if it beats persistence in cross-validation."""
    if evaluation["cv_mae_mean"] < evaluation["cv_persistence_mae_mean"]:
        return "gradient_boosting"
    return "persistence"


def fit_final(features: pd.DataFrame, horizon: int, cfg: ModelConfig,
              pollutant: str = "pm2_5", method: str = "gradient_boosting") -> dict:
    """Retrain the point and quantile models on all data."""
    target = get_pollutant(pollutant).target
    X, y = make_supervised(features, horizon, target)
    X = select_features(X, pollutant)
    rs = cfg.random_state
    point = (gradient_boosting(target, random_state=rs) if method == "gradient_boosting"
             else PersistenceBaseline(target))
    return {
        "method": method,
        "point": point.fit(X, y),
        "q10": gradient_boosting(target, "quantile", QUANTILES[0], rs).fit(X, y),
        "q90": gradient_boosting(target, "quantile", QUANTILES[1], rs).fit(X, y),
        "features": list(X.columns),
    }


def train(features: pd.DataFrame, cfg: ModelConfig) -> tuple[dict, dict]:
    """Evaluate, then train, every pollutant and horizon.

    Returns (model bundle, evaluations[pollutant][horizon]). Feature importance is computed
    for the first horizon only (it is the most expensive step).
    """
    missing = [p for p in cfg.pollutants if get_pollutant(p).target not in features]
    if missing:
        raise ValueError(f"missing target columns for: {', '.join(missing)}")
    first = min(cfg.horizons)
    evaluations = {
        p: {h: evaluate_horizon(features, h, cfg, p, importance=h == first) for h in cfg.horizons}
        for p in cfg.pollutants
    }
    for by_h in evaluations.values():
        for ev in by_h.values():
            ev["selected_method"] = select_method(ev)
    pm_target = get_pollutant(cfg.pollutants[0]).target
    bundle = {
        "version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "trained_until": features.dropna(subset=[pm_target]).index.max().date().isoformat(),
        "pollutants": {
            p: {h: fit_final(features, h, cfg, p, evaluations[p][h]["selected_method"])
                for h in cfg.horizons}
            for p in cfg.pollutants
        },
        "metrics": {
            p: {h: _public(ev) for h, ev in by_h.items()} for p, by_h in evaluations.items()
        },
    }
    return bundle, evaluations


def _public(evaluation: dict) -> dict:
    return {k: v for k, v in evaluation.items() if k not in ("predictions", "importances")}
