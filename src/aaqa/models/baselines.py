"""Baseline models: a machine-learning model is only worth something if it beats them."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aaqa.features.build import TARGET


class PersistenceBaseline:
    """"Tomorrow will be like today": predicts the value of day D."""

    def __init__(self, target: str = TARGET):
        self.target = target

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame):
        return X[self.target].to_numpy()


class ClimatologyBaseline:
    """Historical mean for the month of the target day."""

    def __init__(self, horizon: int):
        self.horizon = horizon
        self.monthly_: pd.Series | None = None

    def _target_months(self, X: pd.DataFrame):
        return (X.index + pd.Timedelta(days=self.horizon)).month

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.monthly_ = y.groupby(self._target_months(X)).mean()
        self.overall_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame):
        months = pd.Series(self._target_months(X))
        return months.map(self.monthly_).fillna(self.overall_).to_numpy()


def ridge_baseline():
    """Regularised linear regression on the same features as the main model."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
