"""
Model definitions for the interpretable-vs-heavy comparison.

Two families:

Interpretable / linear-ish (the ones small-data nutrition work should reach for):
    - Linear Regression   (OLS, the plain baseline)
    - Elastic Net         (regularised linear, handles correlated predictors)
    - SVR                 (RBF support vector regression, still compact)

Heavy / "default fancy" (what people often reach for by reflex):
    - Random Forest
    - Gradient Boosting

Every model that is scale-sensitive is wrapped in a Pipeline with
StandardScaler so the comparison is fair. Hyper-parameter grids are kept small
and sensible for a dataset of a few hundred rows.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Which models we treat as "interpretable" for the write-up.
INTERPRETABLE = {"Linear Regression", "Elastic Net", "SVR"}


def build_models() -> dict[str, Pipeline]:
    """Return a dict of {name: estimator}. Scale-sensitive models get a scaler."""
    return {
        "Linear Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Elastic Net": Pipeline(
            [("scaler", StandardScaler()), ("model", ElasticNet(max_iter=10000))]
        ),
        "SVR": Pipeline(
            [("scaler", StandardScaler()), ("model", SVR(kernel="rbf"))]
        ),
        "Random Forest": Pipeline(
            [("model", RandomForestRegressor(random_state=42))]
        ),
        "Gradient Boosting": Pipeline(
            [("model", GradientBoostingRegressor(random_state=42))]
        ),
    }


def param_grids() -> dict[str, dict]:
    """Small hyper-parameter grids for nested cross-validation.

    Keys use the ``model__`` prefix to address the estimator step inside each
    Pipeline.
    """
    return {
        "Linear Regression": {},  # nothing to tune
        "Elastic Net": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0],
            "model__l1_ratio": [0.1, 0.5, 0.9],
        },
        "SVR": {
            "model__C": [1.0, 10.0, 100.0],
            "model__gamma": ["scale", 0.1, 0.01],
            "model__epsilon": [0.1, 0.2],
        },
        "Random Forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 3],
        },
        "Gradient Boosting": {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [2, 3],
        },
    }
