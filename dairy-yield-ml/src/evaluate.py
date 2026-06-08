"""
Evaluation via nested cross-validation.

Why nested CV: with only a few hundred rows and models that need
hyper-parameter tuning, evaluating on the same folds you tuned on leaks
information and flatters the heavy models. Nested CV keeps tuning (inner loop)
strictly separate from performance estimation (outer loop), so every reported
number is an honest out-of-fold estimate.

Metrics:
    RMSE - root mean squared error (kg), same units as milk yield
    MAE  - mean absolute error (kg)
    R2   - coefficient of determination
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def nested_cv_predict(
    estimator,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    outer_splits: int = 5,
    inner_splits: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, list[float]]:
    """Return (out-of-fold predictions, per-fold RMSE list) from nested CV.

    Computed in a single pass over the outer folds: for each fold we tune on the
    training part (inner CV), predict the held-out part, store those predictions,
    and record that fold's RMSE. This gives both the pooled OOF predictions and
    the fold-to-fold spread without running cross-validation twice.

    If ``param_grid`` is empty the estimator is used as-is (no inner search).
    """
    outer = KFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    X_arr = X.to_numpy()
    y_arr = y.to_numpy()
    oof = np.empty(len(y_arr), dtype=float)
    fold_rmse: list[float] = []

    inner = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)

    for train_idx, test_idx in outer.split(X_arr):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        if param_grid:
            search = GridSearchCV(
                clone(estimator),
                param_grid,
                cv=inner,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            search.fit(X_tr, y_tr)
            best = search.best_estimator_
        else:
            best = clone(estimator)
            best.fit(X_tr, y_tr)

        preds = best.predict(X_te)
        oof[test_idx] = preds
        fold_rmse.append(_rmse(y_te, preds))

    return oof, fold_rmse


def evaluate_all(
    models: dict,
    grids: dict,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Run nested CV for every model.

    Reports the pooled out-of-fold metrics plus the per-fold RMSE spread (mean
    and SD across outer folds), so a small gap between models can be read
    against the cross-validation noise rather than taken as a point estimate.

    Returns:
        results : DataFrame with RMSE, MAE, R2, RMSE_fold_mean, RMSE_fold_sd
        oof_preds : dict {model_name: out-of-fold predictions}
    """
    rows = []
    oof_preds: dict[str, np.ndarray] = {}
    y_arr = y.to_numpy()

    for name, est in models.items():
        preds, fold_rmse = nested_cv_predict(est, grids.get(name, {}), X, y, seed=seed)
        oof_preds[name] = preds
        fold_rmse = np.asarray(fold_rmse)
        rows.append(
            dict(
                model=name,
                RMSE=_rmse(y_arr, preds),
                MAE=_mae(y_arr, preds),
                R2=_r2(y_arr, preds),
                RMSE_fold_mean=float(fold_rmse.mean()),
                RMSE_fold_sd=float(fold_rmse.std()),
            )
        )

    results = pd.DataFrame(rows).set_index("model").sort_values("RMSE")
    return results, oof_preds
