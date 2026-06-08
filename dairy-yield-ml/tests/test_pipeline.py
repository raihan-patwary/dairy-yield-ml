"""
Tests for the dairy milk-yield pipeline.

Run with:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import FEATURES, LACTATION_STATS, TARGET, _make_placeholder, get_xy, load_data
from src.evaluate import _mae, _r2, _rmse, evaluate_all
from src.models import INTERPRETABLE, build_models, param_grids


# ---------- data ----------

def test_placeholder_has_expected_columns():
    df = _make_placeholder()
    for col in FEATURES + [TARGET, "cow_id"]:
        assert col in df.columns


def test_placeholder_row_count():
    df = _make_placeholder(n_per_parity=60)
    assert len(df) == 60 * len(LACTATION_STATS)


def test_placeholder_reproduces_published_means():
    # the placeholder should land near the published per-lactation yield means
    df = _make_placeholder(n_per_parity=400, seed=1)
    for parity, s in LACTATION_STATS.items():
        target_mean = s["yield_"][0]
        got = df.loc[df.parity == parity, TARGET].mean()
        # within 6% of the published mean
        assert abs(got - target_mean) / target_mean < 0.06


def test_load_data_returns_placeholder_when_no_csv(tmp_path):
    fake = tmp_path / "does_not_exist.csv"
    df, is_placeholder = load_data(raw_path=fake)
    assert is_placeholder is True
    assert len(df) > 0


def test_get_xy_shapes():
    df = _make_placeholder()
    X, y = get_xy(df)
    assert X.shape[0] == y.shape[0] == len(df)
    assert list(X.columns) == FEATURES


def test_get_xy_raises_on_missing_columns():
    bad = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(KeyError):
        get_xy(bad)


# ---------- models ----------

def test_build_models_returns_five():
    models = build_models()
    assert len(models) == 5
    assert INTERPRETABLE.issubset(set(models.keys()))


def test_every_model_has_a_grid_key():
    models = build_models()
    grids = param_grids()
    for name in models:
        assert name in grids


# ---------- metrics ----------

def test_rmse_zero_for_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert _rmse(y, y) == 0.0


def test_mae_known_value():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert _mae(y_true, y_pred) == pytest.approx(2.0)


def test_r2_perfect_is_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert _r2(y, y) == pytest.approx(1.0)


# ---------- end to end ----------

def test_evaluate_all_smoke():
    # small, fast end-to-end run on the placeholder
    df = _make_placeholder(n_per_parity=20, seed=7)
    X, y = get_xy(df)
    models = build_models()
    grids = param_grids()
    results, oof = evaluate_all(models, grids, X, y)
    assert len(results) == 5
    assert {"RMSE", "MAE", "R2"}.issubset(results.columns)
    for name in models:
        assert len(oof[name]) == len(y)
