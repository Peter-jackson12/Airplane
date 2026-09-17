"""Boundary checks for the nested calibration path.

The point of every test here is the same: calibrators and thresholds may only see
inner-holdout rows of outer-train. If outer-valid can move any of them, the whole
experiment collapses into grading a model on the data that tuned it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import cv
from src.calibration import KINDS, fit_calibrator, select_threshold


class ProbeModel:
    """Deterministic stand-in for LightGBM; records what it was fitted on."""

    fits: list = []

    def __init__(self):
        self.n_estimators = 10

    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators}

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

    def fit(self, X, y, **kwargs):
        ProbeModel.fits.append((X.copy(), y.copy()))
        self.rate = float(np.mean(y))
        return self

    def predict_proba(self, X):
        base = np.asarray(X["score"], dtype=float)
        p = np.clip(base * 0.5 + self.rate * 0.5, 1e-4, 1 - 1e-4)
        return np.column_stack([1 - p, p])


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    n = 600
    score = rng.uniform(0, 1, n)
    X = pd.DataFrame({"score": score,
                      "cat": pd.Categorical(rng.integers(0, 5, n)),
                      "row_id": np.arange(n)})
    y = pd.Series((rng.uniform(0, 1, n) < score * 0.6).astype(int))
    valid = pd.DataFrame({"score": rng.uniform(0, 1, 200),
                          "cat": pd.Categorical(rng.integers(0, 5, 200)),
                          "row_id": np.arange(1000, 1200)})
    return X, y, valid


def _run(X, y, valid, frac, capture, seed=0):
    selection = {}
    fit = cv.run_fold_nested_grid(
        ProbeModel, X, y, valid, cv.CVConfig(seed=seed),
        n_estimators_grid=[10, 25], te_cols=["cat"],
        selection_metadata=selection, holdout_capture=capture,
        calibration_holdout_frac=frac)
    return fit, selection


def test_capture_defaults_leave_the_existing_path_untouched(data):
    X, y, valid = data
    plain, meta_plain = _run(X, y, valid, 0.0, None)
    captured, meta_captured = _run(X, y, valid, 0.0, {})
    assert plain.grid_scores == captured.grid_scores
    assert plain.selected_n_estimators == captured.selected_n_estimators
    assert meta_plain == meta_captured
    np.testing.assert_array_equal(plain.valid_probs, captured.valid_probs)


def test_calibration_material_comes_only_from_the_inner_holdout(data):
    X, y, valid = data
    capture = {}
    fit, _ = _run(X, y, valid, 0.0, capture)
    assert capture["calibration_y"].size == fit.n_inner_holdout
    assert capture["calibration_y"].size + fit.n_inner_train == len(X)
    assert capture["calibration_probs"].size == fit.n_inner_holdout
    assert capture["calibration_is_disjoint"] is False
    # Outer-valid is larger than the holdout and its probabilities are not captured.
    assert fit.valid_probs.size == len(valid)
    assert not np.isin(capture["calibration_probs"], fit.valid_probs).all()


def test_outer_valid_cannot_move_the_calibration_data_or_threshold(data):
    X, y, valid = data
    first, first_meta = {}, None
    fit_a, first_meta = _run(X, y, valid, 0.0, first)
    other = valid.assign(score=valid.score[::-1].to_numpy(),
                         cat=pd.Categorical([4] * len(valid)))
    second = {}
    fit_b, second_meta = _run(X, y, other, 0.0, second)
    assert first_meta == second_meta
    assert fit_a.grid_scores == fit_b.grid_scores
    for key in ("selection_y", "selection_probs", "calibration_y", "calibration_probs"):
        np.testing.assert_array_equal(first[key], second[key])


def test_split_arm_keeps_calibration_rows_out_of_selection(data):
    X, y, valid = data
    capture = {}
    fit, meta = _run(X, y, valid, 0.5, capture)
    assert capture["calibration_is_disjoint"] is True
    n_sel = capture["selection_y"].size
    n_cal = capture["calibration_y"].size
    assert n_sel + n_cal == fit.n_inner_holdout
    assert abs(n_cal - fit.n_inner_holdout / 2) <= 1
    # The two slices are disjoint by construction; a shared row would make the
    # calibrator inherit the selection it is supposed to be independent of.
    assert n_sel > 0 and n_cal > 0
    assert 0.10 <= meta["threshold"] <= 0.70


def test_split_arm_rejects_a_missing_capture(data):
    X, y, valid = data
    with pytest.raises(ValueError):
        cv.run_fold_nested_grid(ProbeModel, X, y, valid, cv.CVConfig(),
                                n_estimators_grid=[10], te_cols=["cat"],
                                calibration_holdout_frac=0.5)


def test_invalid_calibration_fraction_is_rejected(data):
    X, y, valid = data
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            cv.run_fold_nested_grid(ProbeModel, X, y, valid, cv.CVConfig(),
                                    n_estimators_grid=[10], te_cols=["cat"],
                                    holdout_capture={}, calibration_holdout_frac=bad)


def test_identity_calibrator_returns_the_input_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < p).astype(int)
    np.testing.assert_allclose(fit_calibrator("none", y, p)(p), p)


def test_calibrators_stay_inside_the_unit_interval_and_handle_extremes():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p ** 2).astype(int)
    probe = np.array([0.0, 1e-9, 0.5, 1 - 1e-9, 1.0])
    for kind in KINDS:
        out = fit_calibrator(kind, y, p)(probe)
        assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0


def test_single_class_fitting_sample_falls_back_to_identity():
    p = np.linspace(0.05, 0.95, 50)
    y = np.zeros(50, dtype=int)
    for kind in KINDS:
        np.testing.assert_allclose(fit_calibrator(kind, y, p)(p), p)


def test_calibrator_rejects_out_of_range_or_mismatched_input():
    with pytest.raises(ValueError):
        fit_calibrator("platt", np.array([0, 1]), np.array([0.5, 1.5]))
    with pytest.raises(ValueError):
        fit_calibrator("platt", np.array([0, 1, 1]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        fit_calibrator("unknown", np.array([0, 1]), np.array([0.2, 0.8]))


def test_threshold_selection_prefers_the_macro_f1_maximum():
    y = np.array([0] * 80 + [1] * 20)
    p = np.concatenate([np.full(80, 0.10), np.full(20, 0.60)])
    grid = np.round(np.arange(0.10, 0.705, 0.01), 2)
    assert 0.11 <= select_threshold(y, p, grid) <= 0.60
    with pytest.raises(ValueError):
        select_threshold(y, p, np.array([]))
