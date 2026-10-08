"""Synthetic contracts for the three-classifier comparison (tutor feedback 1)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src import classifier_compare as cc
from src.cv import CVConfig, _make_inner_split, make_folds, run_fold_nested_grid

CFG = CVConfig(n_splits=3, seed=7, threshold_grid=(0.10, 0.70, 0.01), inner_holdout_frac=0.2)
TE_COLS = ("Tail_Number", "Route", "Origin_Airport", "Destination_Airport", "Airline")


def synthetic(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "row_id": np.arange(n, dtype=float),
        "Month": rng.integers(1, 13, n),
        "Dep_Hour": rng.integers(0, 24, n),
        "Arr_Hour": rng.integers(0, 24, n),
        "Distance": rng.normal(800, 300, n),
        "weather_origin_tmpf": np.where(rng.random(n) < 0.1, np.nan, rng.normal(50, 15, n)),
    })
    for col, k in [("Tail_Number", 40), ("Route", 30), ("Origin_Airport", 8),
                   ("Destination_Airport", 8), ("Airline", 5)]:
        X[col] = pd.Categorical(rng.integers(0, k, n))
    logit = -1.5 + 0.004 * (X.Distance - 800) + 0.4 * (X.Airline.astype(int) == 2)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


class SpyClassifier(ClassifierMixin, BaseEstimator):
    """Records which row_ids reach fit(); predicts a constant-ish probability."""

    seen: list = []

    def __init__(self, shift: float = 0.0):
        self.shift = shift

    def fit(self, X, y):
        SpyClassifier.seen.append(("fit", X["row_id"].to_numpy().copy()))
        self.classes_ = np.array([0, 1])
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.clip(self.rate_ + self.shift + 0.001 * (X["Distance"].to_numpy() > 800), 0.01, 0.99)
        return np.column_stack([1 - p, p])


def test_outer_valid_rows_never_reach_fit_and_selection_ignores_valid():
    X, y = synthetic()
    folds = make_folds(y, CFG)
    tr, va = folds[0]
    X_tr, y_tr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
    X_va = X.iloc[va].reset_index(drop=True)
    SpyClassifier.seen = []
    grid = [{"shift": 0.0}, {"shift": 0.05}, {"shift": -0.05}]
    fit_a = cc.run_fold_nested_params(lambda p, cols: SpyClassifier(**p), grid,
                                      X_tr, y_tr, X_va, CFG, fold=0, te_cols=TE_COLS,
                                      te_m=20.0, te_inner_splits=5)
    valid_ids = set(X_va.row_id)
    holdout_ids = set(X_tr.row_id.iloc[fit_a.inner_holdout_positions])
    for _, ids in SpyClassifier.seen:
        assert not (set(ids) & valid_ids)
        assert not (set(ids) & holdout_ids)
    # Corrupting outer-valid features cannot change selection or threshold.
    X_va_bad = X_va.copy()
    X_va_bad["Distance"] = 1e6
    fit_b = cc.run_fold_nested_params(lambda p, cols: SpyClassifier(**p), grid,
                                      X_tr, y_tr, X_va_bad, CFG, fold=0, te_cols=TE_COLS,
                                      te_m=20.0, te_inner_splits=5)
    assert fit_a.selected_params == fit_b.selected_params
    assert fit_a.threshold == fit_b.threshold
    assert [g["inner_holdout_log_loss"] for g in fit_a.grid_scores] == \
        [g["inner_holdout_log_loss"] for g in fit_b.grid_scores]


def test_outer_valid_labels_cannot_change_selection_in_outer_loop():
    """Flip every label of outer fold 0: fold-0 selection must not move."""
    X, y = synthetic()
    folds = make_folds(y, CFG)
    tr, va = folds[0]
    y_flip = y.copy()
    y_flip.iloc[va] = 1 - y_flip.iloc[va]
    res = []
    for yy in (y, y_flip):
        fit = cc.run_fold_nested_params(
            lambda p, cols: cc.build_logistic(p, cols, seed=0), cc.lr_param_grid(),
            X.iloc[tr].reset_index(drop=True), yy.iloc[tr].reset_index(drop=True),
            X.iloc[va].reset_index(drop=True), CFG, fold=0, te_cols=TE_COLS,
            te_m=20.0, te_inner_splits=5)
        res.append(fit)
    assert res[0].selected_params == res[1].selected_params
    assert res[0].threshold == res[1].threshold
    np.testing.assert_allclose(res[0].valid_probs, res[1].valid_probs)


def test_logistic_preprocessing_is_fitted_on_inner_train_only():
    X, y = synthetic()
    X.loc[X.row_id < 60, "Distance"] = 50_000.0  # make the median sensitive to row choice
    folds = make_folds(y, CFG)
    tr, va = folds[1]
    X_tr, y_tr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
    fit = cc.run_fold_nested_params(
        lambda p, cols: cc.build_logistic(p, cols, seed=0), cc.lr_param_grid(),
        X_tr, y_tr, X.iloc[va].reset_index(drop=True), CFG, fold=1,
        te_cols=TE_COLS, te_m=20.0, te_inner_splits=5)
    inner_tr, _ = _make_inner_split(len(X_tr), y_tr, CFG, 1, None)
    np.testing.assert_array_equal(np.sort(inner_tr), np.sort(fit.inner_train_positions))
    numeric = fit.model.named_steps["columns"].named_transformers_["remainder"]
    names = list(numeric.feature_names_in_)
    imputer, scaler = numeric.named_steps["impute"], numeric.named_steps["scale"]
    j = names.index("weather_origin_tmpf")
    expected_median = np.nanmedian(X_tr.weather_origin_tmpf.iloc[inner_tr])
    assert imputer.statistics_[j] == pytest.approx(expected_median)
    assert imputer.statistics_[j] != pytest.approx(np.nanmedian(X.weather_origin_tmpf))
    k = names.index("Distance")
    assert scaler.mean_[k] == pytest.approx(X_tr.Distance.iloc[inner_tr].mean())
    # High-cardinality raw codes are represented only by inner-boundary TE.
    assert "Tail_Number" not in names and "TE_Tail_Number" in names
    onehot = fit.model.named_steps["columns"].named_transformers_["onehot"]
    assert list(onehot.feature_names_in_) == list(cc.LR_ONEHOT_COLS)


def test_random_forest_runs_with_missing_values_and_selects_from_grid():
    X, y = synthetic()
    folds = make_folds(y, CFG)
    tr, va = folds[2]
    fit = cc.run_fold_nested_params(
        lambda p, cols: cc.build_random_forest(p, cols, seed=1, n_jobs=1, n_estimators=10),
        cc.rf_param_grid(), X.iloc[tr].reset_index(drop=True),
        y.iloc[tr].reset_index(drop=True), X.iloc[va].reset_index(drop=True), CFG,
        fold=2, te_cols=TE_COLS, te_m=20.0, te_inner_splits=5)
    assert fit.selected_params in cc.rf_param_grid()
    assert len(fit.grid_scores) == len(cc.rf_param_grid())
    assert np.isfinite(fit.valid_probs).all()


def test_generic_harness_matches_run_fold_nested_grid_for_lightgbm():
    """The LR/RF harness and the repository LightGBM path share one boundary."""
    X, y = synthetic(n=800, seed=3)
    folds = make_folds(y, CFG)
    tr, va = folds[0]
    X_tr, y_tr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
    X_va = X.iloc[va].reset_index(drop=True)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=7,
                  random_state=0, verbose=-1, n_jobs=1)
    grid = [5, 20, 60]
    meta: dict = {}
    ref = run_fold_nested_grid(lambda: lgb.LGBMClassifier(**params), X_tr, y_tr, X_va, CFG,
                               fold=0, n_estimators_grid=grid, te_cols=TE_COLS, te_m=20.0,
                               te_inner_splits=5, selection_metadata=meta)
    mine = cc.run_fold_nested_params(
        lambda p, cols: lgb.LGBMClassifier(**params, **p),
        [{"n_estimators": k} for k in grid], X_tr, y_tr, X_va, CFG, fold=0,
        te_cols=TE_COLS, te_m=20.0, te_inner_splits=5)
    assert mine.selected_params == {"n_estimators": ref.selected_n_estimators}
    assert mine.threshold == meta["threshold"]
    assert [g["inner_holdout_log_loss"] for g in mine.grid_scores] == \
        pytest.approx([ref.grid_scores[k] for k in grid])
    np.testing.assert_allclose(mine.valid_probs, ref.valid_probs)


def test_fold_fingerprint_check():
    _, y = synthetic()
    folds = make_folds(y, CFG)
    fp = cc.check_fold_fingerprint(folds, len(y), None)
    from src.weather_model import fold_fingerprint as weather_fp
    assert fp == weather_fp(folds, len(y))
    assert cc.check_fold_fingerprint(folds, len(y), fp) == fp
    other = make_folds(y, CVConfig(n_splits=3, seed=8))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        cc.check_fold_fingerprint(other, len(y), fp)
    broken = [(tr, va[:-1]) for tr, va in folds]
    with pytest.raises(ValueError):
        cc.fold_fingerprint(broken, len(y))


def test_class_metrics_and_paired_deltas():
    m = cc.class_metrics(tn=50, fp=10, fn=20, tp=20)
    assert m["precision_delayed"] == pytest.approx(20 / 30)
    assert m["recall_delayed"] == pytest.approx(0.5)
    assert m["recall_not_delayed"] == pytest.approx(50 / 60)
    rows = []
    for seed in (42, 1, 7):
        for i, model in enumerate(cc.MODEL_KEYS):
            rows.append({"seed": seed, "model": model, "fold_fingerprint": f"fp{seed}",
                         "n_rows": 100, "elapsed_sec": 1.0,
                         **{k: 0.5 - 0.01 * i for k in cc.SUMMARY_METRICS}})
    runs = pd.DataFrame(rows)
    paired = cc.paired_deltas(runs, (42, 1, 7))
    assert len(paired) == 6
    assert paired.macro_f1_nested_delta_model_minus_lightgbm.lt(0).all()
    summary = cc.summarize(runs, paired)
    assert summary["paired_deltas_model_minus_lightgbm"]["random_forest"]["log_loss"]["mean"] \
        == pytest.approx(-0.02)
    bad = runs.copy()
    bad.loc[bad.model.eq("random_forest") & bad.seed.eq(1), "fold_fingerprint"] = "other"
    with pytest.raises(ValueError, match="different outer folds"):
        cc.paired_deltas(bad, (42, 1, 7))
