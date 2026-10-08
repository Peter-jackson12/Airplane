"""Synthetic contracts for the three-classifier comparison (tutor feedback 1)."""
from __future__ import annotations

from types import SimpleNamespace

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


# =============================================================================
# Review follow-ups (2026-10-08): leakage, determinism and identity contracts
# =============================================================================

SPEC = SimpleNamespace(cat_cols=TE_COLS, inner_splits=5, te_drop_original=False)


def _fold_frames(seed: int = 0, fold: int = 0, n: int = 600):
    X, y = synthetic(n=n, seed=seed)
    folds = make_folds(y, CFG)
    tr, va = folds[fold]
    return X, y, tr, va


def _small_lgbm_driver(monkeypatch):
    from notebooks import run_classifier_comparison as base_cmp
    monkeypatch.setattr(base_cmp.runner, "N_ESTIMATORS_GRID", [5, 20, 60])
    monkeypatch.setattr(base_cmp.runner, "model_factory", lambda: lgb.LGBMClassifier(
        objective="binary", learning_rate=0.1, num_leaves=7, random_state=0,
        verbose=-1, n_jobs=1))
    return base_cmp


@pytest.mark.parametrize("model_key", ["random_forest", "lightgbm"])
def test_outer_valid_label_flip_cannot_change_rf_or_lgbm_fold(monkeypatch, model_key):
    """Driver fold paths: flipping every outer-valid label of fold 0 leaves the
    fold-0 selection, threshold, inner scores and outer-valid probabilities unchanged."""
    base_cmp = _small_lgbm_driver(monkeypatch)
    X, y, tr, va = _fold_frames(seed=11)
    y_flip = y.copy()
    y_flip.iloc[va] = 1 - y_flip.iloc[va]
    out = []
    for yy in (y, y_flip):
        args = (X.iloc[tr].reset_index(drop=True), yy.iloc[tr].reset_index(drop=True),
                X.iloc[va].reset_index(drop=True), CFG, 0, SPEC)
        if model_key == "lightgbm":
            out.append(base_cmp.lgbm_fold(*args))
        else:
            out.append(base_cmp.sklearn_fold(model_key, *args, seed=42,
                                             rf_n_estimators=10, rf_n_jobs=1))
    a, b = out
    assert not y.iloc[va].equals(y_flip.iloc[va])
    assert a["selected_params"] == b["selected_params"]
    assert a["threshold"] == b["threshold"]
    assert [g["inner_holdout_log_loss"] for g in a["grid_scores"]] == \
        [g["inner_holdout_log_loss"] for g in b["grid_scores"]]
    np.testing.assert_array_equal(a["valid_probs"], b["valid_probs"])


def test_inner_holdout_labels_do_not_change_inner_train_te_or_preprocessors(monkeypatch):
    """Given the inner split, changing only inner-holdout labels must leave inner-train
    TE, holdout and outer-valid TE, the fitted LR preprocessors/coefficients and the
    fitted RF unchanged.

    The inner split itself is stratified on outer-train labels (allowed: no
    outer-valid label is involved), so it is pinned here to isolate the property.
    """
    X, y, tr, va = _fold_frames(seed=2, fold=1)
    X_tr, y_tr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
    X_va = X.iloc[va].reset_index(drop=True)
    pinned = _make_inner_split(len(X_tr), y_tr, CFG, 1, None)
    monkeypatch.setattr(cc, "_make_inner_split", lambda *args: pinned)
    inner_ho = pinned[1]
    y_alt = y_tr.copy()
    y_alt.iloc[inner_ho] = 1 - y_alt.iloc[inner_ho]

    a, b = (cc.inner_boundary_frames(X_tr, yy, X_va, CFG, fold=1, te_cols=TE_COLS,
                                     te_m=20.0, te_inner_splits=5) for yy in (y_tr, y_alt))
    X_in_a, y_in_a, X_ho_a, y_ho_a, X_va_a = a[:5]
    X_in_b, y_in_b, X_ho_b, y_ho_b, X_va_b = b[:5]
    pd.testing.assert_frame_equal(X_in_a, X_in_b)
    pd.testing.assert_series_equal(y_in_a, y_in_b)
    pd.testing.assert_frame_equal(X_ho_a, X_ho_b)  # holdout TE comes from inner-train labels
    pd.testing.assert_frame_equal(X_va_a, X_va_b)
    assert not y_ho_a.equals(y_ho_b)

    def lr_fit(yy):
        return cc.run_fold_nested_params(
            lambda p, cols: cc.build_logistic(p, cols, seed=0), [{"C": 1.0}],
            X_tr, yy, X_va, CFG, fold=1, te_cols=TE_COLS, te_m=20.0, te_inner_splits=5)

    lr_a, lr_b = lr_fit(y_tr), lr_fit(y_alt)
    cols_a = lr_a.model.named_steps["columns"]
    cols_b = lr_b.model.named_steps["columns"]
    num_a = cols_a.named_transformers_["remainder"]
    num_b = cols_b.named_transformers_["remainder"]
    np.testing.assert_array_equal(num_a.named_steps["impute"].statistics_,
                                  num_b.named_steps["impute"].statistics_)
    np.testing.assert_array_equal(num_a.named_steps["scale"].mean_,
                                  num_b.named_steps["scale"].mean_)
    oh_a, oh_b = cols_a.named_transformers_["onehot"], cols_b.named_transformers_["onehot"]
    for ca, cb in zip(oh_a.categories_, oh_b.categories_):
        np.testing.assert_array_equal(ca, cb)
    np.testing.assert_array_equal(lr_a.model.named_steps["model"].coef_,
                                  lr_b.model.named_steps["model"].coef_)
    np.testing.assert_array_equal(lr_a.valid_probs, lr_b.valid_probs)

    def rf_fit(yy):
        return cc.run_fold_nested_params(
            lambda p, cols: cc.build_random_forest(p, cols, seed=3, n_jobs=1, n_estimators=10),
            [{"min_samples_leaf": 5, "max_features": 0.5}], X_tr, yy, X_va, CFG, fold=1,
            te_cols=TE_COLS, te_m=20.0, te_inner_splits=5)

    np.testing.assert_array_equal(rf_fit(y_tr).valid_probs, rf_fit(y_alt).valid_probs)


def test_onehot_categories_and_infrequent_levels_are_learned_from_inner_train_only():
    X, y, tr, va = _fold_frames(seed=4, fold=0, n=900)
    X_tr, y_tr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
    X_va = X.iloc[va].reset_index(drop=True)
    inner_tr, inner_ho = _make_inner_split(len(X_tr), y_tr, CFG, 0, None)
    # Dep_Hour=77: 18 inner-train rows (< min_frequency 20) + 4 inner-holdout rows.
    # Counted over the whole outer-train it would be frequent (22 >= 20).
    X_tr.loc[np.sort(inner_tr)[:18], "Dep_Hour"] = 77
    X_tr.loc[np.sort(inner_ho)[:4], "Dep_Hour"] = 77
    # Dep_Hour=88 exists only in outer-valid (unknown at fit time).
    X_va.loc[:29, "Dep_Hour"] = 88
    assert int((X_tr.Dep_Hour == 77).sum()) == 22 >= cc.LR_ONEHOT_MIN_FREQUENCY
    fit = cc.run_fold_nested_params(
        lambda p, cols: cc.build_logistic(p, cols, seed=0), [{"C": 1.0}],
        X_tr, y_tr, X_va, CFG, fold=0, te_cols=TE_COLS, te_m=20.0, te_inner_splits=5)
    np.testing.assert_array_equal(np.sort(inner_tr), np.sort(fit.inner_train_positions))
    onehot = fit.model.named_steps["columns"].named_transformers_["onehot"]
    inner = X_tr.iloc[inner_tr]
    for j, col in enumerate(onehot.feature_names_in_):
        values = inner[col].astype(float)
        np.testing.assert_array_equal(onehot.categories_[j], np.sort(values.unique()))
        counts = values.value_counts()
        expected_rare = np.sort(counts[counts < cc.LR_ONEHOT_MIN_FREQUENCY].index.to_numpy())
        got_rare = onehot.infrequent_categories_[j]
        got_rare = np.array([]) if got_rare is None else np.sort(got_rare)
        np.testing.assert_array_equal(got_rare, expected_rare)
    j = list(onehot.feature_names_in_).index("Dep_Hour")
    assert 77.0 in onehot.categories_[j] and 77.0 in onehot.infrequent_categories_[j]
    assert 88.0 not in onehot.categories_[j]
    assert np.isfinite(fit.valid_probs).all() and len(fit.valid_probs) == len(X_va)


def test_driver_refuses_fold_fingerprint_mismatch_before_fitting(monkeypatch):
    from notebooks import run_classifier_comparison as base_cmp
    from notebooks import run_classifier_tuning as tuning
    X, y = synthetic(n=300, seed=6)
    calls = []
    monkeypatch.setattr(base_cmp, "sklearn_fold", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(base_cmp, "lgbm_fold", lambda *a, **k: calls.append(a))
    for model_key in cc.MODEL_KEYS:
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            base_cmp.evaluate_model(model_key, X, y, seed=42, spec=SPEC, expected_fp="0" * 64,
                                    rf_n_estimators=10, rf_n_jobs=1)
    monkeypatch.setattr(tuning, "rf_tuned_fold", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(tuning, "lgbm_tuned_fold", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        tuning.evaluate_unit("weather_on", "random_forest_tuned", X, y, seed=42, spec=SPEC,
                             expected_fp="0" * 64, state={"folds": []}, ckpt_dir=None,
                             save_state=lambda: None, rf_n_estimators=10, rf_n_jobs=1)
    assert calls == []


def test_random_forest_is_deterministic_across_n_jobs():
    X, y, tr, va = _fold_frames(seed=8, fold=2)

    def fit(jobs):
        return cc.run_fold_nested_params(
            lambda p, cols: cc.build_random_forest(p, cols, seed=42, n_jobs=jobs,
                                                   n_estimators=20),
            cc.rf_param_grid(), X.iloc[tr].reset_index(drop=True),
            y.iloc[tr].reset_index(drop=True), X.iloc[va].reset_index(drop=True), CFG,
            fold=2, te_cols=TE_COLS, te_m=20.0, te_inner_splits=5)

    a, b = fit(1), fit(4)
    assert a.selected_params == b.selected_params and a.threshold == b.threshold
    # The fitted trees are identical for any n_jobs (seeded per tree) ...
    forest_a, forest_b = (f.model.named_steps["model"] for f in (a, b))
    for ta, tb in zip(forest_a.estimators_, forest_b.estimators_):
        np.testing.assert_array_equal(ta.tree_.feature, tb.tree_.feature)
        np.testing.assert_array_equal(ta.tree_.threshold, tb.tree_.threshold)
        np.testing.assert_array_equal(ta.tree_.value, tb.tree_.value)
    # ... while the parallel sum of per-tree probabilities may be accumulated in a
    # different order, which moves predict_proba by a few ulp (consistent with the
    # <=5.6e-17 RF inner LogLoss differences recorded in the tuning reproduction check).
    assert [g["inner_holdout_log_loss"] for g in a.grid_scores] == pytest.approx(
        [g["inner_holdout_log_loss"] for g in b.grid_scores], rel=0, abs=1e-12)
    np.testing.assert_allclose(a.valid_probs, b.valid_probs, rtol=0, atol=1e-12)


def test_logistic_grid_scores_record_per_candidate_convergence():
    X, y, tr, va = _fold_frames(seed=1)
    fit = cc.run_fold_nested_params(
        lambda p, cols: cc.build_logistic(p, cols, seed=0), cc.lr_param_grid(),
        X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True),
        X.iloc[va].reset_index(drop=True), CFG, fold=0, te_cols=TE_COLS,
        te_m=20.0, te_inner_splits=5)
    assert len(fit.grid_scores) == len(cc.LR_C_GRID)
    for g in fit.grid_scores:
        assert isinstance(g["n_iter"], int) and 1 <= g["n_iter"] <= cc.LR_MAX_ITER
        assert g["converged"] is (g["n_iter"] < cc.LR_MAX_ITER)
    # Recording only: selection is still the minimal inner-holdout LogLoss.
    best = min(fit.grid_scores, key=lambda g: g["inner_holdout_log_loss"])
    assert fit.selected_params == best["params"]
    # Tree models keep the 20261008 grid_scores shape.
    rf = cc.run_fold_nested_params(
        lambda p, cols: cc.build_random_forest(p, cols, seed=1, n_jobs=1, n_estimators=5),
        [{"min_samples_leaf": 5, "max_features": 0.5}], X.iloc[tr].reset_index(drop=True),
        y.iloc[tr].reset_index(drop=True), X.iloc[va].reset_index(drop=True), CFG, fold=0,
        te_cols=TE_COLS, te_m=20.0, te_inner_splits=5)
    assert set(rf.grid_scores[0]) == {"params", "inner_holdout_log_loss", "fit_sec"}


def test_target_and_row_key_hashes_are_additional_identity_fields():
    _, y = synthetic(n=200, seed=3)
    y.index = pd.Index(np.arange(1000, 1200))
    folds = make_folds(y, CFG)
    from src.weather_model import fold_fingerprint as weather_fp
    t, k = cc.target_sha256(y), cc.row_key_sha256(y.index)
    assert len(t) == len(k) == 64
    y_flip = y.copy()
    y_flip.iloc[0] = 1 - y_flip.iloc[0]
    assert cc.target_sha256(y_flip) != t
    assert cc.row_key_sha256(y.index[::-1]) != k
    assert cc.row_key_sha256(pd.Index([str(i) for i in y.index])) != k
    # The fold fingerprint keeps its recorded definition (validation assignment only).
    assert cc.fold_fingerprint(folds, len(y)) == weather_fp(folds, len(y))
    with pytest.raises(ValueError, match="binary"):
        cc.target_sha256([0, 2])
    with pytest.raises(ValueError, match="unique"):
        cc.row_key_sha256([1, 1])


def test_comparison_checkpoint_identity_includes_code_hashes():
    from notebooks import run_classifier_comparison as base_cmp
    from src.weather_model import stable_json_hash
    X, y = synthetic(n=50, seed=0)
    code = base_cmp.code_sha256_lf()
    assert set(code) == set(base_cmp.CODE_FILES)
    kw = dict(name="baseline_recovery_v2_x", git_sha="abc", smoke=True, sample=50, X=X, y=y,
              grids={"g": 1}, models=cc.MODEL_KEYS)
    ident = base_cmp.experiment_identity(**kw, code_sha=code)
    assert ident["code_sha256_lf"] == code
    assert ident["target_sha256"] == cc.target_sha256(y)
    assert ident["row_key_sha256"] == cc.row_key_sha256(y.index)
    changed = dict(code, **{"src/features.py": "0" * 64})
    assert stable_json_hash(ident) != stable_json_hash(
        base_cmp.experiment_identity(**kw, code_sha=changed))
