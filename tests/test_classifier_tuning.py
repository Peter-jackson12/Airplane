"""Synthetic contracts for the wider classifier tuning follow-up (tuning grids,
config selection, grid-edge flags, paired deltas and the weather_off RF path)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from notebooks import run_classifier_tuning as tuning
from src import classifier_compare as cc
from src.cv import make_folds
from tests.test_classifier_comparison import CFG, TE_COLS, synthetic

SPEC = SimpleNamespace(cat_cols=TE_COLS, inner_splits=5, te_drop_original=False)


def test_lgbm_tuning_grid_is_predeclared_and_contains_20261008_config():
    grid = cc.lgbm_tuning_grid()
    assert len(grid) == 24
    assert len({cc.compact_json(g) for g in grid}) == 24
    assert all(set(g) == set(cc.TUNE_LGBM_KEYS) for g in grid)
    base = {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20,
            "subsample": 0.8, "subsample_freq": 0}
    assert base in grid
    bagged = [g for g in grid if g["subsample_freq"] == 1]
    assert len(bagged) == 6 and all(g["subsample"] == 0.8 for g in bagged)
    assert {g["num_leaves"] for g in bagged} == {63, 127}
    merged = cc.lgbm_config_params({"learning_rate": 0.05, "random_state": 42, "max_depth": 8},
                                   bagged[0])
    assert merged["subsample_freq"] == 1 and merged["random_state"] == 42
    assert merged["max_depth"] == 8 and merged["learning_rate"] == bagged[0]["learning_rate"]
    with pytest.raises(ValueError, match="unexpected LightGBM config keys"):
        cc.lgbm_config_params({}, {"learning_rate": 0.1})
    with pytest.raises(ValueError, match="n_estimators"):
        cc.lgbm_config_params({"n_estimators": 5}, base)
    # The runner merges the unchanged repository params (random_state 42).
    params = cc.lgbm_config_params(tuning.lgbm_base_params(), base)
    import rerun_all_phases as runner
    assert {**runner.LGBM_PARAMS, "min_child_samples": 20, "subsample_freq": 0} == params


def test_rf_tuning_grid_uses_float_max_features():
    grid = cc.rf_tuning_grid()
    assert len(grid) == 9
    assert {"min_samples_leaf": 25, "max_features": 0.5} in grid
    # max_features=1.0 must be the float fraction (all features), never int 1.
    assert all(isinstance(g["max_features"], float) for g in grid)
    assert {g["max_features"] for g in grid} == {0.5, 0.7, 1.0}


def test_grid_edge_flags():
    axes = {"a": (1, 2, 3), "b": (20, 100)}
    assert cc.grid_edge_flags({"a": 2, "b": 20}, axes) == {"a": None, "b": "low"}
    assert cc.grid_edge_flags({"a": 3, "b": 100}, axes) == {"a": "high", "b": "high"}
    assert cc.grid_edge_flags({"a": 1, "b": 100, "extra": 0}, axes)["a"] == "low"
    with pytest.raises(ValueError, match="not in declared axis"):
        cc.grid_edge_flags({"a": 4, "b": 20}, axes)
    lgbm_axes = cc.lgbm_tuning_axes([10, 600, 50])
    assert lgbm_axes["n_estimators"] == (10, 50, 600)
    flags = cc.grid_edge_flags({"learning_rate": 0.05, "num_leaves": 127,
                                "min_child_samples": 20, "n_estimators": 600}, lgbm_axes)
    assert flags == {"learning_rate": None, "num_leaves": "high",
                     "min_child_samples": "low", "n_estimators": "high"}


def test_select_best_config_min_score_first_tie_and_scalar_records():
    scores = {0: 0.5, 1: 0.3, 2: 0.3, 3: 0.4}
    seen = []

    def evaluate(cfg):
        seen.append(cfg["i"])
        return {"inner_holdout_log_loss": scores[cfg["i"]], "threshold": 0.2,
                "valid_probs": np.full(3, cfg["i"], dtype=float)}

    sel = cc.select_best_config([{"i": i} for i in range(4)], evaluate)
    assert seen == [0, 1, 2, 3]
    assert sel.best_index == 1
    np.testing.assert_array_equal(sel.best["valid_probs"], np.ones(3))
    assert all("valid_probs" not in r for r in sel.records)
    assert [r["config"]["i"] for r in sel.records] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="empty"):
        cc.select_best_config([], evaluate)
    with pytest.raises(ValueError, match="non-finite"):
        cc.select_best_config([{"i": 0}], lambda c: {"inner_holdout_log_loss": np.nan})


def small_lgbm_grid(monkeypatch):
    grid = [
        {"learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 20,
         "subsample": 0.8, "subsample_freq": 0},
        {"learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 5,
         "subsample": 0.8, "subsample_freq": 1},
    ]
    monkeypatch.setattr(cc, "lgbm_tuning_grid", lambda: grid)
    monkeypatch.setattr(cc, "lgbm_tuning_axes", lambda k: {
        "learning_rate": (0.03, 0.1), "num_leaves": (7, 15),
        "min_child_samples": (5, 20), "n_estimators": tuple(sorted(k))})
    monkeypatch.setattr(tuning.runner, "N_ESTIMATORS_GRID", [5, 20, 60])
    monkeypatch.setattr(tuning, "lgbm_base_params", lambda: {
        "objective": "binary", "max_depth": 4, "colsample_bytree": 0.8,
        "random_state": 42, "verbose": -1, "n_jobs": 1})
    return grid


def test_lgbm_tuned_fold_ignores_outer_valid_labels_and_features(monkeypatch):
    grid = small_lgbm_grid(monkeypatch)
    X, y = synthetic(n=800, seed=5)
    folds = make_folds(y, CFG)
    tr, va = folds[0]
    X_tr, y_tr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
    X_va = X.iloc[va].reset_index(drop=True)
    a = tuning.lgbm_tuned_fold(X_tr, y_tr, X_va, CFG, 0, SPEC)
    X_bad = X_va.copy()
    X_bad["Distance"] = -1e6
    b = tuning.lgbm_tuned_fold(X_tr, y_tr, X_bad, CFG, 0, SPEC)
    assert a["selected_params"] == b["selected_params"]
    assert a["threshold"] == b["threshold"]
    assert a["best_inner_log_loss"] == b["best_inner_log_loss"]
    assert {k: v for k, v in a["selected_params"].items() if k != "n_estimators"} in grid
    assert a["selected_params"]["n_estimators"] in (5, 20, 60)
    assert len(a["grid_scores"]) == len(grid)
    assert a["best_inner_log_loss"] == min(g["inner_holdout_log_loss"] for g in a["grid_scores"])
    assert set(a["grid_edge"]) == {"learning_rate", "num_leaves", "min_child_samples",
                                   "n_estimators"}
    assert len(a["valid_probs"]) == len(va) and np.isfinite(a["valid_probs"]).all()


def test_rf_weather_off_path_shares_inner_boundary(monkeypatch):
    monkeypatch.setattr(cc, "rf_tuning_grid", lambda: [
        {"min_samples_leaf": 5, "max_features": 0.5},
        {"min_samples_leaf": 10, "max_features": 1.0}])
    monkeypatch.setattr(cc, "rf_tuning_axes", lambda: {
        "min_samples_leaf": (5, 10), "max_features": (0.5, 1.0)})
    X_on, y = synthetic(n=700, seed=9)
    X_off = X_on.drop(columns=["weather_origin_tmpf"])
    folds = make_folds(y, CFG)
    tr, va = folds[1]
    out = {}
    for cond, X in (("weather_on", X_on), ("weather_off", X_off)):
        out[cond] = tuning.rf_tuned_fold(
            X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True),
            X.iloc[va].reset_index(drop=True), CFG, 1, SPEC,
            rf_n_estimators=10, rf_n_jobs=1, tag=cond)
    assert out["weather_on"]["n_model_features"] == out["weather_off"]["n_model_features"] + 1
    assert out["weather_on"]["n_inner_holdout"] == out["weather_off"]["n_inner_holdout"]
    for res in out.values():
        assert res["selected_params"] in cc.rf_tuning_grid()
        assert len(res["valid_probs"]) == len(va)


def test_paired_seed_deltas():
    def table(shift, fp="fp"):
        return pd.DataFrame([{"seed": s, "fold_fingerprint": f"{fp}{s}", "n_rows": 10,
                              "log_loss": 0.4 + shift + 0.001 * i, "roc_auc": 0.6 - shift}
                             for i, s in enumerate((42, 1, 7))])
    res = cc.paired_seed_deltas(table(0.01), table(0.0), (42, 1, 7), ("log_loss", "roc_auc"),
                                label="a-b")
    assert res["metrics"]["log_loss"]["mean"] == pytest.approx(0.01)
    assert res["metrics"]["roc_auc"]["values_by_seed"]["1"] == pytest.approx(-0.01)
    assert res["metrics"]["log_loss"]["sign_consistent_across_seeds"] is True
    with pytest.raises(ValueError, match="different outer folds"):
        cc.paired_seed_deltas(table(0.0, "x"), table(0.0), (42, 1, 7), ("log_loss",), label="x")
    with pytest.raises(ValueError, match="seed 3 missing"):
        cc.paired_seed_deltas(table(0.0), table(0.0), (3,), ("log_loss",), label="x")


def test_existing_comparison_budget_unchanged():
    """The 20261008 comparison constants must not move (its evidence stays reproducible)."""
    assert cc.LR_C_GRID == (0.001, 0.01, 0.1, 1.0, 10.0)
    assert cc.RF_MIN_SAMPLES_LEAF_GRID == (5, 25, 100)
    assert cc.RF_MAX_FEATURES_GRID == ("sqrt", 0.5)
    assert cc.RF_N_ESTIMATORS == 300 and cc.RF_RANDOM_STATE == 42
    assert len(cc.rf_param_grid()) == 6 and len(cc.lr_param_grid()) == 5
