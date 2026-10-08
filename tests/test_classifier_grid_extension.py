"""Synthetic contracts for the grid-extension follow-up (extended grids, prefix
tree-count scoring equivalence, leakage boundary, edge flags, reproduction)."""
from __future__ import annotations

from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from notebooks import run_classifier_grid_extension as ext
from src import classifier_compare as cc
from src import classifier_grid_extension as ge
from src.cv import make_folds, run_fold_nested_grid
from tests.test_classifier_comparison import CFG, TE_COLS, synthetic

SPEC = SimpleNamespace(cat_cols=TE_COLS, inner_splits=5, te_drop_original=False)
BASE = {"objective": "binary", "max_depth": 4, "colsample_bytree": 0.8,
        "random_state": 42, "verbose": -1, "n_jobs": 1, "min_child_samples": 5}


def test_lgbm_ext_grid_is_predeclared():
    grid = ge.lgbm_ext_grid()
    assert len(grid) == 18 and len({cc.compact_json(g) for g in grid}) == 18
    assert {g["learning_rate"] for g in grid} == {0.01, 0.02, 0.03}
    assert {g["num_leaves"] for g in grid} == {127, 255, 511}
    assert {g["min_child_samples"] for g in grid} == {20, 100}
    assert all(g["subsample_freq"] == 0 and set(g) == set(cc.TUNE_LGBM_KEYS) for g in grid)
    assert ge.trees_for_config(grid[0]) == [10, 25, 50, 75, 100, 150, 300, 600, 1000, 1500]
    assert ge.trees_for_config({"learning_rate": 0.03})[-1] == 600
    assert ge.trees_for_config({"learning_rate": 0.02})[-1] == 1000
    with pytest.raises(ValueError, match="tree cap"):
        ge.trees_for_config({"learning_rate": 0.5})
    # runner merges unchanged repository params and only adds the thread count
    import rerun_all_phases as runner
    assert ext.lgbm_base_params() == {**runner.LGBM_PARAMS, "n_jobs": ext.LGBM_N_JOBS}
    assert ext.LGBM_N_JOBS + 8 <= 14


def test_rf_ext_grid_contains_requested_and_weather_on_optimum():
    grid = ge.rf_ext_grid()
    assert len(grid) == 15
    assert all(isinstance(g["max_features"], float) for g in grid)
    requested = {(leaf, mf) for leaf in (50, 100, 200, 400) for mf in (0.5, 0.7, 1.0)}
    assert requested <= {(g["min_samples_leaf"], g["max_features"]) for g in grid}
    assert {"min_samples_leaf": 25, "max_features": 0.5} in grid


def test_edge_flags_use_the_selected_configs_own_tree_candidates():
    cfg = {"learning_rate": 0.02, "num_leaves": 255, "min_child_samples": 20,
           "subsample": 0.8, "subsample_freq": 0}
    trees = ge.trees_for_config(cfg)
    flags = ge.lgbm_edge_flags({**cfg, "n_estimators": 1000}, trees)
    assert flags == {"learning_rate": None, "num_leaves": None, "min_child_samples": "low",
                     "n_estimators": "high"}
    flags = ge.lgbm_edge_flags({**cfg, "n_estimators": 300}, trees)
    assert flags["n_estimators"] is None
    flags = ge.rf_ext_axes()
    assert flags["min_samples_leaf"][0] == 25 and flags["min_samples_leaf"][-1] == 400


SMALL_AXES = {"learning_rate": (0.05, 0.1), "num_leaves": (7, 15),
              "min_child_samples": (5, 20)}


def make_small(monkeypatch=None):
    configs = [
        {"learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 20,
         "subsample": 0.8, "subsample_freq": 0},
        {"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 5,
         "subsample": 0.8, "subsample_freq": 0},
    ]

    def trees_of(config):
        return [5, 20, 60] if config["learning_rate"] == 0.1 else [5, 20, 60, 90]

    def make_model(config, n_estimators):
        return lgb.LGBMClassifier(**{**BASE, **config, "n_estimators": int(n_estimators)})

    return configs, trees_of, make_model


def split(n=800, seed=5, fold=0):
    X, y = synthetic(n=n, seed=seed)
    tr, va = make_folds(y, CFG)[fold]
    return (X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True),
            X.iloc[va].reset_index(drop=True))


def run_ext(X_tr, y_tr, X_va, fold=0):
    configs, trees_of, make_model = make_small()
    return ge.lgbm_ext_fold(make_model, configs, X_tr, y_tr, X_va, CFG, fold=fold,
                            te_cols=TE_COLS, te_m=20.0, te_inner_splits=5, trees_of=trees_of,
                            axes=SMALL_AXES)


def test_prefix_scoring_matches_run_fold_nested_grid():
    """Same (config, tree count) choice, inner scores, threshold and outer-valid probs as
    the repository's per-count refits (run_fold_nested_grid) for the selected config."""
    X_tr, y_tr, X_va = split()
    res = run_ext(X_tr, y_tr, X_va)
    configs, trees_of, make_model = make_small()
    for rec, config in zip(res["grid_scores"], configs):
        grid = trees_of(config)
        meta: dict = {}
        fit = run_fold_nested_grid(
            lambda: lgb.LGBMClassifier(**{**BASE, **config}), X_tr, y_tr, X_va, CFG, fold=0,
            n_estimators_grid=grid, te_cols=TE_COLS, te_m=20.0, te_inner_splits=5,
            selection_metadata=meta)
        assert rec["selected_n_estimators"] == fit.selected_n_estimators
        for k in grid:
            assert rec["n_estimators_scores"][str(k)] == pytest.approx(fit.grid_scores[k], abs=1e-9)
        if res["selected_params"]["learning_rate"] == config["learning_rate"]:
            np.testing.assert_allclose(res["valid_probs"], fit.valid_probs, atol=1e-9)
            assert res["threshold"] == meta["threshold"]
    assert res["best_inner_log_loss"] == min(r["inner_holdout_log_loss"] for r in res["grid_scores"])
    assert {k: v for k, v in res["selected_params"].items() if k != "n_estimators"} in configs
    assert set(res["grid_edge"]) == {"learning_rate", "num_leaves", "min_child_samples",
                                     "n_estimators"}
    assert len(res["valid_probs"]) == len(X_va) and np.isfinite(res["valid_probs"]).all()


def test_selection_ignores_outer_valid_features_and_labels_are_not_an_input():
    X_tr, y_tr, X_va = split()
    a = run_ext(X_tr, y_tr, X_va)
    X_bad = X_va.copy()
    X_bad["Distance"] = -1e6
    b = run_ext(X_tr, y_tr, X_bad)
    assert a["selected_params"] == b["selected_params"]
    assert a["threshold"] == b["threshold"] and a["best_inner_log_loss"] == b["best_inner_log_loss"]
    assert not np.allclose(a["valid_probs"], b["valid_probs"])
    import inspect
    assert "y_valid" not in inspect.signature(ge.lgbm_ext_fold).parameters


def test_bagging_configs_are_refused():
    X_tr, y_tr, X_va = split(n=400)
    configs, trees_of, make_model = make_small()
    bad = [{**configs[0], "subsample_freq": 1}]
    with pytest.raises(ValueError, match="subsample_freq=0"):
        ge.lgbm_ext_fold(make_model, bad, X_tr, y_tr, X_va, CFG, fold=0, te_cols=TE_COLS,
                         te_m=20.0, te_inner_splits=5, trees_of=trees_of)


def test_ties_keep_smallest_tree_count_and_first_config():
    X_tr, y_tr, X_va = split(n=400)
    configs, _, make_model = make_small()
    # identical configs => identical scores; the first one must be kept
    same = [configs[0], dict(configs[0])]
    res = ge.lgbm_ext_fold(make_model, same, X_tr, y_tr, X_va, CFG, fold=0, te_cols=TE_COLS,
                           te_m=20.0, te_inner_splits=5, trees_of=lambda c: [5, 20], axes=SMALL_AXES)
    assert res["grid_scores"][0]["inner_holdout_log_loss"] == res["grid_scores"][1][
        "inner_holdout_log_loss"]


def test_rf_ext_fold_selects_inside_grid_for_both_weather_conditions(monkeypatch):
    monkeypatch.setattr(ge, "EXT_RF_MIN_SAMPLES_LEAF_GRID", (5, 10))
    monkeypatch.setattr(ge, "EXT_RF_MAX_FEATURES_GRID", (0.5, 1.0))
    X_on, y = synthetic(n=700, seed=9)
    X_off = X_on.drop(columns=["weather_origin_tmpf"])
    tr, va = make_folds(y, CFG)[1]
    out = {}
    for cond, X in (("weather_on", X_on), ("weather_off", X_off)):
        out[cond] = ext.rf_ext_fold(
            X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True),
            X.iloc[va].reset_index(drop=True), CFG, 1, SPEC,
            rf_n_estimators=10, rf_n_jobs=1, tag=cond)
    assert out["weather_on"]["n_model_features"] == out["weather_off"]["n_model_features"] + 1
    for res in out.values():
        assert res["selected_params"] in ge.rf_ext_grid()
        assert len(res["valid_probs"]) == len(va)
        assert set(res["grid_edge"]) == {"min_samples_leaf", "max_features"}


def test_reproduction_checks_compare_shared_configs_only():
    shared = {"learning_rate": 0.03, "num_leaves": 127, "min_child_samples": 20,
              "subsample": 0.8, "subsample_freq": 0}
    other = {**shared, "num_leaves": 255}
    mine = [{"config": shared, "n_estimators_scores": {"150": 0.4, "1000": 0.5}},
            {"config": other, "n_estimators_scores": {"150": 0.9}}]
    ref = [{"config": shared, "n_estimators_scores": {"150": 0.4000001, "600": 0.41}}]
    folds = pd.DataFrame([{"seed": 42, "condition": "weather_on", "model": "lightgbm_ext",
                           "fold": 1, "grid_scores": cc.compact_json(mine)}])
    tfolds = pd.DataFrame([{"seed": 42, "condition": "weather_on", "model": "lightgbm_tuned",
                            "fold": 1, "grid_scores": cc.compact_json(ref)}])
    out = ext.reproduction_checks(folds, tfolds)
    assert out["lightgbm"]["42/weather_on/fold1"]["n_compared"] == 1
    assert out["lightgbm_max_abs_diff"] == pytest.approx(1e-7)
    rf_mine = [{"params": {"min_samples_leaf": 25, "max_features": 0.5},
                "inner_holdout_log_loss": 0.43}, {"params": {"min_samples_leaf": 400,
                                                              "max_features": 0.5},
                                                  "inner_holdout_log_loss": 0.5}]
    rf_ref = [{"params": {"min_samples_leaf": 25, "max_features": 0.5},
               "inner_holdout_log_loss": 0.43}]
    rf_folds = pd.DataFrame([{"seed": 42, "condition": "weather_off", "model": "random_forest_ext",
                              "fold": 1, "grid_scores": cc.compact_json(rf_mine)}])
    rf_t = pd.DataFrame([{"seed": 42, "condition": "weather_off", "model": "random_forest_tuned",
                          "fold": 1, "grid_scores": cc.compact_json(rf_ref)}])
    assert ext.reproduction_checks(rf_folds, rf_t)["random_forest_max_abs_diff"] == 0.0


def test_parts_and_names_and_thread_budget():
    assert ext.PARTS["lgbm"] == (("weather_on", "lightgbm_ext"),)
    assert [u[0] for u in ext.PARTS["rf"]] == ["weather_off", "weather_on"]
    assert ext.BASE_NAME == "baseline_recovery_v2_classifier_grid_ext_20261008"
    for path in (ext.TUNING_RUNS, ext.TUNING_FOLDS, ext.TUNING_MANIFEST):
        assert path.name.startswith("baseline_recovery_v2_classifier_tuning_20261008")
    grids = ext.grids_description("lgbm", 300, 8, "reason")
    assert grids["lightgbm_ext"]["n_configurations"] == 18
    assert grids["lightgbm_ext"]["trim_reason"] == "reason"
    assert ext.grids_description("rf", 300, 8, None)["random_forest_ext"]["n_configurations"] == 15
