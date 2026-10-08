"""Grid-extension follow-up to the 20261008 classifier tuning (pure logic).

The 20261008 tuning left several selections on grid edges. This module declares
the extended, predeclared budgets and the fold function for LightGBM that
trains each configuration ONCE up to its largest tree count and scores every
predeclared tree count as a prefix of that model (``num_iteration``).

Why the prefix scoring is the same selection as ``run_fold_nested_grid``:
for gbdt without row bagging (``subsample_freq=0``) the first k trees of a model
trained with K >= k trees are exactly the k-tree model of the same seed (the
feature-fraction RNG sequence does not depend on the total tree count). The
test suite checks this equivalence against ``src.cv.run_fold_nested_grid``.
Configurations that use row bagging are refused to keep that guarantee.

No data loading and no file writes here. Outer-valid labels are never an
argument of any selection function.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from . import classifier_compare as cc
from .cv import CVConfig

# --- LightGBM extended grid (experiment A) ---------------------------------------
EXT_LGBM_LEARNING_RATES: tuple[float, ...] = (0.01, 0.02, 0.03)
EXT_LGBM_NUM_LEAVES: tuple[int, ...] = (127, 255, 511)
EXT_LGBM_MIN_CHILD_SAMPLES: tuple[int, ...] = (20, 100)
EXT_LGBM_SUBSAMPLE = 0.8
#: Tree-count candidates (union over learning rates); each learning rate uses the
#: candidates up to its own predeclared cap.
EXT_LGBM_N_ESTIMATORS: tuple[int, ...] = (10, 25, 50, 75, 100, 150, 300, 600, 1000, 1500)
#: Largest tree count per learning rate (smaller rates need more trees; the
#: 20261008 optimum at lr=0.03 was 150 trees, interior to a 600 cap).
EXT_LGBM_MAX_TREES_BY_LR: dict[float, int] = {0.01: 1500, 0.02: 1000, 0.03: 600}

# --- Random Forest extended grid (experiment B) ----------------------------------
#: ``50..400`` is the requested extension; 25 is kept so that the 20261008
#: weather_on optimum (leaf 25) stays available when the same grid is used for
#: both weather conditions (a grid without 25 would disadvantage weather_on).
EXT_RF_MIN_SAMPLES_LEAF_GRID: tuple[int, ...] = (25, 50, 100, 200, 400)
EXT_RF_REQUESTED_LEAF_GRID: tuple[int, ...] = (50, 100, 200, 400)
EXT_RF_MAX_FEATURES_GRID: tuple[float, ...] = (0.5, 0.7, 1.0)


def lgbm_ext_grid() -> list[dict[str, Any]]:
    """18 configurations (3 learning rates x 3 leaf counts x 2 min_child_samples).

    ``subsample`` stays 0.8 but ``subsample_freq=0`` (row bagging inactive), the
    repository parameterisation of the 20261008 runs: no evidence suggested
    that activating bagging helps (it was not selected in 20261008).
    """
    return [{"learning_rate": float(lr), "num_leaves": int(nl), "min_child_samples": int(mcs),
             "subsample": EXT_LGBM_SUBSAMPLE, "subsample_freq": 0}
            for lr in EXT_LGBM_LEARNING_RATES for nl in EXT_LGBM_NUM_LEAVES
            for mcs in EXT_LGBM_MIN_CHILD_SAMPLES]


def trees_for_config(config: dict[str, Any], max_trees_by_lr: dict[float, int] | None = None
                     ) -> list[int]:
    caps = EXT_LGBM_MAX_TREES_BY_LR if max_trees_by_lr is None else max_trees_by_lr
    cc.require(float(config["learning_rate"]) in caps, "no tree cap for this learning rate")
    cap = caps[float(config["learning_rate"])]
    return [k for k in EXT_LGBM_N_ESTIMATORS if k <= cap]


def lgbm_ext_axes() -> dict[str, tuple]:
    return {"learning_rate": EXT_LGBM_LEARNING_RATES, "num_leaves": EXT_LGBM_NUM_LEAVES,
            "min_child_samples": EXT_LGBM_MIN_CHILD_SAMPLES}


def lgbm_edge_flags(selected: dict[str, Any], config_trees: Sequence[int],
                    axes: dict[str, tuple] | None = None) -> dict[str, str | None]:
    """Edge flags on the three config axes plus ``n_estimators`` against the
    selected configuration's own tree-count candidates."""
    full = {**(lgbm_ext_axes() if axes is None else axes), "n_estimators": tuple(config_trees)}
    return cc.grid_edge_flags(selected, full)


def rf_ext_grid(leaves: Sequence[int] = EXT_RF_MIN_SAMPLES_LEAF_GRID) -> list[dict[str, Any]]:
    return [{"min_samples_leaf": int(leaf), "max_features": float(mf)}
            for leaf in leaves for mf in EXT_RF_MAX_FEATURES_GRID]


def rf_ext_axes(leaves: Sequence[int] = EXT_RF_MIN_SAMPLES_LEAF_GRID) -> dict[str, tuple]:
    return {"min_samples_leaf": tuple(leaves), "max_features": EXT_RF_MAX_FEATURES_GRID}


def lgbm_ext_fold(
    make_model: Callable[[dict[str, Any], int], Any],
    configs: Sequence[dict[str, Any]],
    X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, cfg: CVConfig,
    *, fold: int, te_cols: Sequence[str], te_m: float, te_inner_splits: int,
    te_drop_original: bool = False,
    trees_of: Callable[[dict[str, Any]], Sequence[int]] = trees_for_config,
    axes: dict[str, tuple] | None = None,
    on_config: Callable[[dict[str, Any], int, float, float], None] | None = None,
) -> dict[str, Any]:
    """Select (configuration, tree count) by inner-holdout LogLoss, then score
    outer-valid once with the selected prefix model.

    ``make_model(config, n_estimators)`` must return an unfitted LightGBM
    classifier. Ties keep the first configuration, and within a configuration
    the smallest tree count (``<``), as in ``run_fold_nested_grid``.
    """
    cc.require(len(configs) > 0, "empty configuration grid")
    cc.require(all(c["subsample_freq"] == 0 for c in configs),
               "prefix scoring requires subsample_freq=0 (no row bagging)")
    X_in, y_in, X_ho, y_ho, X_va, _inner_tr, _inner_ho = cc.inner_boundary_frames(
        X_train, y_train, X_valid, cfg, fold=fold, te_cols=te_cols, te_m=te_m,
        te_inner_splits=te_inner_splits, te_drop_original=te_drop_original)
    cc.require(list(X_ho.columns) == list(X_in.columns) == list(X_va.columns),
               "inner/outer feature columns differ")
    records: list[dict[str, Any]] = []
    best_ll, best = np.inf, None
    for config in configs:
        trees = sorted(int(k) for k in trees_of(config))
        t0 = time.perf_counter()
        model = make_model(dict(config), max(trees))
        model.fit(X_in, y_in)
        fit_sec = time.perf_counter() - t0
        scores: dict[int, float] = {}
        probs_by_k: dict[int, np.ndarray] = {}
        for k in trees:
            probs = np.asarray(model.predict_proba(X_ho, num_iteration=k), dtype=float)[:, 1]
            scores[k] = float(log_loss(y_ho, probs, labels=[0, 1]))
            probs_by_k[k] = probs
        k_best = min(trees, key=lambda k: (scores[k], k))  # first (smallest k) on ties
        score = scores[k_best]
        cc.require(np.isfinite(score), f"non-finite inner-holdout LogLoss for {config}")
        records.append({
            "config": dict(config), "inner_holdout_log_loss": score,
            "selected_n_estimators": int(k_best),
            "n_estimators_scores": {str(k): float(v) for k, v in scores.items()},
            "n_estimators_candidates": trees,
            "fit_sec": float(time.perf_counter() - t0), "fit_only_sec": float(fit_sec),
        })
        if on_config is not None:
            on_config(dict(config), int(k_best), score, float(time.perf_counter() - t0))
        if score < best_ll:
            best_ll = score
            best = {"model": model, "k": int(k_best), "config": dict(config),
                    "probs_ho": probs_by_k[k_best], "trees": trees,
                    "index": len(records) - 1}
        else:
            del model
    assert best is not None
    threshold = cc.select_threshold(np.asarray(y_ho), best["probs_ho"], cfg)
    valid_probs = np.asarray(best["model"].predict_proba(X_va, num_iteration=best["k"]),
                             dtype=float)[:, 1]
    selected = {**best["config"], "n_estimators": best["k"]}
    return {
        "valid_probs": valid_probs, "threshold": float(threshold),
        "selected_params": selected, "grid_scores": records,
        "best_inner_log_loss": float(best_ll),
        "n_inner_train": int(len(X_in)), "n_inner_holdout": int(len(X_ho)),
        "n_model_features": len(getattr(best["model"], "feature_name_", [])),
        "grid_edge": lgbm_edge_flags(selected, best["trees"], axes),
        "extra": {},
    }
