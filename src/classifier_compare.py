"""Fair three-classifier comparison harness (tutor feedback item 1).

Logistic Regression, Random Forest and LightGBM are compared on the SAME
labeled rows, target, outer folds and seeds. Every model goes through the same
per-fold boundary as ``src.cv.run_fold_nested_grid``:

1. outer-train is split into inner-train / inner-holdout with
   ``src.cv._make_inner_split`` (identical formula and seed);
2. target encoding of the declared categorical columns is computed inside that
   inner boundary (``oof_target_encode``), and outer-valid is encoded one-way
   from inner-train labels only (``smoothed_target_encode``);
3. each candidate of a small model-specific grid is fitted on inner-train
   only; its model-specific preprocessing (imputation, scaling, one-hot) is part
   of the fitted sklearn pipeline, so it also only sees inner-train rows;
4. the candidate with minimal inner-holdout LogLoss is selected (same criterion
   that the LightGBM path uses for ``n_estimators``);
5. the decision threshold is the argmax of inner-holdout Macro F1 over
   ``cfg.thresholds()`` (same as ``selection_metadata`` in the LightGBM path);
6. the selected inner-train model scores outer-valid exactly once.

Outer-valid labels are never passed to any selection function. This module
does not load data and does not write files; the runner is
``notebooks/run_classifier_comparison.py``.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, NamedTuple, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .cv import CVConfig, _make_inner_split
from .features import oof_target_encode, smoothed_target_encode

MODEL_KEYS = ("lightgbm", "logistic_regression", "random_forest")
BASELINE_MODEL = "lightgbm"

# --- Logistic Regression budget ------------------------------------------------
#: Inverse L2 strength. Five log-spaced points; the selection metric is
#: inner-holdout LogLoss (same as LightGBM tree-count selection).
LR_C_GRID: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
LR_MAX_ITER = 2000
#: Low-cardinality columns one-hot encoded for LR only. A linear model cannot
#: represent non-monotone month/hour/airline effects from integer codes; trees
#: can split on them directly, so the tree models keep the integer form.
LR_ONEHOT_COLS: tuple[str, ...] = ("Airline", "Month", "Dep_Hour", "Arr_Hour")
LR_ONEHOT_MIN_FREQUENCY = 20
#: High-cardinality categoricals (thousands of levels). For LR their raw integer
#: codes are meaningless, so only the inner-boundary TE_* columns represent
#: them (the raw code columns are dropped inside the LR pipeline).
LR_TE_ONLY_COLS: tuple[str, ...] = (
    "Tail_Number", "Route", "Origin_Airport", "Destination_Airport",
)

# --- Random Forest budget ------------------------------------------------------
RF_N_ESTIMATORS = 300
RF_MIN_SAMPLES_LEAF_GRID: tuple[int, ...] = (5, 25, 100)
RF_MAX_FEATURES_GRID: tuple[Any, ...] = ("sqrt", 0.5)
RF_N_JOBS = 16
#: Fixed model seed, analogous to LightGBM whose random_state stayed 42 for all
#: three split seeds in the 20260922 weather comparison. Split seeds (42/1/7)
#: vary only outer/inner/TE splits for every model.
RF_RANDOM_STATE = 42
LR_RANDOM_STATE = 42  # lbfgs is deterministic; recorded for completeness


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def lr_param_grid() -> list[dict[str, Any]]:
    return [{"C": float(c)} for c in LR_C_GRID]


def rf_param_grid() -> list[dict[str, Any]]:
    return [{"min_samples_leaf": int(leaf), "max_features": mf}
            for leaf in RF_MIN_SAMPLES_LEAF_GRID for mf in RF_MAX_FEATURES_GRID]


# =============================================================================
# Model-specific preprocessing (fitted inside the pipeline => inner-train only)
# =============================================================================


def categories_to_float(X: pd.DataFrame) -> pd.DataFrame:
    """Replace pandas ``category`` columns by their (integer) category values.

    The categorical vocabulary was built target-free by ``encode_categoricals``
    over the whole input bundle (existing repository contract), so the integer
    values are consistent between inner-train, inner-holdout and outer-valid.
    No statistics are learned here.
    """
    out = X.copy()
    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(float)
    return out


def build_logistic(params: dict[str, Any], columns: Sequence[str], *,
                   seed: int) -> Pipeline:
    """Imputer(median)+indicator -> StandardScaler for numeric/TE columns,
    one-hot for low-cardinality columns, raw high-cardinality codes dropped."""
    columns = list(columns)
    onehot = [c for c in LR_ONEHOT_COLS if c in columns]
    dropped = [c for c in LR_TE_ONLY_COLS if c in columns]
    for col in dropped:
        require(f"TE_{col}" in columns,
                f"LR drops raw {col} but its inner-boundary TE_{col} is missing")
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                 keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    transformer = ColumnTransformer(
        [
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     min_frequency=LR_ONEHOT_MIN_FREQUENCY,
                                     sparse_output=False), onehot),
            ("drop_high_card_codes", "drop", dropped),
        ],
        remainder=numeric,
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("categories", FunctionTransformer(categories_to_float,
                                           feature_names_out="one-to-one")),
        ("columns", transformer),
        ("model", LogisticRegression(C=float(params["C"]), max_iter=LR_MAX_ITER,
                                     solver="lbfgs", random_state=seed)),
    ])


def build_random_forest(params: dict[str, Any], columns: Sequence[str], *,
                        seed: int, n_jobs: int = RF_N_JOBS,
                        n_estimators: int = RF_N_ESTIMATORS) -> Pipeline:
    """Same feature matrix as LightGBM (integer category values + TE_*);
    missing values handled natively by sklearn>=1.4 trees (no imputation)."""
    del columns
    return Pipeline([
        ("categories", FunctionTransformer(categories_to_float,
                                           feature_names_out="one-to-one")),
        ("model", RandomForestClassifier(
            n_estimators=int(n_estimators),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            n_jobs=int(n_jobs),
            random_state=int(seed),
        )),
    ])


# =============================================================================
# Shared nested selection boundary
# =============================================================================


class ParamFit(NamedTuple):
    model: Any
    valid_probs: np.ndarray
    selected_params: dict[str, Any]
    grid_scores: list[dict[str, Any]]
    threshold: float
    n_inner_train: int
    n_inner_holdout: int
    inner_train_positions: np.ndarray
    inner_holdout_positions: np.ndarray


def select_threshold(y_holdout: np.ndarray, p_holdout: np.ndarray, cfg: CVConfig) -> float:
    """argmax of inner-holdout Macro F1 (first max wins) — mirrors run_fold_nested_grid."""
    grid = cfg.thresholds()
    scores = [f1_score(y_holdout, p_holdout >= th, average="macro") for th in grid]
    return float(grid[int(np.argmax(scores))])


def inner_boundary_frames(
    X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, cfg: CVConfig,
    *, fold: int, te_cols: Sequence[str], te_m: float, te_inner_splits: int,
    te_drop_original: bool = False,
):
    """Inner split + inner-boundary TE exactly as ``run_fold_nested_grid`` does."""
    require(len(X_train) == len(y_train), "X_train/y_train length mismatch")
    require(not any(str(c).startswith("TE_") for X in (X_train, X_valid) for c in X.columns),
            "pre-encoded TE columns are not allowed")
    inner_tr, inner_ho = _make_inner_split(len(X_train), y_train, cfg, fold, None)
    te = oof_target_encode(
        X_train, y_train, inner_tr, inner_ho,
        cols=te_cols, m=te_m, inner_splits=te_inner_splits,
        seed=cfg.seed, drop_original=False,
    )
    X_in, y_in, X_ho = te.X_train, te.y_train, te.X_valid
    y_ho = y_train.iloc[inner_ho].reset_index(drop=True)
    present = [c for c in te_cols if c in X_valid.columns]
    X_va = X_valid.copy().reset_index(drop=True)
    for col in present:
        X_va[f"TE_{col}"] = smoothed_target_encode(X_in[col], y_in, X_va[col], m=te_m)
    if te_drop_original and present:
        X_in, X_ho, X_va = (X.drop(columns=present) for X in (X_in, X_ho, X_va))
    return X_in, y_in, X_ho, y_ho, X_va, inner_tr, inner_ho


def run_fold_nested_params(
    build_model: Callable[[dict[str, Any], Sequence[str]], Any],
    param_grid: Sequence[dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    cfg: CVConfig,
    *,
    fold: int,
    te_cols: Sequence[str],
    te_m: float,
    te_inner_splits: int,
    te_drop_original: bool = False,
    on_candidate: Callable[[dict[str, Any], float, float], None] | None = None,
) -> ParamFit:
    """Generalisation of ``run_fold_nested_grid`` from an ``n_estimators`` grid
    to an arbitrary parameter grid. Outer-valid labels are not an argument."""
    require(len(param_grid) > 0, "empty parameter grid")
    X_in, y_in, X_ho, y_ho, X_va, inner_tr, inner_ho = inner_boundary_frames(
        X_train, y_train, X_valid, cfg, fold=fold, te_cols=te_cols, te_m=te_m,
        te_inner_splits=te_inner_splits, te_drop_original=te_drop_original,
    )
    columns = list(X_in.columns)
    require(list(X_ho.columns) == columns and list(X_va.columns) == columns,
            "inner/outer feature columns differ")

    import time
    scores: list[dict[str, Any]] = []
    best_score, best_model, best_params, best_probs = np.inf, None, None, None
    for params in param_grid:
        t0 = time.perf_counter()
        model = build_model(dict(params), columns)
        model.fit(X_in, y_in)
        probs = np.asarray(model.predict_proba(X_ho), dtype=float)[:, 1]
        score = float(log_loss(y_ho, probs, labels=[0, 1]))
        elapsed = time.perf_counter() - t0
        scores.append({"params": dict(params), "inner_holdout_log_loss": score,
                       "fit_sec": float(elapsed)})
        if on_candidate is not None:
            on_candidate(dict(params), score, elapsed)
        if score < best_score:
            best_score, best_model, best_params, best_probs = score, model, dict(params), probs
        else:
            del model
    assert best_model is not None and best_params is not None
    threshold = select_threshold(np.asarray(y_ho), best_probs, cfg)
    valid_probs = np.asarray(best_model.predict_proba(X_va), dtype=float)[:, 1]
    return ParamFit(best_model, valid_probs, best_params, scores, threshold,
                    int(len(X_in)), int(len(X_ho)), inner_tr, inner_ho)


# =============================================================================
# Fold identity, metrics and summaries
# =============================================================================


def fold_fingerprint(folds: Sequence[tuple[np.ndarray, np.ndarray]], n_rows: int) -> str:
    """Identical to ``src.weather_model.fold_fingerprint`` (validation assignment hash)."""
    assignment = np.full(n_rows, -1, dtype=np.int16)
    counts = np.zeros(n_rows, dtype=np.int8)
    for fold, (_, valid) in enumerate(folds):
        valid = np.asarray(valid)
        require(bool((valid >= 0).all() and (valid < n_rows).all()), "invalid fold indices")
        assignment[valid] = fold
        np.add.at(counts, valid, 1)
    require(bool(np.all(counts == 1) and np.all(assignment >= 0)),
            "each row must belong to exactly one outer validation fold")
    return hashlib.sha256(assignment.tobytes()).hexdigest()


def check_fold_fingerprint(folds, n_rows: int, expected: str | None) -> str:
    got = fold_fingerprint(folds, n_rows)
    if expected is not None:
        require(got == expected,
                f"outer fold fingerprint mismatch: got {got}, expected {expected}")
    return got


def class_metrics(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    """Per-class metrics from a confusion matrix; Delayed=1 is the positive class."""
    def ratio(a: float, b: float) -> float:
        return float(a / b) if b else 0.0
    p1, r1 = ratio(tp, tp + fp), ratio(tp, tp + fn)
    p0, r0 = ratio(tn, tn + fn), ratio(tn, tn + fp)
    f1_1 = ratio(2 * p1 * r1, p1 + r1)
    f1_0 = ratio(2 * p0 * r0, p0 + r0)
    return {
        "precision_delayed": p1, "recall_delayed": r1, "f1_delayed": f1_1,
        "precision_not_delayed": p0, "recall_not_delayed": r0, "f1_not_delayed": f1_0,
        "macro_f1_from_cm": (f1_1 + f1_0) / 2,
    }


SUMMARY_METRICS = (
    "macro_f1_nested", "log_loss", "roc_auc", "f1_at_050",
    "precision_delayed", "recall_delayed", "f1_delayed", "f1_not_delayed",
)


def paired_deltas(runs: pd.DataFrame, seeds: Sequence[int],
                  metrics: Sequence[str] = SUMMARY_METRICS) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        cell = runs[runs.seed.eq(seed)].set_index("model")
        require(set(cell.index) == set(MODEL_KEYS), f"incomplete model set for seed {seed}")
        require(cell.fold_fingerprint.nunique() == 1, "models used different outer folds")
        require(cell.n_rows.nunique() == 1, "models used different evaluation rows")
        base = cell.loc[BASELINE_MODEL]
        for model in MODEL_KEYS:
            if model == BASELINE_MODEL:
                continue
            row = {"seed": int(seed), "model": model, "baseline": BASELINE_MODEL,
                   "fold_fingerprint": base.fold_fingerprint}
            for metric in metrics:
                row[f"{metric}_delta_model_minus_lightgbm"] = (
                    float(cell.loc[model, metric]) - float(base[metric]))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame, paired: pd.DataFrame,
              metrics: Sequence[str] = SUMMARY_METRICS) -> dict[str, Any]:
    models = {}
    for model in MODEL_KEYS:
        sub = runs[runs.model.eq(model)].sort_values("seed")
        models[model] = {
            metric: {
                "mean": float(sub[metric].mean()),
                "std": float(sub[metric].std(ddof=1)) if len(sub) > 1 else None,
                "values_by_seed": {str(int(r.seed)): float(getattr(r, metric))
                                   for r in sub.itertuples()},
            } for metric in metrics
        }
        models[model]["elapsed_sec_total"] = float(sub.elapsed_sec.sum())
    deltas = {}
    for model in MODEL_KEYS:
        if model == BASELINE_MODEL:
            continue
        sub = paired[paired.model.eq(model)].sort_values("seed")
        deltas[model] = {}
        for metric in metrics:
            col = f"{metric}_delta_model_minus_lightgbm"
            deltas[model][metric] = {
                "mean": float(sub[col].mean()),
                "std": float(sub[col].std(ddof=1)) if len(sub) > 1 else None,
                "values_by_seed": {str(int(r.seed)): float(getattr(r, col))
                                   for r in sub.itertuples()},
                "sign_consistent_across_seeds": bool(
                    (sub[col] > 0).all() or (sub[col] < 0).all()),
            }
    return {"models": models, "paired_deltas_model_minus_lightgbm": deltas}


def compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
