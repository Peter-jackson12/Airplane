"""Per-classifier post-hoc probability calibration inside the outer-train boundary.

Follow-up to the 20261008 classifier comparison/tuning runs. For every outer
fold the previously selected configuration (selected on the inner holdout of
that same outer-train, so reusing it is leak-free) is refitted on the same
inner-train rows through the same inner boundary
(``src.classifier_compare.inner_boundary_frames``: identical inner split and
inner-boundary target encoding). Three calibrator arms are then built:

* ``none``     -- the raw model probability (reproduces the recorded run);
* ``platt``    -- logistic regression on logit(p) (same form as src/calibration.py);
* ``isotonic`` -- monotone step function (sklearn IsotonicRegression).

Each fitted calibrator comes from one of two calibration-data sources, both
strictly inside outer-train:

* ``inner_holdout``         -- the selected model's inner-holdout scores. These
  rows already selected the configuration and the threshold, and the
  re-selected threshold is chosen on the same rows the calibrator was fitted on
  (double use; kept as a cheap comparison arm, analogous to the 20260917
  "shared" arm).
* ``inner_train_crossfit``  -- K-fold cross-fitted scores on inner-train: each
  inner-train part is scored by a model of the same configuration fitted on the
  other parts, with target encoding recomputed inside that sub-boundary. The
  calibrator never sees inner-holdout rows, so the threshold re-selected on the
  calibrated inner holdout uses rows disjoint from calibrator fitting
  (preferred arm).

The decision threshold of every arm is re-selected as the argmax of Macro F1
on the (calibrated) inner-holdout probabilities. No function in this module
receives outer-valid labels; scoring is done by the caller on pooled
outer-valid predictions.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, NamedTuple, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import classifier_compare as cc
from .cv import CVConfig
from .features import oof_target_encode

CALIBRATORS = ("none", "platt", "isotonic")
CALIBRATION_SOURCES = ("inner_holdout", "inner_train_crossfit")
#: (arm, calibrator, calibration data source) in fixed order.
ARMS: tuple[tuple[str, str, str], ...] = (
    ("none", "none", "none"),
    ("platt_inner_holdout", "platt", "inner_holdout"),
    ("isotonic_inner_holdout", "isotonic", "inner_holdout"),
    ("platt_crossfit", "platt", "inner_train_crossfit"),
    ("isotonic_crossfit", "isotonic", "inner_train_crossfit"),
)
ARM_NAMES = tuple(a[0] for a in ARMS)
CROSSFIT_K = 5
#: Offset of the cross-fitting split seed (cfg.seed + fold + offset); distinct
#: from the inner split (cfg.seed + fold) and from src.cv's 10_000 offset.
CROSSFIT_SEED_OFFSET = 20_000
#: Platt works on logit(p); probabilities are clipped to this epsilon first
#: (same constant as src/calibration.py).
LOGIT_EPS = 1e-6
#: Isotonic outputs are clipped to [ISO_CLIP, 1 - ISO_CLIP] so that a step of
#: exactly 0 or 1 cannot produce an unbounded LogLoss; clipped rows are counted.
ISO_CLIP = 1e-6
ECE_EQUAL_FREQ_BINS = 15
ECE_EQUAL_WIDTH_BINS = 10  # comparable to src.oof ``ece_10`` used on 20260917

require = cc.require


# =============================================================================
# Reliability bins and ECE
# =============================================================================


def _check_yp(y, p) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    require(y.ndim == 1 and p.ndim == 1 and y.size == p.size and y.size > 0,
            "y/p must be equal-length nonempty 1-D arrays")
    require(bool(np.isfinite(p).all() and p.min() >= 0.0 and p.max() <= 1.0),
            "probabilities must be finite and within [0, 1]")
    require(bool(np.isin(y, (0.0, 1.0)).all()), "labels must be binary 0/1")
    return y, p


def _bins_frame(y: np.ndarray, p: np.ndarray, idx: np.ndarray, edges: np.ndarray,
                binning: str) -> pd.DataFrame:
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        rows.append({
            "binning": binning, "bin": b,
            "lower": float(edges[b]), "upper": float(edges[b + 1]), "n": n,
            "mean_probability": float(p[m].mean()) if n else np.nan,
            "observed_rate": float(y[m].mean()) if n else np.nan,
        })
    out = pd.DataFrame(rows)
    out["abs_gap"] = (out.mean_probability - out.observed_rate).abs()
    return out


def equal_width_bins(y, p, n_bins: int = ECE_EQUAL_WIDTH_BINS) -> pd.DataFrame:
    """Fixed-width bins [k/n, (k+1)/n); the last bin includes 1 (as src.oof)."""
    require(isinstance(n_bins, int) and n_bins >= 1, "n_bins must be a positive int")
    y, p = _check_yp(y, p)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return _bins_frame(y, p, idx, edges, f"equal_width_{n_bins}")


def equal_frequency_bins(y, p, n_bins: int = ECE_EQUAL_FREQ_BINS) -> pd.DataFrame:
    """Quantile bins of the evaluated probabilities.

    Edges are the k/n quantiles of ``p``; a row goes to bin
    ``searchsorted(interior_edges, p, side='right')``. Tied probabilities always
    share a bin, so with many ties (e.g. isotonic steps) some bins are empty and
    the remaining ones are larger; empty bins contribute nothing to the ECE.
    """
    y, p = _check_yp(y, p)
    idx, edges = equal_frequency_assign(p, n_bins)
    return _bins_frame(y, p, idx, edges, f"equal_frequency_{n_bins}")


def equal_frequency_assign(p, n_bins: int = ECE_EQUAL_FREQ_BINS) -> tuple[np.ndarray, np.ndarray]:
    """(bin index per row, quantile edges); equal probabilities get equal indices."""
    require(isinstance(n_bins, int) and n_bins >= 1, "n_bins must be a positive int")
    p = np.asarray(p, dtype=float)
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    return np.searchsorted(edges[1:-1], p, side="right"), edges


def ece_from_bins(bins: pd.DataFrame) -> float:
    """sum_b n_b |mean_p_b - rate_b| / N over nonempty bins."""
    nonempty = bins[bins.n > 0]
    require(len(nonempty) > 0, "no nonempty bins")
    return float((nonempty.n * nonempty.abs_gap).sum() / nonempty.n.sum())


def probability_metrics(y, p) -> dict[str, float]:
    """Threshold-free metrics of one probability vector."""
    y, p = _check_yp(y, p)
    ef = equal_frequency_bins(y, p)
    ew = equal_width_bins(y, p)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(np.mean((p - y) ** 2)),
        "ece_ef15": ece_from_bins(ef),
        "ece_ew10": ece_from_bins(ew),
        "ef15_nonempty_bins": int((ef.n > 0).sum()),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan"),
        "mean_probability": float(p.mean()),
        "calibration_bias": float(p.mean() - y.mean()),
        "n_distinct_probabilities": int(np.unique(p).size),
    }


# =============================================================================
# Calibrators
# =============================================================================


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(p / (1.0 - p))


class Calibrator(NamedTuple):
    kind: str
    transform: Callable[[np.ndarray], np.ndarray]
    info: dict[str, Any]


def fit_calibrator(kind: str, y, p) -> Calibrator:
    """Fit a calibrator on (label, raw probability) pairs supplied by the caller.

    The caller is responsible for passing only outer-train rows; this function
    is the single entry point for calibrator fitting so that tests can spy on it.
    """
    require(kind in CALIBRATORS, f"unknown calibrator {kind}")
    y, p = _check_yp(y, p)
    y = y.astype(int)
    info: dict[str, Any] = {"kind": kind, "n_fit": int(y.size), "positives_fit": int(y.sum())}
    if kind == "none":
        return Calibrator(kind, lambda q: np.asarray(q, dtype=float), info)
    require(np.unique(y).size == 2, "calibrator fitting sample must contain both classes")
    if kind == "platt":
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(_logit(p).reshape(-1, 1), y)
        info.update(coef=float(model.coef_[0, 0]), intercept=float(model.intercept_[0]))
        return Calibrator(kind, lambda q: model.predict_proba(
            _logit(np.asarray(q, dtype=float)).reshape(-1, 1))[:, 1], info)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p, y)
    info.update(n_steps=int(np.unique(iso.predict(np.unique(p))).size))
    return Calibrator(kind, lambda q: np.clip(iso.predict(np.asarray(q, dtype=float)),
                                              ISO_CLIP, 1.0 - ISO_CLIP), info)


def select_threshold(y_inner, p_inner, cfg: CVConfig) -> float:
    """Inner-data threshold selection (argmax Macro F1, first max) -- the same
    rule as ``cc.select_threshold`` / ``run_fold_nested_grid``. Single entry point
    so that tests can verify it only ever receives inner-holdout labels."""
    return cc.select_threshold(np.asarray(y_inner), np.asarray(p_inner, dtype=float), cfg)


# =============================================================================
# Selected-configuration model builders
# =============================================================================

#: Experiment model keys (LR from the 20261008 comparison; RF/LightGBM from the
#: 20261008 tuning run, weather_on condition).
MODEL_KEYS = ("logistic_regression", "random_forest_tuned", "lightgbm_tuned")


def selected_model_builder(
    model_key: str, params: dict[str, Any], *, lgbm_base_params: dict[str, Any] | None = None,
    rf_n_jobs: int = 6, lgbm_n_jobs: int = 4, rf_n_estimators: int = cc.RF_N_ESTIMATORS,
) -> Callable[[Sequence[str]], Any]:
    """Unfitted-estimator factory for one previously selected configuration.

    The configurations were selected on the inner holdout of the same outer-train
    by the 20261008 comparison (LR) and tuning (RF, LightGBM) runs.
    """
    params = dict(params)
    if model_key == "logistic_regression":
        require(set(params) == {"C"}, f"unexpected LR params {params}")
        return lambda columns: cc.build_logistic(params, columns, seed=cc.LR_RANDOM_STATE)
    if model_key == "random_forest_tuned":
        require(set(params) == {"min_samples_leaf", "max_features"}, f"unexpected RF params {params}")
        return lambda columns: cc.build_random_forest(
            params, columns, seed=cc.RF_RANDOM_STATE, n_jobs=int(rf_n_jobs),
            n_estimators=int(rf_n_estimators))
    if model_key == "lightgbm_tuned":
        import lightgbm as lgb
        require(lgbm_base_params is not None, "LightGBM needs the repository base params")
        require(set(params) == set(cc.TUNE_LGBM_KEYS) | {"n_estimators"},
                f"unexpected LightGBM params {params}")
        merged = cc.lgbm_config_params(dict(lgbm_base_params),
                                       {k: params[k] for k in cc.TUNE_LGBM_KEYS})
        merged.update(n_estimators=int(params["n_estimators"]), n_jobs=int(lgbm_n_jobs))
        return lambda columns: lgb.LGBMClassifier(**merged)
    raise ValueError(f"unknown model key {model_key}")


# =============================================================================
# One outer fold
# =============================================================================


def crossfit_splits(y_inner: pd.Series | np.ndarray, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    require(k >= 2, "cross-fitting needs k >= 2")
    y_arr = np.asarray(y_inner)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed))
    return [(np.asarray(a), np.asarray(b)) for a, b in skf.split(np.zeros(len(y_arr)), y_arr)]


def _predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X), dtype=float)[:, 1]


def crossfit_scores(
    build_model: Callable[[Sequence[str]], Any],
    X_inner_raw: pd.DataFrame, y_inner_raw: pd.Series, *, k: int, seed: int,
    te_cols: Sequence[str], te_m: float, te_inner_splits: int, te_seed: int,
    te_drop_original: bool,
) -> np.ndarray:
    """K-fold out-of-fold scores on inner-train only (raw, pre-TE frames).

    For each part, target encoding is recomputed inside the sub-boundary
    (``oof_target_encode`` with the part as its one-way encoded 'valid' side),
    exactly mirroring how the inner holdout is encoded from inner-train.
    """
    require(not any(str(c).startswith("TE_") for c in X_inner_raw.columns),
            "cross-fitting expects raw (pre-TE) inner-train rows")
    X_inner_raw = X_inner_raw.reset_index(drop=True)
    y_inner_raw = y_inner_raw.reset_index(drop=True)
    oof = np.full(len(y_inner_raw), np.nan)
    present = [c for c in te_cols if c in X_inner_raw.columns]
    for sub_tr, sub_ho in crossfit_splits(y_inner_raw, k, seed):
        te = oof_target_encode(X_inner_raw, y_inner_raw, sub_tr, sub_ho, cols=te_cols, m=te_m,
                               inner_splits=te_inner_splits, seed=te_seed, drop_original=False)
        X_fit, X_part = te.X_train, te.X_valid
        if te_drop_original and present:
            X_fit, X_part = X_fit.drop(columns=present), X_part.drop(columns=present)
        model = build_model(list(X_fit.columns))
        model.fit(X_fit, te.y_train)
        oof[sub_ho] = _predict(model, X_part)
        del model
    require(bool(np.isfinite(oof).all()), "cross-fitted scores incomplete")
    return oof


def calibrate_fold(
    build_model: Callable[[Sequence[str]], Any],
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
    crossfit_k: int = CROSSFIT_K,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Refit the selected configuration and build every calibration arm.

    ``build_model(columns)`` returns an unfitted estimator of the already
    selected configuration. Outer-valid labels are not an argument: calibrators
    are fitted on inner-holdout or cross-fitted inner-train scores and the
    threshold of each arm is re-selected on inner-holdout probabilities only.
    """
    t0 = time.perf_counter()
    X_in, y_in, X_ho, y_ho, X_va, inner_tr, inner_ho = cc.inner_boundary_frames(
        X_train, y_train, X_valid, cfg, fold=fold, te_cols=te_cols, te_m=te_m,
        te_inner_splits=te_inner_splits, te_drop_original=te_drop_original)
    columns = list(X_in.columns)
    require(list(X_ho.columns) == columns and list(X_va.columns) == columns,
            "inner/outer feature columns differ")
    model = build_model(columns)
    model.fit(X_in, y_in)
    p_ho = _predict(model, X_ho)
    p_va = _predict(model, X_va)
    del model
    base_sec = time.perf_counter() - t0
    y_ho_np = np.asarray(y_ho, dtype=int)

    t1 = time.perf_counter()
    X_inner_raw = X_train.iloc[inner_tr].reset_index(drop=True)
    y_inner_raw = y_train.iloc[inner_tr].reset_index(drop=True)
    require(bool(np.array_equal(np.asarray(y_inner_raw), np.asarray(y_in))),
            "inner-train label order mismatch")
    p_cf = crossfit_scores(
        build_model, X_inner_raw, y_inner_raw, k=crossfit_k,
        seed=cfg.seed + fold + CROSSFIT_SEED_OFFSET, te_cols=te_cols, te_m=te_m,
        te_inner_splits=te_inner_splits, te_seed=cfg.seed, te_drop_original=te_drop_original)
    crossfit_sec = time.perf_counter() - t1
    if log is not None:
        log(f"    fold={fold + 1} base fit {base_sec:.1f}s, crossfit k={crossfit_k} {crossfit_sec:.1f}s")

    sources = {
        "none": None,
        "inner_holdout": (y_ho_np, p_ho),
        "inner_train_crossfit": (np.asarray(y_inner_raw, dtype=int), p_cf),
    }
    arms: dict[str, dict[str, Any]] = {}
    for arm, kind, source in ARMS:
        if kind == "none":
            cal = fit_calibrator("none", y_ho_np, p_ho)
        else:
            y_fit, p_fit = sources[source]
            cal = fit_calibrator(kind, y_fit, p_fit)
        q_ho = np.asarray(cal.transform(p_ho), dtype=float)
        q_va = np.asarray(cal.transform(p_va), dtype=float)
        threshold = select_threshold(y_ho_np, q_ho, cfg)
        ho = probability_metrics(y_ho_np, q_ho)
        arms[arm] = {
            "calibrator": kind, "calibration_data": source,
            "valid_probs": q_va,
            "threshold": float(threshold),
            "calibrator_info": cal.info,
            "inner_holdout_log_loss": ho["log_loss"],
            "inner_holdout_ece_ef15": ho["ece_ef15"],
            "inner_holdout_macro_f1_at_threshold": float(
                f1_score(y_ho_np, q_ho >= threshold, average="macro")),
            "valid_rows_at_clip": int(np.sum((q_va <= ISO_CLIP) | (q_va >= 1 - ISO_CLIP)))
            if kind == "isotonic" else 0,
        }
    return {
        "arms": arms,
        "raw_valid_probs": p_va,
        "raw_inner_holdout_log_loss": float(log_loss(y_ho_np, p_ho, labels=[0, 1])),
        "crossfit_inner_train_log_loss": float(log_loss(np.asarray(y_inner_raw), p_cf, labels=[0, 1])),
        "n_inner_train": int(len(X_in)), "n_inner_holdout": int(len(X_ho)),
        "n_model_features": len(columns),
        "base_fit_sec": float(base_sec), "crossfit_sec": float(crossfit_sec),
    }


# =============================================================================
# Pooled outer-valid scoring (the only place that reads outer-valid labels)
# =============================================================================


def probs_sha256(p: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(p, dtype=float).tobytes()).hexdigest()


def score_pooled(y, oof: np.ndarray, folds: Sequence[tuple[np.ndarray, np.ndarray]],
                 thresholds: Sequence[float]) -> dict[str, Any]:
    """Pooled outer-valid metrics with per-fold pre-selected thresholds."""
    y_arr = np.asarray(y, dtype=int)
    oof = np.asarray(oof, dtype=float)
    require(len(thresholds) == len(folds), "one threshold per fold")
    require(bool(np.isfinite(oof).all()), "OOF probabilities incomplete")
    pred = np.zeros_like(y_arr)
    for (_, va), th in zip(folds, thresholds):
        pred[va] = oof[va] >= float(th)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_arr, pred, labels=[0, 1]).ravel())
    per_class = cc.class_metrics(tn, fp, fn, tp)
    macro = float(f1_score(y_arr, pred, average="macro"))
    require(abs(per_class["macro_f1_from_cm"] - macro) < 1e-9, "macro F1 / CM mismatch")
    # Pooled AUC mixes per-fold calibration maps; the within-fold mean is
    # invariant to any strictly monotone per-fold calibrator (Platt).
    auc_within = float(np.mean([roc_auc_score(y_arr[va], oof[va]) for _, va in folds]))
    return {
        **probability_metrics(y_arr, oof),
        "roc_auc_within_fold_mean": auc_within,
        "macro_f1_nested": macro,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        **{k: v for k, v in per_class.items() if k != "macro_f1_from_cm"},
        "per_fold_thresholds": [float(t) for t in thresholds],
    }


def reliability_table(y, oof: np.ndarray) -> pd.DataFrame:
    return pd.concat([equal_frequency_bins(y, oof), equal_width_bins(y, oof)], ignore_index=True)


RUN_METRICS = ("log_loss", "brier", "ece_ef15", "ece_ew10", "roc_auc", "roc_auc_within_fold_mean",
               "macro_f1_nested",
               "precision_delayed", "recall_delayed", "f1_delayed", "calibration_bias")


def paired_vs_none(runs: pd.DataFrame, metrics: Sequence[str] = RUN_METRICS) -> pd.DataFrame:
    """Per seed x model x calibrated arm: metric(arm) - metric(none), same folds."""
    rows = []
    for (seed, model), cell in runs.groupby(["seed", "model"], sort=True):
        cell = cell.set_index("arm")
        require("none" in cell.index, f"missing none arm for {seed}/{model}")
        require(cell.fold_fingerprint.nunique() == 1, "arms used different outer folds")
        base = cell.loc["none"]
        for arm in cell.index:
            if arm == "none":
                continue
            for metric in metrics:
                rows.append({"seed": int(seed), "model": model, "arm": arm,
                             "calibrator": cell.loc[arm, "calibrator"],
                             "calibration_data": cell.loc[arm, "calibration_data"],
                             "metric": metric, "value_none": float(base[metric]),
                             "value_arm": float(cell.loc[arm, metric]),
                             "delta_arm_minus_none": float(cell.loc[arm, metric]) - float(base[metric]),
                             "fold_fingerprint": base.fold_fingerprint})
    return pd.DataFrame(rows)


def stats(values: Sequence[float]) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else None,
            "min": float(v.min()), "max": float(v.max())}


def summarize_deltas(paired: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (model, arm, metric), sub in paired.groupby(["model", "arm", "metric"], sort=True):
        sub = sub.sort_values("seed")
        d = sub.delta_arm_minus_none.to_numpy()
        out.setdefault(model, {}).setdefault(arm, {})[metric] = {
            **stats(d),
            "values_by_seed": {str(int(s)): float(v) for s, v in zip(sub.seed, d)},
            "sign_consistent_across_seeds": bool((d > 0).all() or (d < 0).all()),
        }
    return out
