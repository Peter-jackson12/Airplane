"""Probability calibrators fitted strictly inside outer-train.

The fitting data always comes from the inner holdout that `run_fold_nested_grid()`
carves out of outer-train. Outer-valid rows never reach `fit_calibrator()`.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

KINDS = ("none", "platt", "isotonic")
_EPS = 1e-6


def _check(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or y.size != p.size or y.size == 0:
        raise ValueError("Calibration inputs must be equal-length nonempty 1-D arrays")
    if not np.isfinite(p).all() or p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("Calibration probabilities must be finite and within [0, 1]")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Calibration labels must be binary")
    return y, p


def _logit(p):
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def fit_calibrator(kind: str, y, p) -> Callable[[np.ndarray], np.ndarray]:
    """Return a mapping from raw probability to calibrated probability.

    `none` is the identity and exists so every arm runs the same code path.
    A single-class fitting sample cannot support a calibrator, so it falls back to
    the identity and the caller records that the arm was not fitted.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown calibrator: {kind}")
    y, p = _check(y, p)
    if kind == "none" or np.unique(y).size < 2:
        return lambda q: np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    if kind == "platt":
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(_logit(p).reshape(-1, 1), y)
        return lambda q: model.predict_proba(_logit(np.asarray(q, dtype=float)).reshape(-1, 1))[:, 1]
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(p, y)
    return lambda q: np.clip(model.predict(np.asarray(q, dtype=float)), 0.0, 1.0)


def select_threshold(y, p, grid) -> float:
    """Pick the macro-F1 maximising threshold; ties go to the smallest threshold."""
    y, p = _check(y, p)
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("Threshold grid must be a nonempty 1-D array")
    scores = [f1_score(y, p >= th, average="macro", labels=[0, 1], zero_division=0)
              for th in grid]
    return float(grid[int(np.argmax(scores))])
