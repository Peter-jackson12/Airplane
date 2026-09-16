"""Row-level OOF contracts and descriptive diagnostics; no model fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, roc_auc_score

from .features import TARGET_MAP

OOF_SCHEMA_VERSION = 2
BASE_COLUMNS = ["run_id", "phase_key", "seed", "source_row", "ID", "fold",
                "y_true", "probability", "threshold", "prediction", "n_estimators"]


def labeled_identity(raw: pd.DataFrame) -> pd.DataFrame:
    """Use the same label eligibility as split_labeled; source_row is zero-based."""
    target = raw["Delay"]
    mask = target.notna() & target.astype("string").str.strip().ne("")
    selected = raw.loc[mask].reset_index(drop=True)
    y = selected.Delay.map(TARGET_MAP)
    if y.isna().any():
        raise ValueError("Unknown target label")
    if selected.ID.isna().any() or selected.ID.duplicated().any():
        raise ValueError("OOF requires nonmissing unique IDs")
    result = pd.DataFrame({"source_row": np.flatnonzero(mask), "ID": selected.ID,
                           "y_true": y.astype(int)})
    columns = [c for c in raw.columns if c not in ("ID", "Delay")]
    flags = selected[columns].isna().astype("int8")
    flags.columns = [f"raw_missing__{c}" for c in columns]
    return pd.concat([result, flags], axis=1)


def build_oof_rows(raw, y, probs, folds, thresholds, trees, *, run_id, phase_key,
                   seed, features=None):
    identity = labeled_identity(raw)
    if not np.array_equal(identity.y_true.to_numpy(), np.asarray(y)):
        raise ValueError("OOF identity/target alignment mismatch")
    n = len(identity)
    if len(probs) != n or len(thresholds) != len(folds) or len(trees) != len(folds):
        raise ValueError("OOF dimensions mismatch")
    counts = np.zeros(n, dtype=int)
    fold_id = np.full(n, -1, dtype=int)
    row_threshold = np.full(n, np.nan)
    row_trees = np.zeros(n, dtype=int)
    for k, ((tr, va), threshold, tree) in enumerate(zip(folds, thresholds, trees)):
        va = np.asarray(va)
        if not np.issubdtype(va.dtype, np.integer) or (va < 0).any() or (va >= n).any():
            raise ValueError("Invalid OOF validation indices")
        if np.intersect1d(tr, va).size:
            raise ValueError("OOF train/valid overlap")
        np.add.at(counts, va, 1)
        fold_id[va], row_threshold[va], row_trees[va] = k, threshold, tree
    if not np.all(counts == 1):
        raise ValueError("Every labeled row must be scored exactly once")
    identity["run_id"], identity["phase_key"], identity["seed"] = run_id, phase_key, seed
    identity["fold"], identity["probability"] = fold_id, np.asarray(probs)
    identity["threshold"], identity["n_estimators"] = row_threshold, row_trees
    identity["prediction"] = (np.asarray(probs) >= row_threshold).astype(int)
    if features is not None:
        if len(features) != n:
            raise ValueError("OOF feature alignment mismatch")
        # LabelEncoder converts the MISSING token to an integer; use its saved mapping.
        for name in ("Airline", "Dep_Hour", "Arr_Hour"):
            if name in features:
                s = features[name].reset_index(drop=True)
                if name == "Airline":
                    if "missing_category_codes" not in features.attrs:
                        raise ValueError("Missing category-code metadata for Airline")
                    code = features.attrs["missing_category_codes"].get(name)
                    flag = s.isna() | (s.eq(code) if code is not None else False)
                else:
                    flag = s.isna() | s.lt(0)
                identity[f"feature_missing__{name}"] = flag.astype("int8")
    extra = [c for c in identity if c not in BASE_COLUMNS]
    result = identity[BASE_COLUMNS + extra]
    validate_oof_rows(result, run_id=run_id, phase_key=phase_key, seed=seed,
                      n_rows=n, n_splits=len(folds))
    return result


def validate_oof_rows(rows, *, run_id, phase_key, seed, n_rows, n_splits):
    if list(rows.columns[:len(BASE_COLUMNS)]) != BASE_COLUMNS or len(rows) != n_rows or not n_rows:
        raise ValueError("OOF schema/row count mismatch")
    extras = list(rows.columns[len(BASE_COLUMNS):])
    if not extras or any(not c.startswith(("raw_missing__", "feature_missing__")) for c in extras):
        raise ValueError("OOF missingness schema mismatch")
    if rows.isna().any().any():
        raise ValueError("OOF contains missing values")
    for name, value in (("run_id", run_id), ("phase_key", phase_key), ("seed", seed)):
        if not rows[name].eq(value).all():
            raise ValueError(f"OOF {name} mismatch")
    if rows.ID.duplicated().any() or rows.source_row.duplicated().any():
        raise ValueError("OOF duplicate identity")
    if not rows.source_row.is_monotonic_increasing or (rows.source_row < 0).any():
        raise ValueError("OOF source row order mismatch")
    for col in ("source_row", "fold", "n_estimators"):
        x = rows[col].to_numpy(dtype=float)
        if not np.isfinite(x).all() or not np.equal(x, np.floor(x)).all():
            raise ValueError(f"OOF invalid integer: {col}")
    if set(rows.fold) != set(range(n_splits)) or (rows.n_estimators <= 0).any():
        raise ValueError("OOF fold/tree mismatch")
    for col in ("y_true", "prediction", *extras):
        if not rows[col].isin([0, 1]).all():
            raise ValueError(f"OOF invalid binary: {col}")
    for col in ("probability", "threshold"):
        if not rows[col].between(0, 1).all():
            raise ValueError(f"OOF invalid {col}")
    if not np.array_equal(rows.prediction, (rows.probability >= rows.threshold).astype(int)):
        raise ValueError("OOF prediction/threshold mismatch")
    if (rows.groupby("fold")[["threshold", "n_estimators"]].nunique() != 1).any().any():
        raise ValueError("OOF fold settings are inconsistent")


def calibration_bins(y, p, n_bins=10):
    """Fixed-width bins, [left,right), final bin includes 1; empty means stay NaN."""
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    indices = np.minimum((p * n_bins).astype(int), n_bins - 1)
    result = []
    for k in range(n_bins):
        m = indices == k
        result.append(dict(bin=k, lower=k / n_bins, upper=(k + 1) / n_bins,
                           n=int(m.sum()), mean_probability=float(p[m].mean()) if m.any() else np.nan,
                           observed_rate=float(y[m].mean()) if m.any() else np.nan))
    return pd.DataFrame(result)


def metrics(rows):
    y, p, pred = (rows[c].to_numpy() for c in ("y_true", "probability", "prediction"))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    bins = calibration_bins(y, p)
    ratio = lambda a, b: float(a / b) if b else np.nan
    return dict(n=len(y), positives=int(y.sum()), positive_rate=float(y.mean()),
                mean_probability=float(p.mean()), calibration_bias=float(p.mean() - y.mean()),
                macro_f1=float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
                log_loss=float(log_loss(y, p, labels=[0, 1])),
                roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
                brier=float(np.mean((p - y) ** 2)),
                ece_10=float((bins.n * (bins.mean_probability - bins.observed_rate).abs()).sum() / len(y)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                fpr=ratio(fp, tn + fp), fnr=ratio(fn, tp + fn),
                error_rate=ratio(fp + fn, len(y)))


def diagnostic_groups(rows):
    yield "overall", "all", np.ones(len(rows), dtype=bool)
    raw_cols = [c for c in rows if c.startswith("raw_missing__")]
    for col in raw_cols + [c for c in rows if c.startswith("feature_missing__")]:
        for value, label in ((0, "observed"), (1, "missing")):
            yield col, label, rows[col].eq(value).to_numpy()
    counts = rows[raw_cols].sum(axis=1)
    for value in sorted(counts.unique()):
        yield "raw_missing_count", str(value), counts.eq(value).to_numpy()
    dep, arr = "raw_missing__Estimated_Departure_Time", "raw_missing__Estimated_Arrival_Time"
    if dep in rows and arr in rows:
        for d, a, label in ((0, 0, "both_observed"), (1, 0, "departure_missing"),
                            (0, 1, "arrival_missing"), (1, 1, "both_missing")):
            yield "raw_time_pattern", label, (rows[dep].eq(d) & rows[arr].eq(a)).to_numpy()


def diagnose(rows):
    scores, bins = [], []
    for dimension, group, mask in diagnostic_groups(rows):
        context = dict(phase_key=rows.phase_key.iloc[0], seed=int(rows.seed.iloc[0]),
                       dimension=dimension, group=group)
        part = rows.loc[mask]
        if part.empty:
            scores.append({**context, "n": 0, "positives": 0})
            continue
        scores.append({**context, **metrics(part)})
        table = calibration_bins(part.y_true, part.probability)
        for k, v in context.items():
            table[k] = v
        bins.append(table)
    return pd.DataFrame(scores), pd.concat(bins, ignore_index=True)
