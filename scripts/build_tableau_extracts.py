"""Build Tableau-ready extracts for three portfolio dashboards (read-only, no training).

Dashboards
1. 지연 패턴 탐색 — labeled rows of data/train.csv (255,001 rows).
2. 날씨와 지연 — labeled, date-attributed rows of the 10-minute weather join
   (180,332 rows), plus the tracked weather on/off confusion matrices.
3. 모델이 틀리는 곳 — the existing P6_clean row-level OOF predictions
   (255,001 labeled rows x 3 split seeds) and the tracked classifier comparison.
   Also every tracked classifier run on the weather_eval rows in one long table
   (comparison, tuning, grid extension, calibration arms) and the calibration
   reliability bins (``model_comparison_all.csv``, ``model_calibration_reliability.csv``).

Nothing is fitted. Weather rows are loaded with the comparison runner's own
validated loader (``notebooks.run_weather_model_comparison.load_inputs``), which
re-checks the join manifest SHA-256 values before returning frames.

Outputs
- ``output/tableau/*.csv``: aggregated tables only (tracked).
- ``output/tableau/tableau_extracts_manifest.json``: input hashes, row counts,
  and reconciliation results.
- ``data/tableau/flights_rowlevel.csv``: optional row-level table (ignored by
  Git through ``data/``; never commit it — redistribution terms unconfirmed).

Run::

    PYTHONUTF8=1 uv run --locked --offline python -u scripts/build_tableau_extracts.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.weather_model import ROLES  # noqa: E402

RAW_PATH = ROOT / "data/train.csv"
MAPPING_PATH = ROOT / "output/baseline_recovery_v2_weather_scope_fix_20260918_mapping_table.csv"
WEATHER_RUNS_PATH = ROOT / "output/baseline_recovery_v2_weather_model_compare_20260922_weather_model_runs.csv"
CLASSIFIER_RUNS_PATH = ROOT / "output/baseline_recovery_v2_classifier_compare_20261008_runs.csv"
CLASSIFIER_SOURCES = {
    # experiment -> (runs CSV, summary JSON); all share the 180,332 weather_eval rows,
    # outer folds (fingerprint-checked by each runner) and split seeds 42/1/7.
    "classifier_compare_20261008": (
        "output/baseline_recovery_v2_classifier_compare_20261008_runs.csv",
        "output/baseline_recovery_v2_classifier_compare_20261008_summary.json"),
    "classifier_tuning_20261008": (
        "output/baseline_recovery_v2_classifier_tuning_20261008_runs.csv",
        "output/baseline_recovery_v2_classifier_tuning_20261008_summary.json"),
    "classifier_grid_ext_20261008_lgbm": (
        "output/baseline_recovery_v2_classifier_grid_ext_20261008_lgbm_runs.csv",
        "output/baseline_recovery_v2_classifier_grid_ext_20261008_lgbm_summary.json"),
    "classifier_grid_ext_20261008_rf": (
        "output/baseline_recovery_v2_classifier_grid_ext_20261008_rf_runs.csv",
        "output/baseline_recovery_v2_classifier_grid_ext_20261008_rf_summary.json"),
    "classifier_calibration_20261008": (
        "output/baseline_recovery_v2_classifier_calibration_20261008_runs.csv",
        "output/baseline_recovery_v2_classifier_calibration_20261008_summary.json"),
}
CALIBRATION_BINS_PATH = ROOT / "output/baseline_recovery_v2_classifier_calibration_20261008_reliability_bins.csv"
# model key -> (family, display label, search budget label, number of configurations)
CLASSIFIER_MODEL_SPECS = {
    "lightgbm": ("lightgbm", "LightGBM",
                 "고정 설정(learning_rate 0.05, num_leaves 63) × 트리 수 8개", 1),
    "logistic_regression": ("logistic_regression", "Logistic Regression", "규제 강도 C 5개", 5),
    "random_forest": ("random_forest", "Random Forest",
                      "min_samples_leaf 3개 × max_features 2개 (300그루)", 6),
    "lightgbm_tuned": ("lightgbm", "LightGBM",
                       "24개 설정 × 트리 수 8개 (학습률 0.03~0.1, 잎 31~127)", 24),
    "random_forest_tuned": ("random_forest", "Random Forest",
                            "9개 설정 (min_samples_leaf 10/25/50 × max_features 0.5/0.7/1.0)", 9),
    "lightgbm_ext": ("lightgbm", "LightGBM",
                     "18개 설정 × 트리 수 최대 10개 (학습률 0.01~0.03, 잎 127/255/511, 상한 1500)", 18),
    "random_forest_ext": ("random_forest", "Random Forest",
                          "15개 설정 (min_samples_leaf 25~400 × max_features 0.5/0.7/1.0)", 15),
}
CALIBRATION_ARM_LABELS = {
    "none": "보정 없음",
    "platt_inner_holdout": "Platt · 내부 검증 행 적합(선택 행 재사용)",
    "isotonic_inner_holdout": "Isotonic · 내부 검증 행 적합(선택 행 재사용)",
    "platt_crossfit": "Platt · 내부 학습 교차적합",
    "isotonic_crossfit": "Isotonic · 내부 학습 교차적합",
}
BINNING_LABELS = {"equal_frequency_15": "동일 빈도 15구간(주 지표)",
                  "equal_width_10": "동일 폭 10구간(보조)"}
COMPARISON_METRICS = ("macro_f1_nested", "log_loss", "roc_auc", "precision_delayed",
                      "recall_delayed", "f1_delayed", "f1_not_delayed", "brier", "ece_ef15",
                      "ece_ew10", "calibration_bias", "mean_probability")
OOF_GROUPS_PATH = ROOT / "output/baseline_recovery_v2_oof_20260916_v2_groups.csv"
OOF_RUN_PREFIX = "baseline_recovery_v2_oof_20260916_v2"
OOF_PHASE = "P6_clean"
SEEDS = (42, 1, 7)
DEFAULT_OUT = ROOT / "output/tableau"
DEFAULT_ROWLEVEL = ROOT / "data/tableau/flights_rowlevel.csv"

DEP, ARR = "Estimated_Departure_Time", "Estimated_Arrival_Time"
ROUTE_MIN_ROWS_D3 = 100
ROUTE_OTHER_LABEL = "(기타: 라벨 100행 미만 노선)"
MISSING_LABEL = "(결측)"
EXPECTED = {
    "labeled_rows": 255_001, "labeled_delayed": 45_000,
    "weather_rows": 180_332, "weather_delayed": 31_805,
    "both_time_missing_rows": 3_031, "both_time_missing_delayed": 519,
}

POPULATIONS = {
    "labeled_all": "라벨 전체 255,001행 (data/train.csv의 Delay 비결측 행)",
    "weather_eval": "날씨 평가 집단 180,332행 (날짜 귀속 + 라벨, 10분 가정 결합)",
    "both_time_missing": "원본 출발·도착 예정 시각 모두 결측인 라벨 3,031행",
    "oof_p6_clean": "P6_clean OOF: 라벨 255,001행 × 분할 시드 3개(42/1/7)",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def lower_priority() -> None:
    """Run at idle/low CPU priority so a concurrent long experiment is not slowed."""
    try:
        if os.name == "nt":
            import ctypes
            idle = 0x00000040  # IDLE_PRIORITY_CLASS
            kernel32 = ctypes.windll.kernel32
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), idle)
        else:
            os.nice(19)
    except Exception:  # pragma: no cover - best effort only
        pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else resolved.as_posix()


# =============================================================================
# Pure functions (tested on synthetic data in tests/test_tableau_extracts.py)
# =============================================================================

def delayed_flag(delay: pd.Series) -> pd.Series:
    """Map the raw label to 0/1. Missing labels are rejected (unlabeled is not negative)."""
    require(delay.notna().all(), "unlabeled rows must be removed before aggregation")
    values = delay.astype(str).str.strip()
    require(values.isin(["Delayed", "Not_Delayed"]).all(), "unexpected Delay label")
    return values.eq("Delayed").astype("int64")


def fill_category(values: pd.Series, missing_label: str = MISSING_LABEL) -> pd.Series:
    """Keep raw categories; make missing an explicit bucket so denominators stay whole."""
    out = values.astype("object").where(values.notna(), missing_label)
    return out.astype(str)


def hour_bucket(hhmm: pd.Series) -> pd.DataFrame:
    """Raw HHMM -> hour bucket. Missing raw time stays its own bucket (not imputed).

    ``2400`` keeps hour 24 (raw notation) rather than being silently rolled over.
    """
    val = pd.to_numeric(hhmm, errors="coerce")
    hour = np.floor(val / 100.0)
    order = hour.fillna(-1).astype("int64")
    label = order.map(lambda h: "결측(원본 시각 없음)" if h < 0
                      else ("24 (2400 표기)" if h == 24 else f"{h:02d}"))
    return pd.DataFrame({"hour_order": order.to_numpy(), "hour_label": label.to_numpy()})


def time_pattern(dep: pd.Series, arr: pd.Series) -> pd.Series:
    """Same grouping as src/oof.py raw_time_pattern (raw missingness of both times)."""
    d, a = dep.isna().to_numpy(), arr.isna().to_numpy()
    out = np.select([~d & ~a, d & ~a, ~d & a, d & a],
                    ["both_observed", "departure_missing", "arrival_missing", "both_missing"],
                    default="invalid")
    return pd.Series(out, index=dep.index)


def rate_table(frame: pd.DataFrame, by: list[str], *, population: str,
               delayed_col: str = "delayed") -> pd.DataFrame:
    """Group counts with explicit denominators. NaN keys are rejected (fill first)."""
    require(not frame[by].isna().any().any(), f"NaN group keys in {by}; fill before grouping")
    require(frame[delayed_col].isin([0, 1]).all(), "delayed flag must be 0/1")
    grouped = frame.groupby(by, sort=True, observed=True)[delayed_col]
    out = grouped.agg(n_rows="size", n_delayed="sum").reset_index()
    out["n_not_delayed"] = out.n_rows - out.n_delayed
    out["delay_rate"] = out.n_delayed / out.n_rows
    total = len(frame)
    out["share_of_population"] = out.n_rows / total
    out.insert(0, "population", population)
    out.insert(1, "population_rows", total)
    out.insert(2, "population_delayed", int(frame[delayed_col].sum()))
    return out


# ---- weather bins -----------------------------------------------------------

TRACE = 0.0001
WEATHER_BIN_SPECS = {
    # feature: (Korean label, unit, edges [lower, upper) with the last upper inclusive=inf)
    "tmpf": ("기온", "°F", [-np.inf, 32, 50, 68, 86, np.inf]),
    "sknt": ("풍속", "knot", [0, 6, 11, 16, 21, 31, np.inf]),
    "vsby": ("시정", "mile", [0, 1, 3, 5, 10, np.inf]),
    "p01i": ("1시간 강수", "inch", [0, 0.05, 0.10, np.inf]),
    "age_minutes": ("관측 나이", "분", [0, 30, 60, 91]),
}


def _interval_label(lo: float, hi: float, unit: str) -> str:
    if np.isinf(lo):
        return f"< {hi:g} {unit}"
    if np.isinf(hi):
        return f"≥ {lo:g} {unit}"
    return f"[{lo:g}, {hi:g}) {unit}"


def bin_weather_values(values: pd.Series, matched: pd.Series, feature: str) -> pd.DataFrame:
    """Assign every row exactly one bin, including unmatched and field-missing rows.

    - matched == 0                    -> '미결합(관측 보고 없음)'
    - matched == 1 and value NaN      -> '결합됨·필드 결측(M)'
    - sknt == 0                       -> '0 knot(무풍 보고)'
    - p01i == 0                       -> '0 inch(강수 없음 보고)'
    - p01i == 0.0001                  -> '미량(trace, 0.0001 표기)' (not a measured amount)
    - otherwise half-open numeric intervals from WEATHER_BIN_SPECS
    """
    label_ko, unit, edges = WEATHER_BIN_SPECS[feature]
    v = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    m = pd.to_numeric(matched, errors="raise").to_numpy()
    require(np.isin(m, [0, 1]).all(), "matched must be 0/1")
    require(np.isnan(v[m == 0]).all(), f"{feature}: unmatched rows must not carry values")
    n = len(v)
    label = np.full(n, None, dtype=object)
    order = np.full(n, -99, dtype=np.int64)
    kind = np.full(n, None, dtype=object)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)

    def assign(mask, lab, ordv, knd, lo=np.nan, hi=np.nan):
        mask = mask & pd.isna(label)
        label[mask] = lab
        order[mask] = ordv
        kind[mask] = knd
        lower[mask] = lo
        upper[mask] = hi

    assign(m == 0, "미결합(관측 보고 없음)", 900, "unmatched")
    assign((m == 1) & np.isnan(v), "결합됨·필드 결측(M)", 800, "field_missing")
    finite = (m == 1) & ~np.isnan(v)
    start = 0
    if feature == "sknt":
        assign(finite & (v == 0), "0 knot(무풍 보고)", 0, "special", 0, 0)
        start = 1
    if feature == "p01i":
        assign(finite & (v == 0), "0 inch(강수 없음 보고)", 0, "special", 0, 0)
        assign(finite & np.isclose(v, TRACE, rtol=0, atol=1e-9),
               "미량(trace, 0.0001 표기)", 1, "special", TRACE, TRACE)
        start = 2
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if feature == "p01i" and i == 0:
            mask = finite & (v > TRACE) & (v < hi)
            lab, lo = f"(0.0001, {hi:g}) {unit}", TRACE
        elif feature == "sknt" and i == 0:
            mask = finite & (v > 0) & (v < hi)
            lab = f"(0, {hi:g}) {unit}"
        elif feature == "age_minutes" and np.isfinite(hi) and i == len(edges) - 2:
            mask = finite & (v >= lo) & (v < hi)
            lab = f"[{lo:g}, {hi - 1:g}] {unit}"
        else:
            mask = finite & (v >= lo) & (v < hi)
            lab = _interval_label(lo, hi, unit)
        assign(mask, lab, start + i, "value", lo, hi)
    require(not pd.isna(label).any(),
            f"{feature}: {int(pd.isna(label).sum())} rows fell outside every bin")
    return pd.DataFrame({"bin_label": label, "bin_order": order, "bin_kind": kind,
                         "bin_lower": lower, "bin_upper": upper})


def matched_status(origin_matched: pd.Series, dest_matched: pd.Series) -> pd.DataFrame:
    o = origin_matched.to_numpy().astype(int)
    d = dest_matched.to_numpy().astype(int)
    label = np.select([(o == 1) & (d == 1), (o == 1) & (d == 0), (o == 0) & (d == 1)],
                      ["양쪽 공항 결합", "출발 공항만 결합", "도착 공항만 결합"], "양쪽 미결합")
    order = np.select([(o == 1) & (d == 1), (o == 1) & (d == 0), (o == 0) & (d == 1)],
                      [0, 1, 2], 3)
    kind = np.select([(o == 1) & (d == 1), (o == 1) | (d == 1)], ["value", "partial"], "unmatched")
    return pd.DataFrame({"bin_label": label, "bin_order": order, "bin_kind": kind,
                         "bin_lower": np.nan, "bin_upper": np.nan})


def weather_bins_long(weather: pd.DataFrame, delayed: pd.Series, *,
                      population: str = "weather_eval") -> pd.DataFrame:
    """Long table: one row per (role, feature, bin). Each (role, feature) sums to all rows."""
    require(len(weather) == len(delayed), "weather/label length mismatch")
    y = pd.Series(np.asarray(delayed), name="delayed")
    parts = []
    for role in ROLES:
        matched = weather[f"weather_{role}_matched"].reset_index(drop=True)
        for feature in WEATHER_BIN_SPECS:
            col = f"weather_{role}_{feature}"
            bins = bin_weather_values(weather[col].reset_index(drop=True), matched, feature)
            frame = pd.concat([bins, y], axis=1)
            frame["role"] = role
            frame["feature"] = feature
            parts.append(frame)
    both = matched_status(weather["weather_origin_matched"], weather["weather_destination_matched"])
    both = pd.concat([both, y], axis=1)
    both["role"] = "both"
    both["feature"] = "matched_status"
    parts.append(both)
    rows = pd.concat(parts, ignore_index=True)
    rows["bin_lower"] = rows.bin_lower.fillna(-9999.0)  # groupby-safe sentinel, restored below
    rows["bin_upper"] = rows.bin_upper.fillna(-9999.0)
    keys = ["role", "feature", "bin_order", "bin_label", "bin_kind", "bin_lower", "bin_upper"]
    grouped = rows.groupby(keys, sort=True)["delayed"].agg(n_rows="size", n_delayed="sum").reset_index()
    for c in ("bin_lower", "bin_upper"):
        # infinite edges are written blank so Tableau keeps the column numeric
        grouped[c] = grouped[c].where((grouped[c] != -9999.0) & np.isfinite(grouped[c]), np.nan)
    grouped["n_not_delayed"] = grouped.n_rows - grouped.n_delayed
    grouped["delay_rate"] = grouped.n_delayed / grouped.n_rows
    total = len(weather)
    grouped["share_of_population"] = grouped.n_rows / total
    labels = {f: WEATHER_BIN_SPECS[f][0] for f in WEATHER_BIN_SPECS}
    units = {f: WEATHER_BIN_SPECS[f][1] for f in WEATHER_BIN_SPECS}
    labels["matched_status"], units["matched_status"] = "결합 상태", ""
    grouped.insert(2, "feature_label_ko", grouped.feature.map(labels))
    grouped.insert(3, "unit", grouped.feature.map(units))
    role_ko = {"origin": "출발 공항", "destination": "도착 공항", "both": "출발+도착"}
    grouped.insert(1, "role_ko", grouped.role.map(role_ko))
    grouped.insert(0, "population", population)
    grouped.insert(1, "population_rows", total)
    grouped.insert(2, "population_delayed", int(y.sum()))
    check = grouped.groupby(["role", "feature"]).n_rows.sum()
    require((check == total).all(), "weather bins do not cover the population")
    return grouped


# ---- model evidence ---------------------------------------------------------

CELL_SPECS = (("tn", "Not_Delayed", "Not_Delayed", "TN"), ("fp", "Not_Delayed", "Delayed", "FP"),
              ("fn", "Delayed", "Not_Delayed", "FN"), ("tp", "Delayed", "Delayed", "TP"))


def class_metrics(tn, fp, fn, tp) -> dict:
    tn, fp, fn, tp = (np.asarray(x, dtype=float) for x in (tn, fp, fn, tp))
    with np.errstate(divide="ignore", invalid="ignore"):
        prec_d = tp / (tp + fp)
        rec_d = tp / (tp + fn)
        f1_d = 2 * tp / (2 * tp + fp + fn)
        prec_n = tn / (tn + fn)
        rec_n = tn / (tn + fp)
        f1_n = 2 * tn / (2 * tn + fn + fp)
        fpr = fp / (fp + tn)
        err = (fp + fn) / (tn + fp + fn + tp)
    return {"precision_delayed": prec_d, "recall_delayed": rec_d, "f1_delayed": f1_d,
            "precision_not_delayed": prec_n, "recall_not_delayed": rec_n,
            "f1_not_delayed": f1_n, "macro_f1_from_counts": (f1_d + f1_n) / 2,
            "false_positive_rate": fpr, "error_rate": err}


def confusion_long(runs: pd.DataFrame, *, experiment: str, variant_col: str,
                   population: str) -> pd.DataFrame:
    """Runs CSV (one row per seed x variant) -> one row per confusion cell."""
    out = []
    for _, r in runs.iterrows():
        total = int(r.tn + r.fp + r.fn + r.tp)
        require(total == int(r.n_rows), f"{experiment}: confusion total != n_rows")
        require(int(r.fn + r.tp) == int(r.positive_rows), f"{experiment}: positives mismatch")
        for col, actual, pred, cell in CELL_SPECS:
            out.append({"population": population, "experiment": experiment,
                        "seed": int(r.seed), "variant": r[variant_col], "cell": cell,
                        "actual": actual, "predicted": pred, "n": int(r[col]),
                        "n_rows": total, "actual_class_rows": int(r.positive_rows)
                        if actual == "Delayed" else total - int(r.positive_rows),
                        "share_of_rows": int(r[col]) / total})
    return pd.DataFrame(out)


def metrics_long(runs: pd.DataFrame, *, experiment: str, variant_col: str,
                 population: str, metrics: tuple[str, ...]) -> pd.DataFrame:
    derived = class_metrics(runs.tn, runs.fp, runs.fn, runs.tp)
    out = []
    for i, (_, r) in enumerate(runs.iterrows()):
        values = {m: float(r[m]) for m in metrics}
        values.update({k: float(v[i]) for k, v in derived.items()})
        for metric, value in values.items():
            out.append({"population": population, "experiment": experiment,
                        "seed": int(r.seed), "variant": r[variant_col], "metric": metric,
                        "value": value, "n_rows": int(r.n_rows),
                        "positive_rows": int(r.positive_rows),
                        "source": "tracked runs CSV" if metric in metrics
                        else "derived from tracked tn/fp/fn/tp"})
    return pd.DataFrame(out)


def oof_error_table(frame: pd.DataFrame, by: str, *, population: str) -> pd.DataFrame:
    """Per seed x group confusion counts from row-level OOF predictions.

    ``frame`` columns: seed, y_true (0/1), prediction (0/1), and ``by``.
    """
    require(frame[by].notna().all(), f"NaN in {by}")
    f = frame.assign(
        tn=((frame.y_true == 0) & (frame.prediction == 0)).astype("int64"),
        fp=((frame.y_true == 0) & (frame.prediction == 1)).astype("int64"),
        fn=((frame.y_true == 1) & (frame.prediction == 0)).astype("int64"),
        tp=((frame.y_true == 1) & (frame.prediction == 1)).astype("int64"))
    g = f.groupby(["seed", by], sort=True)[["tn", "fp", "fn", "tp"]].sum().reset_index()
    g = g.rename(columns={by: "group"})
    g["n_rows"] = g[["tn", "fp", "fn", "tp"]].sum(axis=1)
    g["n_delayed"] = g.fn + g.tp
    g["delay_rate"] = g.n_delayed / g.n_rows
    for k, v in class_metrics(g.tn, g.fp, g.fn, g.tp).items():
        if k in {"precision_delayed", "recall_delayed", "false_positive_rate", "error_rate"}:
            g[k] = v
    seed_rows = f.groupby("seed").size()
    g["seed_population_rows"] = g.seed.map(seed_rows)
    g.insert(0, "population", population)
    g.insert(1, "dimension", by)
    return g


def error_stability_table(per_row: pd.DataFrame, by: str, *, population: str) -> pd.DataFrame:
    """``per_row``: one row per labeled row with error_seed_count (0..3) and ``by``."""
    require(per_row.error_seed_count.between(0, 3).all(), "error_seed_count outside 0..3")
    counts = pd.crosstab(per_row[by], per_row.error_seed_count)
    counts = counts.reindex(columns=[0, 1, 2, 3], fill_value=0)
    counts.columns = [f"n_wrong_{c}_of_3" for c in counts.columns]
    counts = counts.reset_index().rename(columns={by: "group"})
    counts["n_rows"] = counts[[f"n_wrong_{c}_of_3" for c in range(4)]].sum(axis=1)
    counts["share_wrong_all_3"] = counts.n_wrong_3_of_3 / counts.n_rows
    counts["share_wrong_none"] = counts.n_wrong_0_of_3 / counts.n_rows
    counts.insert(0, "population", population)
    counts.insert(1, "dimension", by)
    return counts


def comparison_long(runs: pd.DataFrame, *, experiment: str, population: str) -> pd.DataFrame:
    """One classifier runs CSV -> one row per (seed, condition, model, arm, metric).

    Runs without a ``condition`` column are weather_on (the 20261008 comparison); runs
    without an ``arm`` column are uncalibrated. Only metrics present in the CSV are kept.
    Delayed precision/recall/F1 present in the CSV must match its tn/fp/fn/tp.
    """
    require({"seed", "model", "n_rows", "positive_rows", "tn", "fp", "fn", "tp"} <= set(runs.columns),
            f"{experiment}: runs CSV lacks required columns")
    unknown = set(runs.model) - set(CLASSIFIER_MODEL_SPECS)
    require(not unknown, f"{experiment}: unknown models {sorted(unknown)}")
    require(((runs.tn + runs.fp + runs.fn + runs.tp) == runs.n_rows).all(),
            f"{experiment}: confusion total != n_rows")
    require(((runs.fn + runs.tp) == runs.positive_rows).all(), f"{experiment}: positives mismatch")
    derived = class_metrics(runs.tn, runs.fp, runs.fn, runs.tp)
    for col in ("precision_delayed", "recall_delayed", "f1_delayed", "f1_not_delayed"):
        if col in runs.columns:
            require(np.allclose(runs[col].to_numpy(float), np.asarray(derived[col], float),
                                rtol=0, atol=1e-12), f"{experiment}: {col} differs from tn/fp/fn/tp")
    metrics = [m for m in COMPARISON_METRICS if m in runs.columns]
    out = []
    for _, r in runs.iterrows():
        family, label, budget, n_cfg = CLASSIFIER_MODEL_SPECS[r.model]
        arm = r["arm"] if "arm" in runs.columns else "none"
        require(arm in CALIBRATION_ARM_LABELS, f"{experiment}: unknown arm {arm}")
        base = {"population": population, "experiment": experiment,
                "condition": r["condition"] if "condition" in runs.columns else "weather_on",
                "model": r.model, "model_family": family, "model_label": label,
                "search_budget": budget, "n_configurations": n_cfg,
                "calibration_arm": arm, "calibration_arm_ko": CALIBRATION_ARM_LABELS[arm],
                "calibrator": r["calibrator"] if "calibrator" in runs.columns else "none",
                "calibration_data": r["calibration_data"] if "calibration_data" in runs.columns
                else "none",
                "seed": int(r.seed), "n_rows": int(r.n_rows), "positive_rows": int(r.positive_rows)}
        for metric in metrics:
            out.append({**base, "metric": metric, "value": float(r[metric])})
    return pd.DataFrame(out)


def summary_means(summary: dict, experiment: str) -> dict[tuple[str, str, str], dict[str, float]]:
    """(condition, model, arm) -> {metric: 3-seed mean} as recorded in a tracked summary JSON."""
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    if "levels" in summary:  # calibration: levels[model][arm][metric]
        for model, arms in summary["levels"].items():
            for arm, mets in arms.items():
                out[("weather_on", model, arm)] = {k: v["mean"] for k, v in mets.items()}
        return out
    for key, mets in summary["models"].items():
        condition, model = key.split("/", 1) if "/" in key else ("weather_on", key)
        out[(condition, model, "none")] = {k: v["mean"] for k, v in mets.items()
                                           if isinstance(v, dict) and "mean" in v}
    require(out, f"{experiment}: no model means in summary")
    return out


def reconcile_comparison(long: pd.DataFrame, means: dict, *, experiment: str,
                         seeds: tuple[int, ...], rows: int, positives: int) -> dict[str, bool]:
    """Every (condition, model, arm) has the expected seeds/denominators and its 3-seed
    means of Macro F1 / LogLoss / ROC-AUC equal the tracked summary JSON."""
    checks: dict[str, bool] = {}
    part = long[long.experiment == experiment]
    for (cond, model, arm), g in part.groupby(["condition", "model", "calibration_arm"]):
        tag = f"comparison_{experiment}_{cond}_{model}_{arm}"
        checks[f"{tag}_seeds"] = sorted(g.seed.unique().tolist()) == sorted(seeds)
        checks[f"{tag}_denominators"] = bool((g.n_rows == rows).all() and (g.positive_rows == positives).all())
        ref = means.get((cond, model, arm))
        ok = ref is not None
        for metric in ("macro_f1_nested", "log_loss", "roc_auc"):
            vals = g.loc[g.metric == metric, "value"]
            ok = ok and len(vals) == len(seeds) and abs(float(vals.mean()) - float(ref[metric])) < 1e-12
        checks[f"{tag}_means_vs_summary"] = bool(ok)
    checks[f"comparison_{experiment}_covers_summary"] = (
        set(means) == set(map(tuple, part[["condition", "model", "calibration_arm"]]
                              .drop_duplicates().to_numpy().tolist())))
    return checks


def reliability_long(bins: pd.DataFrame, *, population: str) -> pd.DataFrame:
    """Calibration reliability bins (one row per seed x model x arm x binning x bin), with
    counts that can be summed across seeds: ``n_delayed`` and ``sum_probability``.

    Empty equal-width bins keep ``n = 0`` and blank probability/rate columns.
    """
    need = {"seed", "model", "arm", "calibrator", "calibration_data", "binning", "bin", "lower",
            "upper", "n", "mean_probability", "observed_rate", "abs_gap"}
    require(need <= set(bins.columns), "reliability bins lack required columns")
    require(set(bins.binning) <= set(BINNING_LABELS), "unknown binning")
    empty = bins.n == 0
    require(bins.loc[~empty, ["mean_probability", "observed_rate", "abs_gap"]].notna().all().all(),
            "non-empty bin without probability/rate")
    pos = (bins.observed_rate * bins.n).fillna(0.0)
    require(float((pos - pos.round()).abs().max()) < 1e-3, "observed_rate x n is not a count")
    out = bins.copy()
    out["n_delayed"] = pos.round().astype("int64")
    out["sum_probability"] = (bins.mean_probability * bins.n).fillna(0.0)
    out["gap_signed"] = bins.mean_probability - bins.observed_rate
    out.insert(0, "population", population)
    out.insert(1, "experiment", "classifier_calibration_20261008")
    out.insert(4, "model_label", out.model.map(lambda m: CLASSIFIER_MODEL_SPECS[m][1]))
    out.insert(6, "calibration_arm_ko", out.arm.map(CALIBRATION_ARM_LABELS))
    out.insert(10, "binning_ko", out.binning.map(BINNING_LABELS))
    require(out.model_label.notna().all() and out.calibration_arm_ko.notna().all(),
            "unknown model or arm in reliability bins")
    return out.rename(columns={"arm": "calibration_arm"})


def reconcile_reliability(rel: pd.DataFrame, runs: pd.DataFrame, *, rows: int,
                          positives: int) -> dict[str, bool]:
    """Bins cover every evaluated row once per binning, and n-weighted |gap| reproduces
    the run-level ECE recorded in the runs CSV."""
    checks: dict[str, bool] = {}
    ece_col = {"equal_frequency_15": "ece_ef15", "equal_width_10": "ece_ew10"}
    ref = runs.set_index(["seed", "model", "arm"])
    for (seed, model, arm, binning), g in rel.groupby(["seed", "model", "calibration_arm", "binning"]):
        tag = f"reliability_{model}_{arm}_{binning}_seed{seed}"
        checks[f"{tag}_rows"] = int(g.n.sum()) == rows and int(g.n_delayed.sum()) == positives
        ece = float((g.n * g.abs_gap.fillna(0.0)).sum() / g.n.sum())
        checks[f"{tag}_ece_vs_runs"] = abs(ece - float(ref.loc[(seed, model, arm), ece_col[binning]])) < 1e-9
    checks["reliability_covers_all_runs"] = (
        len(rel.groupby(["seed", "model", "calibration_arm"])) == len(runs))
    return checks


def write_csv(frame: pd.DataFrame, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n", encoding="utf-8", float_format="%.10g")
    os.replace(tmp, path)
    data = path.read_bytes()
    require(b"\r" not in data, f"CR found in {path}")
    return {"path": rel(path), "rows": int(len(frame)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


# =============================================================================
# Loading (local data; not exercised by Git-only tests)
# =============================================================================

def load_labeled_raw() -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    lab = raw.loc[raw.Delay.notna()].copy()
    lab["delayed"] = delayed_flag(lab.Delay)
    return lab.reset_index(drop=True)


def add_dims(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["airline"] = fill_category(df.Airline)
    df["carrier_code"] = fill_category(df["Carrier_Code(IATA)"])
    dep, arr = hour_bucket(df[DEP]), hour_bucket(df[ARR])
    df["dep_hour_order"], df["dep_hour_label"] = dep.hour_order.to_numpy(), dep.hour_label.to_numpy()
    df["arr_hour_order"], df["arr_hour_label"] = arr.hour_order.to_numpy(), arr.hour_label.to_numpy()
    df["raw_time_pattern"] = time_pattern(df[DEP], df[ARR])
    df["route"] = df.Origin_Airport.astype(str) + "-" + df.Destination_Airport.astype(str)
    df["month"] = df.Month.astype("int64")
    return df


def load_airports() -> pd.DataFrame:
    m = pd.read_csv(MAPPING_PATH)
    require(m.iata.is_unique, "mapping iata not unique")
    return m[["iata", "state", "lat", "lon", "verification_tier"]].rename(
        columns={"iata": "airport", "state": "airport_state",
                 "verification_tier": "weather_station_tier"})


def load_weather_population():
    from notebooks.run_weather_model_comparison import load_inputs, JOIN_RUN
    from src.weather_full import read_joined_csv
    from src.weather_model import labeled_mask
    adopted, weather, manifest, _scenario, paths = load_inputs(JOIN_RUN)
    dates = read_joined_csv(paths["joined_weather"], usecols=["ID", "attributed_date"])
    require(dates.ID.astype(str).tolist() == weather.ID.astype(str).tolist(),
            "attributed_date order mismatch")
    mask = labeled_mask(adopted).to_numpy()
    w = weather.loc[mask].reset_index(drop=True)
    a = adopted.loc[mask].reset_index(drop=True)
    w["attributed_date"] = dates.attributed_date.to_numpy()[mask]
    w["delayed"] = delayed_flag(a.Delay).to_numpy()
    inputs = {k: {"path": rel(v)} for k, v in paths.items()}
    inputs["joined_weather"]["sha256"] = manifest["outputs"]["10"]["sha256"]
    return w, inputs


def load_oof() -> tuple[pd.DataFrame, dict]:
    frames, info = [], {}
    for seed in SEEDS:
        files = sorted((ROOT / "output" / f"{OOF_RUN_PREFIX}_seed{seed}_oof").glob(f"{OOF_PHASE}-*.csv.gz"))
        require(len(files) == 1, f"expected one {OOF_PHASE} OOF file for seed {seed}, got {files}")
        f = pd.read_csv(files[0], usecols=["seed", "ID", "fold", "y_true", "probability",
                                           "threshold", "prediction"], dtype={"ID": str})
        require((f.seed == seed).all(), "seed column mismatch")
        require(f.ID.is_unique, "duplicate OOF IDs")
        frames.append(f)
        info[str(seed)] = {"path": rel(files[0]), "sha256": sha256(files[0]), "rows": int(len(f))}
    return pd.concat(frames, ignore_index=True), info


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rowlevel", type=Path, default=DEFAULT_ROWLEVEL)
    parser.add_argument("--route-min-rows", type=int, default=1,
                        help="drop routes with fewer labeled rows from the route table")
    args = parser.parse_args()
    lower_priority()
    require(RAW_PATH.is_file(), "data/train.csv missing; real data required")
    out = args.out
    written: dict[str, dict] = {}
    recon: dict[str, object] = {}

    # ---------------- Dashboard 1: labeled rows ----------------
    lab = add_dims(load_labeled_raw())
    require(len(lab) == EXPECTED["labeled_rows"] and lab.delayed.sum() == EXPECTED["labeled_delayed"],
            "labeled denominator drift")
    pop = "labeled_all"
    airports = load_airports()

    airline = rate_table(lab, ["airline"], population=pop)
    written["d1_delay_by_airline"] = write_csv(airline, out / "d1_delay_by_airline.csv")

    month = rate_table(lab, ["month"], population=pop)
    written["d1_delay_by_month"] = write_csv(month, out / "d1_delay_by_month.csv")

    hours = []
    for role, prefix in (("departure", "dep"), ("arrival", "arr")):
        t = rate_table(lab, [f"{prefix}_hour_order", f"{prefix}_hour_label"], population=pop)
        t = t.rename(columns={f"{prefix}_hour_order": "hour_order", f"{prefix}_hour_label": "hour_label"})
        t.insert(3, "time_role", role)
        t.insert(4, "time_role_ko", "예정 출발 시각" if role == "departure" else "예정 도착 시각")
        hours.append(t)
    hours = pd.concat(hours, ignore_index=True)
    written["d1_delay_by_hour"] = write_csv(hours, out / "d1_delay_by_hour.csv")

    ap_parts = []
    for role, col in (("origin", "Origin_Airport"), ("destination", "Destination_Airport")):
        t = rate_table(lab.assign(airport=lab[col]), ["airport"], population=pop)
        t.insert(3, "airport_role", role)
        t.insert(4, "airport_role_ko", "출발 공항" if role == "origin" else "도착 공항")
        t["rank_by_rows"] = t.n_rows.rank(method="first", ascending=False).astype(int)
        ap_parts.append(t)
    ap = pd.concat(ap_parts, ignore_index=True).merge(airports, on="airport", how="left")
    recon["airports_without_coordinates"] = sorted(ap.loc[ap.lat.isna(), "airport"].unique().tolist())
    written["d1_delay_by_airport"] = write_csv(ap, out / "d1_delay_by_airport.csv")

    route = rate_table(lab, ["route", "Origin_Airport", "Destination_Airport"], population=pop)
    route = route.rename(columns={"Origin_Airport": "origin_airport",
                                  "Destination_Airport": "destination_airport"})
    route["rank_by_rows"] = route.n_rows.rank(method="first", ascending=False).astype(int)
    coords = airports.set_index("airport")[["lat", "lon"]]
    route["origin_lat"] = route.origin_airport.map(coords.lat)
    route["origin_lon"] = route.origin_airport.map(coords.lon)
    route["destination_lat"] = route.destination_airport.map(coords.lat)
    route["destination_lon"] = route.destination_airport.map(coords.lon)
    recon["route_table_rows_before_min_filter"] = int(route.n_rows.sum())
    route = route.loc[route.n_rows >= args.route_min_rows].sort_values("rank_by_rows")
    written["d1_delay_by_route"] = write_csv(route, out / "d1_delay_by_route.csv")

    am = rate_table(lab, ["airline", "month"], population=pop)
    written["d1_delay_by_airline_month"] = write_csv(am, out / "d1_delay_by_airline_month.csv")

    # ---------------- Dashboard 2: weather population ----------------
    weather, weather_inputs = load_weather_population()
    require(len(weather) == EXPECTED["weather_rows"] and weather.delayed.sum() == EXPECTED["weather_delayed"],
            "weather denominator drift")
    wbins = weather_bins_long(weather, weather.delayed)
    written["d2_weather_bins_long"] = write_csv(wbins, out / "d2_weather_bins_long.csv")

    wmonth = weather.assign(year_month=weather.attributed_date.astype(str).str[:7])
    ym = rate_table(wmonth, ["year_month"], population="weather_eval")
    written["d2_delay_by_year_month"] = write_csv(ym, out / "d2_delay_by_year_month.csv")

    # ---------------- Model evidence (D2 + D3) ----------------
    wruns = pd.read_csv(WEATHER_RUNS_PATH)
    cruns = pd.read_csv(CLASSIFIER_RUNS_PATH)
    metric_cols = ("macro_f1_nested", "log_loss", "roc_auc", "f1_at_050", "deployment_threshold")
    conf = pd.concat([
        confusion_long(wruns, experiment="weather_on_off_lightgbm", variant_col="condition",
                       population="weather_eval"),
        confusion_long(cruns, experiment="classifier_compare_weather_on", variant_col="model",
                       population="weather_eval"),
    ], ignore_index=True)
    written["model_confusion_long"] = write_csv(conf, out / "model_confusion_long.csv")
    mets = pd.concat([
        metrics_long(wruns, experiment="weather_on_off_lightgbm", variant_col="condition",
                     population="weather_eval", metrics=metric_cols),
        metrics_long(cruns, experiment="classifier_compare_weather_on", variant_col="model",
                     population="weather_eval", metrics=metric_cols),
    ], ignore_index=True)
    written["model_metrics_long"] = write_csv(mets, out / "model_metrics_long.csv")

    # All classifier runs (comparison, tuning, grid extension, calibration arms) in one long table
    comparison_checks: dict[str, bool] = {}
    comp_parts = []
    for experiment, (runs_rel, summary_rel) in CLASSIFIER_SOURCES.items():
        runs = pd.read_csv(ROOT / runs_rel)
        summary = json.loads((ROOT / summary_rel).read_text(encoding="utf-8"))
        require(summary.get("smoke_only_not_evidence") is False, f"{experiment}: smoke summary")
        part = comparison_long(runs, experiment=experiment, population="weather_eval")
        comp_parts.append(part)
        comparison_checks.update(reconcile_comparison(
            part, summary_means(summary, experiment), experiment=experiment, seeds=SEEDS,
            rows=EXPECTED["weather_rows"], positives=EXPECTED["weather_delayed"]))
    comp_all = pd.concat(comp_parts, ignore_index=True)
    written["model_comparison_all"] = write_csv(comp_all, out / "model_comparison_all.csv")

    cal_runs = pd.read_csv(ROOT / CLASSIFIER_SOURCES["classifier_calibration_20261008"][0])
    rel_bins = reliability_long(pd.read_csv(CALIBRATION_BINS_PATH), population="weather_eval")
    comparison_checks.update(reconcile_reliability(
        rel_bins, cal_runs, rows=EXPECTED["weather_rows"], positives=EXPECTED["weather_delayed"]))
    written["model_calibration_reliability"] = write_csv(
        rel_bins, out / "model_calibration_reliability.csv")

    # ---------------- Dashboard 3: row-level OOF (P6_clean, 255,001 x 3) ----------------
    oof, oof_info = load_oof()
    keyed = lab.set_index(lab.ID.astype(str))
    for seed in SEEDS:
        ids = oof.loc[oof.seed == seed, "ID"]
        require(len(ids) == len(lab) and set(ids) == set(keyed.index),
                f"OOF seed {seed} rows differ from labeled rows")
    route_counts = lab.route.value_counts()
    keyed["route_100"] = keyed.route.where(keyed.route.map(route_counts) >= ROUTE_MIN_ROWS_D3,
                                           ROUTE_OTHER_LABEL)
    recon["d3_routes_kept"] = int((route_counts >= ROUTE_MIN_ROWS_D3).sum())
    recon["d3_rows_in_kept_routes"] = int(route_counts[route_counts >= ROUTE_MIN_ROWS_D3].sum())
    dims = ["airline", "dep_hour_label", "month", "Origin_Airport", "route_100", "raw_time_pattern"]
    oofd = oof.join(keyed[dims + ["delayed"]], on="ID")
    require((oofd.y_true == oofd.delayed).all(), "OOF y_true differs from raw label")
    oofd = oofd.rename(columns={"Origin_Airport": "origin_airport", "dep_hour_label": "dep_hour",
                                "route_100": "route"})
    dim_names = ["airline", "dep_hour", "month", "origin_airport", "route", "raw_time_pattern"]
    err = pd.concat([oof_error_table(oofd, d, population="oof_p6_clean") for d in dim_names],
                    ignore_index=True)
    err["group"] = err.group.astype(str)
    hour_sort = dict(zip(lab.dep_hour_label, lab.dep_hour_order))
    err["group_sort"] = np.select(
        [err.dimension.eq("month"), err.dimension.eq("dep_hour")],
        [pd.to_numeric(err.group, errors="coerce"), err.group.map(hour_sort).astype(float)], np.nan)
    written["d3_oof_errors_by_group"] = write_csv(err, out / "d3_oof_errors_by_group.csv")

    wrong = (oofd.y_true != oofd.prediction).astype(int)
    per_row = oofd.assign(wrong=wrong).groupby("ID").agg(
        error_seed_count=("wrong", "sum"), n_seeds=("seed", "nunique"),
        **{d: (d, "first") for d in dim_names})
    require((per_row.n_seeds == len(SEEDS)).all(), "row missing a seed")
    stab = pd.concat([error_stability_table(per_row, d, population="oof_p6_clean") for d in dim_names],
                     ignore_index=True)
    stab["group"] = stab.group.astype(str)
    written["d3_oof_error_stability"] = write_csv(stab, out / "d3_oof_error_stability.csv")

    # ---------------- Overview ----------------
    btm = lab.raw_time_pattern.eq("both_missing")
    weather_ids = set(weather.ID.astype(str))
    overview = pd.DataFrame([
        {"population": "labeled_all", "population_ko": POPULATIONS["labeled_all"],
         "n_rows": len(lab), "n_delayed": int(lab.delayed.sum())},
        {"population": "weather_eval", "population_ko": POPULATIONS["weather_eval"],
         "n_rows": len(weather), "n_delayed": int(weather.delayed.sum())},
        {"population": "both_time_missing", "population_ko": POPULATIONS["both_time_missing"],
         "n_rows": int(btm.sum()), "n_delayed": int(lab.loc[btm, "delayed"].sum())},
    ])
    overview["n_not_delayed"] = overview.n_rows - overview.n_delayed
    overview["delay_rate"] = overview.n_delayed / overview.n_rows
    overview["raw_total_rows"] = int(pd.read_csv(RAW_PATH, usecols=["ID"]).shape[0])
    written["overview_populations"] = write_csv(overview, out / "overview_populations.csv")

    # ---------------- Reconciliation ----------------
    def sums(t):
        return [int(t.n_rows.sum()), int(t.n_delayed.sum())]
    target = [EXPECTED["labeled_rows"], EXPECTED["labeled_delayed"]]
    checks = {
        "d1_airline": sums(airline) == target,
        "d1_month": sums(month) == target,
        "d1_hour_departure": sums(hours[hours.time_role == "departure"]) == target,
        "d1_hour_arrival": sums(hours[hours.time_role == "arrival"]) == target,
        "d1_airport_origin": sums(ap[ap.airport_role == "origin"]) == target,
        "d1_airport_destination": sums(ap[ap.airport_role == "destination"]) == target,
        "d1_route_before_filter": recon["route_table_rows_before_min_filter"] == target[0],
        "d1_airline_month": sums(am) == target,
    }
    for (dim, seed), t in err.groupby(["dimension", "seed"]):
        checks[f"d3_errors_{dim}_seed{seed}"] = sums(t) == target
    for dim, t in stab.groupby("dimension"):
        checks[f"d3_stability_{dim}"] = int(t.n_rows.sum()) == target[0]
    wt = [EXPECTED["weather_rows"], EXPECTED["weather_delayed"]]
    for (role, feature), t in wbins.groupby(["role", "feature"]):
        checks[f"d2_bins_{role}_{feature}"] = sums(t) == wt
    checks["d2_year_month"] = sums(ym) == wt
    # weather confusion equals the runs CSV cell by cell
    for _, r in wruns.iterrows():
        cells = conf[(conf.experiment == "weather_on_off_lightgbm") & (conf.seed == r.seed)
                     & (conf.variant == r.condition)].set_index("cell").n
        checks[f"weather_confusion_seed{r.seed}_{r.condition}"] = (
            [int(cells[c]) for c in ("TN", "FP", "FN", "TP")] == [int(r.tn), int(r.fp), int(r.fn), int(r.tp)]
            and int(cells.sum()) == wt[0] and int(cells["FN"] + cells["TP"]) == wt[1])
    for _, r in cruns.iterrows():
        cells = conf[(conf.experiment == "classifier_compare_weather_on") & (conf.seed == r.seed)
                     & (conf.variant == r.model)].set_index("cell").n
        checks[f"classifier_confusion_seed{r.seed}_{r.model}"] = (
            [int(cells[c]) for c in ("TN", "FP", "FN", "TP")] == [int(r.tn), int(r.fp), int(r.fn), int(r.tp)]
            and int(cells.sum()) == wt[0])
    checks.update(comparison_checks)
    # OOF recomputation equals tracked groups CSV
    groups = pd.read_csv(OOF_GROUPS_PATH)
    g = groups[(groups.phase_key == OOF_PHASE)]
    for seed in SEEDS:
        ov = g[(g.seed == seed) & (g.dimension == "overall")].iloc[0]
        e = err[(err.seed == seed) & (err.dimension == "airline")][["tn", "fp", "fn", "tp"]].sum()
        checks[f"oof_overall_seed{seed}_vs_groups_csv"] = (
            [int(e.tn), int(e.fp), int(e.fn), int(e.tp)] == [int(ov.tn), int(ov.fp), int(ov.fn), int(ov.tp)])
        for _, gr in g[(g.seed == seed) & (g.dimension == "raw_time_pattern")].iterrows():
            mine = err[(err.seed == seed) & (err.dimension == "raw_time_pattern") & (err.group == gr.group)]
            checks[f"oof_time_pattern_{gr.group}_seed{seed}_vs_groups_csv"] = (
                len(mine) == 1 and [int(mine.iloc[0][c]) for c in ("tn", "fp", "fn", "tp")]
                == [int(gr[c]) for c in ("tn", "fp", "fn", "tp")])
    bm = err[(err.dimension == "raw_time_pattern") & (err.group == "both_missing")]
    recon["both_time_missing_recall_by_seed"] = {str(int(s)): float(v) for s, v in
                                                 zip(bm.seed, bm.recall_delayed)}
    recon["both_time_missing_recall_3seed_mean"] = float(bm.recall_delayed.mean())
    checks["both_time_missing_rows"] = (int(btm.sum()) == EXPECTED["both_time_missing_rows"] and
                                        int(lab.loc[btm, "delayed"].sum()) == EXPECTED["both_time_missing_delayed"])
    checks["both_time_missing_in_weather_eval_is_0"] = not (set(lab.loc[btm, "ID"].astype(str)) & weather_ids)
    recon["both_time_missing_in_weather_eval"] = len(set(lab.loc[btm, "ID"].astype(str)) & weather_ids)
    for role in ROLES:
        recon[f"weather_{role}_unmatched_rows"] = int((weather[f"weather_{role}_matched"] == 0).sum())
        recon[f"weather_{role}_p01i_trace_rows"] = int(
            np.isclose(weather[f"weather_{role}_p01i"].to_numpy(dtype=float), TRACE, rtol=0, atol=1e-9).sum())
    recon["checks"] = checks
    recon["all_checks_passed"] = all(checks.values())

    # ---------------- Row-level (local only) ----------------
    rl = lab[["ID", "month", "Day_of_Month", "dep_hour_order", "dep_hour_label", "arr_hour_order",
              "arr_hour_label", "raw_time_pattern", "Origin_Airport", "Destination_Airport", "route",
              "airline", "carrier_code", "Distance", "delayed"]].copy()
    rl = rl.rename(columns={"Origin_Airport": "origin_airport", "Destination_Airport": "destination_airport",
                            "Day_of_Month": "day_of_month", "Distance": "distance_miles"})
    for role, col in (("origin", "origin_airport"), ("destination", "destination_airport")):
        rl[f"{role}_lat"] = rl[col].map(coords.lat)
        rl[f"{role}_lon"] = rl[col].map(coords.lon)
    wcols = ["ID", "attributed_date"] + [c for c in weather.columns if c.startswith("weather_")]
    rl = rl.merge(weather[wcols].assign(ID=weather.ID.astype(str)), on="ID", how="left")
    rl.insert(rl.columns.get_loc("delayed") + 1, "in_weather_eval", rl.attributed_date.notna().astype(int))
    piv = oof.pivot(index="ID", columns="seed", values=["probability", "prediction"])
    for seed in SEEDS:
        rl[f"oof_p6clean_prob_seed{seed}"] = rl.ID.map(piv[("probability", seed)])
        rl[f"oof_p6clean_pred_seed{seed}"] = rl.ID.map(piv[("prediction", seed)]).astype(int)
    rl["oof_p6clean_error_seed_count"] = rl.ID.map(per_row.error_seed_count).astype(int)
    require(len(rl) == EXPECTED["labeled_rows"] and int(rl.in_weather_eval.sum()) == EXPECTED["weather_rows"],
            "row-level denominators drift")
    rowlevel_info = write_csv(rl, args.rowlevel)
    rowlevel_info["tracked"] = False

    manifest = {
        "schema_version": 1,
        "generator": "scripts/build_tableau_extracts.py",
        "training_performed": False,
        "populations": POPULATIONS,
        "inputs": {
            "raw_train": {"path": rel(RAW_PATH), "sha256": sha256(RAW_PATH)},
            "airport_coordinates": {"path": rel(MAPPING_PATH), "sha256": sha256(MAPPING_PATH),
                                    "note": "lat/lon from the mwgg airports list cached during station mapping"},
            "weather": weather_inputs,
            "weather_runs": {"path": rel(WEATHER_RUNS_PATH), "sha256": sha256(WEATHER_RUNS_PATH)},
            "classifier_runs": {"path": rel(CLASSIFIER_RUNS_PATH), "sha256": sha256(CLASSIFIER_RUNS_PATH)},
            "classifier_comparison_all": {
                experiment: {"runs": {"path": runs_rel, "sha256": sha256(ROOT / runs_rel)},
                             "summary": {"path": summary_rel, "sha256": sha256(ROOT / summary_rel)}}
                for experiment, (runs_rel, summary_rel) in CLASSIFIER_SOURCES.items()},
            "calibration_reliability_bins": {"path": rel(CALIBRATION_BINS_PATH),
                                             "sha256": sha256(CALIBRATION_BINS_PATH)},
            "oof_groups": {"path": rel(OOF_GROUPS_PATH), "sha256": sha256(OOF_GROUPS_PATH)},
            "oof_rowlevel": oof_info,
        },
        "outputs": written,
        "rowlevel_local_only": rowlevel_info,
        "reconciliation": recon,
    }
    mpath = out / "tableau_extracts_manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                     newline="\n")
    print(json.dumps({"outputs": {k: v["rows"] for k, v in written.items()},
                      "rowlevel": rowlevel_info, "all_checks_passed": recon["all_checks_passed"],
                      "failed": [k for k, v in checks.items() if not v],
                      "recon": {k: v for k, v in recon.items() if k != "checks"}},
                     ensure_ascii=False, indent=2))
    require(recon["all_checks_passed"], "reconciliation failed")


if __name__ == "__main__":
    main()
