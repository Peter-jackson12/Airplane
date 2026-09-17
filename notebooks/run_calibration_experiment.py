"""Nested probability calibration: fit inside outer-train, score on outer-valid.

Every calibrator is fitted on the inner holdout that `run_fold_nested_grid()` carves
out of outer-train, then applied to the untouched outer-valid probabilities. The
calibrated threshold is chosen on the same inner holdout. Outer-valid labels never
take part in fitting or selection, so this is not "fit on the pooled OOF and grade
the improvement on that same OOF".

Two arms differ only in which inner-holdout rows feed the calibrator:

  shared  the whole inner holdout, which also chose the tree count and threshold
  split   a disjoint half that took no part in tree-count or threshold selection

Run from the repository root. `--sample` is a smoke run and never performance
evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rerun_all_phases as runner
from src.calibration import KINDS, fit_calibrator, select_threshold
from src.cv import make_folds, run_fold_nested_grid
from src.features import load_data
from src.oof import calibration_bins, labeled_identity, metrics
from importlib.metadata import version

from src.run_store import file_digest

ARMS = {"shared": 0.0, "split": 0.5}
CODE_FILES = ["src/cv.py", "src/calibration.py", "src/features.py", "src/oof.py",
              "rerun_all_phases.py", "notebooks/run_calibration_experiment.py"]
TIME_COLUMNS = ("raw_missing__Estimated_Departure_Time",
                "raw_missing__Estimated_Arrival_Time")


def phase_spec(key: str):
    for spec in runner.PHASES:
        if spec.key == key:
            return spec
    raise ValueError(f"Unknown phase: {key}")


def run_phase_seed(raw, spec, seed, arm, cache):
    """Return per-fold records and a {calibrator: pooled OOF probability} mapping."""
    frac = ARMS[arm]
    cfg = _apply_seed(seed)
    X_lab, y, X_unlab = runner.build_features(raw, spec)
    folds = make_folds(y, cfg)
    te_cols = spec.cat_cols if spec.te else ()

    probabilities = {kind: np.full(len(y), np.nan) for kind in KINDS}
    thresholds = {kind: np.full(len(y), np.nan) for kind in KINDS}
    records = []
    for fold, (tr, va) in enumerate(folds):
        started = time.perf_counter()
        capture, selection = {}, {}
        fit = run_fold_nested_grid(
            runner.model_factory,
            X_lab.iloc[tr].reset_index(drop=True),
            y.iloc[tr].reset_index(drop=True),
            X_lab.iloc[va].reset_index(drop=True),
            cfg,
            fold=fold,
            n_estimators_grid=runner.N_ESTIMATORS_GRID,
            te_cols=te_cols,
            te_m=runner.TE_SMOOTHING_M,
            te_inner_splits=spec.inner_splits,
            te_drop_original=spec.te_drop_original,
            selection_metadata=selection,
            holdout_capture=capture,
            calibration_holdout_frac=frac,
        )
        if capture["calibration_is_disjoint"] != bool(frac):
            raise ValueError("Calibration slice flag does not match the requested arm")
        grid = cfg.thresholds()
        for kind in KINDS:
            calibrate = fit_calibrator(kind, capture["calibration_y"],
                                       capture["calibration_probs"])
            # Threshold comes from the selection slice, never from outer-valid.
            threshold = select_threshold(capture["selection_y"],
                                         calibrate(capture["selection_probs"]), grid)
            probabilities[kind][va] = calibrate(fit.valid_probs)
            thresholds[kind][va] = threshold
            records.append(dict(
                phase_key=spec.key, seed=seed, arm=arm, calibrator=kind, fold=fold,
                n_estimators=fit.selected_n_estimators, threshold=threshold,
                n_inner_train=fit.n_inner_train, n_inner_holdout=fit.n_inner_holdout,
                n_calibration_rows=int(capture["calibration_y"].size),
                n_selection_rows=int(capture["selection_y"].size),
                uncalibrated_threshold=selection["threshold"],
                elapsed_sec=round(time.perf_counter() - started, 1),
            ))
        print(f"   fold {fold + 1}/{len(folds)} | {spec.key} seed {seed} arm {arm} "
              f"| n_estimators {fit.selected_n_estimators} "
              f"| {time.perf_counter() - started:.1f}s", flush=True)

    for kind in KINDS:
        if np.isnan(probabilities[kind]).any() or np.isnan(thresholds[kind]).any():
            raise ValueError("Every labeled row must be scored exactly once")
    return records, probabilities, thresholds, y


def _apply_seed(seed):
    """Match `rerun_all_phases.main()`: the seed drives both the folds and LightGBM.

    The production entry point sets these two globals together. Bypassing it and
    setting only the fold seed would leave every seed sharing one model seed, so the
    three runs would not be the independent repeats the protocol assumes.
    """
    runner.CFG = replace(runner.CFG, seed=seed)
    runner.LGBM_PARAMS["random_state"] = seed
    return runner.CFG


def score(y, probability, threshold, identity):
    rows = pd.DataFrame({"y_true": np.asarray(y, dtype=int),
                         "probability": probability,
                         "prediction": (probability >= threshold).astype(int)})
    result = {"overall": metrics(rows)}
    dep, arr = (identity[c].to_numpy() for c in TIME_COLUMNS)
    for d, a, label in ((0, 0, "both_observed"), (1, 0, "departure_missing"),
                        (0, 1, "arrival_missing"), (1, 1, "both_missing")):
        mask = (dep == d) & (arr == a)
        result[label] = metrics(rows.loc[mask]) if mask.any() else {"n": 0}
    return result, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline_recovery_v2_calibration_20260917")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--seeds", default="42,1,7")
    ap.add_argument("--phases", default="P6_fixed,P6_clean")
    ap.add_argument("--arms", default="shared,split")
    args = ap.parse_args()
    if not args.name.startswith("baseline_recovery_v2"):
        print("[!] name must start with baseline_recovery_v2")
        return 1

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if any(a not in ARMS for a in arms):
        print(f"[!] arms must be within {sorted(ARMS)}")
        return 1

    source = ROOT / "data/train.csv"
    if not source.exists():
        print(f"[!] real data not found: {source}")
        return 1
    raw = load_data(source)
    if args.sample is not None:
        raw = raw.head(args.sample).reset_index(drop=True)
    identity = labeled_identity(raw)

    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    folds_rows, score_rows, bin_rows = [], [], []
    cache: dict = {}
    for seed in seeds:
        for phase in phases:
            spec = phase_spec(phase)
            for arm in arms:
                records, probabilities, thresholds, y = run_phase_seed(
                    raw, spec, seed, arm, cache)
                folds_rows.extend(records)
                for kind in KINDS:
                    grouped, rows = score(y, probabilities[kind], thresholds[kind],
                                          identity)
                    for group, values in grouped.items():
                        score_rows.append(dict(phase_key=phase, seed=seed, arm=arm,
                                               calibrator=kind, group=group, **values))
                    table = calibration_bins(rows.y_true, rows.probability)
                    table["phase_key"], table["seed"] = phase, seed
                    table["arm"], table["calibrator"] = arm, kind
                    bin_rows.append(table)

    runs = pd.DataFrame(folds_rows)
    scores = pd.DataFrame(score_rows)
    bins = pd.concat(bin_rows, ignore_index=True)
    runs.to_csv(out / f"{args.name}_folds.csv", index=False)
    scores.to_csv(out / f"{args.name}_scores.csv", index=False)
    bins.to_csv(out / f"{args.name}_calibration_bins.csv", index=False)

    measures = ["macro_f1", "log_loss", "roc_auc", "brier", "ece_10",
                "calibration_bias", "precision", "recall", "fnr"]
    keys = ["phase_key", "arm", "calibrator", "group"]
    summary = scores.groupby(keys)[measures].agg(["mean", "std", "count"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.reset_index().to_csv(out / f"{args.name}_summary.csv", index=False)

    manifest = dict(
        name=args.name,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        sample=args.sample, seeds=seeds, phases=phases, arms=arms,
        calibrators=list(KINDS),
        boundary=("calibrators and thresholds are fitted on inner-holdout rows of "
                  "outer-train; outer-valid is scored once and never fitted"),
        n_labeled_rows=int(len(identity)),
        source_sha256=file_digest(source),
        code_sha256={f: file_digest(ROOT / f) for f in CODE_FILES},
        cv=runner.CFG.describe(),
        lgbm_params=dict(runner.LGBM_PARAMS),
        grid=runner.N_ESTIMATORS_GRID,
        versions={name: version(name) for name in
                  ("lightgbm", "pandas", "numpy", "scikit-learn")},
        files=[f"{args.name}_{s}.csv" for s in
               ("folds", "scores", "calibration_bins", "summary")],
    )
    (out / f"{args.name}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f">> saved: {args.name}_summary.csv (+ folds/scores/calibration_bins/manifest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
