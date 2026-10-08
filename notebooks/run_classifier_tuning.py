"""Wider predeclared tuning of LightGBM and Random Forest + RF weather on/off.

Follow-up to ``notebooks/run_classifier_comparison.py`` (20261008 comparison).
Same population (180,332 labeled weather-join rows, P6_clean), same target,
same outer folds (make_folds, seeds 42/1/7, fingerprint-checked against the
20261008 comparison) and the same nested inner boundary:

* LightGBM (weather_on): 24 predeclared configurations
  (``cc.lgbm_tuning_grid``). Each configuration runs the unchanged repository
  ``run_fold_nested_grid`` over ``runner.N_ESTIMATORS_GRID``; the configuration
  with minimal inner-holdout LogLoss (its own best tree count) is selected and
  its inner-holdout Macro F1 threshold is used. No early stopping.
* Random Forest (weather_on and weather_off): 9 configurations
  (``cc.rf_tuning_grid``), 300 trees, through ``cc.run_fold_nested_params``.

Outer-valid labels are never passed to any selection code. Logistic
Regression is not rerun; its 20261008 runs (same folds) are the reference.

Usage:
    python -u notebooks/run_classifier_tuning.py                       # full run
    python -u notebooks/run_classifier_tuning.py --sample 20000 \
        --out-dir <scratch> --name baseline_recovery_v2_classifier_tuning_smoke_20261008
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402

import rerun_all_phases as runner  # noqa: E402
from notebooks import run_classifier_comparison as base_cmp  # noqa: E402
from notebooks import run_weather_model_comparison as weather_cmp  # noqa: E402
from src import classifier_compare as cc  # noqa: E402
from src.cv import evaluate_oof, make_folds, run_fold_nested_grid  # noqa: E402
from src.weather_model import SEEDS, WEATHER_MODEL_FEATURES, digest, stable_json_hash  # noqa: E402

DEFAULT_NAME = "baseline_recovery_v2_classifier_tuning_20261008"
COMPARE_NAME = "baseline_recovery_v2_classifier_compare_20261008"
COMPARE_RUNS = ROOT / "output" / f"{COMPARE_NAME}_runs.csv"
COMPARE_FOLDS = ROOT / "output" / f"{COMPARE_NAME}_folds.csv"
COMPARE_MANIFEST = ROOT / "output" / f"{COMPARE_NAME}_manifest.json"
WEATHER_RUNS = base_cmp.REFERENCE_RUNS  # 20260922 LightGBM weather_off / weather_on

#: (condition, model) evaluation units, in execution order within each seed.
UNITS: tuple[tuple[str, str], ...] = (
    ("weather_on", "random_forest_tuned"),
    ("weather_off", "random_forest_tuned"),
    ("weather_on", "lightgbm_tuned"),
)
CODE_FILES = (
    "src/classifier_compare.py",
    "notebooks/run_classifier_tuning.py",
    "notebooks/run_classifier_comparison.py",
    "notebooks/run_weather_model_comparison.py",
    "src/weather_model.py",
    "src/weather_full.py",
    "src/features.py",
    "src/cv.py",
    "rerun_all_phases.py",
)
IDENTITY_CODE_FILES = ("src/classifier_compare.py", "src/cv.py", "src/features.py",
                       "src/weather_model.py", "src/weather_full.py", "rerun_all_phases.py",
                       "notebooks/run_weather_model_comparison.py")
METRICS = cc.SUMMARY_METRICS
log = base_cmp.log
atomic_json = base_cmp.atomic_json
atomic_csv = base_cmp.atomic_csv


def lower_process_priority() -> str:
    """Below-normal OS priority keeps the machine usable without changing thread
    counts (LightGBM keeps its default threads, so numerics match 20261008)."""
    if sys.platform == "win32":
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        kernel32.SetPriorityClass.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        ok = kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
        return "below_normal" if ok else "unchanged"
    try:
        os.nice(10)
        return "nice+10"
    except OSError:
        return "unchanged"


def lgbm_base_params() -> dict:
    params = dict(runner.LGBM_PARAMS)
    cc.require(params.get("random_state") == 42, "LightGBM model seed drift")
    return params


def protocol_description(cfg, spec) -> dict:
    out = base_cmp.protocol_description(cfg, spec)
    out.update({
        "hyperparameter_selection": ("nested_grid_min_inner_holdout_log_loss over predeclared "
                                     "configurations (LightGBM: configuration x n_estimators)"),
        "tree_selection_lightgbm": ("run_fold_nested_grid per configuration (n_estimators grid, "
                                    "no early stopping); configuration chosen by its best inner LogLoss"),
        "threshold_selection": ("inner-holdout Macro F1 argmax of the selected model "
                                "(outer_train_holdout)"),
    })
    return out


def grids_description(rf_n_estimators: int, rf_n_jobs: int) -> dict:
    return {
        "lightgbm_tuned": {
            "base_params": lgbm_base_params(),
            "configurations": cc.lgbm_tuning_grid(),
            "n_configurations": len(cc.lgbm_tuning_grid()),
            "n_estimators_grid": list(runner.N_ESTIMATORS_GRID),
            "axes": {k: list(v) for k, v in cc.lgbm_tuning_axes(runner.N_ESTIMATORS_GRID).items()},
            "structure": ("full factorial learning_rate{0.03,0.05,0.1} x num_leaves{31,63,127} x "
                          "min_child_samples{20,100} without row bagging (subsample_freq=0) + "
                          "row-bagging variant subsample=0.8/subsample_freq=1 for "
                          "learning_rate{0.03,0.05,0.1} x num_leaves{63,127} at min_child_samples=20"),
            "fixed": "max_depth=8, colsample_bytree=0.8, random_state=42 for every split seed",
            "includes_20261008_config": {"learning_rate": 0.05, "num_leaves": 63,
                                         "min_child_samples": 20, "subsample": 0.8,
                                         "subsample_freq": 0},
            "threads": "LightGBM default (unchanged from 20261008)",
        },
        "random_forest_tuned": {
            "n_estimators": int(rf_n_estimators),
            "configurations": cc.rf_tuning_grid(),
            "axes": {k: list(v) for k, v in cc.rf_tuning_axes().items()},
            "n_jobs": int(rf_n_jobs), "class_weight": None, "bootstrap": True,
            "random_state": cc.RF_RANDOM_STATE,
            "random_state_policy": "fixed 42 for all split seeds (as in 20261008)",
            "missing_values": "native sklearn tree NaN support (no imputation)",
            "includes_20261008_selected_config": {"min_samples_leaf": 25, "max_features": 0.5},
        },
    }


# =============================================================================
# One fold of each tuned model
# =============================================================================


def lgbm_tuned_fold(X_tr, y_tr, X_va, cfg, fold, spec) -> dict:
    base = lgbm_base_params()

    def evaluate(config: dict) -> dict:
        params = cc.lgbm_config_params(base, config)
        meta: dict = {}
        t0 = time.perf_counter()
        fit = run_fold_nested_grid(
            lambda: lgb.LGBMClassifier(**params), X_tr, y_tr, X_va, cfg, fold=fold,
            n_estimators_grid=runner.N_ESTIMATORS_GRID,
            te_cols=spec.cat_cols, te_m=runner.TE_SMOOTHING_M,
            te_inner_splits=spec.inner_splits, te_drop_original=spec.te_drop_original,
            selection_metadata=meta,
        )
        cc.require("threshold" in meta, "nested threshold was not selected")
        best_ll = float(min(fit.grid_scores.values()))
        elapsed = time.perf_counter() - t0
        log(f"    lgbm fold={fold + 1} {config} k*={fit.selected_n_estimators} "
            f"inner_LL={best_ll:.6f} ({elapsed:.1f}s)")
        return {
            "inner_holdout_log_loss": best_ll,
            "selected_n_estimators": int(fit.selected_n_estimators),
            "n_estimators_scores": {str(k): float(v) for k, v in fit.grid_scores.items()},
            "threshold": float(meta["threshold"]),
            "fit_sec": float(elapsed),
            "n_inner_train": int(fit.n_inner_train), "n_inner_holdout": int(fit.n_inner_holdout),
            "n_model_features": len(getattr(fit.model, "feature_name_", [])),
            "valid_probs": np.asarray(fit.valid_probs, dtype=float),
        }

    sel = cc.select_best_config(cc.lgbm_tuning_grid(), evaluate)
    best = sel.best
    selected = {**sel.records[sel.best_index]["config"],
                "n_estimators": best["selected_n_estimators"]}
    return {
        "valid_probs": best["valid_probs"], "threshold": best["threshold"],
        "selected_params": selected,
        "grid_scores": sel.records,
        "best_inner_log_loss": best["inner_holdout_log_loss"],
        "n_inner_train": best["n_inner_train"], "n_inner_holdout": best["n_inner_holdout"],
        "n_model_features": best["n_model_features"],
        "grid_edge": cc.grid_edge_flags(selected, cc.lgbm_tuning_axes(runner.N_ESTIMATORS_GRID)),
        "extra": {},
    }


def rf_tuned_fold(X_tr, y_tr, X_va, cfg, fold, spec, *, rf_n_estimators, rf_n_jobs,
                  tag: str) -> dict:
    def build(params, columns):
        return cc.build_random_forest(params, columns, seed=cc.RF_RANDOM_STATE,
                                      n_jobs=rf_n_jobs, n_estimators=rf_n_estimators)

    def on_candidate(params, score, elapsed):
        log(f"    {tag} fold={fold + 1} {params} inner_LL={score:.6f} ({elapsed:.1f}s)")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = cc.run_fold_nested_params(
            build, cc.rf_tuning_grid(), X_tr, y_tr, X_va, cfg, fold=fold,
            te_cols=spec.cat_cols, te_m=runner.TE_SMOOTHING_M,
            te_inner_splits=spec.inner_splits, te_drop_original=spec.te_drop_original,
            on_candidate=on_candidate,
        )
    warn_msgs = sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught})
    for msg in warn_msgs:
        log(f"    WARNING {msg}")
    names = list(fit.model.named_steps["categories"].get_feature_names_out())
    return {
        "valid_probs": fit.valid_probs, "threshold": fit.threshold,
        "selected_params": fit.selected_params,
        "grid_scores": [dict(g) for g in fit.grid_scores],
        "best_inner_log_loss": float(min(g["inner_holdout_log_loss"] for g in fit.grid_scores)),
        "n_inner_train": fit.n_inner_train, "n_inner_holdout": fit.n_inner_holdout,
        "n_model_features": len(names),
        "grid_edge": cc.grid_edge_flags(fit.selected_params, cc.rf_tuning_axes()),
        "extra": {"warnings": warn_msgs},
    }


# =============================================================================
# Unit evaluation with per-fold checkpoints
# =============================================================================


def unit_key(seed: int, condition: str, model: str) -> str:
    return f"{seed}_{condition}_{model}"


def evaluate_unit(condition, model_key, X, y, *, seed, spec, expected_fp, state, ckpt_dir,
                  save_state, rf_n_estimators, rf_n_jobs) -> tuple[dict, list[dict]]:
    cfg = replace(runner.CFG, seed=int(seed))
    folds = make_folds(y, cfg)
    fingerprint = cc.check_fold_fingerprint(folds, len(y), expected_fp)
    key = unit_key(seed, condition, model_key)
    done_folds = {int(r["fold"]): r for r in state["folds"] if r["unit"] == key}
    oof = np.full(len(y), np.nan)
    for fold, (tr, va) in enumerate(folds):
        probs_path = ckpt_dir / f"{key}_fold{fold + 1}_valid_probs.npy"
        if fold + 1 in done_folds:
            probs = np.load(probs_path)
            row = done_folds[fold + 1]
            cc.require(len(probs) == len(va) and hashlib.sha256(probs.tobytes()).hexdigest()
                       == row["valid_probs_sha256"], f"checkpoint probs mismatch {key} fold {fold + 1}")
            oof[va] = probs
            log(f"  RESUME {key} fold={fold + 1}")
            continue
        t0 = time.perf_counter()
        X_tr = X.iloc[tr].reset_index(drop=True)
        y_tr = y.iloc[tr].reset_index(drop=True)
        X_va = X.iloc[va].reset_index(drop=True)
        if model_key == "lightgbm_tuned":
            res = lgbm_tuned_fold(X_tr, y_tr, X_va, cfg, fold, spec)
        else:
            res = rf_tuned_fold(X_tr, y_tr, X_va, cfg, fold, spec, rf_n_estimators=rf_n_estimators,
                                rf_n_jobs=rf_n_jobs, tag=f"rf[{condition}]")
        probs = np.asarray(res["valid_probs"], dtype=float)
        cc.require(len(probs) == len(va) and bool(np.isfinite(probs).all()), "bad valid probs")
        np.save(probs_path, probs)
        oof[va] = probs
        row = {
            "unit": key, "seed": int(seed), "condition": condition, "model": model_key,
            "fold": fold + 1,
            "n_outer_train": int(len(tr)), "n_outer_valid": int(len(va)),
            "n_inner_train": res["n_inner_train"], "n_inner_holdout": res["n_inner_holdout"],
            "selected_params": cc.compact_json(res["selected_params"]),
            "grid_edge": cc.compact_json(res["grid_edge"]),
            "threshold": float(res["threshold"]),
            "best_inner_holdout_log_loss": float(res["best_inner_log_loss"]),
            "grid_scores": cc.compact_json(res["grid_scores"]),
            "n_model_features": int(res["n_model_features"]),
            "extra": cc.compact_json(res["extra"]),
            "valid_probs_sha256": hashlib.sha256(probs.tobytes()).hexdigest(),
            "fold_elapsed_sec": float(time.perf_counter() - t0),
        }
        state["folds"].append(row)
        save_state()
        log(f"  {key} fold={fold + 1} params={res['selected_params']} edge={res['grid_edge']} "
            f"thr={res['threshold']:.2f} ({row['fold_elapsed_sec']:.1f}s)")

    fold_rows = sorted((r for r in state["folds"] if r["unit"] == key), key=lambda r: r["fold"])
    cc.require(len(fold_rows) == len(folds), f"incomplete folds for {key}")
    cc.require(bool(np.isfinite(oof).all()), "OOF probabilities incomplete/nonfinite")
    thresholds = [float(r["threshold"]) for r in fold_rows]
    score = evaluate_oof(y, oof, folds, cfg, label=key, per_fold_thresholds=thresholds)
    cm = score["confusion_matrix"]
    per_class = cc.class_metrics(cm["tn"], cm["fp"], cm["fn"], cm["tp"])
    cc.require(abs(per_class["macro_f1_from_cm"] - score["macro_f1"]) < 1e-9,
               "confusion matrix / macro F1 inconsistency")
    run = {
        "seed": int(seed), "condition": condition, "model": model_key,
        "n_rows": int(len(y)), "positive_rows": int(y.sum()), "positive_rate": float(y.mean()),
        "n_input_columns": int(X.shape[1]),
        "n_model_features_fold1": int(fold_rows[0]["n_model_features"]),
        "macro_f1_nested": float(score["macro_f1"]),
        "log_loss": float(score["log_loss"]),
        "roc_auc": float(score["roc_auc"]),
        "f1_at_050": float(score["f1_at_050"]),
        "naive_macro_f1": float(score["naive_macro_f1"]),
        "threshold_optimism": float(score["threshold_optimism"]),
        "deployment_threshold": float(score["deployment_threshold"]),
        "per_fold_thresholds": cc.compact_json(thresholds),
        "selected_params": cc.compact_json([json.loads(r["selected_params"]) for r in fold_rows]),
        "grid_edge": cc.compact_json([json.loads(r["grid_edge"]) for r in fold_rows]),
        "tn": cm["tn"], "fp": cm["fp"], "fn": cm["fn"], "tp": cm["tp"],
        **{k: v for k, v in per_class.items() if k != "macro_f1_from_cm"},
        "fold_fingerprint": fingerprint,
        "fold_fingerprint_matches_reference": (None if expected_fp is None
                                               else bool(fingerprint == expected_fp)),
        "elapsed_sec": float(sum(r["fold_elapsed_sec"] for r in fold_rows)),
        "protocol": cc.compact_json(protocol_description(cfg, spec)),
    }
    return run, fold_rows


# =============================================================================
# Reference tables and summaries
# =============================================================================


def compare_runs() -> pd.DataFrame:
    ref = pd.read_csv(COMPARE_RUNS)
    cc.require(set(ref.model) == set(cc.MODEL_KEYS), "20261008 comparison model set")
    return ref


def weather_lgbm_runs() -> pd.DataFrame:
    """20260922 LightGBM weather_off/weather_on runs with per-class metrics from the CM."""
    ref = pd.read_csv(WEATHER_RUNS).copy()
    for name in ("precision_delayed", "recall_delayed", "f1_delayed", "f1_not_delayed"):
        ref[name] = [cc.class_metrics(int(r.tn), int(r.fp), int(r.fn), int(r.tp))[name]
                     for r in ref.itertuples()]
    return ref


def selection_report(folds_df: pd.DataFrame) -> dict:
    out: dict = {}
    for (condition, model), sub in folds_df.groupby(["condition", "model"], sort=True):
        params = [json.loads(p) for p in sub.selected_params]
        edges = [json.loads(e) for e in sub.grid_edge]
        counts = pd.Series([cc.compact_json(p) for p in params]).value_counts()
        edge_counts: dict = {}
        for axis in edges[0]:
            vals = pd.Series([e[axis] if e[axis] is not None else "interior" for e in edges])
            edge_counts[axis] = {str(k): int(v) for k, v in vals.value_counts().items()}
        out[f"{condition}/{model}"] = {
            "n_folds": int(len(sub)),
            "selected_config_counts": {k: int(v) for k, v in counts.items()},
            "edge_counts_by_axis": edge_counts,
            "folds_with_any_edge": int(sum(any(v is not None for v in e.values()) for e in edges)),
            "folds_with_edge_excluding_two_level_axes": int(sum(
                any(v is not None for k, v in e.items() if k != "min_child_samples")
                for e in edges)),
        }
    return out


def reproduction_checks(folds_df: pd.DataFrame, compare_folds: pd.DataFrame) -> dict:
    """The tuning grids contain the 20261008 LightGBM config and RF winner; their
    inner-holdout scores and thresholds must reproduce the 20261008 fold records."""
    base_cfg = {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20,
                "subsample": 0.8, "subsample_freq": 0}
    out = {"lightgbm_20261008_config": {}, "random_forest_leaf25_mf05": {}}
    max_lgbm, max_rf = 0.0, 0.0
    for r in folds_df.itertuples():
        key = f"{r.seed}/fold{r.fold}"
        if r.model == "lightgbm_tuned":
            ref = compare_folds[(compare_folds.model == "lightgbm") & (compare_folds.seed == r.seed)
                                & (compare_folds.fold == r.fold)].iloc[0]
            mine = [g for g in json.loads(r.grid_scores) if g["config"] == base_cfg]
            cc.require(len(mine) == 1, "base LightGBM config missing from tuning grid")
            ref_scores = {str(g["params"]["n_estimators"]): g["inner_holdout_log_loss"]
                          for g in json.loads(ref.grid_scores)}
            diff = max(abs(mine[0]["n_estimators_scores"][k] - v) for k, v in ref_scores.items())
            max_lgbm = max(max_lgbm, diff)
            out["lightgbm_20261008_config"][key] = {
                "max_abs_inner_ll_diff": diff,
                "threshold_equal": float(mine[0]["threshold"]) == float(ref.threshold),
            }
        elif r.model == "random_forest_tuned" and r.condition == "weather_on":
            ref = compare_folds[(compare_folds.model == "random_forest") & (compare_folds.seed == r.seed)
                                & (compare_folds.fold == r.fold)].iloc[0]
            target = {"min_samples_leaf": 25, "max_features": 0.5}
            mine = [g for g in json.loads(r.grid_scores) if g["params"] == target]
            theirs = [g for g in json.loads(ref.grid_scores) if g["params"] == target]
            cc.require(len(mine) == 1 and len(theirs) == 1, "RF reference config missing")
            diff = abs(mine[0]["inner_holdout_log_loss"] - theirs[0]["inner_holdout_log_loss"])
            max_rf = max(max_rf, diff)
            out["random_forest_leaf25_mf05"][key] = {"abs_inner_ll_diff": diff}
    out["lightgbm_max_abs_diff"] = max_lgbm
    out["random_forest_max_abs_diff"] = max_rf
    out["lightgbm_thresholds_all_equal"] = bool(all(
        v["threshold_equal"] for v in out["lightgbm_20261008_config"].values()))
    return out


def build_comparisons(runs: pd.DataFrame) -> list[dict]:
    ref = compare_runs()
    wref = weather_lgbm_runs()

    def pick(frame, **kw):
        sub = frame
        for k, v in kw.items():
            sub = sub[sub[k] == v]
        cc.require(len(sub) == len(SEEDS), f"reference rows missing for {kw}")
        return sub

    lgbm_t = pick(runs, condition="weather_on", model="lightgbm_tuned")
    rf_on = pick(runs, condition="weather_on", model="random_forest_tuned")
    rf_off = pick(runs, condition="weather_off", model="random_forest_tuned")
    lr = pick(ref, model="logistic_regression")
    lgbm_b = pick(ref, model="lightgbm")
    rf_b = pick(ref, model="random_forest")
    w_on = pick(wref, condition="weather_on").assign(n_rows=lambda d: d.n_rows.astype(int))
    w_off = pick(wref, condition="weather_off").assign(n_rows=lambda d: d.n_rows.astype(int))
    specs = [
        (rf_on, lgbm_t, "weather_on: random_forest_tuned - lightgbm_tuned"),
        (lgbm_t, lr, "weather_on: lightgbm_tuned - logistic_regression(20261008)"),
        (rf_on, lr, "weather_on: random_forest_tuned - logistic_regression(20261008)"),
        (lgbm_t, lgbm_b, "weather_on: lightgbm_tuned - lightgbm(20261008 fixed config)"),
        (rf_on, rf_b, "weather_on: random_forest_tuned - random_forest(20261008 6-config grid)"),
        (rf_on, rf_off, "random_forest_tuned: weather_on - weather_off"),
        (w_on, w_off, "lightgbm(20260922 fixed config): weather_on - weather_off"),
    ]
    return [cc.paired_seed_deltas(a, b, SEEDS, METRICS, label=label) for a, b, label in specs]


def flatten_comparisons(comparisons: list[dict]) -> pd.DataFrame:
    rows = []
    for comp in comparisons:
        for metric, stats in comp["metrics"].items():
            for seed, val in stats["values_by_seed"].items():
                rows.append({"comparison": comp["comparison"], "metric": metric,
                             "seed": int(seed), "delta": val})
    return pd.DataFrame(rows)


def model_summary(runs: pd.DataFrame) -> dict:
    out = {}
    for (condition, model), sub in runs.groupby(["condition", "model"], sort=True):
        sub = sub.sort_values("seed")
        out[f"{condition}/{model}"] = {
            **{m: {"mean": float(sub[m].mean()),
                   "std": float(sub[m].std(ddof=1)) if len(sub) > 1 else None,
                   "values_by_seed": {str(int(r.seed)): float(getattr(r, m)) for r in sub.itertuples()}}
               for m in METRICS},
            "elapsed_sec_total": float(sub.elapsed_sec.sum()),
        }
    return out


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--sample", type=int, default=None,
                        help="SMOKE ONLY: stratified subsample of N labeled rows")
    parser.add_argument("--out-dir", default=str(ROOT / "output"))
    parser.add_argument("--rf-n-jobs", type=int, default=cc.RF_N_JOBS)
    parser.add_argument("--rf-n-estimators", type=int, default=cc.RF_N_ESTIMATORS,
                        help="only overridable together with --sample")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                        help="only overridable together with --sample")
    parser.add_argument("--normal-priority", action="store_true",
                        help="do not lower the OS process priority")
    args = parser.parse_args()
    sys.stdout.reconfigure(newline="\n")
    sys.stderr.reconfigure(newline="\n")
    cc.require(bool(re.fullmatch(r"baseline_recovery_v2_[A-Za-z0-9_]+", args.name)),
               "invalid run name")
    cc.require(1 <= args.rf_n_jobs <= 16, "RF n_jobs must be within 1..16")
    smoke = args.sample is not None
    seeds = tuple(int(s) for s in args.seeds.split(",") if s)
    if not smoke:
        cc.require(seeds == tuple(SEEDS), "full run uses seeds 42/1/7")
        cc.require(args.rf_n_estimators == cc.RF_N_ESTIMATORS, "RF budget is fixed for the full run")
        cc.require("smoke" not in args.name, "full run name must not contain 'smoke'")
    else:
        cc.require("smoke" in args.name, "sample runs must use a name containing 'smoke'")
        cc.require(set(seeds) <= set(SEEDS) and seeds, "unknown seeds")
    priority = "normal" if args.normal_priority else lower_process_priority()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "runs": out_dir / f"{args.name}_runs.csv",
        "folds": out_dir / f"{args.name}_folds.csv",
        "paired": out_dir / f"{args.name}_paired_deltas.csv",
        "summary": out_dir / f"{args.name}_summary.json",
        "manifest": out_dir / f"{args.name}_manifest.json",
    }
    cc.require(not any(p.exists() for p in final_paths.values()),
               "use a fresh --name; final evidence is never overwritten")

    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                        text=True).splitlines()
    code_sha = {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                for p in CODE_FILES}
    log(f"git HEAD {git_sha}; dirty entries: {git_dirty}; priority={priority}")

    t_load = time.perf_counter()
    adopted, weather, _join_manifest, _scenario, source_paths = weather_cmp.load_inputs(
        weather_cmp.JOIN_RUN)
    X_off, X_on, y, spec = weather_cmp.build_pair(adopted, weather)
    del adopted, weather
    cc.require(spec.key == "P6_clean", "base phase drift")
    cc.require(list(X_on.columns[-len(WEATHER_MODEL_FEATURES):]) == list(WEATHER_MODEL_FEATURES),
               "weather_on feature suffix drift")
    cc.require(not set(WEATHER_MODEL_FEATURES) & set(X_off.columns), "weather_off has weather features")
    log(f"loaded weather_on {X_on.shape}, weather_off {X_off.shape}, positives={int(y.sum())} "
        f"in {time.perf_counter() - t_load:.1f}s")

    expected_fps: dict[int, str | None] = {s: None for s in SEEDS}
    if smoke:
        idx = (y.groupby(y, group_keys=False)
               .apply(lambda g: g.sample(frac=args.sample / len(y), random_state=0)).index)
        idx = np.sort(np.asarray(idx))
        X_on = X_on.iloc[idx].reset_index(drop=True)
        X_off = X_off.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
        log(f"SMOKE sample: {len(y)} rows (not performance evidence; fingerprint check skipped)")
    else:
        cc.require(len(y) == 180_332, "full population must be 180,332 labeled rows")
        manifest_ref = json.loads(COMPARE_MANIFEST.read_text(encoding="utf-8"))
        expected_fps = {s: str(manifest_ref["fold_fingerprints"][str(s)]) for s in SEEDS}
    matrices = {"weather_on": X_on, "weather_off": X_off}

    identity = {
        "name": args.name, "smoke": smoke, "sample": args.sample, "seeds": list(seeds),
        "rows": int(len(y)), "positives": int(y.sum()),
        "columns_sha256": {k: stable_json_hash(list(v.columns)) for k, v in matrices.items()},
        "grids": grids_description(args.rf_n_estimators, args.rf_n_jobs),
        "units": [list(u) for u in UNITS],
        # Fold-result-relevant code only; the runner's reporting code is hashed in
        # the manifest, so a reporting-only fix can resume from the checkpoint.
        "code_sha256_lf": {p: h for p, h in code_sha.items() if p in IDENTITY_CODE_FILES},
    }
    ckpt_dir = (out_dir if smoke else ROOT / "data" / "classifier_compare") / f"{args.name}_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "checkpoint.json"
    state = {"identity_sha256": stable_json_hash(identity), "runs": [], "folds": []}
    if ckpt.exists():
        loaded = json.loads(ckpt.read_text(encoding="utf-8"))
        cc.require(loaded.get("identity_sha256") == state["identity_sha256"],
                   "checkpoint belongs to a different experiment identity")
        state = loaded
        log(f"RESUME checkpoint: {len(state['runs'])} units, {len(state['folds'])} folds")

    def save_state() -> None:
        atomic_json(ckpt, state)

    done = {(int(r["seed"]), r["condition"], r["model"]) for r in state["runs"]}
    for seed in seeds:
        for condition, model_key in UNITS:
            if (seed, condition, model_key) in done:
                log(f"RESUME seed={seed} {condition}/{model_key}")
                continue
            log(f"START seed={seed} {condition}/{model_key}")
            run, _ = evaluate_unit(
                condition, model_key, matrices[condition], y, seed=seed, spec=spec,
                expected_fp=expected_fps[seed], state=state, ckpt_dir=ckpt_dir,
                save_state=save_state, rf_n_estimators=args.rf_n_estimators,
                rf_n_jobs=args.rf_n_jobs)
            state["runs"].append(run)
            save_state()
            log(f"DONE seed={seed} {condition}/{model_key} MacroF1={run['macro_f1_nested']:.6f} "
                f"LL={run['log_loss']:.6f} AUC={run['roc_auc']:.6f} "
                f"P1={run['precision_delayed']:.4f} R1={run['recall_delayed']:.4f} "
                f"({run['elapsed_sec']:.0f}s)")

    runs = pd.DataFrame(state["runs"]).sort_values(["seed", "condition", "model"]).reset_index(drop=True)
    folds_df = (pd.DataFrame(state["folds"]).sort_values(["seed", "condition", "model", "fold"])
                .reset_index(drop=True))
    cc.require(runs.groupby("seed").fold_fingerprint.nunique().eq(1).all(), "outer fold mismatch")

    selection = selection_report(folds_df)
    comparisons, reproduction = None, None
    if not smoke:
        comparisons = build_comparisons(runs)
        reproduction = reproduction_checks(folds_df, pd.read_csv(COMPARE_FOLDS))
        paired = flatten_comparisons(comparisons)

    summary = {
        "schema_version": 1, "name": args.name, "smoke_only_not_evidence": smoke,
        "question": ("With wider predeclared grids, does the RF > LightGBM ordering of the "
                     "20261008 comparison hold, and does Random Forest gain from the 14 weather "
                     "features on the same rows/folds (paired weather_on - weather_off)?"),
        "population": {"description": "P6_clean (+14 weather features for weather_on), labeled "
                                      "date-attributed rows of the 10-minute full weather join",
                       "rows": int(len(y)), "positive_rows": int(y.sum()),
                       "positive_rate": float(y.mean()),
                       "input_columns": {k: int(v.shape[1]) for k, v in matrices.items()},
                       "sample": args.sample},
        "seeds": list(seeds),
        "fold_fingerprints": {str(int(r.seed)): r.fold_fingerprint
                              for r in runs.drop_duplicates("seed").itertuples()},
        "reference_fold_fingerprints_20261008": {str(k): v for k, v in expected_fps.items()},
        "protocol": protocol_description(replace(runner.CFG, seed=SEEDS[0]), spec),
        "grids": grids_description(args.rf_n_estimators, args.rf_n_jobs),
        "models": model_summary(runs),
        "selection": selection,
        "paired_comparisons": comparisons,
        "reproduction_of_20261008_inside_tuning_grids": reproduction,
        "references": {
            "logistic_regression_and_untuned_models": {
                "path": COMPARE_RUNS.relative_to(ROOT).as_posix(), "sha256": digest(COMPARE_RUNS)},
            "lightgbm_weather_on_off": {
                "path": WEATHER_RUNS.relative_to(ROOT).as_posix(), "sha256": digest(WEATHER_RUNS)},
        },
        "remaining_candidates": [
            "Post-hoc probability calibration (e.g. Platt/Isotonic on an inner-holdout split) "
            "was out of scope for this round; LogLoss here includes calibration quality.",
            "LightGBM weather on/off was not retuned; its on/off reference is the 20260922 fixed-config run.",
        ],
        "interpretation_guardrails": [
            "Same labeled rows, target, outer folds (fingerprint-checked against 20261008) and seeds.",
            "Configurations, tree counts and thresholds are selected on the inner holdout of each outer-train only; outer-valid labels select nothing.",
            "Grids are predeclared, finite budgets (LightGBM 24 configs x 8 tree counts, RF 9 configs at 300 trees); not a global optimisation.",
            "The RF on/off comparison uses the tuned grid in both conditions while the LightGBM on/off reference uses one fixed config; the two deltas are not a like-for-like tuning comparison.",
            "Static stratified CV on one sample period; not a time-ordered or future-deployment estimate.",
            "Three seeds are split repetitions of the same rows, not independent datasets or confidence intervals.",
        ],
    }

    atomic_csv(final_paths["runs"], runs)
    atomic_csv(final_paths["folds"], folds_df)
    if not smoke:
        atomic_csv(final_paths["paired"], paired)
    atomic_json(final_paths["summary"], summary)

    import sklearn
    manifest = {
        "schema_version": 1, "name": args.name, "successful": True,
        "smoke_only_not_evidence": smoke,
        "git_sha": git_sha, "git_status_porcelain_at_start": git_dirty,
        "python_version": platform.python_version(),
        "package_versions": {"pandas": pd.__version__, "numpy": np.__version__,
                             "scikit-learn": sklearn.__version__, "lightgbm": lgb.__version__},
        "cpu_count": os.cpu_count(), "process_priority": priority,
        "uv_lock_sha256": digest(ROOT / "uv.lock"),
        "inputs": {k: {"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p),
                       "bytes": p.stat().st_size} for k, p in source_paths.items()},
        "reference_runs": {
            "compare_runs": {"path": COMPARE_RUNS.relative_to(ROOT).as_posix(), "sha256": digest(COMPARE_RUNS)},
            "compare_folds": {"path": COMPARE_FOLDS.relative_to(ROOT).as_posix(), "sha256": digest(COMPARE_FOLDS)},
            "compare_manifest": {"path": COMPARE_MANIFEST.relative_to(ROOT).as_posix(),
                                 "sha256": digest(COMPARE_MANIFEST)},
            "weather_runs": {"path": WEATHER_RUNS.relative_to(ROOT).as_posix(), "sha256": digest(WEATHER_RUNS)},
        },
        "join_name": weather_cmp.JOIN_RUN,
        "code_sha256_lf": code_sha,
        "experiment_identity": identity,
        "experiment_identity_sha256": stable_json_hash(identity),
        "fold_fingerprints": summary["fold_fingerprints"],
        "reference_fold_fingerprints_20261008": summary["reference_fold_fingerprints_20261008"],
        "fold_fingerprints_equal_reference": (None if smoke else bool(all(
            summary["fold_fingerprints"][str(s)] == expected_fps[s] for s in SEEDS))),
        "grids": summary["grids"],
        "protocol": summary["protocol"],
        "outputs": {k: {"path": p.name, "sha256": digest(p), "bytes": p.stat().st_size}
                    for k, p in final_paths.items() if k != "manifest" and p.exists()},
        "checkpoint": str(ckpt.relative_to(ROOT).as_posix()) if ckpt.is_relative_to(ROOT) else str(ckpt),
        "fits_models": True, "network_requests": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_json(final_paths["manifest"], manifest)
    log("FINISHED " + json.dumps({
        "runs": final_paths["runs"].name,
        "lightgbm_reproduction_max_abs_diff": None if reproduction is None else reproduction["lightgbm_max_abs_diff"],
        "rf_reproduction_max_abs_diff": None if reproduction is None else reproduction["random_forest_max_abs_diff"],
    }))


if __name__ == "__main__":
    main()
