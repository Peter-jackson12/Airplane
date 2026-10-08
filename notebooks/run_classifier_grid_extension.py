"""Grid-edge extension of the 20261008 classifier tuning.

Follow-up to ``notebooks/run_classifier_tuning.py``. Same population (180,332
labeled weather-join rows, P6_clean), target, outer folds (make_folds, seeds
42/1/7; fingerprints checked against the recorded 20261008 ones) and the same
nested inner boundary. Parts (separate output names, may run as two processes):

* ``--part lgbm`` (weather_on): learning_rate {0.01,0.02,0.03} x num_leaves
  {127,255,511} x min_child_samples {20,100} (18 configurations,
  subsample_freq=0), tree counts up to 1500 (per-learning-rate cap), selected
  by inner-holdout LogLoss. Each configuration is trained once to its largest
  tree count and each candidate count is scored as a prefix
  (``src.classifier_grid_extension``); outcome-equivalent to
  ``run_fold_nested_grid`` (tested).
* ``--part rf``: Random Forest, 300 trees, min_samples_leaf {25,50,100,200,400}
  x max_features {0.5,0.7,1.0}, for weather_off AND weather_on with the SAME
  grid (25 added to the requested 50..400 so that weather_on's earlier optimum
  stays available).
* ``--combine``: reads both finished parts and writes the cross-part paired
  deltas (LightGBM-extended vs RF-extended, RF on/off).

Outer-valid labels never enter any selection code. This runner imports, but
does not modify, ``notebooks/run_classifier_tuning.py`` and
``src/classifier_compare.py``.
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
from notebooks import run_classifier_tuning as tuning  # noqa: E402
from notebooks import run_weather_model_comparison as weather_cmp  # noqa: E402
from src import classifier_compare as cc  # noqa: E402
from src import classifier_grid_extension as ge  # noqa: E402
from src.cv import evaluate_oof, make_folds  # noqa: E402
from src.weather_model import SEEDS, WEATHER_MODEL_FEATURES, digest, stable_json_hash  # noqa: E402

BASE_NAME = "baseline_recovery_v2_classifier_grid_ext_20261008"
TUNING_NAME = "baseline_recovery_v2_classifier_tuning_20261008"
TUNING_RUNS = ROOT / "output" / f"{TUNING_NAME}_runs.csv"
TUNING_FOLDS = ROOT / "output" / f"{TUNING_NAME}_folds.csv"
TUNING_MANIFEST = ROOT / "output" / f"{TUNING_NAME}_manifest.json"
COMPARE_RUNS = tuning.COMPARE_RUNS
COMPARE_MANIFEST = tuning.COMPARE_MANIFEST

LGBM_N_JOBS = 6
PARTS = {
    "lgbm": (("weather_on", "lightgbm_ext"),),
    "rf": (("weather_off", "random_forest_ext"), ("weather_on", "random_forest_ext")),
}
CODE_FILES = (
    "src/classifier_grid_extension.py",
    "notebooks/run_classifier_grid_extension.py",
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
#: Fold-result-relevant code (reporting-only runner edits can still resume).
IDENTITY_CODE_FILES = (
    "src/classifier_grid_extension.py", "src/classifier_compare.py", "src/cv.py",
    "src/features.py", "src/weather_model.py", "src/weather_full.py", "rerun_all_phases.py",
    "notebooks/run_weather_model_comparison.py",
)
METRICS = cc.SUMMARY_METRICS
log = tuning.log
atomic_json = tuning.atomic_json
atomic_csv = tuning.atomic_csv


def lgbm_base_params() -> dict:
    params = tuning.lgbm_base_params()
    params["n_jobs"] = LGBM_N_JOBS
    return params


def protocol_description(cfg, spec) -> dict:
    out = tuning.protocol_description(cfg, spec)
    out["tree_selection_lightgbm"] = (
        "per configuration: one fit up to its largest predeclared tree count, every candidate "
        "count scored as a prefix (num_iteration) on the inner holdout, no early stopping; "
        "(configuration, count) chosen by minimal inner-holdout LogLoss")
    return out


def grids_description(part: str, rf_n_estimators: int, rf_n_jobs: int, trim_reason: str | None) -> dict:
    if part == "lgbm":
        configs = ge.lgbm_ext_grid()
        return {"lightgbm_ext": {
            "base_params": lgbm_base_params(),
            "configurations": configs, "n_configurations": len(configs),
            "n_estimators_candidates_union": list(ge.EXT_LGBM_N_ESTIMATORS),
            "max_trees_by_learning_rate": {str(k): v for k, v in ge.EXT_LGBM_MAX_TREES_BY_LR.items()},
            "n_estimators_by_config": [ge.trees_for_config(c) for c in configs],
            "axes": {k: list(v) for k, v in ge.lgbm_ext_axes().items()},
            "subsample_handling": ("subsample=0.8 declared but subsample_freq=0 (row bagging "
                                   "inactive), as in the 20261008 selected configurations; "
                                   "bagging was not selected in 20261008 and the prefix scoring "
                                   "requires it off"),
            "fixed": "max_depth=8 (caps a tree at 256 leaves: num_leaves 255 and 511 are both "
                     "effectively unconstrained), colsample_bytree=0.8, random_state=42",
            "threads": f"n_jobs={LGBM_N_JOBS} (shared machine); 20261008 used LightGBM default threads",
            "trim_reason": trim_reason,
        }}
    return {"random_forest_ext": {
        "n_estimators": int(rf_n_estimators),
        "configurations": ge.rf_ext_grid(), "n_configurations": len(ge.rf_ext_grid()),
        "axes": {k: list(v) for k, v in ge.rf_ext_axes().items()},
        "requested_leaf_grid": list(ge.EXT_RF_REQUESTED_LEAF_GRID),
        "added_to_requested": "min_samples_leaf=25 (weather_on optimum in 20261008) so the same "
                              "grid serves both conditions",
        "n_jobs": int(rf_n_jobs), "class_weight": None, "bootstrap": True,
        "random_state": cc.RF_RANDOM_STATE,
        "missing_values": "native sklearn tree NaN support (no imputation)",
        "trim_reason": trim_reason,
    }}


# =============================================================================
# Fold functions
# =============================================================================


def lgbm_ext_fold(X_tr, y_tr, X_va, cfg, fold, spec) -> dict:
    base = lgbm_base_params()

    def make_model(config: dict, n_estimators: int):
        params = cc.lgbm_config_params(base, config)
        params["n_estimators"] = int(n_estimators)
        return lgb.LGBMClassifier(**params)

    def on_config(config, k, score, elapsed):
        log(f"    lgbm fold={fold + 1} lr={config['learning_rate']} nl={config['num_leaves']} "
            f"mcs={config['min_child_samples']} k*={k} inner_LL={score:.6f} ({elapsed:.1f}s)")

    return ge.lgbm_ext_fold(
        make_model, ge.lgbm_ext_grid(), X_tr, y_tr, X_va, cfg, fold=fold,
        te_cols=spec.cat_cols, te_m=runner.TE_SMOOTHING_M,
        te_inner_splits=spec.inner_splits, te_drop_original=spec.te_drop_original,
        on_config=on_config)


def rf_ext_fold(X_tr, y_tr, X_va, cfg, fold, spec, *, rf_n_estimators, rf_n_jobs, tag) -> dict:
    def build(params, columns):
        return cc.build_random_forest(params, columns, seed=cc.RF_RANDOM_STATE,
                                      n_jobs=rf_n_jobs, n_estimators=rf_n_estimators)

    def on_candidate(params, score, elapsed):
        log(f"    {tag} fold={fold + 1} {params} inner_LL={score:.6f} ({elapsed:.1f}s)")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = cc.run_fold_nested_params(
            build, ge.rf_ext_grid(), X_tr, y_tr, X_va, cfg, fold=fold,
            te_cols=spec.cat_cols, te_m=runner.TE_SMOOTHING_M,
            te_inner_splits=spec.inner_splits, te_drop_original=spec.te_drop_original,
            on_candidate=on_candidate)
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
        "grid_edge": cc.grid_edge_flags(fit.selected_params, ge.rf_ext_axes()),
        "extra": {"warnings": warn_msgs},
    }


# =============================================================================
# Unit evaluation with per-fold checkpoints
# =============================================================================


def evaluate_unit(condition, model_key, X, y, *, seed, spec, expected_fp, state, ckpt_dir,
                  save_state, rf_n_estimators, rf_n_jobs) -> dict:
    cfg = replace(runner.CFG, seed=int(seed))
    folds = make_folds(y, cfg)
    fingerprint = cc.check_fold_fingerprint(folds, len(y), expected_fp)
    key = tuning.unit_key(seed, condition, model_key)
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
        if model_key == "lightgbm_ext":
            res = lgbm_ext_fold(X_tr, y_tr, X_va, cfg, fold, spec)
        else:
            res = rf_ext_fold(X_tr, y_tr, X_va, cfg, fold, spec, rf_n_estimators=rf_n_estimators,
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
    return {
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
        "target_sha256": cc.target_sha256(y),
        "row_key_sha256": cc.row_key_sha256(y.index),
        "elapsed_sec": float(sum(r["fold_elapsed_sec"] for r in fold_rows)),
        "protocol": cc.compact_json(protocol_description(cfg, spec)),
    }


# =============================================================================
# References, reproduction checks and comparisons
# =============================================================================


def reproduction_checks(folds_df: pd.DataFrame, tuning_folds: pd.DataFrame) -> dict:
    """Configurations shared with the 20261008 tuning grids must reproduce their
    recorded inner-holdout LogLoss (RF: every shared (leaf, max_features) cell;
    LightGBM: lr=0.03, num_leaves=127, mcs in {20,100}, tree counts <= 600)."""
    out: dict = {"lightgbm": {}, "random_forest": {}}
    max_l, max_r = 0.0, 0.0
    for r in folds_df.itertuples():
        key = f"{r.seed}/{r.condition}/fold{r.fold}"
        if r.model == "lightgbm_ext":
            ref = tuning_folds[(tuning_folds.model == "lightgbm_tuned") & (tuning_folds.seed == r.seed)
                               & (tuning_folds.fold == r.fold)].iloc[0]
            ref_cfgs = {cc.compact_json(g["config"]): g for g in json.loads(ref.grid_scores)}
            diffs = []
            for g in json.loads(r.grid_scores):
                other = ref_cfgs.get(cc.compact_json(g["config"]))
                if other is None:
                    continue
                for k, v in other["n_estimators_scores"].items():
                    if k in g["n_estimators_scores"]:
                        diffs.append(abs(g["n_estimators_scores"][k] - v))
            cc.require(len(diffs) > 0, "no LightGBM config shared with 20261008 tuning")
            out["lightgbm"][key] = {"n_compared": len(diffs), "max_abs_inner_ll_diff": max(diffs)}
            max_l = max(max_l, max(diffs))
        elif r.model == "random_forest_ext":
            ref = tuning_folds[(tuning_folds.model == "random_forest_tuned") & (tuning_folds.seed == r.seed)
                               & (tuning_folds.condition == r.condition)
                               & (tuning_folds.fold == r.fold)].iloc[0]
            ref_cells = {cc.compact_json(g["params"]): g["inner_holdout_log_loss"]
                         for g in json.loads(ref.grid_scores)}
            diffs = [abs(g["inner_holdout_log_loss"] - ref_cells[cc.compact_json(g["params"])])
                     for g in json.loads(r.grid_scores) if cc.compact_json(g["params"]) in ref_cells]
            cc.require(len(diffs) > 0, "no RF config shared with 20261008 tuning")
            out["random_forest"][key] = {"n_compared": len(diffs), "max_abs_inner_ll_diff": max(diffs)}
            max_r = max(max_r, max(diffs))
    out["lightgbm_max_abs_diff"] = max_l
    out["random_forest_max_abs_diff"] = max_r
    return out


def pick(frame: pd.DataFrame, **kw) -> pd.DataFrame:
    sub = frame
    for k, v in kw.items():
        sub = sub[sub[k] == v]
    cc.require(len(sub) == len(SEEDS), f"reference rows missing for {kw}")
    return sub


def build_comparisons(part: str, runs: pd.DataFrame) -> list[dict]:
    t_runs = pd.read_csv(TUNING_RUNS)
    c_runs = pd.read_csv(COMPARE_RUNS)
    specs = []
    if part == "lgbm":
        lg = pick(runs, condition="weather_on", model="lightgbm_ext")
        specs = [
            (lg, pick(t_runs, condition="weather_on", model="lightgbm_tuned"),
             "weather_on: lightgbm_ext - lightgbm_tuned(20261008, 24 configs)"),
            (lg, pick(c_runs, model="lightgbm"),
             "weather_on: lightgbm_ext - lightgbm(20261008 fixed config)"),
            (lg, pick(t_runs, condition="weather_on", model="random_forest_tuned"),
             "weather_on: lightgbm_ext - random_forest_tuned(20261008, 9 configs)"),
            (lg, pick(c_runs, model="logistic_regression"),
             "weather_on: lightgbm_ext - logistic_regression(20261008)"),
        ]
    else:
        off = pick(runs, condition="weather_off", model="random_forest_ext")
        on = pick(runs, condition="weather_on", model="random_forest_ext")
        t_on = pick(t_runs, condition="weather_on", model="random_forest_tuned")
        t_off = pick(t_runs, condition="weather_off", model="random_forest_tuned")
        specs = [
            (on, off, "random_forest_ext (identical 15-config grid): weather_on - weather_off"),
            (off, t_off, "weather_off: random_forest_ext - random_forest_tuned(20261008, leaf 10/25/50)"),
            (on, t_on, "weather_on: random_forest_ext - random_forest_tuned(20261008, leaf 10/25/50)"),
            (t_on, off, "GRIDS DIFFER: weather_on random_forest_tuned(20261008) - weather_off random_forest_ext"),
            (t_on, t_off, "random_forest_tuned(20261008 grids): weather_on - weather_off [reference]"),
        ]
    return [cc.paired_seed_deltas(a, b, SEEDS, METRICS, label=label) for a, b, label in specs]


# =============================================================================
# Combine mode
# =============================================================================


def combine(out_dir: Path, base: str) -> None:
    paths = {p: out_dir / f"{base}_{p}_runs.csv" for p in ("lgbm", "rf")}
    cc.require(all(p.exists() for p in paths.values()), "both parts must be finished first")
    manifests = {p: json.loads((out_dir / f"{base}_{p}_manifest.json").read_text(encoding="utf-8"))
                 for p in paths}
    cc.require(all(m["successful"] and not m["smoke_only_not_evidence"] for m in manifests.values()),
               "parts must be successful non-smoke runs")
    lg = pd.read_csv(paths["lgbm"])
    rf = pd.read_csv(paths["rf"])
    comps = [
        cc.paired_seed_deltas(pick(rf, condition="weather_on", model="random_forest_ext"),
                              pick(lg, condition="weather_on", model="lightgbm_ext"), SEEDS, METRICS,
                              label="weather_on: random_forest_ext - lightgbm_ext"),
    ]
    final = {"paired": out_dir / f"{base}_combined_paired_deltas.csv",
             "summary": out_dir / f"{base}_combined_summary.json"}
    cc.require(not any(p.exists() for p in final.values()), "combined outputs already exist")
    atomic_csv(final["paired"], tuning.flatten_comparisons(comps))
    atomic_json(final["summary"], {
        "schema_version": 1, "name": f"{base}_combined", "smoke_only_not_evidence": False,
        "inputs": {p: {"path": paths[p].name, "sha256": digest(paths[p]),
                       "manifest_sha256": digest(out_dir / f"{base}_{p}_manifest.json")} for p in paths},
        "paired_comparisons": comps,
        "note": "Cross-part comparison; both parts share rows/folds/seeds (fingerprint-checked).",
    })
    log(f"combined written: {final['summary'].name}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=sorted(PARTS))
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--base-name", default=BASE_NAME)
    parser.add_argument("--sample", type=int, default=None,
                        help="SMOKE ONLY: stratified subsample of N labeled rows")
    parser.add_argument("--out-dir", default=str(ROOT / "output"))
    parser.add_argument("--rf-n-jobs", type=int, default=8)
    parser.add_argument("--rf-n-estimators", type=int, default=cc.RF_N_ESTIMATORS,
                        help="only overridable together with --sample")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                        help="only overridable together with --sample")
    parser.add_argument("--trim-reason", default=None,
                        help="recorded verbatim in the manifest when a predeclared trim is applied")
    parser.add_argument("--normal-priority", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(newline="\n")
    sys.stderr.reconfigure(newline="\n")
    cc.require(bool(re.fullmatch(r"baseline_recovery_v2_[A-Za-z0-9_]+", args.base_name)),
               "invalid run name")
    out_dir = Path(args.out_dir).resolve()
    if args.combine:
        combine(out_dir, args.base_name)
        return
    cc.require(args.part is not None, "--part is required")
    cc.require(1 <= args.rf_n_jobs <= 16, "RF n_jobs must be within 1..16")
    part = args.part
    name = f"{args.base_name}_{part}"
    smoke = args.sample is not None
    seeds = tuple(int(s) for s in args.seeds.split(",") if s)
    if not smoke:
        cc.require(seeds == tuple(SEEDS), "full run uses seeds 42/1/7")
        cc.require(args.rf_n_estimators == cc.RF_N_ESTIMATORS, "RF budget is fixed for the full run")
        cc.require("smoke" not in name, "full run name must not contain 'smoke'")
    else:
        cc.require("smoke" in name, "sample runs must use a base name containing 'smoke'")
        cc.require(set(seeds) <= set(SEEDS) and seeds, "unknown seeds")
    priority = "normal" if args.normal_priority else tuning.lower_process_priority()

    out_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "runs": out_dir / f"{name}_runs.csv",
        "folds": out_dir / f"{name}_folds.csv",
        "paired": out_dir / f"{name}_paired_deltas.csv",
        "summary": out_dir / f"{name}_summary.json",
        "manifest": out_dir / f"{name}_manifest.json",
    }
    cc.require(not any(p.exists() for p in final_paths.values()),
               "use a fresh name; final evidence is never overwritten")

    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                        text=True).splitlines()
    code_sha = {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                for p in CODE_FILES}
    log(f"git HEAD {git_sha}; dirty entries: {git_dirty}; priority={priority}")

    t_load = time.perf_counter()
    adopted, weather, _jm, _scenario, source_paths = weather_cmp.load_inputs(weather_cmp.JOIN_RUN)
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
        tuning_ref = json.loads(TUNING_MANIFEST.read_text(encoding="utf-8"))
        cc.require(all(str(tuning_ref["fold_fingerprints"][str(s)]) == expected_fps[s] for s in SEEDS),
                   "20261008 tuning and comparison fingerprints disagree")
    matrices = {"weather_on": X_on, "weather_off": X_off}
    units = PARTS[part]

    identity = {
        "name": name, "part": part, "smoke": smoke, "sample": args.sample, "seeds": list(seeds),
        "rows": int(len(y)), "positives": int(y.sum()),
        "columns_sha256": {k: stable_json_hash(list(v.columns)) for k, v in matrices.items()},
        "target_sha256": cc.target_sha256(y), "row_key_sha256": cc.row_key_sha256(y.index),
        "grids": grids_description(part, args.rf_n_estimators, args.rf_n_jobs, args.trim_reason),
        "units": [list(u) for u in units],
        "code_sha256_lf": {p: h for p, h in code_sha.items() if p in IDENTITY_CODE_FILES},
    }
    ckpt_dir = (out_dir if smoke else ROOT / "data" / "classifier_compare") / f"{name}_ckpt"
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
        for condition, model_key in units:
            if (seed, condition, model_key) in done:
                log(f"RESUME seed={seed} {condition}/{model_key}")
                continue
            log(f"START seed={seed} {condition}/{model_key}")
            run = evaluate_unit(
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

    selection = tuning.selection_report(folds_df)
    comparisons, reproduction, paired = None, None, None
    if not smoke:
        comparisons = build_comparisons(part, runs)
        reproduction = reproduction_checks(folds_df, pd.read_csv(TUNING_FOLDS))
        paired = tuning.flatten_comparisons(comparisons)

    summary = {
        "schema_version": 1, "name": name, "part": part, "smoke_only_not_evidence": smoke,
        "question": ("Do the choices left on grid edges in 20261008 persist when the grids are "
                     "extended (LightGBM: smaller learning rates, more leaves, more trees; "
                     "Random Forest: larger min_samples_leaf), and does the RF weather on/off "
                     "delta change under identical grids?"),
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
        "grids": grids_description(part, args.rf_n_estimators, args.rf_n_jobs, args.trim_reason),
        "models": tuning.model_summary(runs),
        "selection": selection,
        "paired_comparisons": comparisons,
        "reproduction_of_20261008_inside_extended_grids": reproduction,
        "references": {
            "tuning_20261008_runs": {"path": TUNING_RUNS.relative_to(ROOT).as_posix(),
                                     "sha256": digest(TUNING_RUNS)},
            "compare_20261008_runs": {"path": COMPARE_RUNS.relative_to(ROOT).as_posix(),
                                      "sha256": digest(COMPARE_RUNS)},
        },
        "interpretation_guardrails": [
            "Same labeled rows, target, outer folds (fingerprint-checked against 20261008) and seeds.",
            "Configurations, tree counts and thresholds are selected on the inner holdout of each outer-train only; outer-valid labels select nothing.",
            "Grids are predeclared finite budgets, not a global optimisation; an interior selection only means the neighbouring declared values were worse on the inner holdout.",
            "LightGBM keeps max_depth=8, which caps trees at 256 leaves; num_leaves 255 and 511 therefore cannot differ materially.",
            "Static stratified CV on one sample period; not a time-ordered or future-deployment estimate.",
            "Three seeds are split repetitions of the same rows, not independent datasets or confidence intervals.",
            "Paired deltas against 20261008 results compare different grid budgets; labelled GRIDS DIFFER where applicable.",
        ],
    }

    atomic_csv(final_paths["runs"], runs)
    atomic_csv(final_paths["folds"], folds_df)
    if not smoke:
        atomic_csv(final_paths["paired"], paired)
    atomic_json(final_paths["summary"], summary)

    import sklearn
    manifest = {
        "schema_version": 1, "name": name, "part": part, "successful": True,
        "smoke_only_not_evidence": smoke,
        "git_sha": git_sha, "git_status_porcelain_at_start": git_dirty,
        "python_version": platform.python_version(),
        "package_versions": {"pandas": pd.__version__, "numpy": np.__version__,
                             "scikit-learn": sklearn.__version__, "lightgbm": lgb.__version__},
        "cpu_count": os.cpu_count(), "process_priority": priority,
        "thread_budget": {"lightgbm_n_jobs": LGBM_N_JOBS, "rf_n_jobs": int(args.rf_n_jobs)},
        "uv_lock_sha256": digest(ROOT / "uv.lock"),
        "inputs": {k: {"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p),
                       "bytes": p.stat().st_size} for k, p in source_paths.items()},
        "reference_runs": {
            "tuning_runs": {"path": TUNING_RUNS.relative_to(ROOT).as_posix(), "sha256": digest(TUNING_RUNS)},
            "tuning_folds": {"path": TUNING_FOLDS.relative_to(ROOT).as_posix(), "sha256": digest(TUNING_FOLDS)},
            "tuning_manifest": {"path": TUNING_MANIFEST.relative_to(ROOT).as_posix(),
                                "sha256": digest(TUNING_MANIFEST)},
            "compare_runs": {"path": COMPARE_RUNS.relative_to(ROOT).as_posix(), "sha256": digest(COMPARE_RUNS)},
            "compare_manifest": {"path": COMPARE_MANIFEST.relative_to(ROOT).as_posix(),
                                 "sha256": digest(COMPARE_MANIFEST)},
        },
        "join_name": weather_cmp.JOIN_RUN,
        "code_sha256_lf": code_sha,
        "experiment_identity": identity,
        "experiment_identity_sha256": stable_json_hash(identity),
        "fold_fingerprints": summary["fold_fingerprints"],
        "reference_fold_fingerprints_20261008": summary["reference_fold_fingerprints_20261008"],
        "fold_fingerprints_equal_reference": (None if smoke else bool(all(
            summary["fold_fingerprints"][str(s)] == expected_fps[s] for s in SEEDS))),
        "target_sha256": cc.target_sha256(y), "row_key_sha256": cc.row_key_sha256(y.index),
        "grids": summary["grids"], "protocol": summary["protocol"],
        "trim_reason": args.trim_reason,
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
