"""Fair Logistic Regression / Random Forest / LightGBM comparison (tutor feedback 1).

Population: the final weather comparison population (180,332 labeled,
date-attributed rows; P6_clean preprocessing + the 14 frozen weather features,
i.e. the ``weather_on`` matrix of notebooks/run_weather_model_comparison.py).
All three models share rows, target, outer folds (make_folds, seeds 42/1/7,
5 outer folds) and the nested inner boundary of src/classifier_compare.py.

LightGBM is rerun with the unchanged repository protocol
(``run_fold_nested_grid`` + ``runner.N_ESTIMATORS_GRID`` + inner-holdout
threshold) and checked against the existing 20260922 weather_on evidence.

Usage:
    python -u notebooks/run_classifier_comparison.py                  # full run
    python -u notebooks/run_classifier_comparison.py --sample 20000 \
        --out-dir <scratch> --name baseline_recovery_v2_classifier_compare_smoke_20261008
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

import rerun_all_phases as runner  # noqa: E402
from notebooks import run_weather_model_comparison as weather_cmp  # noqa: E402
from src import classifier_compare as cc  # noqa: E402
from src.cv import evaluate_oof, make_folds, run_fold_nested_grid  # noqa: E402
from src.weather_model import SEEDS, WEATHER_MODEL_FEATURES, digest, stable_json_hash  # noqa: E402

DEFAULT_NAME = "baseline_recovery_v2_classifier_compare_20261008"
REFERENCE_RUNS = (ROOT / "output" /
                  "baseline_recovery_v2_weather_model_compare_20260922_weather_model_runs.csv")
REFERENCE_CONDITION = "weather_on"
CODE_FILES = (
    "src/classifier_compare.py",
    "notebooks/run_classifier_comparison.py",
    "notebooks/run_weather_model_comparison.py",
    "src/weather_model.py",
    "src/weather_full.py",
    "src/features.py",
    "src/cv.py",
    "rerun_all_phases.py",
)


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def protocol_description(cfg, spec) -> dict:
    """Truthful protocol record (CVConfig.describe()'s inner_early_stopping label
    is NOT used: no model uses early stopping here)."""
    return {
        "n_splits": cfg.n_splits,
        "shuffle": cfg.shuffle,
        "seed": cfg.seed,
        "inner_holdout_frac": cfg.inner_holdout_frac,
        "threshold_grid": list(cfg.threshold_grid),
        "n_thresholds": int(cfg.thresholds().size),
        "metric_aggregation": cfg.metric_aggregation,
        "early_stopping": False,
        "inner_early_stopping": False,
        "hyperparameter_selection": "nested_grid_min_inner_holdout_log_loss",
        "tree_selection_lightgbm": "nested_grid_n_estimators",
        "threshold_selection": "outer_train_holdout_inner_holdout_macro_f1_argmax",
        "selected_model_refit_on_full_outer_train": False,
        "outer_valid_labels_used_for_selection": False,
        "target_encoding": {
            "cols": list(spec.cat_cols), "m": runner.TE_SMOOTHING_M,
            "inner_splits": spec.inner_splits, "boundary": "inner_train_only",
            "drop_original": spec.te_drop_original,
        },
        "probability_calibration": "none",
    }


def grids_description(rf_n_estimators: int, rf_n_jobs: int) -> dict:
    return {
        "lightgbm": {"n_estimators_grid": list(runner.N_ESTIMATORS_GRID),
                     "fixed_params": dict(runner.LGBM_PARAMS),
                     "random_state_policy": ("fixed 42 for all split seeds, as in the 20260922 "
                                             "weather runner (rerun_all_phases.main() not called)"),
                     "bagging_note": ("subsample=0.8 is inactive because subsample_freq=0 "
                                      "(LightGBM default); colsample_bytree=0.8 is active"),
                     "features": "P6_clean weather_on matrix + inner-boundary TE_* "
                                 "(raw categoricals kept as LightGBM category dtype)"},
        "logistic_regression": {
            "C_grid": list(cc.LR_C_GRID), "penalty": "l2", "solver": "lbfgs",
            "max_iter": cc.LR_MAX_ITER, "class_weight": None,
            "onehot_cols": list(cc.LR_ONEHOT_COLS),
            "onehot_min_frequency": cc.LR_ONEHOT_MIN_FREQUENCY,
            "high_cardinality_raw_codes_dropped_TE_only": list(cc.LR_TE_ONLY_COLS),
            "numeric": "SimpleImputer(median, add_indicator=True) -> StandardScaler",
            "fit_scope": "inner_train only (pipeline)",
            "random_state": cc.LR_RANDOM_STATE,
        },
        "random_forest": {
            "n_estimators": int(rf_n_estimators),
            "min_samples_leaf_grid": list(cc.RF_MIN_SAMPLES_LEAF_GRID),
            "max_features_grid": list(cc.RF_MAX_FEATURES_GRID),
            "n_jobs": int(rf_n_jobs), "class_weight": None, "bootstrap": True,
            "missing_values": "native sklearn tree NaN support (no imputation)",
            "categoricals": "integer category values (target-free global vocabulary) + inner-boundary TE_*",
            "random_state": cc.RF_RANDOM_STATE,
            "random_state_policy": "fixed 42 for all split seeds (analogous to LightGBM)",
        },
    }


def lgbm_fold(X_tr, y_tr, X_va, cfg, fold, spec) -> dict:
    selection: dict = {}
    fit = run_fold_nested_grid(
        runner.model_factory, X_tr, y_tr, X_va, cfg, fold=fold,
        n_estimators_grid=runner.N_ESTIMATORS_GRID,
        te_cols=spec.cat_cols, te_m=runner.TE_SMOOTHING_M,
        te_inner_splits=spec.inner_splits, te_drop_original=spec.te_drop_original,
        selection_metadata=selection,
    )
    cc.require("threshold" in selection, "nested threshold was not selected")
    return {
        "valid_probs": fit.valid_probs,
        "threshold": float(selection["threshold"]),
        "selected_params": {"n_estimators": int(fit.selected_n_estimators)},
        "grid_scores": [{"params": {"n_estimators": int(k)}, "inner_holdout_log_loss": float(v)}
                        for k, v in fit.grid_scores.items()],
        "best_inner_log_loss": float(min(fit.grid_scores.values())),
        "n_inner_train": fit.n_inner_train, "n_inner_holdout": fit.n_inner_holdout,
        "n_model_features": len(getattr(fit.model, "feature_name_", [])),
        "feature_names": list(getattr(fit.model, "feature_name_", [])),
        "extra": {},
    }


def sklearn_fold(model_key, X_tr, y_tr, X_va, cfg, fold, spec, *, seed,
                 rf_n_estimators, rf_n_jobs) -> dict:
    if model_key == "logistic_regression":
        grid = cc.lr_param_grid()
        def build(params, columns):
            return cc.build_logistic(params, columns, seed=cc.LR_RANDOM_STATE)
    else:
        grid = cc.rf_param_grid()
        def build(params, columns):
            return cc.build_random_forest(params, columns, seed=cc.RF_RANDOM_STATE,
                                          n_jobs=rf_n_jobs,
                                          n_estimators=rf_n_estimators)

    def on_candidate(params, score, elapsed):
        log(f"    {model_key} fold={fold + 1} {params} inner_LL={score:.6f} ({elapsed:.1f}s)")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = cc.run_fold_nested_params(
            build, grid, X_tr, y_tr, X_va, cfg, fold=fold,
            te_cols=spec.cat_cols, te_m=runner.TE_SMOOTHING_M,
            te_inner_splits=spec.inner_splits, te_drop_original=spec.te_drop_original,
            on_candidate=on_candidate,
        )
    warn_msgs = sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught})
    for msg in warn_msgs:
        log(f"    WARNING {msg}")
    extra: dict = {"warnings": warn_msgs}
    if model_key == "logistic_regression":
        names = list(fit.model.named_steps["columns"].get_feature_names_out())
        extra["n_iter"] = int(np.max(fit.model.named_steps["model"].n_iter_))
        extra["converged"] = bool(extra["n_iter"] < cc.LR_MAX_ITER)
    else:
        names = list(fit.model.named_steps["categories"].get_feature_names_out())
    return {
        "valid_probs": fit.valid_probs, "threshold": fit.threshold,
        "selected_params": fit.selected_params,
        "grid_scores": [{k: v for k, v in g.items()} for g in fit.grid_scores],
        "best_inner_log_loss": float(min(g["inner_holdout_log_loss"] for g in fit.grid_scores)),
        "n_inner_train": fit.n_inner_train, "n_inner_holdout": fit.n_inner_holdout,
        "n_model_features": len(names), "feature_names": names, "extra": extra,
    }


def evaluate_model(model_key, X, y, *, seed, spec, expected_fp, rf_n_estimators,
                   rf_n_jobs) -> tuple[dict, list[dict]]:
    cfg = replace(runner.CFG, seed=int(seed))
    folds = make_folds(y, cfg)
    fingerprint = cc.check_fold_fingerprint(folds, len(y), expected_fp)
    oof = np.full(len(y), np.nan)
    fold_rows, thresholds, input_names = [], [], []
    started = time.perf_counter()
    for fold, (tr, va) in enumerate(folds):
        t0 = time.perf_counter()
        X_tr = X.iloc[tr].reset_index(drop=True)
        y_tr = y.iloc[tr].reset_index(drop=True)
        X_va = X.iloc[va].reset_index(drop=True)
        if model_key == "lightgbm":
            res = lgbm_fold(X_tr, y_tr, X_va, cfg, fold, spec)
        else:
            res = sklearn_fold(model_key, X_tr, y_tr, X_va, cfg, fold, spec, seed=seed,
                               rf_n_estimators=rf_n_estimators, rf_n_jobs=rf_n_jobs)
        oof[va] = res["valid_probs"]
        thresholds.append(res["threshold"])
        if not input_names:
            input_names = res["feature_names"]
        fold_rows.append({
            "seed": int(seed), "model": model_key, "fold": fold + 1,
            "n_outer_train": int(len(tr)), "n_outer_valid": int(len(va)),
            "n_inner_train": res["n_inner_train"], "n_inner_holdout": res["n_inner_holdout"],
            "selected_params": cc.compact_json(res["selected_params"]),
            "threshold": res["threshold"],
            "best_inner_holdout_log_loss": res["best_inner_log_loss"],
            "grid_scores": cc.compact_json(res["grid_scores"]),
            "n_model_features": res["n_model_features"],
            "extra": cc.compact_json(res["extra"]),
            "fold_elapsed_sec": float(time.perf_counter() - t0),
        })
        log(f"  {model_key} seed={seed} fold={fold + 1} params={res['selected_params']} "
            f"thr={res['threshold']:.2f} ({fold_rows[-1]['fold_elapsed_sec']:.1f}s)")

    cc.require(bool(np.isfinite(oof).all()), "OOF probabilities incomplete/nonfinite")
    score = evaluate_oof(y, oof, folds, cfg, label=model_key,
                         per_fold_thresholds=thresholds)
    cm = score["confusion_matrix"]
    per_class = cc.class_metrics(cm["tn"], cm["fp"], cm["fn"], cm["tp"])
    cc.require(abs(per_class["macro_f1_from_cm"] - score["macro_f1"]) < 1e-9,
               "confusion matrix / macro F1 inconsistency")
    row = {
        "seed": int(seed), "model": model_key,
        "n_rows": int(len(y)), "positive_rows": int(y.sum()),
        "positive_rate": float(y.mean()),
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
        "tn": cm["tn"], "fp": cm["fp"], "fn": cm["fn"], "tp": cm["tp"],
        **{k: v for k, v in per_class.items() if k != "macro_f1_from_cm"},
        "fold_fingerprint": fingerprint,
        "fold_fingerprint_matches_reference": (None if expected_fp is None
                                               else bool(fingerprint == expected_fp)),
        "elapsed_sec": float(time.perf_counter() - started),
        "protocol": cc.compact_json(protocol_description(cfg, runner_spec_cache["spec"])),
    }
    return row, fold_rows


runner_spec_cache: dict = {}


def reference_rows() -> pd.DataFrame:
    ref = pd.read_csv(REFERENCE_RUNS)
    ref = ref[ref.condition.eq(REFERENCE_CONDITION)].set_index("seed")
    cc.require(sorted(ref.index.tolist()) == sorted(SEEDS), "reference weather_on seeds")
    return ref


def lightgbm_reproduction(runs: pd.DataFrame, folds: pd.DataFrame, ref: pd.DataFrame) -> dict:
    out = {"reference_file": REFERENCE_RUNS.relative_to(ROOT).as_posix(),
           "reference_sha256": digest(REFERENCE_RUNS),
           "reference_condition": REFERENCE_CONDITION, "by_seed": {}}
    all_exact = True
    for seed in SEEDS:
        mine = runs[(runs.model == "lightgbm") & (runs.seed == seed)].iloc[0]
        r = ref.loc[seed]
        sel = [p["n_estimators"] for p in json.loads(mine.selected_params)]
        entry = {
            "fold_fingerprint_equal": mine.fold_fingerprint == r.fold_fingerprint,
            "selected_n_estimators_equal": sel == json.loads(r.selected_n_estimators),
            "per_fold_thresholds_equal": (json.loads(mine.per_fold_thresholds)
                                          == json.loads(r.per_fold_thresholds)),
            "confusion_matrix_equal": all(int(mine[k]) == int(r[k]) for k in ("tn", "fp", "fn", "tp")),
            "abs_diff": {m: abs(float(mine[m]) - float(r[m]))
                         for m in ("macro_f1_nested", "log_loss", "roc_auc", "f1_at_050")},
            "n_features_rerun": int(mine.n_model_features_fold1),
            "n_features_reference": int(r.n_features),
        }
        entry["exact"] = bool(entry["fold_fingerprint_equal"] and entry["selected_n_estimators_equal"]
                              and entry["per_fold_thresholds_equal"] and entry["confusion_matrix_equal"]
                              and max(entry["abs_diff"].values()) < 1e-9)
        all_exact &= entry["exact"]
        out["by_seed"][str(seed)] = entry
    out["all_exact"] = bool(all_exact)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--sample", type=int, default=None,
                        help="SMOKE ONLY: stratified subsample of N labeled rows")
    parser.add_argument("--out-dir", default=str(ROOT / "output"))
    parser.add_argument("--models", default=",".join(cc.MODEL_KEYS))
    parser.add_argument("--rf-n-jobs", type=int, default=cc.RF_N_JOBS)
    parser.add_argument("--rf-n-estimators", type=int, default=cc.RF_N_ESTIMATORS,
                        help="only overridable together with --sample")
    args = parser.parse_args()
    sys.stdout.reconfigure(newline="\n")
    sys.stderr.reconfigure(newline="\n")
    cc.require(bool(re.fullmatch(r"baseline_recovery_v2_[A-Za-z0-9_]+", args.name)),
               "invalid run name")
    smoke = args.sample is not None
    models = tuple(m for m in args.models.split(",") if m)
    cc.require(set(models) <= set(cc.MODEL_KEYS) and models, "unknown model key")
    if not smoke:
        cc.require(models == cc.MODEL_KEYS, "full run must include all three models")
        cc.require(args.rf_n_estimators == cc.RF_N_ESTIMATORS,
                   "RF budget is fixed for the full run")
        cc.require("smoke" not in args.name, "full run name must not contain 'smoke'")
    else:
        cc.require("smoke" in args.name, "sample runs must use a name containing 'smoke'")

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
    log(f"git HEAD {git_sha}; dirty entries: {git_dirty}")

    t_load = time.perf_counter()
    adopted, weather, join_manifest, _scenario, source_paths = weather_cmp.load_inputs(
        weather_cmp.JOIN_RUN)
    _X_off, X, y, spec = weather_cmp.build_pair(adopted, weather)
    del adopted, weather, _X_off
    runner_spec_cache["spec"] = spec
    cc.require(spec.key == "P6_clean", "base phase drift")
    cc.require(list(X.columns[-len(WEATHER_MODEL_FEATURES):]) == list(WEATHER_MODEL_FEATURES),
               "weather_on feature suffix drift")
    log(f"loaded weather_on matrix {X.shape}, positives={int(y.sum())} "
        f"in {time.perf_counter() - t_load:.1f}s")

    expected_fps: dict[int, str | None] = {s: None for s in SEEDS}
    ref = None
    if smoke:
        rng_idx = (y.groupby(y, group_keys=False)
                   .apply(lambda g: g.sample(frac=args.sample / len(y), random_state=0)).index)
        rng_idx = np.sort(np.asarray(rng_idx))
        X = X.iloc[rng_idx].reset_index(drop=True)
        y = y.iloc[rng_idx].reset_index(drop=True)
        log(f"SMOKE sample: {len(y)} rows (not performance evidence; fingerprint check skipped)")
    else:
        cc.require(len(y) == 180_332, "full population must be 180,332 labeled rows")
        ref = reference_rows()
        expected_fps = {s: str(ref.loc[s, "fold_fingerprint"]) for s in SEEDS}

    identity = {
        "name": args.name, "git_sha": git_sha, "smoke": smoke, "sample": args.sample,
        "rows": int(len(y)), "positives": int(y.sum()),
        "columns_sha256": stable_json_hash(list(X.columns)),
        "grids": grids_description(args.rf_n_estimators, args.rf_n_jobs),
        "seeds": list(SEEDS), "models": list(models),
    }
    ckpt_dir = (out_dir if smoke else ROOT / "data" / "classifier_compare") / f"{args.name}_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "checkpoint.json"
    state = {"identity_sha256": stable_json_hash(identity), "rows": [], "folds": []}
    if ckpt.exists():
        loaded = json.loads(ckpt.read_text(encoding="utf-8"))
        cc.require(loaded.get("identity_sha256") == state["identity_sha256"],
                   "checkpoint belongs to a different experiment identity")
        state = loaded
    done = {(int(r["seed"]), r["model"]) for r in state["rows"]}

    for seed in SEEDS:
        for model_key in models:
            if (seed, model_key) in done:
                log(f"RESUME seed={seed} model={model_key}")
                continue
            log(f"START seed={seed} model={model_key}")
            row, fold_rows = evaluate_model(
                model_key, X, y, seed=seed, spec=spec, expected_fp=expected_fps[seed],
                rf_n_estimators=args.rf_n_estimators, rf_n_jobs=args.rf_n_jobs)
            state["rows"].append(row)
            state["folds"].extend(fold_rows)
            atomic_json(ckpt, state)
            log(f"DONE seed={seed} model={model_key} MacroF1={row['macro_f1_nested']:.6f} "
                f"LL={row['log_loss']:.6f} AUC={row['roc_auc']:.6f} "
                f"P1={row['precision_delayed']:.4f} R1={row['recall_delayed']:.4f} "
                f"({row['elapsed_sec']:.0f}s)")

    runs = pd.DataFrame(state["rows"])
    runs["model"] = pd.Categorical(runs.model, categories=list(cc.MODEL_KEYS), ordered=True)
    runs = runs.sort_values(["seed", "model"]).reset_index(drop=True)
    runs["model"] = runs.model.astype(str)
    folds_df = pd.DataFrame(state["folds"]).sort_values(["seed", "model", "fold"]).reset_index(drop=True)
    cc.require(runs.groupby("seed").fold_fingerprint.nunique().eq(1).all(), "outer fold mismatch")

    complete = set(models) == set(cc.MODEL_KEYS)
    paired = cc.paired_deltas(runs, SEEDS) if complete else pd.DataFrame()
    aggregate = cc.summarize(runs, paired) if complete else {}
    reproduction = lightgbm_reproduction(runs, folds_df, ref) if (ref is not None) else None

    summary = {
        "schema_version": 1, "name": args.name, "smoke_only_not_evidence": smoke,
        "question": ("Do Logistic Regression and Random Forest, each with its own "
                     "preprocessing and a small inner selection budget, match LightGBM "
                     "on the same rows/folds/seeds of the weather_on population?"),
        "population": {"description": "P6_clean + 14 weather features (weather_on), labeled "
                                      "date-attributed rows of the 10-minute full weather join",
                       "rows": int(len(y)), "positive_rows": int(y.sum()),
                       "positive_rate": float(y.mean()), "input_columns": int(X.shape[1]),
                       "sample": args.sample},
        "seeds": list(SEEDS),
        "fold_fingerprints": {str(int(r.seed)): r.fold_fingerprint
                              for r in runs.drop_duplicates("seed").itertuples()},
        "protocol": protocol_description(replace(runner.CFG, seed=SEEDS[0]), spec),
        "grids": grids_description(args.rf_n_estimators, args.rf_n_jobs),
        **aggregate,
        "lightgbm_reproduction_vs_20260922_weather_on": reproduction,
        "interpretation_guardrails": [
            "Same 180,332 labeled rows, target, outer folds (fingerprint-checked) and seeds for all models.",
            "Hyperparameters and thresholds are selected on the inner holdout of each outer-train only; outer-valid labels select nothing.",
            "Grids are small, predeclared budgets (LR 5 C values, RF 6 configs at 300 trees, LightGBM 8 tree counts); this is not a global hyperparameter optimisation of any model.",
            "Probabilities are reported without calibration; LogLoss partly reflects calibration quality (notably for Random Forest).",
            "Static stratified CV on one sample period; not a time-ordered or future-deployment estimate.",
            "Three seeds are split repetitions of the same rows, not independent datasets or confidence intervals.",
        ],
    }

    atomic_csv(final_paths["runs"], runs)
    atomic_csv(final_paths["folds"], folds_df)
    if complete:
        atomic_csv(final_paths["paired"], paired)
    atomic_json(final_paths["summary"], summary)

    import lightgbm
    import sklearn
    manifest = {
        "schema_version": 1, "name": args.name, "successful": True,
        "smoke_only_not_evidence": smoke,
        "git_sha": git_sha, "git_status_porcelain_at_start": git_dirty,
        "python_version": platform.python_version(),
        "package_versions": {"pandas": pd.__version__, "numpy": np.__version__,
                             "scikit-learn": sklearn.__version__,
                             "lightgbm": lightgbm.__version__},
        "cpu_count": os.cpu_count(),
        "uv_lock_sha256": digest(ROOT / "uv.lock"),
        "inputs": {k: {"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p),
                       "bytes": p.stat().st_size} for k, p in source_paths.items()},
        "reference_runs": {"path": REFERENCE_RUNS.relative_to(ROOT).as_posix(),
                           "sha256": digest(REFERENCE_RUNS)},
        "join_name": weather_cmp.JOIN_RUN,
        "code_sha256_lf": {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                           for p in CODE_FILES},
        "experiment_identity": identity,
        "experiment_identity_sha256": stable_json_hash(identity),
        "fold_fingerprints": summary["fold_fingerprints"],
        "reference_fold_fingerprints": {str(k): v for k, v in expected_fps.items()},
        "grids": summary["grids"],
        "protocol": summary["protocol"],
        "outputs": {k: {"path": p.name, "sha256": digest(p), "bytes": p.stat().st_size}
                    for k, p in final_paths.items() if k != "manifest" and p.exists()},
        "checkpoint": str(ckpt.relative_to(ROOT)) if ckpt.is_relative_to(ROOT) else str(ckpt),
        "fits_models": True, "network_requests": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_json(final_paths["manifest"], manifest)
    log("FINISHED " + json.dumps({
        "runs": final_paths["runs"].name,
        "lightgbm_reproduction_all_exact": None if reproduction is None else reproduction["all_exact"],
    }))


if __name__ == "__main__":
    main()
