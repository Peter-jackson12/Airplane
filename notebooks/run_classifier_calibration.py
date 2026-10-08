"""Per-classifier probability calibration on the weather_on population.

Same population (180,332 labeled weather-join rows, P6_clean + 14 weather
features), target, outer folds (make_folds, seeds 42/1/7; fingerprints checked
against the 20261008 comparison manifest) and inner boundary as the 20261008
classifier comparison/tuning runs.

Configurations are NOT re-searched: each fold reuses the configuration that
was selected inside that fold's inner boundary (inner-holdout LogLoss) by
* Logistic Regression: baseline_recovery_v2_classifier_compare_20261008_folds.csv
* Random Forest / LightGBM (weather_on): baseline_recovery_v2_classifier_tuning_20261008_folds.csv
The refit is verified against the recorded runs (valid_probs_sha256 /
thresholds / inner-holdout LogLoss / pooled metrics).

Calibration arms (src/classifier_calibration.py): none, Platt, Isotonic, each
fitted on (a) the inner holdout (double use) or (b) 5-fold cross-fitted
inner-train scores (preferred). Thresholds are re-selected on the inner holdout.
Outer-valid labels are only used for the final pooled scoring.

Usage:
    python -u notebooks/run_classifier_calibration.py                    # full run
    python -u notebooks/run_classifier_calibration.py --sample 20000 --rf-n-estimators 50 \
        --seeds 42 --out-dir <scratch> --name baseline_recovery_v2_classifier_calibration_smoke_20261008
"""
from __future__ import annotations

import os

# Thread budget on a shared machine: BLAS/OpenMP pools capped before numpy loads.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse  # noqa: E402
from dataclasses import replace  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402

import rerun_all_phases as runner  # noqa: E402
from notebooks import run_classifier_comparison as base_cmp  # noqa: E402
from notebooks import run_classifier_tuning as tuning  # noqa: E402
from notebooks import run_weather_model_comparison as weather_cmp  # noqa: E402
from src import classifier_calibration as ccal  # noqa: E402
from src import classifier_compare as cc  # noqa: E402
from src.cv import make_folds  # noqa: E402
from src.weather_model import SEEDS, WEATHER_MODEL_FEATURES, digest, stable_json_hash  # noqa: E402

DEFAULT_NAME = "baseline_recovery_v2_classifier_calibration_20261008"
OUT = ROOT / "output"
COMPARE_NAME = "baseline_recovery_v2_classifier_compare_20261008"
TUNING_NAME = "baseline_recovery_v2_classifier_tuning_20261008"
REFS = {
    "compare_runs": OUT / f"{COMPARE_NAME}_runs.csv",
    "compare_folds": OUT / f"{COMPARE_NAME}_folds.csv",
    "compare_manifest": OUT / f"{COMPARE_NAME}_manifest.json",
    "tuning_runs": OUT / f"{TUNING_NAME}_runs.csv",
    "tuning_folds": OUT / f"{TUNING_NAME}_folds.csv",
    "tuning_manifest": OUT / f"{TUNING_NAME}_manifest.json",
}
TUNING_PROBS_DIR = ROOT / "data" / "classifier_compare" / f"{TUNING_NAME}_ckpt"
EARLIER_LGBM_CALIBRATION = {
    "report": "output/baseline_recovery_v2_calibration_20260917_report.md",
    "level_summary": "output/baseline_recovery_v2_calibration_20260917_level_summary.csv",
    "note": ("Earlier LightGBM-only calibration study: different population (255,001 labeled rows, "
             "P6_clean/P6_fixed without weather), fixed LightGBM config, ECE with 10 equal-width "
             "bins, calibrator fitted on the inner holdout ('shared') or a split of it "
             "('separate'). Numbers are not comparable with this run; cross-reference only."),
}
CODE_FILES = (
    "src/classifier_calibration.py",
    "notebooks/run_classifier_calibration.py",
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
#: Fold-result-relevant code (checkpoint identity). The runner's reporting code is
#: hashed in the manifest only, so a reporting-only fix can resume.
IDENTITY_CODE_FILES = ("src/classifier_calibration.py", "src/classifier_compare.py", "src/cv.py",
                       "src/features.py", "src/weather_model.py", "src/weather_full.py",
                       "rerun_all_phases.py", "notebooks/run_weather_model_comparison.py")
log = base_cmp.log
atomic_json = base_cmp.atomic_json
atomic_csv = base_cmp.atomic_csv


# =============================================================================
# Recorded selections and references
# =============================================================================


def recorded_selections() -> dict[tuple[str, int, int], dict]:
    """(model, seed, fold) -> recorded fold row (selected params, threshold, ...)."""
    cmp_f = pd.read_csv(REFS["compare_folds"])
    tun_f = pd.read_csv(REFS["tuning_folds"])
    out: dict[tuple[str, int, int], dict] = {}
    for r in cmp_f[cmp_f.model.eq("logistic_regression")].itertuples():
        out[("logistic_regression", int(r.seed), int(r.fold))] = {
            "params": json.loads(r.selected_params), "threshold": float(r.threshold),
            "best_inner_holdout_log_loss": float(r.best_inner_holdout_log_loss),
            "valid_probs_sha256": None, "source": REFS["compare_folds"].name}
    sub = tun_f[tun_f.condition.eq("weather_on")]
    for r in sub.itertuples():
        out[(str(r.model), int(r.seed), int(r.fold))] = {
            "params": json.loads(r.selected_params), "threshold": float(r.threshold),
            "best_inner_holdout_log_loss": float(r.best_inner_holdout_log_loss),
            "valid_probs_sha256": str(r.valid_probs_sha256), "source": REFS["tuning_folds"].name}
    for model in ccal.MODEL_KEYS:
        for seed in SEEDS:
            for fold in range(1, 6):
                cc.require((model, seed, fold) in out, f"recorded selection missing {model}/{seed}/{fold}")
    return out


def reference_runs() -> dict[tuple[str, int], pd.Series]:
    cmp_r = pd.read_csv(REFS["compare_runs"])
    tun_r = pd.read_csv(REFS["tuning_runs"])
    out = {}
    for r in cmp_r[cmp_r.model.eq("logistic_regression")].itertuples(index=False):
        out[("logistic_regression", int(r.seed))] = pd.Series(r._asdict())
    for r in tun_r[tun_r.condition.eq("weather_on")].itertuples(index=False):
        out[(str(r.model), int(r.seed))] = pd.Series(r._asdict())
    return out


def recorded_tuning_probs(model: str, seed: int, fold: int) -> np.ndarray | None:
    path = TUNING_PROBS_DIR / f"{seed}_weather_on_{model}_fold{fold}_valid_probs.npy"
    return np.load(path) if path.is_file() else None


# =============================================================================
# One (seed, model) unit with per-fold checkpoints
# =============================================================================


def fold_key(seed: int, model: str, fold: int) -> str:
    return f"{seed}_{model}_fold{fold}"


def run_fold(model_key, X, y, tr, va, cfg, fold, spec, *, params, build_kwargs) -> dict:
    """One outer fold. Only outer-train labels ``y.iloc[tr]`` are passed on."""
    X_tr = X.iloc[tr].reset_index(drop=True)
    y_tr = y.iloc[tr].reset_index(drop=True)
    X_va = X.iloc[va].reset_index(drop=True)
    build = ccal.selected_model_builder(model_key, params, **build_kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = ccal.calibrate_fold(
            build, X_tr, y_tr, X_va, cfg, fold=fold, te_cols=spec.cat_cols,
            te_m=runner.TE_SMOOTHING_M, te_inner_splits=spec.inner_splits,
            te_drop_original=spec.te_drop_original, log=log)
    res["warnings"] = sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught})
    return res


def evaluate_unit(model_key, X, y, *, seed, spec, expected_fp, selections, state, ckpt_dir,
                  save_state, build_kwargs, smoke) -> tuple[list[dict], list[pd.DataFrame]]:
    cfg = replace(runner.CFG, seed=int(seed))
    folds = make_folds(y, cfg)
    fingerprint = cc.check_fold_fingerprint(folds, len(y), expected_fp)
    oof = {arm: np.full(len(y), np.nan) for arm in ccal.ARM_NAMES}
    for fold, (tr, va) in enumerate(folds):
        key = fold_key(seed, model_key, fold + 1)
        npz = ckpt_dir / f"{key}_valid_probs.npz"
        done = [r for r in state["folds"] if r["fold_key"] == key]
        if done:
            stored = np.load(npz)
            for arm in ccal.ARM_NAMES:
                rec = next(r for r in done if r["arm"] == arm)
                cc.require(ccal.probs_sha256(stored[arm]) == rec["valid_probs_sha256"],
                           f"checkpoint probs mismatch {key}/{arm}")
                oof[arm][va] = stored[arm]
            log(f"  RESUME {key}")
            continue
        sel = selections[(model_key, int(seed), fold + 1)]
        t0 = time.perf_counter()
        res = run_fold(model_key, X, y, tr, va, cfg, fold, spec, params=sel["params"],
                       build_kwargs=build_kwargs)
        for msg in res["warnings"]:
            log(f"    WARNING {msg}")
        raw = np.asarray(res["raw_valid_probs"], dtype=float)
        cc.require(len(raw) == len(va) and bool(np.isfinite(raw).all()), "bad valid probs")
        # Reproduction of the recorded uncalibrated fold.
        raw_sha = ccal.probs_sha256(raw)
        rec_probs = None if smoke else recorded_tuning_probs(model_key, int(seed), fold + 1)
        repro = {
            "recorded_source": sel["source"],
            "raw_valid_probs_sha256_equal_recorded": (None if smoke or sel["valid_probs_sha256"] is None
                                                      else raw_sha == sel["valid_probs_sha256"]),
            "raw_valid_probs_max_abs_diff_recorded": (None if rec_probs is None or len(rec_probs) != len(raw)
                                                      else float(np.max(np.abs(rec_probs - raw)))),
            "none_threshold_equal_recorded": (None if smoke else
                                              res["arms"]["none"]["threshold"] == sel["threshold"]),
            "inner_holdout_log_loss_abs_diff_recorded": (None if smoke else abs(
                res["raw_inner_holdout_log_loss"] - sel["best_inner_holdout_log_loss"])),
        }
        np.savez(npz, **{arm: res["arms"][arm]["valid_probs"] for arm in ccal.ARM_NAMES})
        elapsed = time.perf_counter() - t0
        for arm in ccal.ARM_NAMES:
            a = res["arms"][arm]
            q = np.asarray(a["valid_probs"], dtype=float)
            oof[arm][va] = q
            state["folds"].append({
                "fold_key": key, "seed": int(seed), "model": model_key, "fold": fold + 1,
                "arm": arm, "calibrator": a["calibrator"], "calibration_data": a["calibration_data"],
                "n_outer_train": int(len(tr)), "n_outer_valid": int(len(va)),
                "n_inner_train": res["n_inner_train"], "n_inner_holdout": res["n_inner_holdout"],
                "selected_params": cc.compact_json(sel["params"]),
                "threshold": float(a["threshold"]),
                "recorded_threshold_uncalibrated": float(sel["threshold"]),
                "inner_holdout_log_loss": a["inner_holdout_log_loss"],
                "inner_holdout_ece_ef15": a["inner_holdout_ece_ef15"],
                "inner_holdout_macro_f1_at_threshold": a["inner_holdout_macro_f1_at_threshold"],
                "calibrator_info": cc.compact_json(a["calibrator_info"]),
                "valid_rows_at_iso_clip": a["valid_rows_at_clip"],
                "raw_inner_holdout_log_loss": res["raw_inner_holdout_log_loss"],
                "crossfit_inner_train_log_loss": res["crossfit_inner_train_log_loss"],
                "n_model_features": res["n_model_features"],
                "valid_probs_sha256": ccal.probs_sha256(q),
                "reproduction": cc.compact_json(repro),
                "warnings": cc.compact_json(res["warnings"]),
                "base_fit_sec": res["base_fit_sec"], "crossfit_sec": res["crossfit_sec"],
                "fold_elapsed_sec": float(elapsed),
            })
        save_state()
        th = {arm: res["arms"][arm]["threshold"] for arm in ccal.ARM_NAMES}
        log(f"  {key} params={sel['params']} thr={th} repro={repro} ({elapsed:.1f}s)")

    runs, bins = [], []
    fold_rows = [r for r in state["folds"] if r["seed"] == int(seed) and r["model"] == model_key]
    cc.require(len(fold_rows) == len(folds) * len(ccal.ARM_NAMES), f"incomplete folds {seed}/{model_key}")
    for arm, kind, source in ccal.ARMS:
        rows = sorted((r for r in fold_rows if r["arm"] == arm), key=lambda r: r["fold"])
        thresholds = [r["threshold"] for r in rows]
        score = ccal.score_pooled(y, oof[arm], folds, thresholds)
        runs.append({
            "seed": int(seed), "model": model_key, "arm": arm, "calibrator": kind,
            "calibration_data": source, "n_rows": int(len(y)), "positive_rows": int(y.sum()),
            **{k: v for k, v in score.items() if k != "per_fold_thresholds"},
            "per_fold_thresholds": cc.compact_json(score["per_fold_thresholds"]),
            "fold_fingerprint": fingerprint,
            "fold_fingerprint_matches_reference": (None if expected_fp is None
                                                   else bool(fingerprint == expected_fp)),
            "target_sha256": cc.target_sha256(y),
            "elapsed_sec_unit": float(sum(r["fold_elapsed_sec"] for r in rows)),
        })
        table = ccal.reliability_table(y, oof[arm])
        table.insert(0, "seed", int(seed))
        table.insert(1, "model", model_key)
        table.insert(2, "arm", arm)
        table.insert(3, "calibrator", kind)
        table.insert(4, "calibration_data", source)
        bins.append(table)
    return runs, bins


# =============================================================================
# Summaries
# =============================================================================


def run_level_reproduction(runs: pd.DataFrame, refs: dict) -> dict:
    out, worst = {}, 0.0
    for r in runs[runs.arm.eq("none")].itertuples():
        ref = refs[(r.model, int(r.seed))]
        diffs = {m: abs(float(getattr(r, m)) - float(ref[m]))
                 for m in ("macro_f1_nested", "log_loss", "roc_auc")}
        cm_equal = all(int(getattr(r, k)) == int(ref[k]) for k in ("tn", "fp", "fn", "tp"))
        thr_equal = json.loads(r.per_fold_thresholds) == json.loads(ref["per_fold_thresholds"])
        worst = max(worst, max(diffs.values()))
        out[f"{r.model}/{int(r.seed)}"] = {"abs_diff": diffs, "confusion_matrix_equal": cm_equal,
                                           "per_fold_thresholds_equal": thr_equal}
    return {"by_unit": out, "max_abs_metric_diff": worst,
            "all_confusion_matrices_equal": all(v["confusion_matrix_equal"] for v in out.values()),
            "all_thresholds_equal": all(v["per_fold_thresholds_equal"] for v in out.values())}


def fold_level_reproduction(folds_df: pd.DataFrame) -> dict:
    none = folds_df[folds_df.arm.eq("none")]
    rep = [json.loads(s) for s in none.reproduction]
    out: dict = {}
    for model in sorted(none.model.unique()):
        idx = [i for i, m in enumerate(none.model) if m == model]
        sub = [rep[i] for i in idx]
        hashes = [s["raw_valid_probs_sha256_equal_recorded"] for s in sub]
        diffs = [s["raw_valid_probs_max_abs_diff_recorded"] for s in sub]
        out[model] = {
            "n_folds": len(sub),
            "valid_probs_sha256_equal": (None if all(h is None for h in hashes)
                                         else int(sum(bool(h) for h in hashes))),
            "max_abs_valid_prob_diff": (None if all(d is None for d in diffs)
                                        else float(max(d for d in diffs if d is not None))),
            "thresholds_equal": int(sum(bool(s["none_threshold_equal_recorded"]) for s in sub)),
            "max_inner_holdout_log_loss_abs_diff": float(max(
                s["inner_holdout_log_loss_abs_diff_recorded"] for s in sub)),
        }
    return out


def level_summary(runs: pd.DataFrame) -> dict:
    out: dict = {}
    for (model, arm), sub in runs.groupby(["model", "arm"], sort=True):
        sub = sub.sort_values("seed")
        out.setdefault(model, {})[arm] = {
            m: {**ccal.stats(sub[m]),
                "values_by_seed": {str(int(s)): float(v) for s, v in zip(sub.seed, sub[m])}}
            for m in ccal.RUN_METRICS
        }
    return out


def model_vs_lightgbm(runs: pd.DataFrame, seeds) -> dict:
    """Per arm: model - lightgbm_tuned, paired by seed (same rows/folds)."""
    out: dict = {}
    for arm in ccal.ARM_NAMES:
        base = runs[(runs.arm == arm) & (runs.model == "lightgbm_tuned")]
        for model in ("logistic_regression", "random_forest_tuned"):
            a = runs[(runs.arm == arm) & (runs.model == model)]
            if len(a) != len(seeds) or len(base) != len(seeds):
                continue
            comp = cc.paired_seed_deltas(a, base, seeds, ccal.RUN_METRICS,
                                         label=f"{arm}: {model} - lightgbm_tuned")
            out.setdefault(arm, {})[model] = comp["metrics"]
    return out


def selection_shift(folds_df: pd.DataFrame) -> dict:
    out: dict = {}
    for (model, arm), sub in folds_df.groupby(["model", "arm"], sort=True):
        out.setdefault(model, {})[arm] = {
            "thresholds": {"min": float(sub.threshold.min()), "max": float(sub.threshold.max()),
                           "median": float(sub.threshold.median())},
            "folds_threshold_changed_vs_recorded": int((sub.threshold != sub.recorded_threshold_uncalibrated).sum()),
            "max_valid_rows_at_iso_clip": int(sub.valid_rows_at_iso_clip.max()),
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
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--models", default=",".join(ccal.MODEL_KEYS))
    parser.add_argument("--rf-n-jobs", type=int, default=6)
    parser.add_argument("--lgbm-n-jobs", type=int, default=4)
    parser.add_argument("--rf-n-estimators", type=int, default=cc.RF_N_ESTIMATORS,
                        help="only overridable together with --sample")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                        help="only overridable together with --sample")
    parser.add_argument("--normal-priority", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(newline="\n")
    sys.stderr.reconfigure(newline="\n")
    cc.require(bool(re.fullmatch(r"baseline_recovery_v2_[A-Za-z0-9_]+", args.name)), "invalid run name")
    cc.require(1 <= args.rf_n_jobs <= 8 and 1 <= args.lgbm_n_jobs <= 8, "thread budget exceeded")
    smoke = args.sample is not None
    seeds = tuple(int(s) for s in args.seeds.split(",") if s)
    models = tuple(m for m in args.models.split(",") if m)
    cc.require(set(models) <= set(ccal.MODEL_KEYS) and models, "unknown model key")
    if not smoke:
        cc.require(seeds == tuple(SEEDS), "full run uses seeds 42/1/7")
        cc.require(models == ccal.MODEL_KEYS, "full run uses all three models")
        cc.require(args.rf_n_estimators == cc.RF_N_ESTIMATORS, "RF budget is fixed for the full run")
        cc.require("smoke" not in args.name, "full run name must not contain 'smoke'")
    else:
        cc.require("smoke" in args.name, "sample runs must use a name containing 'smoke'")
        cc.require(set(seeds) <= set(SEEDS) and seeds, "unknown seeds")
    priority = "normal" if args.normal_priority else tuning.lower_process_priority()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "runs": out_dir / f"{args.name}_runs.csv",
        "folds": out_dir / f"{args.name}_folds.csv",
        "reliability_bins": out_dir / f"{args.name}_reliability_bins.csv",
        "paired": out_dir / f"{args.name}_paired_deltas.csv",
        "summary": out_dir / f"{args.name}_summary.json",
        "manifest": out_dir / f"{args.name}_manifest.json",
    }
    cc.require(not any(p.exists() for p in final_paths.values()),
               "use a fresh --name; final evidence is never overwritten")

    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    code_sha = base_cmp.code_sha256_lf(CODE_FILES)
    log(f"git HEAD {git_sha}; dirty entries: {git_dirty}; priority={priority}; "
        f"rf_n_jobs={args.rf_n_jobs} lgbm_n_jobs={args.lgbm_n_jobs} "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")

    t_load = time.perf_counter()
    adopted, weather, _join_manifest, _scenario, source_paths = weather_cmp.load_inputs(weather_cmp.JOIN_RUN)
    _X_off, X, y, spec = weather_cmp.build_pair(adopted, weather)
    del adopted, weather, _X_off
    cc.require(spec.key == "P6_clean", "base phase drift")
    cc.require(list(X.columns[-len(WEATHER_MODEL_FEATURES):]) == list(WEATHER_MODEL_FEATURES),
               "weather_on feature suffix drift")
    log(f"loaded weather_on {X.shape}, positives={int(y.sum())} in {time.perf_counter() - t_load:.1f}s")

    selections = recorded_selections()
    refs = reference_runs()
    expected_fps: dict[int, str | None] = {s: None for s in SEEDS}
    if smoke:
        idx = (y.groupby(y, group_keys=False)
               .apply(lambda g: g.sample(frac=args.sample / len(y), random_state=0)).index)
        idx = np.sort(np.asarray(idx))
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
        log(f"SMOKE sample: {len(y)} rows (not performance evidence; fingerprint/reproduction checks skipped)")
    else:
        cc.require(len(y) == 180_332, "full population must be 180,332 labeled rows")
        ref_manifest = json.loads(REFS["compare_manifest"].read_text(encoding="utf-8"))
        expected_fps = {s: str(ref_manifest["fold_fingerprints"][str(s)]) for s in SEEDS}
        tun_manifest = json.loads(REFS["tuning_manifest"].read_text(encoding="utf-8"))
        cc.require(all(tun_manifest["fold_fingerprints"][str(s)] == expected_fps[s] for s in SEEDS),
                   "tuning and comparison fold fingerprints differ")

    build_kwargs = {"lgbm_base_params": tuning.lgbm_base_params(), "rf_n_jobs": args.rf_n_jobs,
                    "lgbm_n_jobs": args.lgbm_n_jobs, "rf_n_estimators": args.rf_n_estimators}
    protocol = {
        **base_cmp.protocol_description(replace(runner.CFG, seed=SEEDS[0]), spec),
        "hyperparameter_selection": ("reused per fold from the 20261008 runs (selected there by "
                                     "inner-holdout LogLoss inside the same outer-train); not re-searched"),
        "model_fit_scope": "inner-train only (as recorded; no refit on full outer-train)",
        "probability_calibration": {
            "arms": [list(a) for a in ccal.ARMS],
            "platt": "LogisticRegression(lbfgs, C=1.0) on logit(clip(p, 1e-6))",
            "isotonic": f"IsotonicRegression(out_of_bounds='clip'), outputs clipped to [{ccal.ISO_CLIP}, 1-{ccal.ISO_CLIP}]",
            "inner_holdout": ("calibrator fitted on the selected model's inner-holdout scores; the same rows "
                              "also selected the config and the re-selected threshold (double use)"),
            "inner_train_crossfit": (f"{ccal.CROSSFIT_K}-fold stratified cross-fitting within inner-train "
                                     f"(split seed = seed + fold + {ccal.CROSSFIT_SEED_OFFSET}); TE recomputed "
                                     "inside each sub-boundary; sub-models use the same selected config; "
                                     "calibrator never sees inner-holdout or outer-valid rows"),
            "threshold_reselection": "argmax inner-holdout Macro F1 over cfg.thresholds() on calibrated inner-holdout probs",
            "outer_valid_labels_used_for_calibration_or_threshold": False,
        },
        "ece": {"primary": f"{ccal.ECE_EQUAL_FREQ_BINS} equal-frequency bins (quantile edges of the "
                           "evaluated pooled outer-valid probabilities; ties share a bin)",
                "secondary": f"{ccal.ECE_EQUAL_WIDTH_BINS} equal-width bins (as ece_10 in 20260917)"},
        "metric_aggregation": "pooled outer-valid OOF per seed; per-fold thresholds",
    }
    identity = {
        "name": args.name, "smoke": smoke, "sample": args.sample, "seeds": list(seeds),
        "models": list(models), "rows": int(len(y)), "positives": int(y.sum()),
        "columns_sha256": stable_json_hash(list(X.columns)),
        "target_sha256": cc.target_sha256(y), "row_key_sha256": cc.row_key_sha256(y.index),
        "selection_files_sha256": {k: digest(REFS[k]) for k in ("compare_folds", "tuning_folds")},
        "arms": [list(a) for a in ccal.ARMS], "crossfit_k": ccal.CROSSFIT_K,
        "rf_n_estimators": args.rf_n_estimators, "lgbm_n_jobs": args.lgbm_n_jobs,
        "code_sha256_lf": {p: h for p, h in code_sha.items() if p in IDENTITY_CODE_FILES},
    }
    ckpt_dir = (out_dir if smoke else ROOT / "data" / "classifier_compare") / f"{args.name}_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "checkpoint.json"
    state = {"identity_sha256": stable_json_hash(identity), "runs": [], "bins": [], "folds": []}
    if ckpt.exists():
        loaded = json.loads(ckpt.read_text(encoding="utf-8"))
        cc.require(loaded.get("identity_sha256") == state["identity_sha256"],
                   "checkpoint belongs to a different experiment identity")
        state = loaded
        log(f"RESUME checkpoint: {len(state['runs'])} unit-arms, {len(state['folds'])} fold-arms")

    def save_state() -> None:
        atomic_json(ckpt, state)

    t_all = time.perf_counter()
    done = {(int(r["seed"]), r["model"]) for r in state["runs"]}
    for seed in seeds:
        for model_key in models:
            if (seed, model_key) in done:
                log(f"RESUME seed={seed} {model_key}")
                continue
            log(f"START seed={seed} {model_key}")
            t0 = time.perf_counter()
            runs_u, bins_u = evaluate_unit(
                model_key, X, y, seed=seed, spec=spec, expected_fp=expected_fps[seed],
                selections=selections, state=state, ckpt_dir=ckpt_dir, save_state=save_state,
                build_kwargs=build_kwargs, smoke=smoke)
            state["runs"].extend(runs_u)
            state["bins"].extend(json.loads(pd.concat(bins_u).to_json(orient="records")))
            save_state()
            for r in runs_u:
                log(f"DONE seed={seed} {model_key} {r['arm']}: LL={r['log_loss']:.6f} "
                    f"Brier={r['brier']:.6f} ECE15={r['ece_ef15']:.5f} AUC={r['roc_auc']:.6f} "
                    f"MacroF1={r['macro_f1_nested']:.6f} P1={r['precision_delayed']:.4f} "
                    f"R1={r['recall_delayed']:.4f}")
            log(f"unit time {time.perf_counter() - t0:.0f}s")

    runs = pd.DataFrame(state["runs"])
    runs["model"] = pd.Categorical(runs.model, categories=list(ccal.MODEL_KEYS), ordered=True)
    runs["arm"] = pd.Categorical(runs.arm, categories=list(ccal.ARM_NAMES), ordered=True)
    runs = runs.sort_values(["seed", "model", "arm"]).reset_index(drop=True)
    runs["model"], runs["arm"] = runs.model.astype(str), runs.arm.astype(str)
    folds_df = pd.DataFrame(state["folds"]).sort_values(["seed", "model", "fold", "arm"]).reset_index(drop=True)
    bins_df = pd.DataFrame(state["bins"])
    cc.require(runs.groupby("seed").fold_fingerprint.nunique().eq(1).all(), "outer fold mismatch")
    paired = ccal.paired_vs_none(runs)

    repro_runs = None if smoke else run_level_reproduction(runs, refs)
    repro_folds = None if smoke else fold_level_reproduction(folds_df)
    summary = {
        "schema_version": 1, "name": args.name, "smoke_only_not_evidence": smoke,
        "question": ("Do LR, RF and LightGBM differ in probability calibration on the weather_on "
                     "population, and does post-hoc Platt/Isotonic calibration fitted inside outer-train "
                     "change LogLoss/Brier/ECE and Macro F1 at the re-selected inner threshold?"),
        "population": {"rows": int(len(y)), "positive_rows": int(y.sum()),
                       "positive_rate": float(y.mean()), "input_columns": int(X.shape[1]),
                       "sample": args.sample},
        "seeds": list(seeds), "models": list(models),
        "fold_fingerprints": {str(int(r.seed)): r.fold_fingerprint
                              for r in runs.drop_duplicates("seed").itertuples()},
        "reference_fold_fingerprints_20261008": {str(k): v for k, v in expected_fps.items()},
        "fold_fingerprints_equal_reference": (None if smoke else bool(all(
            str(r.fold_fingerprint) == expected_fps[int(r.seed)] for r in runs.itertuples()))),
        "protocol": protocol,
        "levels": level_summary(runs),
        "paired_deltas_arm_minus_none": ccal.summarize_deltas(paired),
        "model_minus_lightgbm_by_arm": model_vs_lightgbm(runs, seeds),
        "threshold_and_clip_summary": selection_shift(folds_df),
        "reproduction_of_recorded_uncalibrated_runs": {"run_level": repro_runs, "fold_level": repro_folds},
        "earlier_lightgbm_calibration_study": EARLIER_LGBM_CALIBRATION,
        "runtime_sec": {"this_process": float(time.perf_counter() - t_all),
                        "sum_fold_elapsed": float(folds_df.drop_duplicates("fold_key").fold_elapsed_sec.sum())},
        "interpretation_guardrails": [
            "Same labeled rows, target, outer folds (fingerprint-checked) and seeds for all models and arms; deltas are paired.",
            "Configurations are reused from the 20261008 inner selections; calibrators and thresholds use outer-train data only.",
            "The inner_holdout arms reuse the rows that already selected the configuration and threshold; the crossfit arms do not.",
            "Crossfit calibrators are fitted on scores of sub-models trained on 80% of inner-train, applied to the model trained on all of inner-train (slight confidence mismatch).",
            "ECE depends on binning; both 15 equal-frequency and 10 equal-width bins are reported.",
            "Three seeds are split repetitions of the same rows, not independent datasets; SD is not a confidence interval.",
            "Static stratified CV on one sample period; not a future-deployment estimate.",
        ],
    }

    atomic_csv(final_paths["runs"], runs)
    atomic_csv(final_paths["folds"], folds_df)
    atomic_csv(final_paths["reliability_bins"], bins_df)
    atomic_csv(final_paths["paired"], paired)
    atomic_json(final_paths["summary"], summary)

    import sklearn
    manifest = {
        "schema_version": 1, "name": args.name, "successful": True, "smoke_only_not_evidence": smoke,
        "git_sha": git_sha, "git_status_porcelain_at_start": git_dirty,
        "python_version": platform.python_version(),
        "package_versions": {"pandas": pd.__version__, "numpy": np.__version__,
                             "scikit-learn": sklearn.__version__, "lightgbm": lgb.__version__},
        "cpu_count": os.cpu_count(), "process_priority": priority,
        "threads": {"rf_n_jobs": args.rf_n_jobs, "lgbm_n_jobs": args.lgbm_n_jobs,
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")},
        "uv_lock_sha256": digest(ROOT / "uv.lock"),
        "inputs": {k: {"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p),
                       "bytes": p.stat().st_size} for k, p in source_paths.items()},
        "reference_runs": {k: {"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)}
                           for k, p in REFS.items()},
        "join_name": weather_cmp.JOIN_RUN,
        "code_sha256_lf": code_sha,
        "experiment_identity": identity,
        "experiment_identity_sha256": stable_json_hash(identity),
        "fold_fingerprints": summary["fold_fingerprints"],
        "reference_fold_fingerprints_20261008": summary["reference_fold_fingerprints_20261008"],
        "fold_fingerprints_equal_reference": summary["fold_fingerprints_equal_reference"],
        "protocol": protocol,
        "outputs": {k: {"path": p.name, "sha256": digest(p), "bytes": p.stat().st_size}
                    for k, p in final_paths.items() if k != "manifest" and p.exists()},
        "checkpoint": (ckpt.relative_to(ROOT).as_posix() if ckpt.is_relative_to(ROOT) else str(ckpt)),
        "fits_models": True, "network_requests": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_json(final_paths["manifest"], manifest)
    log("FINISHED " + json.dumps({
        "runs": final_paths["runs"].name,
        "run_level_max_abs_metric_diff": None if repro_runs is None else repro_runs["max_abs_metric_diff"],
        "fold_level": repro_folds,
    }))


if __name__ == "__main__":
    main()
