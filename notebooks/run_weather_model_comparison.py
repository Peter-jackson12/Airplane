"""Run the submission weather-off vs weather-on paired comparison.

One model family, one existing P6_clean preprocessing pipeline, one predeclared
10-minute weather scenario, and three existing split seeds (42/1/7). The script
never fetches weather and never searches weather feature subsets. A local
checkpoint permits resume after a long model fit; tracked evidence is published
only after all six condition/seed cells succeed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd

import rerun_all_phases as runner
from src.cv import evaluate_oof, make_folds, run_fold_nested_grid
from src.weather_full import read_joined_csv
from src.weather_model import (
    BASE_PHASE_KEY,
    HEADLINE_LATENCY_MINUTES,
    SEEDS,
    WEATHER_MODEL_FEATURES,
    align_raw_to_join,
    attach_weather_features,
    digest,
    extract_weather_features,
    fold_fingerprint,
    joined_usecols,
    labeled_mask,
    require,
    stable_json_hash,
)

ROOT = Path(__file__).resolve().parents[1]
JOIN_RUN = "baseline_recovery_v2_weather_full_join_20260922"
METRICS = ("macro_f1_nested", "log_loss", "roc_auc", "f1_at_050", "recall")
CONDITIONS = ("weather_off", "weather_on")
CODE_FILES = (
    "src/weather_model.py",
    "src/weather_full.py",
    "src/features.py",
    "src/cv.py",
    "rerun_all_phases.py",
    "notebooks/run_weather_model_comparison.py",
)


def validate_name(name: str) -> None:
    import re
    require(bool(re.fullmatch(r"baseline_recovery_v2_[A-Za-z0-9_]+", name)),
            "invalid run name")


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def phase_spec():
    matches = [s for s in runner.PHASES if s.key == BASE_PHASE_KEY]
    require(len(matches) == 1, f"missing/duplicate phase {BASE_PHASE_KEY}")
    spec = matches[0]
    require(spec.te and spec.pseudo is None and spec.safe_preprocessing,
            "P6_clean contract drift")
    return spec


def evaluate_condition(X: pd.DataFrame, y: pd.Series, *, seed: int, spec) -> dict:
    """Fit one condition with exactly the repository's current nested protocol."""
    cfg = replace(runner.CFG, seed=int(seed))
    folds = make_folds(y, cfg)
    fingerprint = fold_fingerprint(folds, len(y))
    oof = np.full(len(y), np.nan)
    thresholds, selected, grids = [], [], []
    feature_names: list[str] = []
    started = time.perf_counter()

    for fold, (tr, va) in enumerate(folds):
        selection: dict = {}
        fit = run_fold_nested_grid(
            runner.model_factory,
            X.iloc[tr].reset_index(drop=True),
            y.iloc[tr].reset_index(drop=True),
            X.iloc[va].reset_index(drop=True),
            cfg,
            fold=fold,
            n_estimators_grid=runner.N_ESTIMATORS_GRID,
            te_cols=spec.cat_cols,
            te_m=runner.TE_SMOOTHING_M,
            te_inner_splits=spec.inner_splits,
            te_drop_original=spec.te_drop_original,
            selection_metadata=selection,
        )
        require("threshold" in selection, "nested threshold was not selected")
        oof[va] = fit.valid_probs
        thresholds.append(float(selection["threshold"]))
        selected.append(int(fit.selected_n_estimators))
        grids.append({str(k): float(v) for k, v in fit.grid_scores.items()})
        if not feature_names:
            feature_names = list(getattr(fit.model, "feature_name_", []))
        print(json.dumps({
            "seed": seed, "fold": fold + 1, "selected_n_estimators": selected[-1],
            "threshold": thresholds[-1],
        }), flush=True)

    require(np.isfinite(oof).all(), "OOF probabilities incomplete/nonfinite")
    score = evaluate_oof(
        y, oof, folds, cfg, label=f"{BASE_PHASE_KEY}",
        feature_names=feature_names, per_fold_thresholds=thresholds,
    )
    return {
        "seed": int(seed),
        "n_rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "positive_rate": float(y.mean()),
        "n_features": int(score["n_features"]),
        "log_loss": float(score["log_loss"]),
        "roc_auc": float(score["roc_auc"]),
        "f1_at_050": float(score["f1_at_050"]),
        "macro_f1_nested": float(score["macro_f1"]),
        "naive_macro_f1": float(score["naive_macro_f1"]),
        "threshold_optimism": float(score["threshold_optimism"]),
        "deployment_threshold": float(score["deployment_threshold"]),
        "per_fold_thresholds": json.dumps(thresholds, separators=(",", ":")),
        "selected_n_estimators": json.dumps(selected, separators=(",", ":")),
        "grid_scores": json.dumps(grids, separators=(",", ":")),
        "tn": int(score["confusion_matrix"]["tn"]),
        "fp": int(score["confusion_matrix"]["fp"]),
        "fn": int(score["confusion_matrix"]["fn"]),
        "tp": int(score["confusion_matrix"]["tp"]),
        "recall": float(score["recall"]),
        "fold_fingerprint": fingerprint,
        "elapsed_sec": float(time.perf_counter() - started),
        "protocol": json.dumps(score["protocol"], sort_keys=True,
                               separators=(",", ":"), ensure_ascii=False),
    }


def load_inputs(join_name: str):
    out = ROOT / "output"
    manifest_path = out / f"{join_name}_full_weather_join_manifest.json"
    require(manifest_path.is_file(), f"join manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == 1 and manifest.get("successful") is True,
            "join manifest is not successful schema v1")
    require(manifest.get("name") == join_name, "join manifest name mismatch")
    require(manifest.get("measured_publication_latency") is False,
            "headline contract expects assumed, not measured, latency")
    require(HEADLINE_LATENCY_MINUTES in manifest.get("latency_scenarios_minutes", []),
            "10-minute join missing")
    output = manifest["outputs"][str(HEADLINE_LATENCY_MINUTES)]
    joined_path = ROOT / output["path"]
    require(joined_path.is_file(), f"joined weather missing: {joined_path}")
    require(joined_path.stat().st_size == int(output["bytes"]), "joined weather byte mismatch")
    require(digest(joined_path) == output["sha256"], "joined weather SHA mismatch")

    summary_path = ROOT / manifest["summary"]["path"]
    require(summary_path.is_file() and digest(summary_path) == manifest["summary"]["sha256"],
            "join summary SHA mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scenario = [s for s in summary["scenarios"]
                if int(s["latency_minutes"]) == HEADLINE_LATENCY_MINUTES]
    require(len(scenario) == 1, "10-minute summary scenario mismatch")

    raw_path = ROOT / manifest["sources"]["raw_train"]["path"]
    require(raw_path.is_file(), "raw train missing")
    require(digest(raw_path) == manifest["sources"]["raw_train"]["sha256"],
            "raw train SHA mismatch")
    raw = pd.read_csv(raw_path, low_memory=False)
    joined = read_joined_csv(joined_path, usecols=joined_usecols())
    require(len(joined) == int(output["rows"]) == manifest["denominators"]["total_adopted_rows"],
            "joined denominator mismatch")
    weather = extract_weather_features(joined)
    adopted = align_raw_to_join(raw, weather.ID)
    return adopted, weather, manifest, scenario[0], {
        "join_manifest": manifest_path,
        "join_summary": summary_path,
        "joined_weather": joined_path,
        "raw_train": raw_path,
    }


def build_pair(adopted: pd.DataFrame, weather: pd.DataFrame):
    spec = phase_spec()
    X_off, y, X_unlab = runner.build_features(adopted, spec)
    X_on, X_on_unlab = attach_weather_features(X_off, X_unlab, adopted, weather)
    require(len(X_off) == len(X_on) == int(labeled_mask(adopted).sum()),
            "paired labeled denominator mismatch")
    require(X_off.index.equals(X_on.index) and y.index.equals(X_off.index),
            "paired row index mismatch")
    require(list(X_on.columns[:len(X_off.columns)]) == list(X_off.columns),
            "weather-on changed baseline columns/order")
    require(list(X_on.columns[len(X_off.columns):]) == list(WEATHER_MODEL_FEATURES),
            "weather-on feature suffix drift")
    require(len(X_unlab) == len(X_on_unlab), "paired unlabeled denominator mismatch")
    return X_off, X_on, y, spec


def identity_payload(*, git_sha: str, manifest: dict, y: pd.Series,
                     X_off: pd.DataFrame, X_on: pd.DataFrame) -> dict:
    return {
        "git_sha": git_sha,
        "join_output_sha256": manifest["outputs"][str(HEADLINE_LATENCY_MINUTES)]["sha256"],
        "raw_train_sha256": manifest["sources"]["raw_train"]["sha256"],
        "base_phase": BASE_PHASE_KEY,
        "latency_minutes": HEADLINE_LATENCY_MINUTES,
        "latency_is_measured": False,
        "seeds": list(SEEDS),
        "weather_features": list(WEATHER_MODEL_FEATURES),
        "labeled_rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "base_columns_sha256": stable_json_hash(list(X_off.columns)),
        "weather_columns_sha256": stable_json_hash(list(X_on.columns)),
        "cfg": runner.CFG.describe(),
        "n_estimators_grid": list(runner.N_ESTIMATORS_GRID),
        "te_smoothing_m": runner.TE_SMOOTHING_M,
        "lgbm_params": runner.LGBM_PARAMS,
    }


def load_checkpoint(path: Path, identity: dict) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("identity_sha256") == stable_json_hash(identity),
            "checkpoint belongs to a different experiment")
    rows = payload.get("rows", [])
    require(isinstance(rows, list), "malformed checkpoint rows")
    seen = set()
    for row in rows:
        key = (int(row["seed"]), row["condition"])
        require(key not in seen and key[0] in SEEDS and key[1] in CONDITIONS,
                "malformed/duplicate checkpoint cell")
        seen.add(key)
    return rows


def save_checkpoint(path: Path, identity: dict, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "identity_sha256": stable_json_hash(identity),
        "identity": identity,
        "rows": rows,
    }, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def summarize(runs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    require(len(runs) == len(SEEDS) * len(CONDITIONS), "incomplete final run matrix")
    require(not runs.duplicated(["seed", "condition"]).any(), "duplicate final run cell")
    paired_rows = []
    for seed in SEEDS:
        pair = runs[runs.seed.eq(seed)].set_index("condition")
        require(set(pair.index) == set(CONDITIONS), "missing paired condition")
        require(pair.loc["weather_off", "fold_fingerprint"] ==
                pair.loc["weather_on", "fold_fingerprint"],
                "paired conditions used different outer folds")
        require(pair.loc["weather_off", "n_rows"] == pair.loc["weather_on", "n_rows"],
                "paired conditions used different evaluation rows")
        row = {"seed": seed,
               "fold_fingerprint": pair.loc["weather_off", "fold_fingerprint"]}
        for metric in METRICS:
            row[f"{metric}_delta_on_minus_off"] = (
                float(pair.loc["weather_on", metric]) -
                float(pair.loc["weather_off", metric])
            )
        paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)

    conditions = {}
    for condition in CONDITIONS:
        sub = runs[runs.condition.eq(condition)]
        conditions[condition] = {
            metric: {
                "mean": float(sub[metric].mean()),
                "std": float(sub[metric].std(ddof=1)),
                "values_by_seed": {str(int(r.seed)): float(getattr(r, metric))
                                   for r in sub.itertuples()},
            }
            for metric in METRICS
        }
    deltas = {}
    for metric in METRICS:
        col = f"{metric}_delta_on_minus_off"
        deltas[metric] = {
            "mean": float(paired[col].mean()),
            "std": float(paired[col].std(ddof=1)),
            "values_by_seed": {str(int(r.seed)): float(getattr(r, col))
                               for r in paired.itertuples()},
        }
    return paired, {"conditions": conditions, "paired_deltas_on_minus_off": deltas}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--join-name", default=JOIN_RUN)
    parser.add_argument("--validate-inputs-only", action="store_true")
    args = parser.parse_args()
    validate_name(args.name)
    require(args.join_name == JOIN_RUN,
            "submission comparison is frozen to the validated 20260922 full join")

    out = ROOT / "output"
    final_paths = {
        "runs": out / f"{args.name}_weather_model_runs.csv",
        "paired": out / f"{args.name}_weather_model_paired_deltas.csv",
        "summary": out / f"{args.name}_weather_model_summary.json",
        "manifest": out / f"{args.name}_weather_model_manifest.json",
    }
    require(not any(p.exists() for p in final_paths.values()),
            "use a fresh --name; final evidence is never overwritten")

    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      text=True).strip()
    require(not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True).strip(), "tracked worktree must be clean")

    adopted, weather, join_manifest, scenario, source_paths = load_inputs(args.join_name)
    X_off, X_on, y, spec = build_pair(adopted, weather)
    require(y.nunique() == 2, "paired label population requires both classes")
    identity = identity_payload(git_sha=git_sha, manifest=join_manifest, y=y,
                                X_off=X_off, X_on=X_on)

    mask = labeled_mask(adopted).reset_index(drop=True)
    labeled_weather = weather.loc[mask].reset_index(drop=True)
    labeled_coverage = {
        "labeled_rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "origin_matched": int(labeled_weather.weather_origin_matched.sum()),
        "destination_matched": int(labeled_weather.weather_destination_matched.sum()),
        "both_matched": int((labeled_weather.weather_origin_matched.eq(1) &
                             labeled_weather.weather_destination_matched.eq(1)).sum()),
    }
    if args.validate_inputs_only:
        print(json.dumps({
            "input_validation": "passed",
            "git_sha": git_sha,
            "join_name": args.join_name,
            "latency_minutes": HEADLINE_LATENCY_MINUTES,
            "latency_is_measured": False,
            "adopted_rows": int(len(adopted)),
            "evaluation_rows": int(len(y)),
            "positive_rows": int(y.sum()),
            "weather_feature_count": len(WEATHER_MODEL_FEATURES),
            "weather_features": list(WEATHER_MODEL_FEATURES),
            "base_feature_count": int(X_off.shape[1]),
            "weather_on_feature_count": int(X_on.shape[1]),
            "labeled_weather_coverage": labeled_coverage,
            "experiment_identity_sha256": stable_json_hash(identity),
        }, ensure_ascii=False, indent=2), flush=True)
        return

    local = ROOT / "data/weather_probe" / f"{args.name}_weather_model"
    local.mkdir(parents=True, exist_ok=True)
    checkpoint = local / "checkpoint.json"
    rows = load_checkpoint(checkpoint, identity)
    done = {(int(r["seed"]), r["condition"]) for r in rows}

    matrices = {"weather_off": X_off, "weather_on": X_on}
    for seed in SEEDS:
        for condition in CONDITIONS:
            key = (seed, condition)
            if key in done:
                print(f"RESUME seed={seed} condition={condition}", flush=True)
                continue
            print(f"START seed={seed} condition={condition}", flush=True)
            row = evaluate_condition(matrices[condition], y, seed=seed, spec=spec)
            row["condition"] = condition
            rows.append(row)
            done.add(key)
            save_checkpoint(checkpoint, identity, rows)
            print(f"DONE seed={seed} condition={condition} "
                  f"F1={row['macro_f1_nested']:.6f} LL={row['log_loss']:.6f}", flush=True)

    runs = pd.DataFrame(rows).sort_values(["seed", "condition"]).reset_index(drop=True)
    paired, aggregate = summarize(runs)
    # A paired submission means all non-weather base columns and every outer fold
    # are shared; nested tree/threshold selection is independently performed inside
    # each condition's outer-train data using the same procedure.
    require(runs.groupby("seed").fold_fingerprint.nunique().eq(1).all(),
            "outer fold mismatch")

    selected_coverage = {
        "origin_matched": int(scenario["origin_matched"]),
        "destination_matched": int(scenario["destination_matched"]),
        "both_matched": int(scenario["both_matched"]),
        "one_matched": int(scenario["one_matched"]),
        "none_matched": int(scenario["none_matched"]),
    }
    summary = {
        "schema_version": 1,
        "name": args.name,
        "base_phase": BASE_PHASE_KEY,
        "headline_latency_minutes": HEADLINE_LATENCY_MINUTES,
        "headline_latency_is_measured": False,
        "seeds": list(SEEDS),
        "weather_model_features": list(WEATHER_MODEL_FEATURES),
        "weather_feature_count": len(WEATHER_MODEL_FEATURES),
        "feature_selection_policy": (
            "Frozen before model fitting from join diagnostics and submission scope; "
            "no score-driven weather feature subset search."
        ),
        "excluded_weather_fields": {
            "gust": "high missingness",
            "snowdepth": "almost entirely missing",
            "wxcodes": "high missingness and categorical expansion",
            "relh": "derived/redundant with temperature and dew point for minimal contract",
            "skyc1": "categorical expansion deferred",
            "metar": "audit text, not an automatic model feature",
        },
        "full_join_10min_coverage": selected_coverage,
        "labeled_population_coverage": labeled_coverage,
        "evaluation_rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "positive_rate": float(y.mean()),
        **aggregate,
        "interpretation_guardrails": [
            "Weather-on and weather-off use identical labeled rows, seeds and outer folds.",
            "Each condition uses the same nested tree/threshold selection procedure; outer-valid labels do not select either.",
            "No row is removed because weather is unmatched or a selected weather field is missing.",
            "The 10-minute publication latency is an assumption, not a measured historical dissemination delay.",
            "The three seed repetitions are not independent datasets or confidence intervals.",
        ],
    }

    atomic_csv(final_paths["runs"], runs)
    atomic_csv(final_paths["paired"], paired)
    atomic_json(final_paths["summary"], summary)

    sources = {
        key: {"path": path.relative_to(ROOT).as_posix(),
              "sha256": digest(path), "bytes": path.stat().st_size}
        for key, path in source_paths.items()
    }
    code = {
        path: hashlib.sha256((ROOT / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in CODE_FILES
    }
    manifest = {
        "schema_version": 1,
        "name": args.name,
        "successful": True,
        "git_sha": git_sha,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "join_name": args.join_name,
        "join_sources": sources,
        "code_sha256_lf": code,
        "uv_lock_sha256": digest(ROOT / "uv.lock"),
        "experiment_identity": identity,
        "experiment_identity_sha256": stable_json_hash(identity),
        "outputs": {
            key: {"path": path.relative_to(ROOT).as_posix(),
                  "sha256": digest(path), "bytes": path.stat().st_size}
            for key, path in final_paths.items() if key != "manifest"
        },
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "fits_models": True,
        "network_requests": 0,
        "target_source": "data/train.csv only; joined weather artifact contains no target/outcome columns",
        "paired_outer_folds": True,
        "same_evaluation_rows": True,
        "weather_rows_filtered_for_model": 0,
        "weather_feature_selection_frozen": True,
        "latency_selected_by_model_score": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_json(final_paths["manifest"], manifest)
    print(json.dumps({
        "successful": True,
        "evaluation_rows": len(y),
        "positive_rows": int(y.sum()),
        "summary": summary["paired_deltas_on_minus_off"],
        "manifest": final_paths["manifest"].relative_to(ROOT).as_posix(),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
