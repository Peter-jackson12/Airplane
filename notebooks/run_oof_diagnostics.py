"""Reproducible OOF export and diagnostics for P6_fixed/P6_clean, three seeds.

Run from the repository root. --sample is a smoke run, never performance evidence.
--analyze-only validates and reads existing artifacts without retraining.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import load_data
from src.oof import diagnose, labeled_identity, metrics
from src.run_store import file_digest, read_oof

PHASES = ["P6_fixed", "P6_clean"]
MEASURES = ["macro_f1", "log_loss", "roc_auc", "brier", "ece_10", "calibration_bias",
            "precision", "recall", "fpr", "fnr", "error_rate"]


def analyze(name, seeds, sample):
    out = ROOT / "output"
    raw = load_data(ROOT / "data/train.csv")
    if sample is not None:
        raw = raw.head(sample).reset_index(drop=True)
    identity = labeled_identity(raw)
    observed_pairs = raw[["Carrier_Code(IATA)", "Airline"]].dropna()
    code_counts = observed_pairs.groupby("Carrier_Code(IATA)").Airline.nunique()
    expected_airline_missing = {}
    for phase in PHASES:
        usable_codes = code_counts.index if phase == "P6_fixed" else code_counts[code_counts == 1].index
        remains_missing = raw.Airline.isna() & ~raw["Carrier_Code(IATA)"].isin(usable_codes)
        expected_airline_missing[phase] = remains_missing.iloc[identity.source_row].astype(int).to_numpy()
    source_sha = file_digest(ROOT / "data/train.csv")
    scores, bin_frames, examples, references, folds = [], [], [], [], []
    stability = {}
    previous = pd.read_csv(out / "preprocessing_full_runs.csv")
    historical = []
    seed_contract = None
    for seed in seeds:
        reference_folds = None
        path = out / f"{name}_seed{seed}.csv"
        summary = pd.read_csv(path, keep_default_na=False)
        if len(summary) != len(PHASES) or set(summary.phase_key) != set(PHASES):
            raise ValueError("Expected exactly the two requested Phases")
        for record in summary.to_dict("records"):
            meta = json.loads(record["run_metadata"])
            if meta["data_sha256"] != source_sha or meta["sample"] != sample or meta["cv"]["seed"] != seed:
                raise ValueError("Analysis source/sample/seed mismatch")
            contract = json.loads(record["run_metadata"])
            contract["cv"].pop("seed")
            contract["params"].pop("random_state")
            if seed_contract is not None and contract != seed_contract:
                raise ValueError("Seed runs have different experimental contracts")
            seed_contract = contract
            rows = read_oof(path, record)
            if reference_folds is not None and not np.array_equal(rows.fold, reference_folds):
                raise ValueError("Phase folds are not paired")
            reference_folds = rows.fold.to_numpy().copy()
            for col in identity:
                if not np.array_equal(rows[col], identity[col]):
                    raise ValueError(f"Source/OOF alignment mismatch: {col}")
            if not np.array_equal(rows["feature_missing__Airline"], expected_airline_missing[record["phase_key"]]):
                raise ValueError("Post-imputation Airline missingness disagrees with independent raw-pair check")
            m = metrics(rows)
            for actual, stored in (("macro_f1", "macro_f1_nested"), ("log_loss", "log_loss"), ("roc_auc", "roc_auc")):
                if abs(m[actual] - float(record[stored])) > 5.01e-7:
                    raise ValueError(f"Saved OOF does not reproduce {stored}")
            for key in ("tn", "fp", "fn", "tp"):
                if m[key] != int(record[key]):
                    raise ValueError("Saved OOF does not reproduce confusion matrix")
            # Independent arithmetic check, separate from sklearn metric helpers.
            y, p = rows.y_true.to_numpy(), rows.probability.to_numpy()
            eps = np.finfo(float).eps
            pc = np.clip(p, eps, 1 - eps)
            ll = -np.mean(y * np.log(pc) + (1 - y) * np.log1p(-pc))
            if not np.isclose(ll, m["log_loss"], rtol=0, atol=1e-12):
                raise ValueError("Independent LogLoss calculation disagrees")
            group_scores, bins = diagnose(rows)
            scores.append(group_scores)
            bin_frames.append(bins)
            for fold, part in rows.groupby("fold"):
                folds.append(dict(phase_key=record["phase_key"], seed=seed, fold=int(fold), **metrics(part)))
            phase = record["phase_key"]
            for error, mask in (("FP", rows.y_true.eq(0) & rows.prediction.eq(1)),
                                ("FN", rows.y_true.eq(1) & rows.prediction.eq(0))):
                # Fixed rule: largest probability error, then ID; examples are not representative samples.
                cases = rows.loc[mask].copy()
                cases["probability_error"] = np.abs(cases.probability - cases.y_true)
                cases = cases.sort_values(["probability_error", "ID"], ascending=[False, True]).head(10)
                cases["error_type"] = error
                context = raw.iloc[cases.source_row.to_numpy()].reset_index(drop=True).drop(columns=["ID", "Delay"])
                examples.append(pd.concat([cases.reset_index(drop=True), context.add_prefix("raw__")], axis=1))
            error_flags = rows.prediction.ne(rows.y_true).astype(int).to_numpy()
            if phase not in stability:
                stability[phase] = rows[["source_row", "ID", "y_true"]].copy()
                stability[phase]["error_seed_count"] = 0
                stability[phase]["probability_min"] = 1.
                stability[phase]["probability_max"] = 0.
            stable = stability[phase]
            stable["error_seed_count"] += error_flags
            stable["probability_min"] = np.minimum(stable.probability_min, p)
            stable["probability_max"] = np.maximum(stable.probability_max, p)
            artifact = path.parent / record["oof_path"]
            references.append(dict(summary=path.name, summary_sha256=file_digest(path),
                                   phase_key=phase, seed=seed, run_id=record["run_id"],
                                   oof_path=record["oof_path"], oof_sha256=file_digest(artifact),
                                   n_rows=len(rows), code_sha256=meta["code_sha256"]))
            if sample is None:
                prior = previous[(previous.phase_key == phase) & (previous.seed == seed)]
                if len(prior) == 1:
                    historical.append(dict(phase_key=phase, seed=seed, **{
                        f"{a}_minus_historical": m[a] - float(prior.iloc[0][b])
                        for a, b in (("macro_f1", "macro_f1_nested"), ("log_loss", "log_loss"), ("roc_auc", "roc_auc"))}))
            print(f"VERIFIED {phase} seed={seed} rows={len(rows)}", flush=True)
    groups = pd.concat(scores, ignore_index=True)
    bins = pd.concat(bin_frames, ignore_index=True)
    groups.to_csv(out / f"{name}_groups.csv", index=False)
    bins.to_csv(out / f"{name}_calibration_bins.csv", index=False)
    pd.DataFrame(folds).to_csv(out / f"{name}_folds.csv", index=False)
    pd.concat(examples, ignore_index=True).to_csv(out / f"{name}_error_examples.csv", index=False)
    aggregated = groups.groupby(["phase_key", "dimension", "group"], dropna=False)[["n", "positive_rate", *MEASURES]].agg(["mean", "std"])
    aggregated.columns = ["_".join(c) for c in aggregated.columns]
    aggregated.to_csv(out / f"{name}_summary.csv")
    keys = ["seed", "dimension", "group"]
    left = groups[groups.phase_key.eq("P6_fixed")].set_index(keys)
    right = groups[groups.phase_key.eq("P6_clean")].set_index(keys)
    paired = right[MEASURES] - left[MEASURES]
    # Postprocessing missingness can change membership by Phase, so it is not paired.
    paired = paired[~paired.index.get_level_values("dimension").str.startswith("feature_missing__")]
    paired.to_csv(out / f"{name}_paired_deltas.csv")
    stable_summary = []
    for phase, frame in stability.items():
        frame["phase_key"], frame["n_seeds"] = phase, len(seeds)
        # Every row remains recoverable; do not treat repeated seeds as new observations.
        frame.to_csv(out / f"{name}_{phase}_error_stability.csv.gz", index=False, compression="gzip")
        for (label, count), part in frame.groupby(["y_true", "error_seed_count"]):
            stable_summary.append(dict(phase_key=phase, y_true=int(label), error_seed_count=int(count), n=len(part)))
    pd.DataFrame(stable_summary).to_csv(out / f"{name}_error_stability_summary.csv", index=False)
    if historical:
        pd.DataFrame(historical).to_csv(out / f"{name}_historical_deltas.csv", index=False)
    plot_diagnostics(groups, bins, out / f"{name}_diagnostics.png", seeds, sample)
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), source="data/train.csv",
                    source_sha256=source_sha, source_rows=len(raw), labeled_rows=len(identity),
                    sample=sample, seeds=seeds, phases=PHASES, runs=references,
                    analysis_code_sha256={n: file_digest(ROOT / n) for n in
                                          ("notebooks/run_oof_diagnostics.py", "src/oof.py", "src/run_store.py")},
                    definitions={"unit": "one labeled source row per phase and seed",
                                 "missing": "raw pandas isna before imputation; ID and Delay excluded from count",
                                 "post_missing": "model-input Airline=MISSING/NaN or hour<0/NaN",
                                 "ece": "10 fixed equal-width bins over [0,1], weighted absolute calibration gap",
                                 "calibration": "diagnostic only; no calibrator fitted",
                                 "example_selection": "top 10 per phase/seed/FP-or-FN by absolute probability error, then ID",
                                 "uncertainty": "seed SD is descriptive, not a confidence interval",
                                 "empty": "n=0; rates undefined; AUC undefined for single-class groups",
                                 "causality": "missingness groups are observational; differences are not causal effects"})
    manifest["artifacts"] = {p.name: file_digest(p) for p in out.glob(f"{name}_*")
                             if p.is_file() and p.suffix in (".csv", ".png", ".gz")}
    (out / f"{name}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(groups[groups.dimension.eq("overall")][["phase_key", "seed", "n", *MEASURES]].to_string(index=False), flush=True)


def plot_diagnostics(groups, bins, path, seeds, sample):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"P6_fixed": "#225EA8", "P6_clean": "#CA6B21"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    fig.suptitle("OOF probability and missingness diagnostics" + (" — SMOKE ONLY" if sample else ""), fontsize=17)
    for phase, style in zip(PHASES, ("o", "s")):
        overall = bins[(bins.dimension == "overall") & (bins.phase_key == phase)]
        for i, seed in enumerate(seeds):
            b = overall[(overall.seed == seed) & (overall.n > 0)]
            axes[0, 0].plot(b.mean_probability, b.observed_rate, marker=style, color=colors[phase],
                            alpha=.6, label=phase if i == 0 else None)
        hist = overall.groupby("bin").n.mean().reindex(range(10), fill_value=0)
        axes[0, 1].step((np.arange(10) + .5) / 10, hist, where="mid", color=colors[phase], label=phase)
    axes[0, 0].plot([0, 1], [0, 1], "--", color="#555555", label="Perfect calibration")
    axes[0, 0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed delay rate",
                    title="Calibration by seed | 10 fixed bins")
    axes[0, 0].legend(fontsize=9)
    axes[0, 1].set(xlim=(0, 1), ylim=(0, None), xlabel="Probability bin midpoint", ylabel="Rows per seed (mean)",
                    title="Probability distribution | same rows across seeds")
    axes[0, 1].legend()
    order = ["both_observed", "departure_missing", "arrival_missing", "both_missing"]
    labels = ["Both\nobserved", "Departure\nmissing", "Arrival\nmissing", "Both\nmissing"]
    for ax, metric, title in ((axes[1, 0], "macro_f1", "Macro F1 by original time missingness"),
                              (axes[1, 1], "brier", "Brier score by original time missingness")):
        for offset, phase in zip((-.12, .12), PHASES):
            part = groups[(groups.dimension == "raw_time_pattern") & (groups.phase_key == phase)]
            stat = part.groupby("group")[metric].agg(["mean", "std"]).reindex(order)
            ax.errorbar(np.arange(4) + offset, stat["mean"], yerr=stat["std"].fillna(0),
                         fmt="o" if phase == "P6_fixed" else "s", capsize=4, color=colors[phase], label=phase)
        ax.set_xticks(range(4), labels)
        ax.set(title=title, ylabel=metric + " (mean ± seed SD)")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="baseline_recovery_v2_oof_20260916_v2")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--seeds", default="42,1,7")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if not args.name.startswith("baseline_recovery_v2_") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        ap.error("name must be a safe baseline_recovery_v2_ stem")
    if args.sample is not None and args.sample <= 0:
        ap.error("sample must be positive")
    seeds = [int(s) for s in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds):
        ap.error("seeds must be nonempty and unique")
    if not (ROOT / "data/train.csv").exists():
        raise FileNotFoundError("Actual data/train.csv is required")
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    if not args.analyze_only:
        for seed in seeds:
            prefix = f"{args.name}_seed{seed}"
            log = out / f"{prefix}.log"
            cmd = [sys.executable, "-u", str(ROOT / "rerun_all_phases.py"), "--seed", str(seed),
                   "--phases", ",".join(PHASES), "--save-oof", "--output-prefix", prefix]
            if args.sample is not None:
                cmd += ["--sample", str(args.sample)]
            print(f"START seed={seed} sample={args.sample} log={log.name}", flush=True)
            with log.open("a", encoding="utf-8") as stream:
                subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"}, check=True)
            print(f"DONE seed={seed}", flush=True)
    analyze(args.name, seeds, args.sample)


if __name__ == "__main__":
    main()
