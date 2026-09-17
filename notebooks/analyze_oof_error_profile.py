"""Describe who the model gets wrong, using only predictions that already exist.

No model is fitted here. The stored row-level OOF files are joined back to the raw
input so the weak groups can be described by their original attributes. Two
questions drive it:

  1. What do the rows with both original times missing actually look like?
  2. What do the rows every seed gets wrong look like, next to the rows every
     seed gets right?

Everything below is a description of observed groups. A group difference is not a
cause: these groups also differ in size, delay rate and composition, so a gap in
recall is a fact about the group, not evidence that missingness produced the error.

Run from the repository root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import load_data
from src.oof import labeled_identity, metrics
from src.run_store import file_digest

PHASES = ["P6_fixed", "P6_clean"]
TIME_COLUMNS = ("raw_missing__Estimated_Departure_Time",
                "raw_missing__Estimated_Arrival_Time")
# Attributes described for each group. Categorical ones get a top-share table,
# numeric ones get quantiles.
CATEGORICAL = ["Month", "Airline", "Carrier_Code(IATA)", "Origin_Airport",
               "Destination_Airport", "Origin_State", "Destination_State"]
NUMERIC = ["Distance", "Day_of_Month"]
TOP_N = 8


def load_oof(name: str, seeds: list[int]) -> pd.DataFrame:
    """Read every stored OOF file for the run and check it against the summary."""
    out = ROOT / "output"
    frames, digests = [], []
    for seed in seeds:
        summary = pd.read_csv(out / f"{name}_seed{seed}.csv", keep_default_na=False)
        for record in summary.to_dict("records"):
            if record["phase_key"] not in PHASES:
                continue
            path = out / record["oof_path"]
            if not path.exists():
                raise FileNotFoundError(
                    f"Row-level OOF is kept locally and is missing here: {path}")
            actual = file_digest(path)
            if actual != record["oof_sha256"]:
                raise ValueError(f"OOF file hash does not match the summary: {path}")
            digests.append({"seed": seed, "phase_key": record["phase_key"],
                            "oof_path": record["oof_path"], "oof_sha256": actual})
            frame = pd.read_csv(path)
            if int(frame.seed.iloc[0]) != seed or frame.phase_key.iloc[0] != record["phase_key"]:
                raise ValueError(f"OOF identity does not match its summary row: {path}")
            frames.append(frame)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(digests)


def verify(oof: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the recorded per-seed scores from the row-level files."""
    checks = []
    for (phase, seed), part in oof.groupby(["phase_key", "seed"]):
        part = part.sort_values("source_row")
        if not np.array_equal(part.source_row.to_numpy(), identity.source_row.to_numpy()):
            raise ValueError(f"{phase} seed {seed}: row positions differ from the raw file")
        if not np.array_equal(part.ID.to_numpy(), identity.ID.to_numpy()):
            raise ValueError(f"{phase} seed {seed}: IDs differ from the raw file")
        if not np.array_equal(part.y_true.to_numpy(), identity.y_true.to_numpy()):
            raise ValueError(f"{phase} seed {seed}: targets differ from the raw file")
        for column in TIME_COLUMNS:
            if not np.array_equal(part[column].to_numpy(), identity[column].to_numpy()):
                raise ValueError(f"{phase} seed {seed}: {column} differs from the raw file")
        checks.append({"phase_key": phase, "seed": seed, **metrics(part)})
    return pd.DataFrame(checks)


def describe_group(raw: pd.DataFrame, rows: pd.Index, label: str,
                   reference: pd.Index) -> pd.DataFrame:
    """Share of each attribute value inside the group, next to the reference group."""
    out = []
    group, other = raw.loc[rows], raw.loc[reference]
    for column in CATEGORICAL:
        counts = group[column].value_counts(dropna=False)
        shares = counts / len(group)
        base = other[column].value_counts(dropna=False) / len(other)
        for value in shares.head(TOP_N).index:
            out.append({"group": label, "attribute": column,
                        "value": "(결측)" if pd.isna(value) else str(value),
                        "n": int(counts[value]), "share": float(shares[value]),
                        "reference_share": float(base.get(value, 0.0)),
                        "share_ratio": float(shares[value] / base[value])
                        if base.get(value, 0.0) else np.nan})
        out.append({"group": label, "attribute": column, "value": "(고유값 수)",
                    "n": int(group[column].nunique(dropna=True)), "share": np.nan,
                    "reference_share": np.nan, "share_ratio": np.nan})
    for column in NUMERIC:
        values = pd.to_numeric(group[column], errors="coerce")
        base = pd.to_numeric(other[column], errors="coerce")
        for name, q in (("중앙값", 0.5), ("하위 25%", 0.25), ("상위 25%", 0.75)):
            out.append({"group": label, "attribute": column, "value": name,
                        "n": int(values.notna().sum()), "share": float(values.quantile(q)),
                        "reference_share": float(base.quantile(q)), "share_ratio": np.nan})
        out.append({"group": label, "attribute": column, "value": "결측률",
                    "n": int(values.isna().sum()), "share": float(values.isna().mean()),
                    "reference_share": float(base.isna().mean()), "share_ratio": np.nan})
    return pd.DataFrame(out)


def missing_cooccurrence(oof_one_seed: pd.DataFrame, mask: np.ndarray,
                         label: str) -> pd.DataFrame:
    """How often each other column is also missing inside the group."""
    columns = [c for c in oof_one_seed if c.startswith("raw_missing__")
               and c not in TIME_COLUMNS]
    inside, outside = oof_one_seed.loc[mask], oof_one_seed.loc[~mask]
    return pd.DataFrame([
        {"group": label, "column": c.replace("raw_missing__", ""),
         "missing_rate": float(inside[c].mean()),
         "reference_missing_rate": float(outside[c].mean()),
         "n_missing": int(inside[c].sum())}
        for c in columns])


def error_stability(oof: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Count, per row, how many seeds got it wrong."""
    part = oof[oof.phase_key == phase]
    wrong = part.assign(wrong=(part.prediction != part.y_true).astype(int))
    table = wrong.pivot_table(index="source_row", columns="seed", values="wrong")
    result = pd.DataFrame({"source_row": table.index,
                           "wrong_seeds": table.sum(axis=1).to_numpy().astype(int)})
    probs = part.pivot_table(index="source_row", columns="seed", values="probability")
    result["mean_probability"] = probs.mean(axis=1).to_numpy()
    result["probability_spread"] = (probs.max(axis=1) - probs.min(axis=1)).to_numpy()
    truth = part.drop_duplicates("source_row").set_index("source_row").y_true
    result["y_true"] = truth.loc[result.source_row].to_numpy()
    return result


def base_rate_tracking(labeled: pd.DataFrame, probability, y, columns, min_rows=2000):
    """Does the average prediction for a group just repeat that group's delay rate?

    If it does, the model is mostly reproducing context-level risk (which carrier,
    which month) rather than telling flights apart inside a context.
    """
    frame = labeled.assign(_p=probability, _y=y)
    rows, correlations = [], []
    for column in columns:
        grouped = frame.groupby(column, dropna=False).agg(
            n=("_y", "size"), observed_rate=("_y", "mean"),
            mean_probability=("_p", "mean"))
        grouped = grouped[grouped.n >= min_rows].sort_values("observed_rate")
        if len(grouped) < 3:
            continue
        for value, r in grouped.iterrows():
            rows.append({"attribute": column,
                         "value": "(결측)" if pd.isna(value) else str(value),
                         "n": int(r.n), "observed_rate": float(r.observed_rate),
                         "mean_probability": float(r.mean_probability)})
        correlations.append({
            "attribute": column, "groups": int(len(grouped)),
            "pearson_r": float(np.corrcoef(grouped.observed_rate,
                                           grouped.mean_probability)[0, 1]),
            "observed_rate_range": float(grouped.observed_rate.max()
                                         - grouped.observed_rate.min()),
            "mean_probability_range": float(grouped.mean_probability.max()
                                            - grouped.mean_probability.min())})
    return pd.DataFrame(rows), pd.DataFrame(correlations)


def stability_summary(stability: pd.DataFrame) -> pd.DataFrame:
    """Where the repeated errors sit relative to the threshold, and how stable."""
    grouped = stability.groupby(["y_true", "wrong_seeds"]).agg(
        n=("source_row", "size"),
        median_probability=("mean_probability", "median"),
        median_seed_spread=("probability_spread", "median"))
    return grouped.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="baseline_recovery_v2_oof_20260916_v2")
    ap.add_argument("--name", default="baseline_recovery_v2_error_profile_20260917")
    ap.add_argument("--seeds", default="42,1,7")
    ap.add_argument("--phase", default="P6_clean", help="기술 분석 대상 조건")
    args = ap.parse_args()
    if not args.name.startswith("baseline_recovery_v2"):
        print("[!] name must start with baseline_recovery_v2")
        return 1
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    data_path = ROOT / "data/train.csv"
    if not data_path.exists():
        print(f"[!] real data not found: {data_path}")
        return 1
    raw = load_data(data_path)
    identity = labeled_identity(raw)
    oof, digests = load_oof(args.source, seeds)
    checks = verify(oof, identity)
    out = ROOT / "output"

    # Labeled rows only, indexed the same way the OOF is.
    labeled = raw.iloc[identity.source_row].reset_index(drop=True)
    one_seed = oof[(oof.phase_key == args.phase) & (oof.seed == seeds[0])] \
        .sort_values("source_row").reset_index(drop=True)
    dep, arr = (one_seed[c].to_numpy() for c in TIME_COLUMNS)
    both_missing = (dep == 1) & (arr == 1)
    both_observed = (dep == 0) & (arr == 0)

    profiles = [
        describe_group(labeled, pd.Index(np.flatnonzero(both_missing)),
                       "양쪽 시각 결측", pd.Index(np.flatnonzero(both_observed))),
    ]
    cooccurrence = missing_cooccurrence(one_seed, both_missing, "양쪽 시각 결측")

    stability = error_stability(oof, args.phase)
    position = pd.Series(np.arange(len(identity)), index=identity.source_row)
    stability["row"] = position.loc[stability.source_row].to_numpy()
    delayed = stability[stability.y_true == 1]
    normal = stability[stability.y_true == 0]
    always_fn = pd.Index(delayed.loc[delayed.wrong_seeds == len(seeds), "row"])
    never_fn = pd.Index(delayed.loc[delayed.wrong_seeds == 0, "row"])
    always_fp = pd.Index(normal.loc[normal.wrong_seeds == len(seeds), "row"])
    never_fp = pd.Index(normal.loc[normal.wrong_seeds == 0, "row"])
    profiles.append(describe_group(labeled, always_fn, "3시드 공통 FN", never_fn))
    profiles.append(describe_group(labeled, always_fp, "3시드 공통 FP", never_fp))
    cooccurrence = pd.concat([
        cooccurrence,
        missing_cooccurrence(one_seed, np.isin(np.arange(len(one_seed)), always_fn),
                             "3시드 공통 FN"),
        missing_cooccurrence(one_seed, np.isin(np.arange(len(one_seed)), always_fp),
                             "3시드 공통 FP"),
    ], ignore_index=True)

    # Probability distribution by group: does the model simply never go high?
    thresholds = one_seed.groupby("fold").threshold.first().to_dict()
    bands = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.01]
    band_rows = []
    for label, mask in (("양쪽 시각 결측", both_missing), ("양쪽 시각 관측", both_observed)):
        part = one_seed.loc[mask]
        cut = pd.cut(part.probability, bands, right=False)
        counts = cut.value_counts().sort_index()
        for interval, count in counts.items():
            inside = part.loc[cut == interval]
            band_rows.append({"group": label, "band": str(interval), "n": int(count),
                              "share": float(count / len(part)),
                              "observed_rate": float(inside.y_true.mean())
                              if count else np.nan})
    bands_frame = pd.DataFrame(band_rows)

    tracking, correlations = base_rate_tracking(
        labeled, one_seed.probability.to_numpy(), one_seed.y_true.to_numpy(),
        ["Month", "Airline", "Origin_Airport", "Carrier_Code(IATA)"])
    tracking.to_csv(out / f"{args.name}_base_rate_tracking.csv", index=False)
    correlations.to_csv(out / f"{args.name}_base_rate_correlation.csv", index=False)
    stability_summary(stability).to_csv(
        out / f"{args.name}_error_stability_summary.csv", index=False)

    profile = pd.concat(profiles, ignore_index=True)
    profile.to_csv(out / f"{args.name}_group_profile.csv", index=False)
    cooccurrence.to_csv(out / f"{args.name}_missing_cooccurrence.csv", index=False)
    bands_frame.to_csv(out / f"{args.name}_probability_bands.csv", index=False)
    stability.drop(columns="row").to_csv(
        out / f"{args.name}_rows_error_stability.csv.gz", index=False,
        compression="gzip")
    checks.to_csv(out / f"{args.name}_score_reproduction.csv", index=False)

    counts = {
        "labeled_rows": int(len(identity)),
        "both_missing": int(both_missing.sum()),
        "both_observed": int(both_observed.sum()),
        "always_fn": int(len(always_fn)), "never_fn": int(len(never_fn)),
        "always_fp": int(len(always_fp)), "never_fp": int(len(never_fp)),
        "always_fn_share_of_delayed": float(len(always_fn) / (stability.y_true == 1).sum()),
        "max_probability_both_missing": float(one_seed.loc[both_missing].probability.max()),
        "max_probability_both_observed": float(one_seed.loc[both_observed].probability.max()),
        "fold_thresholds": {str(k): float(v) for k, v in thresholds.items()},
        "base_rate_correlation": correlations.set_index("attribute").pearson_r.to_dict(),
    }
    lines = [f"# 오분류 기술 분석 — {args.name}", "",
             f"대상 조건 {args.phase}, 시드 {seeds}. 모델을 적합하지 않고 저장된 예측만 사용합니다.",
             "그룹 간 차이는 인과가 아닙니다. 그룹은 크기·지연율·구성이 서로 다릅니다.", "",
             "## 행 수", "",
             "| 항목 | 행 수 |", "|---|---:|"]
    for key, value in counts.items():
        if isinstance(value, int):
            lines.append(f"| {key} | {value:,} |")
    lines += ["", "## 그룹 평균 위험도와 평균 예측 확률의 상관", "",
              "| 속성 | 그룹 수 | 상관계수 | 실제 지연율 폭 | 평균 예측 폭 |",
              "|---|---:|---:|---:|---:|"]
    for _, r in correlations.iterrows():
        lines.append(f"| {r.attribute} | {int(r.groups)} | {r.pearson_r:.4f} "
                     f"| {r.observed_rate_range:.4f} | {r.mean_probability_range:.4f} |")
    lines += ["", "## 확률 구간별 비중", "",
              "| 구간 | 양쪽 시각 관측 | 양쪽 시각 결측 |", "|---|---:|---:|"]
    pivot = bands_frame.pivot(index="band", columns="group", values="share")
    for band, r in pivot.iterrows():
        lines.append(f"| {band} | {r.get('양쪽 시각 관측', float('nan')) * 100:.2f}% "
                     f"| {r.get('양쪽 시각 결측', float('nan')) * 100:.2f}% |")
    lines += ["", "## 시드별 오류 안정성", "",
              "| 정답 | 틀린 시드 수 | 행 수 | 3시드 평균 확률(중앙값) | 시드 간 확률 변동폭(중앙값) |",
              "|---|---:|---:|---:|---:|"]
    for _, r in stability_summary(stability).iterrows():
        lines.append(f"| {'지연' if r.y_true else '정상'} | {int(r.wrong_seeds)} "
                     f"| {int(r.n):,} | {r.median_probability:.4f} "
                     f"| {r.median_seed_spread:.4f} |")
    (out / f"{args.name}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "name": args.name, "source_run": args.source, "phase": args.phase,
        "seeds": seeds, "fits_any_model": False,
        "note": ("Descriptive only. Group differences are not causal; groups differ "
                 "in size, delay rate and composition."),
        "source_sha256": file_digest(data_path),
        "oof_files": digests.to_dict("records"),
        "code_sha256": {f: file_digest(ROOT / f) for f in
                        ("src/oof.py", "src/features.py",
                         "notebooks/analyze_oof_error_profile.py")},
        "counts": counts,
    }
    (out / f"{args.name}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    print(f">> saved: {args.name}_group_profile.csv (+ cooccurrence/bands/stability/manifest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
