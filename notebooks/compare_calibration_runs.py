"""Compare two calibration experiment runs cell by cell.

Written for the case where the same experiment is run again in a different
environment and the question is whether the two agree. Nothing is fitted and
nothing is overwritten; the script reads two result sets and reports the largest
absolute difference for every metric.

It compares on the natural key of the experiment — condition, seed, arm,
calibrator, group — and fails loudly when a key, or a measure column, is present in
one run and not the other, because a silently smaller comparison would look like
agreement. A measure absent from both runs is reported separately and does not
fail, since it is simply not part of that table.

Exit status is 0 when every difference is within --tolerance, 1 otherwise, so the
result can be read without parsing the output.

A difference of a few ULP is not the same finding as a real disagreement, so each
measure also carries a verdict: 동일 (bit-identical), 거의 같음 (nonzero but within
--near), 불일치 (above it). The verdict is reporting only. It never changes the
exit status, which stays governed by --tolerance, so a strict run still fails on a
last-bit difference and the reader can see what kind of difference it was.

Mean-based metrics over hundreds of thousands of rows carry a noise floor near
double-precision epsilon, because a different library version may accumulate the
same values in a different order. That is why --near defaults to 1e-12: far above
that floor, far below any difference that would change a reported figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["phase_key", "seed", "arm", "calibrator", "group"]
MEASURES = ["n", "positives", "macro_f1", "log_loss", "roc_auc", "brier", "ece_10",
            "calibration_bias", "precision", "recall", "fpr", "fnr", "tn", "fp",
            "fn", "tp"]
FOLD_KEYS = ["phase_key", "seed", "arm", "calibrator", "fold"]
FOLD_MEASURES = ["n_estimators", "threshold", "uncalibrated_threshold",
                 "n_inner_train", "n_inner_holdout"]
IDENTICAL, NEAR, DIFFERENT = "동일", "거의 같음", "불일치"


def read(name: str, suffix: str) -> pd.DataFrame:
    path = ROOT / "output" / f"{name}_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"결과 파일이 없습니다: {path}")
    return pd.read_csv(path)


def verdict_for(largest: float, near: float) -> str:
    """Classify a measure by its largest gap. Reporting only; see module docstring."""
    if largest == 0.0:
        return IDENTICAL
    return NEAR if largest <= near else DIFFERENT


def compare(left: pd.DataFrame, right: pd.DataFrame, keys, measures, label, near):
    """Align on keys, then report the largest absolute gap per measure."""
    for frame, side in ((left, "left"), (right, "right")):
        if frame.duplicated(keys).any():
            raise ValueError(f"{label}: {side} 에 중복 키가 있습니다")
    a = left.set_index(keys).sort_index()
    b = right.set_index(keys).sort_index()
    only_left = a.index.difference(b.index)
    only_right = b.index.difference(a.index)
    common = a.index.intersection(b.index)
    columns = [c for c in measures if c in a.columns and c in b.columns]
    # A measure absent from both runs is simply not part of this table. A measure
    # present on one side only is schema drift between the runs, and skipping it
    # would shrink the comparison silently — the same failure the key check guards
    # against — so the two cases are kept apart.
    absent = [c for c in measures if c not in a.columns and c not in b.columns]
    one_sided = [c for c in measures if c not in columns and c not in absent]
    rows = []
    for column in columns:
        x = pd.to_numeric(a.loc[common, column], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(b.loc[common, column], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(x) & np.isnan(y)
        gap = np.where(both_nan, 0.0, np.abs(x - y))
        if np.isnan(gap).any():
            rows.append({"table": label, "measure": column, "compared": int(len(common)),
                         "n_differing": -1, "max_abs_diff": np.nan, "identical": False,
                         "verdict": DIFFERENT, "note": "한쪽만 결측인 값이 있습니다"})
            continue
        largest = float(gap.max()) if gap.size else 0.0
        rows.append({"table": label, "measure": column, "compared": int(len(common)),
                     "n_differing": int((gap > 0).sum()),
                     "max_abs_diff": largest if gap.size else np.nan,
                     "identical": bool(gap.size and largest == 0.0),
                     "verdict": verdict_for(largest, near), "note": ""})
    return (pd.DataFrame(rows), len(only_left), len(only_right), len(common),
            one_sided, absent)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True, help="기준 실행 이름")
    ap.add_argument("--right", required=True, help="비교 실행 이름")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="허용 최대 절대 차이. 기본 0은 완전 일치를 요구한다")
    ap.add_argument("--near", type=float, default=1e-12,
                    help="이 값 이하의 0이 아닌 차이는 '거의 같음'으로 표기한다. "
                         "표기일 뿐이며 종료 코드를 바꾸지 않는다")
    ap.add_argument("--out", help="비교 결과를 저장할 실행 이름(생략하면 저장하지 않음)")
    args = ap.parse_args()
    if args.near < 0:
        print("[!] --near 는 음수일 수 없습니다")
        return 1

    reports, mismatched_keys = [], []
    one_sided_columns, absent_columns = [], []
    for suffix, keys, measures, label in (
            ("scores", KEYS, MEASURES, "scores"),
            ("folds", FOLD_KEYS, FOLD_MEASURES, "folds")):
        try:
            left, right = read(args.left, suffix), read(args.right, suffix)
        except FileNotFoundError as error:
            print(f"[!] {error}")
            return 1
        report, n_left, n_right, n_common, one_sided, absent = compare(
            left, right, keys, measures, label, args.near)
        reports.append(report)
        print(f"[{label}] 공통 키 {n_common}개 "
              f"| {args.left} 에만 {n_left}개 | {args.right} 에만 {n_right}개")
        if n_left or n_right:
            mismatched_keys.append(label)
        if one_sided:
            one_sided_columns.append(f"{label}: {', '.join(one_sided)}")
        if absent:
            absent_columns.append(f"{label}: {', '.join(absent)}")
        print(report.to_string(index=False))
        print()

    combined = pd.concat(reports, ignore_index=True)
    if args.out:
        path = ROOT / "output" / f"{args.out}_comparison.csv"
        combined.to_csv(path, index=False)
        print(f">> saved: {path.name}")

    worst = combined.max_abs_diff.max(skipna=True)
    failed = bool(mismatched_keys) or bool(one_sided_columns) \
        or bool(combined.note.ne("").any()) \
        or (not np.isnan(worst) and worst > args.tolerance)
    counts = combined.verdict.value_counts()
    print(f"최대 절대 차이 {worst:.3e} | 허용치 {args.tolerance:.3e} "
          f"| 판정 {'불일치' if failed else '일치'}")
    print(f"지표 표기: {IDENTICAL} {counts.get(IDENTICAL, 0)}개 "
          f"· {NEAR} {counts.get(NEAR, 0)}개 "
          f"· {DIFFERENT} {counts.get(DIFFERENT, 0)}개 "
          f"(거의 같음 기준 {args.near:.1e})")
    if not counts.get(DIFFERENT, 0) and counts.get(NEAR, 0):
        print(f"[i] 0이 아닌 차이가 모두 {args.near:.1e} 이하입니다. 종료 코드는 "
              f"--tolerance {args.tolerance:.1e} 기준이므로 이 경우에도 1입니다. "
              f"수치가 같다고 볼지는 읽는 사람이 판단합니다.")
    if mismatched_keys:
        print(f"[!] 두 실행의 키 집합이 다릅니다: {', '.join(mismatched_keys)}")
    if one_sided_columns:
        print(f"[!] 한쪽 실행에만 있는 열이 있어 비교하지 못했습니다: "
              f"{'; '.join(one_sided_columns)}")
    if absent_columns:
        print(f"[i] 양쪽 모두에 없어 비교 대상이 아닌 열: {'; '.join(absent_columns)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
