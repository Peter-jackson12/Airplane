"""Compare two calibration experiment runs cell by cell.

Written for the case where the same experiment is run again in a different
environment and the question is whether the two agree. Nothing is fitted and
nothing is overwritten; the script reads two result sets and reports the largest
absolute difference for every metric.

It compares on the natural key of the experiment — condition, seed, arm,
calibrator, group — and fails loudly when a key is present in one run and not the
other, because a silently smaller intersection would look like agreement.

Exit status is 0 when every difference is within --tolerance, 1 otherwise, so the
result can be read without parsing the output.
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


def read(name: str, suffix: str) -> pd.DataFrame:
    path = ROOT / "output" / f"{name}_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"결과 파일이 없습니다: {path}")
    return pd.read_csv(path)


def compare(left: pd.DataFrame, right: pd.DataFrame, keys, measures, label):
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
    rows = []
    for column in columns:
        x = pd.to_numeric(a.loc[common, column], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(b.loc[common, column], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(x) & np.isnan(y)
        gap = np.where(both_nan, 0.0, np.abs(x - y))
        if np.isnan(gap).any():
            rows.append({"table": label, "measure": column, "compared": int(len(common)),
                         "max_abs_diff": np.nan, "identical": False,
                         "note": "한쪽만 결측인 값이 있습니다"})
            continue
        rows.append({"table": label, "measure": column, "compared": int(len(common)),
                     "max_abs_diff": float(gap.max()) if gap.size else np.nan,
                     "identical": bool(gap.size and gap.max() == 0.0), "note": ""})
    return pd.DataFrame(rows), len(only_left), len(only_right), len(common)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True, help="기준 실행 이름")
    ap.add_argument("--right", required=True, help="비교 실행 이름")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="허용 최대 절대 차이. 기본 0은 완전 일치를 요구한다")
    ap.add_argument("--out", help="비교 결과를 저장할 실행 이름(생략하면 저장하지 않음)")
    args = ap.parse_args()

    reports, mismatched_keys = [], []
    for suffix, keys, measures, label in (
            ("scores", KEYS, MEASURES, "scores"),
            ("folds", FOLD_KEYS, FOLD_MEASURES, "folds")):
        try:
            left, right = read(args.left, suffix), read(args.right, suffix)
        except FileNotFoundError as error:
            print(f"[!] {error}")
            return 1
        report, n_left, n_right, n_common = compare(left, right, keys, measures, label)
        reports.append(report)
        print(f"[{label}] 공통 키 {n_common}개 "
              f"| {args.left} 에만 {n_left}개 | {args.right} 에만 {n_right}개")
        if n_left or n_right:
            mismatched_keys.append(label)
        print(report.to_string(index=False))
        print()

    combined = pd.concat(reports, ignore_index=True)
    if args.out:
        path = ROOT / "output" / f"{args.out}_comparison.csv"
        combined.to_csv(path, index=False)
        print(f">> saved: {path.name}")

    worst = combined.max_abs_diff.max(skipna=True)
    failed = bool(mismatched_keys) or bool(combined.note.ne("").any()) \
        or (not np.isnan(worst) and worst > args.tolerance)
    print(f"최대 절대 차이 {worst:.3e} | 허용치 {args.tolerance:.3e} "
          f"| 판정 {'불일치' if failed else '일치'}")
    if mismatched_keys:
        print(f"[!] 두 실행의 키 집합이 다릅니다: {', '.join(mismatched_keys)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
