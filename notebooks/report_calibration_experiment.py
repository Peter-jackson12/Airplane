"""Aggregate the nested calibration experiment into paired, per-seed comparisons.

Differences are taken within the same phase, arm, seed and group, against the
uncalibrated arm. Pairing first and averaging afterwards keeps the three seeds as
three repeats rather than pretending the same rows are independent samples.
The spread reported is a sample standard deviation over three seeds, not a
confidence interval and not a significance test.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MEASURES = ["brier", "ece_10", "log_loss", "macro_f1", "roc_auc", "recall",
            "precision", "calibration_bias"]
LOWER_IS_BETTER = {"brier", "ece_10", "log_loss", "calibration_bias"}
KEYS = ["phase_key", "arm", "seed", "group"]


def paired_deltas(scores: pd.DataFrame) -> pd.DataFrame:
    base = scores[scores.calibrator == "none"].set_index(KEYS)
    out = []
    for kind in scores.calibrator.unique():
        if kind == "none":
            continue
        arm = scores[scores.calibrator == kind].set_index(KEYS)
        common = arm.index.intersection(base.index)
        delta = arm.loc[common, MEASURES] - base.loc[common, MEASURES]
        delta = delta.reset_index()
        delta["calibrator"] = kind
        delta["n"] = arm.loc[common, "n"].to_numpy()
        out.append(delta)
    return pd.concat(out, ignore_index=True)


def across_seeds(frame: pd.DataFrame, value_columns) -> pd.DataFrame:
    keys = [c for c in ("phase_key", "arm", "calibrator", "group") if c in frame]
    agg = frame.groupby(keys)[list(value_columns)].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def markdown_table(deltas: pd.DataFrame, group: str) -> str:
    part = deltas[deltas.group == group].copy()
    lines = ["| 조건 | 방식 | 보정기 | ΔBrier | ΔECE(%p) | ΔLogLoss | ΔMacro F1 | Δ지연 재현율 |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    arm_label = {"shared": "공유형", "split": "분리형"}
    for _, r in part.sort_values(["phase_key", "arm", "calibrator"]).iterrows():
        lines.append(
            f"| {r.phase_key} | {arm_label.get(r.arm, r.arm)} | {r.calibrator} "
            f"| {r.brier_mean:+.6f} ± {r.brier_std:.6f} "
            f"| {r.ece_10_mean * 100:+.4f} ± {r.ece_10_std * 100:.4f} "
            f"| {r.log_loss_mean:+.6f} ± {r.log_loss_std:.6f} "
            f"| {r.macro_f1_mean:+.6f} ± {r.macro_f1_std:.6f} "
            f"| {r.recall_mean * 100:+.2f}%p ± {r.recall_std * 100:.2f} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline_recovery_v2_calibration_20260917")
    args = ap.parse_args()
    out = ROOT / "output"
    scores = pd.read_csv(out / f"{args.name}_scores.csv")
    scores = scores[scores.n > 0].copy()

    deltas = paired_deltas(scores)
    deltas.to_csv(out / f"{args.name}_paired_deltas.csv", index=False)
    delta_summary = across_seeds(deltas, MEASURES)
    delta_summary.to_csv(out / f"{args.name}_paired_delta_summary.csv", index=False)
    level_summary = across_seeds(scores, MEASURES + ["n"])
    level_summary.to_csv(out / f"{args.name}_level_summary.csv", index=False)

    report = [f"# 확률 보정 실험 결과 — {args.name}", "",
              "비교는 같은 조건·방식·시드·그룹 안에서 보정 없음 대비 짝지은 차이입니다.",
              "±는 3시드 표본 표준편차이며 신뢰구간이나 유의성의 근거가 아닙니다.",
              "음수가 좋은 지표: Brier, ECE, LogLoss.", "",
              "## 전체 라벨 행", "", markdown_table(delta_summary, "overall"), "",
              "## 양쪽 시각 결측 그룹", "", markdown_table(delta_summary, "both_missing"), "",
              "## 양쪽 시각 관측 그룹", "", markdown_table(delta_summary, "both_observed"), ""]
    (out / f"{args.name}_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
