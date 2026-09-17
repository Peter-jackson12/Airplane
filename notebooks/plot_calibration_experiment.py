"""Calibration gap and ECE spread for the nested calibration experiment.

Left panel: per-bin calibration gap (actual delay rate minus mean predicted
probability), sample-weighted across the three seeds. The raw reliability curves
sit almost on top of each other, so the gap is plotted directly — it is the
quantity ECE summarises. Right panel: ECE per seed with the seed mean, so the
spread stays visible instead of hiding inside an average. That spread is a
three-seed sample range, not a confidence interval and not a significance test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

_KOREAN = next((n for n in ("NanumGothic", "Malgun Gothic", "Noto Sans CJK KR",
                            "Noto Sans CJK JP")
                if n in {f.name for f in font_manager.fontManager.ttflist}), None)
if _KOREAN:
    matplotlib.rcParams["font.family"] = _KOREAN
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
SERIES = {"none": "#2a78d6", "platt": "#eb6834", "isotonic": "#1baf7a"}
LABEL = {"none": "보정 없음", "platt": "Platt", "isotonic": "Isotonic"}
ARM_LABEL = {"shared": "공유형", "split": "분리형"}
INK, MUTED, GRID = "#1a1a19", "#5c5c58", "#e3e3df"
MIN_ROWS = 500


def weighted_bins(bins: pd.DataFrame) -> pd.DataFrame:
    """Sample-weighted mean across seeds; empty bins stay out."""
    part = bins[bins.n > 0].copy()
    part["wp"] = part.mean_probability * part.n
    part["wo"] = part.observed_rate * part.n
    grouped = part.groupby("bin")[["n", "wp", "wo"]].sum()
    grouped["predicted"] = grouped.wp / grouped.n
    grouped["observed"] = grouped.wo / grouped.n
    grouped["gap"] = (grouped.observed - grouped.predicted) * 100
    return grouped.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline_recovery_v2_calibration_20260917")
    ap.add_argument("--phase", default="P6_clean")
    ap.add_argument("--arm", default="shared")
    args = ap.parse_args()
    out = ROOT / "output"
    bins = pd.read_csv(out / f"{args.name}_calibration_bins.csv")
    scores = pd.read_csv(out / f"{args.name}_scores.csv")
    scores = scores[scores.group == "overall"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    fig.patch.set_facecolor("#fcfcfb")

    selected = bins[(bins.phase_key == args.phase) & (bins.arm == args.arm)]
    left.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    left.text(0.995, 0.02, "확률과 실제가 일치", transform=left.get_yaxis_transform(),
              color=MUTED, fontsize=8.5, ha="right", va="bottom")
    for kind, color in SERIES.items():
        curve = weighted_bins(selected[selected.calibrator == kind])
        curve = curve[curve.n >= MIN_ROWS]
        left.plot(curve.predicted, curve.gap, color=color, lw=2, marker="o",
                  ms=6, mec="#fcfcfb", mew=1.6, zorder=3, label=LABEL[kind])
    left.set_xlabel("예측 확률 (구간 평균)", color=INK, fontsize=9.5)
    left.set_ylabel("실제 지연율 − 예측 확률 (%p)", color=INK, fontsize=9.5)
    left.set_title(f"{args.phase} · {ARM_LABEL[args.arm]} — 구간별 보정 오차",
                   color=INK, fontsize=10.5, loc="left", pad=24)
    left.text(0, 1.012, f"3시드 표본수 가중 · {MIN_ROWS}행 이상 구간만 · 0에 가까울수록 좋음",
              transform=left.transAxes, color=MUTED, fontsize=8.5, va="bottom")
    left.legend(frameon=False, fontsize=9, loc="lower left")

    order = [(p, a) for p in sorted(scores.phase_key.unique())
             for a in sorted(scores.arm.unique())]
    offset = {"none": 0.24, "platt": 0.0, "isotonic": -0.24}
    for row, (phase, arm) in enumerate(order):
        for kind, color in SERIES.items():
            part = scores[(scores.phase_key == phase) & (scores.arm == arm)
                          & (scores.calibrator == kind)]
            y = row + offset[kind]
            values = part.ece_10.to_numpy() * 100
            right.plot(values, np.full(values.size, y), "o", color=color, ms=5,
                       alpha=0.45, mec="none", zorder=2)
            right.plot([values.mean()], [y], "|", color=color, ms=17, mew=2.6,
                       zorder=3)
            if row == len(order) - 1:
                right.annotate(LABEL[kind], (values.mean(), y), color=color,
                               fontsize=9, ha="center", va="bottom",
                               xytext=(0, 10), textcoords="offset points")
    right.set_yticks(range(len(order)),
                     [f"{p}\n{ARM_LABEL[a]}" for p, a in order], fontsize=9)
    right.set_ylim(-0.6, len(order) - 0.15)
    right.set_xlim(0, None)
    right.set_xlabel("ECE (%p, 고정 10구간)", color=INK, fontsize=9.5)
    right.set_title("시드별 ECE와 시드 평균(세로 막대)", color=INK, fontsize=10.5,
                    loc="left", pad=24)
    right.text(0, 1.012, "점 3개는 seed 42·1·7 · 낮을수록 확률이 실제에 가까움",
               transform=right.transAxes, color=MUTED, fontsize=8.5, va="bottom")

    for ax in (left, right):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)

    fig.tight_layout()
    path = out / f"{args.name}_reliability.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    print(f">> saved: {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
