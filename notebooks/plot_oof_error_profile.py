"""Two pictures of why the classifier errs, from the stored predictions only.

Left: where the predicted probabilities land for rows whose two original times are
missing, next to rows where both were observed. The selection threshold is drawn
in, so it is visible how much of each group can ever cross it.

Right: each group's actual delay rate against the average probability the model
gives that group. Points on the diagonal mean the model reproduces the group's
base rate. Neither panel shows a cause; both describe observed groups.
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
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1a1a19", "#5c5c58", "#e3e3df"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline_recovery_v2_error_profile_20260917")
    args = ap.parse_args()
    out = ROOT / "output"
    bands = pd.read_csv(out / f"{args.name}_probability_bands.csv")
    tracking = pd.read_csv(out / f"{args.name}_base_rate_tracking.csv")
    correlation = pd.read_csv(out / f"{args.name}_base_rate_correlation.csv")

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.9))
    fig.patch.set_facecolor("#fcfcfb")

    groups = [("양쪽 시각 관측", BLUE), ("양쪽 시각 결측", ORANGE)]
    order = list(dict.fromkeys(bands.band))
    x = np.arange(len(order))
    width = 0.38
    for i, (label, color) in enumerate(groups):
        part = bands[bands.group == label].set_index("band").reindex(order)
        left.bar(x + (i - 0.5) * width, part.share * 100, width - 0.04,
                 color=color, label=label, zorder=3, edgecolor="#fcfcfb", lw=1)
    left.set_xticks(x, [b.replace("[", "").replace(")", "").replace(", ", "–")
                        for b in order], fontsize=8.5, rotation=30, ha="right")
    left.set_ylabel("그룹 안에서의 비중 (%)", color=INK, fontsize=9.5)
    left.set_xlabel("모델이 매긴 지연 확률", color=INK, fontsize=9.5)
    left.set_title("예측 확률이 어디에 몰려 있는가", color=INK, fontsize=10.5,
                   loc="left", pad=24)
    left.text(0, 1.012, "P6_clean · seed 42 · fold별 선택 임계값은 0.22–0.23",
              transform=left.transAxes, color=MUTED, fontsize=8.5, va="bottom")
    left.axvline(2.5, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
    left.annotate("임계값 →\n오른쪽만 지연으로 분류", (2.62, left.get_ylim()[1] * 0.78),
                  color=MUTED, fontsize=8.5, ha="left", va="top")
    left.legend(frameon=False, fontsize=9, loc="upper right")

    limits = [0.05, 0.28]
    right.plot(limits, limits, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    right.annotate("점선 = 그룹의 평균 지연율을\n그대로 되돌려주는 선", (0.276, 0.088),
                   color=MUTED, fontsize=8.5, ha="right", va="top")
    for attribute, color, label in (("Origin_Airport", BLUE, "출발 공항"),
                                    ("Airline", ORANGE, "항공사")):
        part = tracking[tracking.attribute == attribute]
        r = correlation.loc[correlation.attribute == attribute, "pearson_r"].iloc[0]
        right.scatter(part.observed_rate, part.mean_probability, s=part.n / 260,
                      color=color, alpha=0.55, edgecolor="#fcfcfb", lw=0.8,
                      zorder=3, label=f"{label} ({len(part)}개, r={r:.3f})")
    right.set_xlim(*limits)
    right.set_ylim(*limits)
    right.set_xlabel("그룹의 실제 지연율", color=INK, fontsize=9.5)
    right.set_ylabel("그룹의 평균 예측 확률", color=INK, fontsize=9.5)
    right.set_title("모델은 그룹의 평균 위험도를 되풀이한다", color=INK, fontsize=10.5,
                    loc="left", pad=24)
    right.text(0, 1.012, "2,000행 이상 그룹만 · 점 크기는 행 수 · 상관은 인과가 아님",
               transform=right.transAxes, color=MUTED, fontsize=8.5, va="bottom")
    right.legend(frameon=False, fontsize=9, loc="upper left")

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
    path = out / f"{args.name}.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    print(f">> saved: {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
