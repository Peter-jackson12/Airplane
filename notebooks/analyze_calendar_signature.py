"""Ask what calendar the Month/Day fields actually describe, using only the raw file.

The weather work is blocked on an unknown flight year. No model is fitted here and
no external data is used: the only input is the daily row count for each of the 365
(Month, Day_of_Month) pairs in `data/train.csv`.

Daily counts carry three layers. A monthly level, which is a property of how the
file was sampled rather than of real traffic. A seven-day cycle. And, once both are
removed, date-specific anomalies. US air travel leaves unmistakable marks on
particular dates, and some of those dates are fixed every year while others move
with the weekday — so the movable ones identify the calendar the rows came from.

What this can and cannot do: fixed-date holidays show whether the dates behave like
real US calendar dates at all; movable holidays pin the weekday alignment, which
narrows the year to the set of years sharing that alignment but never to one year.
Counts are evidence about the file's composition, not about traffic, and a holiday
signature is a description of the data, not proof of a cause.

Run from the repository root.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import load_data
from src.run_store import file_digest

MONTH_LENGTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]
# Fixed-date US holidays: present in every year, so they test whether the Month/Day
# fields behave like real calendar dates. (month, day, label)
FIXED = [(7, 4, "독립기념일"), (12, 24, "크리스마스 이브"), (12, 25, "크리스마스"),
         (12, 31, "신정 전야"), (1, 1, "신정"), (10, 31, "핼러윈"),
         (11, 11, "재향군인의 날"), (2, 14, "밸런타인데이")]
Z_STRONG = 2.0


def daily_counts(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day, ordered by day of year on a non-leap calendar."""
    counts = raw.groupby(["Month", "Day_of_Month"]).size().rename("n").reset_index()
    offsets = np.cumsum([0] + MONTH_LENGTHS[:-1])
    counts["doy"] = [offsets[m - 1] + d for m, d in
                     zip(counts.Month, counts.Day_of_Month)]
    return counts.sort_values("doy").reset_index(drop=True)


def decompose(counts: pd.DataFrame) -> pd.DataFrame:
    """Strip the monthly sampling level, then the seven-day cycle."""
    out = counts.copy()
    out["residual"] = out.n - out.groupby("Month").n.transform("mean")
    # 365 = 7*52 + 1, so the day-of-year classes mod 7 are a fixed partition; which
    # class is which weekday is exactly what the movable holidays decide.
    out["cycle_class"] = (out.doy - 1) % 7
    out["anomaly"] = out.residual - out.groupby("cycle_class").residual.transform("mean")
    out["z"] = out.anomaly / out.anomaly.std()
    return out


def variance_explained(values: np.ndarray, groups: np.ndarray) -> float:
    grand = values.mean()
    between = sum(((values[groups == k].mean() - grand) ** 2) * (groups == k).sum()
                  for k in np.unique(groups))
    return float(between / ((values - grand) ** 2).sum())


def permutation_p(values: np.ndarray, groups: np.ndarray, draws: int, seed: int):
    """How often does a random reassignment reach the observed cycle strength?"""
    observed = variance_explained(values, groups)
    rng = np.random.default_rng(seed)
    null = np.array([variance_explained(rng.permutation(values), groups)
                     for _ in range(draws)])
    return observed, float((null >= observed).mean()), float(np.quantile(null, 0.999))


def nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    first = dt.date(year, month, 1)
    return first + dt.timedelta((weekday - first.weekday()) % 7 + 7 * (n - 1))


def last_weekday(year: int, month: int, weekday: int) -> dt.date:
    following = dt.date(year + (month == 12), month % 12 + 1, 1)
    end = following - dt.timedelta(1)
    return end - dt.timedelta((end.weekday() - weekday) % 7)


def movable_holidays(year: int) -> dict[str, dt.date]:
    """US holidays whose date depends on the weekday, so they identify the calendar."""
    return {"추수감사절": nth_weekday(year, 11, 3, 4),
            "블랙프라이데이": nth_weekday(year, 11, 3, 4) + dt.timedelta(1),
            "노동절": nth_weekday(year, 9, 0, 1),
            "현충일": last_weekday(year, 5, 0),
            "마틴루서킹의날": nth_weekday(year, 1, 0, 3),
            "대통령의날": nth_weekday(year, 2, 0, 3)}


def common_year_alignments(low: int = 2005, high: int = 2026) -> dict[int, list[int]]:
    """Non-leap years grouped by the weekday of 1 January."""
    groups: dict[int, list[int]] = {k: [] for k in range(7)}
    for year in range(low, high + 1):
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        if not leap:
            groups[dt.date(year, 1, 1).weekday()].append(year)
    return groups


def score_years(frame: pd.DataFrame, years) -> pd.DataFrame:
    z = {(m, d): value for m, d, value in zip(frame.Month, frame.Day_of_Month, frame.z)}
    rows = []
    for year in years:
        holidays = movable_holidays(year)
        row = {"year": year,
               "jan1_weekday": WEEKDAYS[dt.date(year, 1, 1).weekday()]}
        for name, date in holidays.items():
            row[f"{name}_날짜"] = f"{date.month}/{date.day}"
            row[f"{name}_z"] = round(float(z[(date.month, date.day)]), 2)
        marks = [row[f"{name}_z"] for name in holidays]
        row["저조_개수"] = int(sum(v <= -Z_STRONG for v in marks))
        row["z_합"] = round(float(sum(marks)), 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["저조_개수", "z_합"],
                                          ascending=[False, True])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline_recovery_v2_calendar_signature_20260917")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.name.startswith("baseline_recovery_v2"):
        print("[!] name must start with baseline_recovery_v2")
        return 1

    source = ROOT / "data/train.csv"
    if not source.exists():
        print(f"[!] real data not found: {source}")
        return 1
    raw = load_data(source)
    counts = daily_counts(raw)
    frame = decompose(counts)
    out = ROOT / "output"
    out.mkdir(exist_ok=True)

    leap_day = int(counts.loc[(counts.Month == 2) & (counts.Day_of_Month == 29),
                              "n"].sum())
    month_r2 = variance_explained(counts.n.to_numpy(float),
                                  counts.Month.to_numpy())
    cycle_r2, p_value, null_999 = permutation_p(
        frame.residual.to_numpy(float), frame.cycle_class.to_numpy(),
        args.draws, args.seed)

    fixed_rows = [{"holiday": label, "month": m, "day": d,
                   "n": int(frame.loc[(frame.Month == m) & (frame.Day_of_Month == d),
                                      "n"].iloc[0]),
                   "z": round(float(frame.loc[(frame.Month == m)
                                              & (frame.Day_of_Month == d),
                                              "z"].iloc[0]), 2)}
                  for m, d, label in FIXED]
    fixed = pd.DataFrame(fixed_rows).sort_values("z")

    alignments = common_year_alignments()
    years = [y for group in alignments.values() for y in group]
    scored = score_years(frame, sorted(years))

    strong = frame[frame.z <= -Z_STRONG][["Month", "Day_of_Month", "n", "z"]]
    frame.to_csv(out / f"{args.name}_daily.csv", index=False)
    fixed.to_csv(out / f"{args.name}_fixed_holidays.csv", index=False)
    scored.to_csv(out / f"{args.name}_year_scores.csv", index=False)
    strong.to_csv(out / f"{args.name}_strong_low_days.csv", index=False)

    # Thanksgiving is the deepest and most reliable mark in US air travel, and it
    # moves with the weekday. Every distinct Thursday–Friday pair found here is one
    # calendar the file draws on.
    thursdays = sorted({(movable_holidays(y)["추수감사절"].month,
                         movable_holidays(y)["추수감사절"].day) for y in years})
    signatures = []
    for month, day in thursdays:
        after = dt.date(2019, month, day) + dt.timedelta(1)
        z_day = float(frame.loc[(frame.Month == month)
                                & (frame.Day_of_Month == day), "z"].iloc[0])
        z_after = float(frame.loc[(frame.Month == after.month)
                                  & (frame.Day_of_Month == after.day), "z"].iloc[0])
        signatures.append({"추수감사절_후보": f"{month}/{day}", "당일_z": round(z_day, 2),
                           "다음날_z": round(z_after, 2),
                           "쌍_성립": bool(z_day <= -Z_STRONG and z_after <= -Z_STRONG),
                           "해당_평년": ", ".join(
                               str(y) for y in years
                               if (movable_holidays(y)["추수감사절"].month,
                                   movable_holidays(y)["추수감사절"].day) == (month, day))})
    signature_frame = pd.DataFrame(signatures)
    signature_frame.to_csv(out / f"{args.name}_thanksgiving.csv", index=False)
    matched = signature_frame[signature_frame["쌍_성립"]]

    summary = {
        "name": args.name,
        "fits_any_model": False,
        "uses_external_data": False,
        "source_sha256": file_digest(source),
        "code_sha256": {f: file_digest(ROOT / f) for f in
                        ("src/features.py",
                         "notebooks/analyze_calendar_signature.py")},
        "distinct_days": int(len(counts)),
        "feb29_rows": leap_day,
        "rows": int(counts.n.sum()),
        "month_variance_explained": round(month_r2, 4),
        "weekly_cycle_variance_explained_after_month": round(cycle_r2, 4),
        "weekly_cycle_permutation_p": p_value,
        "weekly_cycle_null_999": round(null_999, 4),
        "permutation_draws": args.draws,
        "thanksgiving_pairs_found": int(len(matched)),
        "thanksgiving_dates": matched["추수감사절_후보"].tolist(),
        "implied_year_sets": matched["해당_평년"].tolist(),
        "note": ("Counts describe how the file was sampled, not real traffic. "
                 "Holiday signatures identify the calendar, never a single year."),
    }
    (out / f"{args.name}_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== 고정 날짜 공휴일 ===")
    print(fixed.to_string(index=False))
    print("\n=== 추수감사절 후보별 목·금 쌍 ===")
    print(signature_frame.to_string(index=False))
    print(f"\n>> saved: {args.name}_daily.csv (+ fixed_holidays/year_scores/"
          f"strong_low_days/thanksgiving/manifest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
