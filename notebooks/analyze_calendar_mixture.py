"""How many calendars are mixed into the file, and could a third be hiding?

`analyze_calendar_signature.py` established that two Thanksgiving Thursday–Friday
pairs are present, so the file draws on at least two calendars. It deliberately
stopped there. This script asks the next question: is "at least two" also "exactly
two", and if a further calendar were present, how small would it have to be to
leave no mark?

Thanksgiving is the instrument, for three reasons. It moves with the weekday, so
each calendar puts it on a different date; it is the deepest recurring dip in US
air travel; and the seven candidate Thursdays are spread across eight days, so the
calendars separate. Labor Day is deliberately not used: its weekend spans Saturday
through Monday, so different calendars put overlapping dips on the same dates and
the signal cannot be attributed.

Two limits are built into what follows. The anomaly scale comes from a
decomposition that removes one weekday cycle, but a mixture superimposes two, so
the weekday means are attenuated and the anomalies are approximate — good for
comparing Thanksgiving dates with each other on one scale, not for absolute
depth. And the share estimate assumes every calendar present would produce the
same dip depth per unit of share; that is an assumption about US travel behaviour,
not something this file can verify.

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
from notebooks.analyze_calendar_signature import (
    WEEKDAYS, common_year_alignments, daily_counts, decompose, movable_holidays)

DIP = -2.0          # z at or below this counts as a dip
DETECTION = [1.0, 2.0]   # bounds are reported at these detection thresholds


def thanksgiving_table(frame: pd.DataFrame, alignments: dict[int, list[int]]):
    """One row per weekday alignment, keyed by its Thanksgiving date."""
    z = {(m, d): float(v) for m, d, v in zip(frame.Month, frame.Day_of_Month, frame.z)}
    rows = []
    for weekday, years in alignments.items():
        if not years:
            continue
        year = years[0]                      # every year in a class shares the calendar
        thursday = movable_holidays(year)["추수감사절"]
        friday = thursday + dt.timedelta(1)
        wednesday = thursday - dt.timedelta(1)
        rows.append({
            "jan1_weekday": WEEKDAYS[weekday],
            "thanksgiving": f"{thursday.month}/{thursday.day}",
            "z_thursday": round(z[(thursday.month, thursday.day)], 2),
            "z_friday": round(z[(friday.month, friday.day)], 2),
            "z_wednesday_before": round(z[(wednesday.month, wednesday.day)], 2),
            "years": ", ".join(map(str, years)),
        })
    table = pd.DataFrame(rows).sort_values("thanksgiving")
    table["dip_pair"] = (table.z_thursday <= DIP) & (table.z_friday <= DIP)
    # A dip on a Thursday that is the previous calendar's Black Friday is not
    # evidence of a separate calendar, so it is marked rather than counted.
    dates = {r.thanksgiving: r for r in table.itertuples()}
    shadow = []
    for row in table.itertuples():
        month, day = map(int, row.thanksgiving.split("/"))
        previous = dt.date(2019, month, day) - dt.timedelta(1)
        key = f"{previous.month}/{previous.day}"
        shadow.append(bool(key in dates and dates[key].dip_pair))
    table["after_a_confirmed_thanksgiving"] = shadow
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline_recovery_v2_calendar_mixture_20260917")
    args = ap.parse_args()
    if not args.name.startswith("baseline_recovery_v2"):
        print("[!] name must start with baseline_recovery_v2")
        return 1

    source = ROOT / "data/train.csv"
    if not source.exists():
        print(f"[!] real data not found: {source}")
        return 1
    frame = decompose(daily_counts(load_data(source)))
    alignments = common_year_alignments()
    table = thanksgiving_table(frame, alignments)

    present = table[table.dip_pair]
    # Shares are proportional to dip depth under the equal-depth assumption above.
    depths = present.z_thursday.abs()
    shares = (depths / depths.sum()).round(4)
    present = present.assign(implied_share=shares.to_numpy())
    per_share_depth = float(depths.sum())    # depth a single calendar would produce

    absent = table[~table.dip_pair & ~table.after_a_confirmed_thanksgiving]
    bounds = {f"z<={t}": round(float(t / per_share_depth), 4) for t in DETECTION}

    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    table.to_csv(out / f"{args.name}_alignments.csv", index=False)

    summary = {
        "name": args.name,
        "fits_any_model": False,
        "uses_external_data": False,
        "source_sha256": file_digest(source),
        "code_sha256": {f: file_digest(ROOT / f) for f in
                        ("src/features.py",
                         "notebooks/analyze_calendar_signature.py",
                         "notebooks/analyze_calendar_mixture.py")},
        "dip_threshold_z": DIP,
        "alignments_with_dip_pair": int(len(present)),
        "alignments_without_dip_pair": int(len(absent)),
        "present": present[["thanksgiving", "z_thursday", "z_friday",
                            "implied_share", "years"]].to_dict("records"),
        "absent_thursday_z": {r.thanksgiving: r.z_thursday for r in absent.itertuples()},
        "implied_single_calendar_depth_z": round(per_share_depth, 2),
        "third_calendar_share_upper_bound": bounds,
        "assumptions": [
            "Every calendar present would produce the same Thanksgiving dip depth "
            "per unit of share.",
            "The anomaly scale removes a single weekday cycle, so a mixture leaves "
            "the depths approximate.",
            "Labor Day is excluded: its weekend spans three days and different "
            "calendars overlap on the same dates.",
        ],
        "note": ("Row counts describe how the file was sampled, not real traffic. "
                 "A share is an estimate under the stated assumptions, not a "
                 "measurement, and an upper bound is not proof of absence."),
    }
    (out / f"{args.name}_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(table.to_string(index=False))
    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n>> saved: {args.name}_alignments.csv, {args.name}_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
