"""Is each row a faithful record, and does anything inside it say which year it is?

Two questions, asked with the raw file alone.

First, integrity. The modelling work has assumed all along that a row is one real
flight record whose fields belong together. Nobody had checked. Real operational
data obeys constraints that a shuffled or synthesised file would break: a route has
one distance, an aircraft belongs to one operator, an airport sits in one state, an
airport code and its numeric id are one and the same thing.

Second, attribution. The file is known to mix at least two calendars, which blocks
attaching weather to an event. If some column differed between those calendars, the
year could be recovered per row without leaving the file. Two candidates are tested
here — the operating carrier, and the routes that carry two distances — by asking
whether either one behaves differently on the two Thanksgiving dates. A calendar
suppressed on its own Thanksgiving leaves the day enriched in the other calendar,
so a column that separates them has to shift.

Neither question is answered by a model, and nothing external is used. A constraint
holding is evidence the fields belong together, not proof; a column failing to
separate the calendars bounds what internal attribution can do, and is not proof
that no column could.

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
from src.run_store import file_digest

MONTH_LENGTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# The two Thanksgivings found by analyze_calendar_signature.py.
THANKSGIVINGS = [(11, 22), (11, 28)]
MIN_ROWS = 400


def day_of_year(frame: pd.DataFrame) -> pd.Series:
    offsets = np.cumsum([0] + MONTH_LENGTHS[:-1])
    return pd.Series([offsets[int(m) - 1] + int(d)
                      for m, d in zip(frame.Month, frame.Day_of_Month)],
                     index=frame.index)


def one_to_one(frame: pd.DataFrame, keys, value, label: str) -> dict:
    """How often one key carries exactly one value, with an example of a breach."""
    part = frame.dropna(subset=list(keys) + [value])
    counts = part.groupby(list(keys))[value].nunique()
    breaches = counts[counts > 1]
    example = None
    if len(breaches):
        key = breaches.index[0]
        mask = np.ones(len(part), dtype=bool)
        for column, k in zip(keys, key if isinstance(key, tuple) else (key,)):
            mask &= (part[column] == k).to_numpy()
        values = sorted(pd.unique(part.loc[mask, value]))[:6]
        example = {"key": " / ".join(map(str, key if isinstance(key, tuple) else (key,))),
                   "values": [float(v) if isinstance(v, (int, float, np.number))
                              else str(v) for v in values]}
    return {"check": label, "groups": int(len(counts)),
            "single_valued": int((counts == 1).sum()),
            "share_single_valued": round(float((counts == 1).mean()), 6),
            "breaches": int(len(breaches)), "example_breach": example}


def separation(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    """Does a column's composition shift on either Thanksgiving?

    Each day's share of a category is compared with that category's share across the
    other November days, as a binomial z. A column that tracks the calendar has to
    move in opposite directions on the two dates.
    """
    november = frame[frame.Month == 11].dropna(subset=[column])
    days = [day for day in day_of_year(november).unique()]
    holiday_doy = {d: day_of_year(pd.DataFrame({"Month": [m], "Day_of_Month": [d2]})).iloc[0]
                   for d, (m, d2) in zip(("first", "second"), THANKSGIVINGS)}
    doy = day_of_year(november)
    base = november[~doy.isin(list(holiday_doy.values()))]
    rows = []
    for value, part in november.groupby(column):
        if len(part) < MIN_ROWS:
            continue
        share = float((base[column] == value).mean())
        row = {"dimension": label, "value": str(value), "november_rows": int(len(part)),
               "baseline_share": round(share, 6)}
        for name, target in holiday_doy.items():
            day = november[doy == target]
            observed = float((day[column] == value).mean())
            error = float(np.sqrt(share * (1 - share) / len(day)))
            row[f"{name}_share"] = round(observed, 6)
            row[f"{name}_z"] = round((observed - share) / error, 2) if error else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # Separation would mean one date pushing the share down and the other up.
    table["separates"] = ((table.first_z <= -2) & (table.second_z >= 2)) | \
                         ((table.first_z >= 2) & (table.second_z <= -2))
    return table.sort_values("november_rows", ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline_recovery_v2_record_integrity_20260917")
    args = ap.parse_args()
    if not args.name.startswith("baseline_recovery_v2"):
        print("[!] name must start with baseline_recovery_v2")
        return 1
    source = ROOT / "data/train.csv"
    if not source.exists():
        print(f"[!] real data not found: {source}")
        return 1
    raw = load_data(source)

    checks = [
        one_to_one(raw, ["Origin_Airport", "Destination_Airport"], "Distance",
                   "노선당 거리"),
        one_to_one(raw, ["Tail_Number"], "Carrier_ID(DOT)", "기체번호당 운항사"),
        one_to_one(raw, ["Origin_Airport"], "Origin_State", "출발공항당 주"),
        one_to_one(raw, ["Destination_Airport"], "Destination_State", "도착공항당 주"),
        one_to_one(raw, ["Origin_Airport"], "Origin_Airport_ID", "공항코드당 공항ID"),
        one_to_one(raw, ["Origin_Airport_ID"], "Origin_Airport", "공항ID당 공항코드"),
        one_to_one(raw, ["Carrier_ID(DOT)"], "Airline", "운항사ID당 항공사명"),
    ]
    integrity = pd.DataFrame([{k: v for k, v in c.items() if k != "example_breach"}
                              for c in checks])

    # Routes carrying two distances are the file's own hint that it spans more than
    # one period, so they are tested as a possible per-row discriminator.
    pair = raw.dropna(subset=["Origin_Airport", "Destination_Airport", "Distance"]).copy()
    pair["route"] = pair.Origin_Airport.astype(str) + "->" + pair.Destination_Airport.astype(str)
    per_route = pair.groupby("route").Distance.nunique()
    dual = per_route[per_route == 2].index
    dual_rows = pair[pair.route.isin(dual)].copy()
    lowest = pair[pair.route.isin(dual)].groupby("route").Distance.min()
    dual_rows["distance_side"] = np.where(
        dual_rows.Distance == dual_rows.route.map(lowest), "lower", "upper")

    carrier = separation(raw, "Carrier_ID(DOT)", "운항사")
    distance = separation(dual_rows, "distance_side", "노선 거리값")
    tables = [t for t in (carrier, distance) if not t.empty]
    separations = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    integrity.to_csv(out / f"{args.name}_constraints.csv", index=False)
    if not separations.empty:
        separations.to_csv(out / f"{args.name}_separation.csv", index=False)

    summary = {
        "name": args.name, "fits_any_model": False, "uses_external_data": False,
        "source_sha256": file_digest(source),
        "code_sha256": {f: file_digest(ROOT / f) for f in
                        ("src/features.py", "notebooks/analyze_record_integrity.py")},
        "constraints": [{k: v for k, v in c.items()} for c in checks],
        "dual_distance_routes": int(len(dual)),
        "dual_distance_rows": int(len(dual_rows)),
        "columns_separating_the_calendars": (
            separations.loc[separations.separates, "value"].tolist()
            if not separations.empty else []),
        "columns_tested_for_separation": ["Carrier_ID(DOT)", "distance_side"],
        "note": ("A constraint holding is evidence the fields belong together, not "
                 "proof. A column failing to separate the calendars bounds internal "
                 "attribution; it does not prove no column could."),
    }
    (out / f"{args.name}_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(integrity.to_string(index=False))
    for check in checks:
        if check["example_breach"]:
            print(f"  · {check['check']} 예외 예: {check['example_breach']}")
    print(f"\n거리값이 2개인 노선 {len(dual)}개 / {len(dual_rows):,}행")
    if not separations.empty:
        print("\n=== 두 추수감사절에서의 구성 변화 ===")
        print(separations.head(28).to_string(index=False))
        print(f"\n두 날짜를 가르는 값: "
              f"{summary['columns_separating_the_calendars'] or '없음'}")
    print(f"\n>> saved: {args.name}_constraints.csv, {args.name}_separation.csv, "
          f"{args.name}_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
