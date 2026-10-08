"""Tutor feedback 2/3/4/10: read-only missing-value, time-restoration, Traffic and
vulnerable-group accounting.

No model is fitted and no target is used to transform data. ``data/train.csv``
and the local row-date attribution file are only read. The pure functions below
reuse the repository's own preprocessing functions (``src/features.py`` and the
``Route`` construction of ``rerun_all_phases.build_features``) so the counts
describe the same transformation the experiments used.

Every accounting dict follows one rule: the population is fixed first and the
identity ``original_missing == restored + residual_missing`` is asserted inside
that same population.

Run (writes output/feedback_missing_audit_20261008.json)::

    PYTHONUTF8=1 uv run --locked --offline python -u scripts/feedback_missing_audit.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    MISSING_HOUR,
    build_time_features,
    build_traffic_features,
    impute_cross,
    restore_time_missing,
)

RAW_PATH = ROOT / "data/train.csv"
ATTRIBUTION_PATH = ROOT / "data/bts/baseline_recovery_v2_row_date_attribution_20260918/row_dates.csv.gz"
GROUPS_PATH = ROOT / "output/baseline_recovery_v2_oof_20260916_v2_groups.csv"
DEFAULT_OUT = ROOT / "output/feedback_missing_audit_20261008.json"
DEP, ARR = "Estimated_Departure_Time", "Estimated_Arrival_Time"


# =============================================================================
# Pure accounting functions (tested on synthetic data)
# =============================================================================


def _mask(df: pd.DataFrame, count_mask) -> np.ndarray:
    if count_mask is None:
        return np.ones(len(df), dtype=bool)
    mask = np.asarray(count_mask, dtype=bool)
    if mask.shape != (len(df),):
        raise ValueError("count_mask length must match df")
    return mask


def _identity(block: dict) -> dict:
    assert block["original_missing"] == block["restored"] + block["residual_missing"], block
    return block


def labeled(df: pd.DataFrame) -> pd.Series:
    """Same rule as src.features.split_labeled (non-null, non-blank Delay)."""
    target = df["Delay"]
    return target.notna() & target.astype("string").str.strip().ne("")


def airline_accounting(df: pd.DataFrame, *, key: str = "Carrier_Code(IATA)",
                       value: str = "Airline", count_mask=None) -> dict:
    """Original/restored/residual ``value`` missing for the legacy ``first`` and
    clean ``unique`` rules of :func:`impute_cross`.

    The mapping is fitted on the whole ``df`` (the "bundle", as the runners do);
    ``count_mask`` only selects the population that is counted afterwards.
    """
    mask = _mask(df, count_mask)
    pairs = ((key, value),)
    legacy = impute_cross(df, pairs=pairs, conflict="first")
    clean = impute_cross(df, pairs=pairs, conflict="unique")

    raw_missing = df[value].isna().to_numpy()
    both = df.dropna(subset=[key, value])
    n_values = both.groupby(key, observed=True)[value].nunique()
    key_n = df[key].map(n_values)
    key_missing = df[key].isna().to_numpy()
    unmapped = (~key_missing) & key_n.isna().to_numpy()
    ambiguous = (~key_missing) & key_n.gt(1).fillna(False).to_numpy()

    def block(after: pd.DataFrame) -> dict:
        restored = raw_missing & after[value].notna().to_numpy()
        residual = raw_missing & after[value].isna().to_numpy()
        return _identity({
            "original_missing": int((raw_missing & mask).sum()),
            "restored": int((restored & mask).sum()),
            "residual_missing": int((residual & mask).sum()),
        })

    leg, cln = block(legacy), block(clean)
    base = raw_missing & mask
    cln["residual_breakdown"] = {
        "key_missing": int((base & key_missing).sum()),
        "key_never_observed_with_value": int((base & unmapped).sum()),
        "key_maps_to_multiple_values": int((base & ambiguous).sum()),
    }
    assert sum(cln["residual_breakdown"].values()) == cln["residual_missing"], cln
    leg["residual_breakdown"] = {
        "key_missing": int((base & key_missing).sum()),
        "key_never_observed_with_value": int((base & unmapped).sum()),
    }
    assert sum(leg["residual_breakdown"].values()) == leg["residual_missing"], leg
    # unique restores a subset of what first restored; the difference is exactly
    # the rows whose key maps to more than one observed value.
    leg_restored = raw_missing & legacy[value].notna().to_numpy()
    cln_restored = raw_missing & clean[value].notna().to_numpy()
    assert not (cln_restored & ~leg_restored).any()
    observed_changed = (~raw_missing) & (legacy[value].ne(df[value]) | clean[value].ne(df[value])).to_numpy()
    return {
        "population_rows": int(mask.sum()),
        "fit_bundle_rows": int(len(df)),
        "legacy_first": leg,
        "clean_unique": cln,
        "legacy_minus_clean_restored": int(((leg_restored & ~cln_restored) & mask).sum()),
        "observed_values_changed": int((observed_changed & mask).sum()),
    }


def add_route(df: pd.DataFrame) -> pd.DataFrame:
    """Same directional key as rerun_all_phases.build_features."""
    out = df.copy()
    out["Route"] = out["Origin_Airport"].astype(str) + "_" + out["Destination_Airport"].astype(str)
    return out


def time_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(before, after) restoration exactly as P6_* (restore_duration+restore_hours)."""
    before = build_time_features(add_route(df), keep_minute=True)
    after = restore_time_missing(before, fill_duration=True, fill_hours=True)
    return before, after


def time_accounting(df: pd.DataFrame, *, count_mask=None, frames=None) -> dict:
    """Departure / arrival hour and local clock gap accounting.

    Restoration statistics (route and global medians) are fitted on all of ``df``;
    ``count_mask`` selects the counted population.
    """
    mask = _mask(df, count_mask)
    before, after = frames if frames is not None else time_frames(df)

    dep0 = before["Dep_Hour"].to_numpy() == MISSING_HOUR
    arr0 = before["Arr_Hour"].to_numpy() == MISSING_HOUR
    dep1 = after["Dep_Hour"].to_numpy() == MISSING_HOUR
    arr1 = after["Arr_Hour"].to_numpy() == MISSING_HOUR
    gap0 = before["Estimated_Duration"].isna().to_numpy()
    gap1 = after["Estimated_Duration"].isna().to_numpy()

    raw_dep_na = df[DEP].isna().to_numpy()
    raw_arr_na = df[ARR].isna().to_numpy()
    # strict=False: the only missing marker is a null/unparseable/negative value.
    assert (dep0 == (pd.to_numeric(df[DEP], errors="coerce").fillna(-1) < 0).to_numpy()).all()

    route_median = before.groupby("Route", observed=True)["Estimated_Duration"].transform("median")
    by_route = gap0 & route_median.notna().to_numpy()
    by_global = gap0 & route_median.isna().to_numpy()
    filled_gap = after["Estimated_Duration"].to_numpy()

    def side(orig, post) -> dict:
        return _identity({
            "original_missing": int((orig & mask).sum()),
            "restored": int((orig & ~post & mask).sum()),
            "residual_missing": int((post & mask).sum()),
        })

    dep = side(dep0, dep1)
    arr = side(arr0, arr1)
    gap = side(gap0, gap1)
    gap["restored_by_route_median"] = int((by_route & mask).sum())
    gap["restored_by_global_median"] = int((by_global & mask).sum())
    assert gap["restored_by_route_median"] + gap["restored_by_global_median"] == gap["restored"]
    gap["restored_value_zero"] = int((gap0 & (filled_gap == 0) & mask).sum())
    # residual hours can only be rows where neither side was observed.
    assert ((dep1 | arr1) <= (dep0 & arr0)).all()
    assert not (dep1 ^ arr1).any()

    def out_of_clock(col: str) -> int:
        v = pd.to_numeric(df[col], errors="coerce")
        bad = ((v // 100 > 23) | (v % 100 > 59)).fillna(False).to_numpy()
        return int((bad & mask).sum())

    return {
        "population_rows": int(mask.sum()),
        "fit_bundle_rows": int(len(df)),
        "raw_departure_null": int((raw_dep_na & mask).sum()),
        "raw_arrival_null": int((raw_arr_na & mask).sum()),
        "raw_both_null": int((raw_dep_na & raw_arr_na & mask).sum()),
        "raw_departure_only_null": int((raw_dep_na & ~raw_arr_na & mask).sum()),
        "raw_arrival_only_null": int((~raw_dep_na & raw_arr_na & mask).sum()),
        "raw_values_outside_clock_dep": out_of_clock(DEP),
        "raw_values_outside_clock_arr": out_of_clock(ARR),
        "dep_hour": dep,
        "arr_hour": arr,
        "local_time_gap": gap,
        "valid_gap_zero_before_restore": int(((before["Estimated_Duration"] == 0).to_numpy() & mask).sum()),
        "global_median_after_route_fill": float(
            before["Estimated_Duration"].fillna(route_median).median()),
        "global_median_before_route_fill": float(before["Estimated_Duration"].median()),
    }


def traffic_accounting(after: pd.DataFrame, *, count_mask=None, years=None) -> dict:
    """Facts about Origin/Dest Traffic built from ``after`` (restored frame)."""
    mask = _mask(after, count_mask)
    frame = after.reset_index(drop=True)
    built = build_traffic_features(frame, exclude_missing_hour=True, add_missing_flag=True)
    result = {"population_rows": int(mask.sum()), "bundle_rows": int(len(frame))}
    for name, airport, hour in (("Origin_Traffic", "Origin_Airport", "Dep_Hour"),
                                ("Dest_Traffic", "Destination_Airport", "Arr_Hour")):
        values = built[name].to_numpy()
        valid = ~np.isnan(values)
        item = {
            "nan_rows": int((~valid & mask).sum()),
            "min_value": float(np.nanmin(values[mask & valid])) if (mask & valid).any() else None,
            "rows_equal_1_only_self": int(((values == 1) & mask).sum()),
        }
        if years is not None:
            keys = ["Month", "Day_of_Month", airport, hour]
            yr = pd.Series(np.asarray(years), index=frame.index)
            sub = frame.loc[valid, keys].assign(_year=yr[valid])
            n_years = sub.groupby(keys, observed=True, dropna=False)["_year"].transform("nunique")
            mixed = np.zeros(len(frame), dtype=bool)
            mixed[np.flatnonzero(valid)] = n_years.gt(1).to_numpy()
            item["rows_in_key_with_both_years"] = int((mixed & mask).sum())
        result[name] = item
    return result


def vulnerable_group_from_groups_csv(groups: pd.DataFrame, phase: str = "P6_clean") -> dict:
    part = groups[(groups.phase_key == phase) & (groups.dimension == "raw_time_pattern")
                  & (groups.group == "both_missing")].sort_values("seed")
    seeds = {}
    for row in part.itertuples():
        seeds[str(int(row.seed))] = {
            "n": int(row.n), "positives": int(row.positives), "tp": int(row.tp),
            "fn": int(row.fn), "recall": float(row.recall),
            "recall_recomputed": float(row.tp / row.positives),
        }
    recalls = [v["recall_recomputed"] for v in seeds.values()]
    return {"phase": phase, "seeds": seeds, "recall_mean_of_seeds": float(np.mean(recalls))}


# =============================================================================
# Driver (real data, read-only)
# =============================================================================


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runner_cross_check(raw: pd.DataFrame) -> dict:
    """Cross-check against the actual P6_clean feature builder (no fitting)."""
    import rerun_all_phases as runner

    spec = next(s for s in runner.PHASES if s.key == "P6_clean")
    X, y, U = runner.build_features(raw, spec)
    out = {}
    for name, frame in (("labeled", X), ("all", pd.concat([X, U], ignore_index=True))):
        miss_code = X.attrs.get("missing_category_codes", {}).get("Airline")
        out[name] = {
            "rows": int(len(frame)),
            "Dep_Hour_Originally_Missing": int(frame.Dep_Hour_Originally_Missing.sum()),
            "Arr_Hour_Originally_Missing": int(frame.Arr_Hour_Originally_Missing.sum()),
            "Local_Time_Gap_Originally_Missing": int(frame.Local_Time_Gap_Originally_Missing.sum()),
            "Dep_Hour_after_restore_missing": int(frame.Dep_Hour.lt(0).sum()),
            "Arr_Hour_after_restore_missing": int(frame.Arr_Hour.lt(0).sum()),
            "Dep_Hour_Missing_flag": int(frame.Dep_Hour_Missing.sum()),
            "Sin_Dep_Hour_nan": int(frame.Sin_Dep_Hour.isna().sum()),
            "Local_Time_Gap_Minutes_nan": int(frame.Local_Time_Gap_Minutes.isna().sum()),
            "Airline_MISSING_category": (int(frame.Airline.astype(int).eq(miss_code).sum())
                                         if miss_code is not None else None),
            "Origin_Traffic_nan": int(frame.Origin_Traffic.isna().sum()),
            "Dest_Traffic_nan": int(frame.Dest_Traffic.isna().sum()),
        }
    return out


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-runner-check", action="store_true")
    args = parser.parse_args(argv)
    if not RAW_PATH.is_file():
        raise FileNotFoundError(f"actual data required: {RAW_PATH}")

    raw = pd.read_csv(RAW_PATH, low_memory=False)
    lab = labeled(raw).to_numpy()
    print(f"raw rows={len(raw):,} labeled={lab.sum():,}", flush=True)

    result: dict = {
        "name": "feedback_missing_audit_20261008",
        "fits_any_model": False,
        "target_used_for": "population split (labeled vs unlabeled) and group positives only",
        "git_head": git_head(),
        "script_sha256": sha256(Path(__file__)),
        "inputs": {"raw_train": {"path": "data/train.csv", "sha256": sha256(RAW_PATH)}},
        "populations": {},
    }

    # --- full 1,000,000-row bundle (preprocessing experiments) -------------
    full_frames = time_frames(raw)
    result["full_bundle"] = {
        "airline": {"all_rows": airline_accounting(raw),
                    "labeled_rows": airline_accounting(raw, count_mask=lab)},
        "time": {"all_rows": time_accounting(raw, frames=full_frames),
                 "labeled_rows": time_accounting(raw, count_mask=lab, frames=full_frames)},
        "traffic": {"all_rows": traffic_accounting(full_frames[1]),
                    "labeled_rows": traffic_accounting(full_frames[1], count_mask=lab)},
    }
    dup_all = int(raw.duplicated().sum())
    dup_no_id = int(raw.drop(columns=["ID"]).duplicated().sum())
    dup_no_id_target = int(raw.drop(columns=["ID", "Delay"]).duplicated().sum())
    result["full_bundle"]["duplicates"] = {
        "exact_rows": dup_all, "excluding_ID": dup_no_id, "excluding_ID_and_Delay": dup_no_id_target}
    print("full bundle done", flush=True)

    # --- date-attributed bundle (weather comparison) -----------------------
    vuln = {}
    if ATTRIBUTION_PATH.is_file():
        result["inputs"]["row_dates"] = {
            "path": str(ATTRIBUTION_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(ATTRIBUTION_PATH)}
        dates = pd.read_csv(ATTRIBUTION_PATH, usecols=["ID", "status", "date_attributed", "attributed_year"])
        assert dates.ID.is_unique and len(dates) == len(raw)
        dates = dates.set_index("ID").loc[raw.ID]
        adopted_mask = dates.date_attributed.astype(bool).to_numpy()
        adopted = raw.loc[adopted_mask].reset_index(drop=True)
        years = dates.attributed_year.to_numpy()[adopted_mask]
        lab_ad = labeled(adopted).to_numpy()
        ad_frames = time_frames(adopted)
        result["date_attributed_bundle"] = {
            "airline": {"all_rows": airline_accounting(adopted),
                        "labeled_rows": airline_accounting(adopted, count_mask=lab_ad)},
            "time": {"all_rows": time_accounting(adopted, frames=ad_frames),
                     "labeled_rows": time_accounting(adopted, count_mask=lab_ad, frames=ad_frames)},
            "traffic": {"all_rows": traffic_accounting(ad_frames[1], years=years),
                        "labeled_rows": traffic_accounting(ad_frames[1], count_mask=lab_ad, years=years)},
        }
        # Traffic of the same labeled+attributed rows in the two bundles.
        full_t = build_traffic_features(full_frames[1].reset_index(drop=True))
        ad_t = build_traffic_features(ad_frames[1].reset_index(drop=True))
        sel = adopted_mask & lab
        f_o = full_t.Origin_Traffic.to_numpy()[sel]
        a_o = ad_t.Origin_Traffic.to_numpy()[lab_ad]
        assert (raw.ID.to_numpy()[sel] == adopted.ID.to_numpy()[lab_ad]).all()
        result["traffic_bundle_comparison_labeled_attributed"] = {
            "rows": int(sel.sum()),
            "origin_equal": int((f_o == a_o).sum()),
            "origin_full_greater": int((f_o > a_o).sum()),
            "origin_full_less": int((f_o < a_o).sum()),
            "origin_median_full": float(np.median(f_o)),
            "origin_median_attributed": float(np.median(a_o)),
        }

        both_null = (raw[DEP].isna() & raw[ARR].isna()).to_numpy()
        target = raw.Delay.astype("string").str.strip()
        grp = both_null & lab
        vuln = {
            "raw_both_time_null_labeled_rows": int(grp.sum()),
            "raw_both_time_null_labeled_delayed": int((grp & target.eq("Delayed").fillna(False).to_numpy()).sum()),
            "in_date_attributed": int((grp & adopted_mask).sum()),
            "in_weather_eval_population": int((grp & adopted_mask & lab).sum()),
            "status_counts": {str(k): int(v) for k, v in
                              pd.Series(dates.status.to_numpy()[grp]).value_counts().items()},
            "any_raw_time_null_in_weather_population": int(
                ((raw[DEP].isna() | raw[ARR].isna()).to_numpy() & adopted_mask & lab).sum()),
        }
        result["populations"] = {
            "all_rows": int(len(raw)), "labeled": int(lab.sum()),
            "date_attributed": int(adopted_mask.sum()),
            "date_attributed_labeled": int(sel.sum()),
            "date_attributed_labeled_delayed": int((sel & target.eq("Delayed").fillna(False).to_numpy()).sum()),
        }
        print("date-attributed bundle done", flush=True)
    else:
        result["date_attributed_bundle"] = "미확인: local row_dates.csv.gz missing"

    groups = pd.read_csv(GROUPS_PATH)
    vuln["oof_groups_csv"] = vulnerable_group_from_groups_csv(groups)
    result["inputs"]["oof_groups"] = {"path": str(GROUPS_PATH.relative_to(ROOT)).replace("\\", "/"),
                                      "sha256": sha256(GROUPS_PATH)}
    result["vulnerable_group_item10"] = vuln

    if not args.skip_runner_check:
        result["runner_cross_check_P6_clean_full_bundle"] = runner_cross_check(raw)
        print("runner cross-check done", flush=True)

    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                        newline="\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
