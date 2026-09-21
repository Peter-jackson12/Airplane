"""Prepare and execute the full adopted-row weather archive collection in bounded shards.

The 21/300-row fetcher is intentionally kept as a sample validator.  Full-scale
collection has different operational needs: an immutable executable plan,
bounded checkpoint size, small failure domains, and cache directories that do
not put ~100k files into one folder.

This module therefore:

1. Rebuilds the adopted 706k-row pool from the existing row-attribution
   evidence and an explicitly named mapping.
2. Produces the *same station/UTC-day windows* as
   fetch_weather_sample_expanded.build_station_day_groups, but materializes
   them as a local request plan before any network call.
3. Orders requests deterministically by UTC day then station and assigns them
   to fixed-size shards.
4. Executes exactly one shard at a time with the existing checkpoint-v3
   budget/retry/crash-accounting engine.  Every shard has its own checkpoint,
   manifest and cache directory.
5. Finalizes only after every shard is complete with zero permanent failures,
   re-validating request identity, windows and cache hashes.

The row-level request plan, shard checkpoints/manifests and raw IEM CSVs live
under data/weather_probe/<name>/ and are intentionally Git-ignored.  Only the
small plan/final summary manifests live under output/.

No target/delay/actual-outcome column is read, and --prepare-only never makes a
network request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from notebooks.fetch_weather_sample import (
    CACHE_DIR as LEGACY_SAMPLE_CACHE_DIR,
    LOOKAHEAD_HOURS,
    LOOKBACK_HOURS,
    digest,
    fetch_attempt,
    validate_cached_window,
)
from notebooks.fetch_weather_sample_expanded import (
    SUBPROCESS_TERMINATION_GRACE_SECONDS,
    SUCCESS_PAUSE_SECONDS,
    checkpoint_has_prior_progress,
    compute_plan_fingerprint,
    execute_with_caps,
    fetch_plan_options,
    load_checkpoint,
    record_budget_limits,
    save_checkpoint,
    verify_plan_fingerprint,
)
from notebooks.scope_weather_collection_refined import ATTRIBUTION_RUN, load_adopted_pool

ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA_VERSION = 1
SHARD_MANIFEST_SCHEMA_VERSION = 1
FINAL_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_SHARD_SIZE = 500
PLAN_COLUMNS = [
    "ordinal",
    "shard_index",
    "station",
    "day",
    "min_prediction_at",
    "max_prediction_at",
    "window_start_utc",
    "window_end_utc",
    "n_prediction_refs",
]


def _validate_name(name: str) -> None:
    if not name.startswith("baseline_recovery_v2") or Path(name).name != name:
        raise ValueError("Invalid --name; use a simple baseline_recovery_v2* run name")


def _json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def _as_root_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _normalize_cache_file(path_value: str | Path) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        return path.as_posix()
    return _as_root_relative(path)


def run_dir(name: str) -> Path:
    return ROOT / "data" / "weather_probe" / name


def plan_path(name: str) -> Path:
    return run_dir(name) / "request_plan.csv"


def plan_manifest_path(name: str) -> Path:
    return ROOT / "output" / f"{name}_full_weather_plan_manifest.json"


def final_manifest_path(name: str) -> Path:
    return ROOT / "output" / f"{name}_full_weather_fetch_manifest.json"


def shard_checkpoint_path(name: str, shard_index: int) -> Path:
    return run_dir(name) / "checkpoints" / f"shard_{shard_index:05d}.json"


def shard_manifest_path(name: str, shard_index: int) -> Path:
    return run_dir(name) / "shards" / f"shard_{shard_index:05d}_manifest.json"


def shard_cache_dir(name: str, shard_index: int) -> Path:
    return run_dir(name) / "cache" / f"shard_{shard_index:05d}"


def build_station_day_plan(pool: pd.DataFrame, shard_size: int = DEFAULT_SHARD_SIZE) -> tuple[pd.DataFrame, dict]:
    """Build the exact station/day request windows the current fetch engine uses.

    This deliberately does *not* use the refined interval-union request-count
    estimate.  A full run must first materialize the windows it will actually
    execute; the historical 132,783 size-based interval estimate remains a
    sizing scenario, not an executable-plan count.
    """
    if shard_size <= 0:
        raise ValueError("shard_size must be > 0")
    required = {"row_tier", "prediction_at", "origin_station", "destination_station"}
    missing = required - set(pool.columns)
    if missing:
        raise ValueError(f"pool is missing required columns: {sorted(missing)}")

    mapping_eligible = pool["row_tier"].eq("confirmed_period")
    utc_resolved = mapping_eligible & pool["prediction_at"].notna()
    safe = pool.loc[utc_resolved, ["origin_station", "destination_station", "prediction_at"]].copy()
    if safe[["origin_station", "destination_station"]].isna().any().any():
        raise ValueError("confirmed_period rows must have both mapped stations")

    origin = safe[["origin_station", "prediction_at"]].rename(columns={"origin_station": "station"})
    destination = safe[["destination_station", "prediction_at"]].rename(
        columns={"destination_station": "station"}
    )
    events = pd.concat([origin, destination], ignore_index=True)
    events["prediction_at"] = pd.to_datetime(events["prediction_at"], utc=True)
    events["day"] = events["prediction_at"].dt.strftime("%Y-%m-%d")

    grouped = (
        events.groupby(["station", "day"], sort=False, as_index=False)
        .agg(
            min_prediction_at=("prediction_at", "min"),
            max_prediction_at=("prediction_at", "max"),
            n_prediction_refs=("prediction_at", "size"),
        )
        .sort_values(["day", "station"], kind="mergesort")
        .reset_index(drop=True)
    )
    grouped["window_start_utc"] = grouped["min_prediction_at"] - pd.Timedelta(hours=LOOKBACK_HOURS)
    grouped["window_end_utc"] = grouped["max_prediction_at"] + pd.Timedelta(hours=LOOKAHEAD_HOURS)
    grouped.insert(0, "ordinal", range(len(grouped)))
    grouped.insert(1, "shard_index", grouped["ordinal"] // shard_size)
    grouped = grouped[PLAN_COLUMNS]

    denominators = {
        "total_adopted_rows": int(len(pool)),
        "mapping_eligible_rows_confirmed_period_both_ends": int(mapping_eligible.sum()),
        "utc_time_resolved_rows": int(utc_resolved.sum()),
        "mapping_eligible_but_utc_unresolved_rows": int((mapping_eligible & ~utc_resolved).sum()),
        "not_mapping_eligible_rows": int((~mapping_eligible).sum()),
        "station_day_request_groups": int(len(grouped)),
        "distinct_stations": int(grouped["station"].nunique()),
        "shard_size": int(shard_size),
        "shard_count": int(grouped["shard_index"].max() + 1) if len(grouped) else 0,
    }
    return grouped, denominators


def groups_from_plan_shard(plan: pd.DataFrame, shard_index: int) -> tuple[list[tuple[str, str]], dict]:
    shard = plan.loc[plan["shard_index"].eq(shard_index)].sort_values("ordinal")
    if shard.empty:
        raise ValueError(f"shard_index={shard_index} is outside the plan")
    groups: dict[tuple[str, str], list[pd.Timestamp]] = {}
    planned: list[tuple[str, str]] = []
    for row in shard.itertuples(index=False):
        key = (str(row.station), str(row.day))
        if key in groups:
            raise ValueError(f"duplicate station/day in request plan: {key}")
        lo = pd.Timestamp(row.min_prediction_at)
        hi = pd.Timestamp(row.max_prediction_at)
        if lo.tzinfo is None or hi.tzinfo is None:
            raise ValueError("plan prediction timestamps must be timezone-aware")
        values = [lo] if lo == hi else [lo, hi]
        calc_start = min(values) - pd.Timedelta(hours=LOOKBACK_HOURS)
        calc_end = max(values) + pd.Timedelta(hours=LOOKAHEAD_HOURS)
        if calc_start != pd.Timestamp(row.window_start_utc) or calc_end != pd.Timestamp(row.window_end_utc):
            raise ValueError(f"plan window no longer matches fetch semantics for {key}")
        groups[key] = values
        planned.append(key)
    return planned, groups


def _source_manifest_row_sha() -> str:
    path = ROOT / "output" / f"{ATTRIBUTION_RUN}_manifest.json"
    manifest = json.loads(path.read_text())
    return str(manifest["row_sha256"])


def prepare_plan(name: str, mapping_name: str, shard_size: int = DEFAULT_SHARD_SIZE) -> dict:
    _validate_name(name)
    local_plan = plan_path(name)
    out_manifest = plan_manifest_path(name)
    if local_plan.exists() or out_manifest.exists():
        raise FileExistsError(
            "Full-weather plan evidence already exists. Reuse it unchanged or use a fresh --name; "
            "existing plan evidence is never overwritten."
        )

    pool = load_adopted_pool(mapping_name)
    plan, denominators = build_station_day_plan(pool, shard_size=shard_size)
    local_plan.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(local_plan, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")

    mapping_path = ROOT / "output" / f"{mapping_name}_mapping_table.csv"
    raw_path = ROOT / "data" / "train.csv"
    manifest = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "name": name,
        "mapping_name": mapping_name,
        "attribution_run": ATTRIBUTION_RUN,
        "attribution_row_sha256": _source_manifest_row_sha(),
        "raw_train_sha256": digest(raw_path),
        "mapping_sha256": digest(mapping_path),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "plan_file": _as_root_relative(local_plan),
        "plan_sha256": digest(local_plan),
        "plan_columns": PLAN_COLUMNS,
        "plan_order": "UTC day ascending, then station ascending; stable mergesort",
        "denominators": denominators,
        "fetch_policy": fetch_plan_options(),
        "cache_reuse_policy": (
            "automatic reuse is exact-window only: a schema-valid existing cache file for the exact "
            "station/start/end window may be reused; merely overlapping/contained historical sample "
            "windows are not treated as complete coverage"
        ),
        "checkpoint_policy": (
            "one checkpoint per fixed-size shard; shard_size is part of this immutable plan and must "
            "not change under the same run name"
        ),
        "fits_any_model": False,
        "uses_external_data": False,
        "target_columns_used": [],
        "limitations": [
            "This materializes the current station/UTC-day fetch semantics. It intentionally does not "
            "claim that the older refined interval-union request estimate is the number this executor "
            "will issue.",
            "prepare-only performs no network request. Actual byte/time totals remain unknown until the "
            "shards are fetched.",
            "Only rows whose origin and destination mapping tier are both confirmed_period and whose "
            "prediction_at resolves are included in the request plan; all adopted rows remain in the "
            "denominator summary.",
        ],
    }
    _json_atomic(out_manifest, manifest)
    return manifest


def load_plan(name: str, mapping_name: str, shard_size: int | None = None) -> tuple[dict, pd.DataFrame]:
    manifest_file = plan_manifest_path(name)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Run --prepare-only first: {manifest_file}")
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("Unsupported full-weather plan schema; use the code that created this plan")
    if manifest.get("mapping_name") != mapping_name:
        raise ValueError(
            f"plan mapping_name={manifest.get('mapping_name')!r}, requested {mapping_name!r}; "
            "use the exact mapping or a fresh run name"
        )
    planned_shard_size = int(manifest["denominators"]["shard_size"])
    if shard_size is not None and int(shard_size) != planned_shard_size:
        raise ValueError(
            f"plan shard_size={planned_shard_size}, requested {shard_size}; shard boundaries are "
            "immutable under one run name"
        )
    local_plan = ROOT / manifest["plan_file"]
    if not local_plan.exists():
        raise FileNotFoundError(f"local request plan is missing: {local_plan}")
    if digest(local_plan) != manifest["plan_sha256"]:
        raise ValueError("local request plan hash no longer matches the recorded plan manifest")

    plan = pd.read_csv(
        local_plan,
        parse_dates=["min_prediction_at", "max_prediction_at", "window_start_utc", "window_end_utc"],
    )
    if list(plan.columns) != PLAN_COLUMNS:
        raise ValueError("request plan columns no longer match the recorded schema")
    if len(plan) != int(manifest["denominators"]["station_day_request_groups"]):
        raise ValueError("request plan row count no longer matches the manifest")
    if len(plan) and not plan["ordinal"].tolist() == list(range(len(plan))):
        raise ValueError("request plan ordinals are not contiguous")
    return manifest, plan


def _validate_existing_cache(station: str, start: pd.Timestamp, end: pd.Timestamp, new_cache_dir: Path) -> Path | None:
    current = validate_cached_window(station, start, end, cache_dir=new_cache_dir)
    if current is not None:
        return current
    # Conservative reuse of older 21/300-row evidence: exact window only.
    return validate_cached_window(station, start, end, cache_dir=LEGACY_SAMPLE_CACHE_DIR)


def execute_shard(
    name: str,
    mapping_name: str,
    shard_index: int,
    *,
    shard_size: int | None,
    max_requests: int,
    max_bytes: int,
    max_seconds: float,
) -> dict:
    plan_manifest, plan = load_plan(name, mapping_name, shard_size=shard_size)
    shard_count = int(plan_manifest["denominators"]["shard_count"])
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count - 1}]")

    planned, groups = groups_from_plan_shard(plan, shard_index)
    checkpoint_file = shard_checkpoint_path(name, shard_index)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = shard_cache_dir(name, shard_index)
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_file)
    resuming = checkpoint_has_prior_progress(checkpoint)
    options = {
        **fetch_plan_options(),
        "full_plan_schema_version": PLAN_SCHEMA_VERSION,
        "full_plan_sha256": plan_manifest["plan_sha256"],
        "shard_index": int(shard_index),
        "shard_size": int(plan_manifest["denominators"]["shard_size"]),
    }
    fingerprint = compute_plan_fingerprint(
        plan_manifest["plan_sha256"], plan_manifest["mapping_sha256"], groups, options
    )
    verify_plan_fingerprint(checkpoint, fingerprint, name=f"{name}/shard_{shard_index:05d}")
    caps = record_budget_limits(
        checkpoint, max_requests=max_requests, max_bytes=max_bytes, max_seconds=max_seconds
    )
    save_checkpoint(checkpoint_file, checkpoint)

    def cache_exists(station: str, start: pd.Timestamp, end: pd.Timestamp) -> Path | None:
        return _validate_existing_cache(station, start, end, cache_dir)

    def attempt(station: str, start: pd.Timestamp, end: pd.Timestamp, *, timeout: float, max_bytes_remaining: int):
        return fetch_attempt(
            station,
            start,
            end,
            timeout=timeout,
            max_bytes_remaining=max_bytes_remaining,
            cache_dir=cache_dir,
        )

    result = execute_with_caps(
        planned,
        groups,
        max_requests=max_requests,
        max_bytes=max_bytes,
        max_seconds=max_seconds,
        cache_exists_fn=cache_exists,
        attempt_fn=attempt,
        digest_fn=digest,
        now_fn=__import__("time").monotonic,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_file,
        sleep_fn=__import__("time").sleep,
        attempt_overhead_seconds=SUBPROCESS_TERMINATION_GRACE_SECONDS,
        success_pause_seconds=SUCCESS_PAUSE_SECONDS,
        progress_every=25,
    )

    fetched = []
    for rec in result["fetched"]:
        item = dict(rec)
        if item.get("cache_file"):
            item["cache_file"] = _normalize_cache_file(item["cache_file"])
        fetched.append(item)

    complete = result["cap_hit"] is None and not result["skipped_cap"]
    successful = complete and not result["failed"] and len(fetched) == len(planned)
    shard_payload = {
        "schema_version": SHARD_MANIFEST_SCHEMA_VERSION,
        "name": name,
        "mapping_name": mapping_name,
        "plan_sha256": plan_manifest["plan_sha256"],
        "shard_index": int(shard_index),
        "request_groups_in_shard": int(len(planned)),
        "resumed_from_checkpoint": bool(resuming),
        "caps": caps,
        "budget_limit_history": checkpoint["budget_limit_history"],
        "plan_fingerprint": fingerprint,
        "complete": bool(complete),
        "successful": bool(successful),
        "fetched_groups": int(len(fetched)),
        "failed_groups": int(len(result["failed"])),
        "skipped_due_to_cap": int(len(result["skipped_cap"])),
        "cap_that_stopped_collection": result["cap_hit"],
        "cumulative_http_attempts": int(result["cumulative_requests"]),
        "cumulative_bytes_downloaded": int(result["cumulative_bytes_measured"]),
        "cumulative_bytes_reserved_against_budget": int(result["cumulative_bytes_reserved"]),
        "cumulative_active_fetch_seconds": float(result["cumulative_seconds"]),
        "unmeasured_byte_attempts": int(result["unmeasured_byte_attempts"]),
        "cache_hit_groups": int(sum(1 for f in fetched if f.get("was_already_cached"))),
        "new_fetch_groups": int(sum(1 for f in fetched if not f.get("was_already_cached"))),
        "checkpoint_file": _as_root_relative(checkpoint_file),
        "cache_directory": _as_root_relative(cache_dir),
        "requests": fetched,
        "failed": result["failed"],
        "skipped": result["skipped_cap"],
    }
    _json_atomic(shard_manifest_path(name, shard_index), shard_payload)
    return shard_payload


def _expected_windows(plan: pd.DataFrame) -> dict[tuple[str, str], tuple[str, str]]:
    expected = {}
    for row in plan.itertuples(index=False):
        expected[(str(row.station), str(row.day))] = (
            pd.Timestamp(row.window_start_utc).isoformat(),
            pd.Timestamp(row.window_end_utc).isoformat(),
        )
    return expected


def finalize(name: str, mapping_name: str, *, verify_cache_hashes: bool = True) -> dict:
    plan_manifest, plan = load_plan(name, mapping_name)
    expected = _expected_windows(plan)
    shard_count = int(plan_manifest["denominators"]["shard_count"])
    shard_index_rows = []
    actual_keys: set[tuple[str, str]] = set()
    totals = {
        "cumulative_http_attempts": 0,
        "cumulative_bytes_downloaded": 0,
        "cumulative_bytes_reserved_against_budget": 0,
        "cumulative_active_fetch_seconds": 0.0,
        "cache_hit_groups": 0,
        "new_fetch_groups": 0,
    }

    for idx in range(shard_count):
        path = shard_manifest_path(name, idx)
        if not path.exists():
            raise ValueError(f"cannot finalize: missing shard manifest {idx}/{shard_count - 1}")
        shard = json.loads(path.read_text())
        if shard.get("schema_version") != SHARD_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"shard {idx} has unsupported manifest schema")
        if shard.get("plan_sha256") != plan_manifest["plan_sha256"]:
            raise ValueError(f"shard {idx} belongs to a different request plan")
        if not shard.get("successful"):
            raise ValueError(
                f"cannot finalize: shard {idx} is not successful "
                f"(cap={shard.get('cap_that_stopped_collection')!r}, failed={shard.get('failed_groups')})"
            )

        for rec in shard["requests"]:
            key = (str(rec["station"]), str(rec["day"]))
            if key in actual_keys:
                raise ValueError(f"duplicate completed request across shards: {key}")
            if key not in expected:
                raise ValueError(f"shard {idx} contains request not present in plan: {key}")
            if (str(rec["window_start_utc"]), str(rec["window_end_utc"])) != expected[key]:
                raise ValueError(f"shard {idx} request window differs from plan for {key}")
            cache = ROOT / rec["cache_file"]
            if not cache.exists():
                raise ValueError(f"shard {idx} cache file is missing: {cache}")
            if verify_cache_hashes and digest(cache) != rec["sha256"]:
                raise ValueError(f"shard {idx} cache hash mismatch for {key}")
            actual_keys.add(key)

        for key in totals:
            totals[key] += shard[key]
        shard_index_rows.append(
            {
                "shard_index": idx,
                "manifest_file": _as_root_relative(path),
                "manifest_sha256": digest(path),
                "request_groups": shard["request_groups_in_shard"],
                "http_attempts": shard["cumulative_http_attempts"],
                "bytes_downloaded": shard["cumulative_bytes_downloaded"],
                "active_fetch_seconds": shard["cumulative_active_fetch_seconds"],
                "cache_hit_groups": shard["cache_hit_groups"],
                "new_fetch_groups": shard["new_fetch_groups"],
            }
        )

    if actual_keys != set(expected):
        missing = len(set(expected) - actual_keys)
        extra = len(actual_keys - set(expected))
        raise ValueError(f"completed shard requests do not equal plan: missing={missing}, extra={extra}")

    payload = {
        "schema_version": FINAL_MANIFEST_SCHEMA_VERSION,
        "name": name,
        "mapping_name": mapping_name,
        "plan_manifest": _as_root_relative(plan_manifest_path(name)),
        "plan_manifest_sha256": digest(plan_manifest_path(name)),
        "plan_sha256": plan_manifest["plan_sha256"],
        "station_day_request_groups": int(len(plan)),
        "shard_size": int(plan_manifest["denominators"]["shard_size"]),
        "shard_count": shard_count,
        "all_shards_complete": True,
        "failed_groups": 0,
        "cache_hashes_reverified": bool(verify_cache_hashes),
        **totals,
        "shards": shard_index_rows,
        "fits_any_model": False,
        "uses_external_data": True,
        "target_columns_used": [],
        "limitations": [
            "This proves completion of the materialized archive-fetch plan, not historical publication "
            "latency or model performance.",
            "Automatic legacy-cache reuse is exact-window only; overlap alone is never treated as full "
            "coverage.",
            "Row-level weather joining and weather-on/off model comparison are separate later stages.",
        ],
    }
    _json_atomic(final_manifest_path(name), payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mapping-name", required=True)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--shard-index", type=int)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    ap.add_argument("--max-requests", type=int, default=1500)
    ap.add_argument("--max-bytes", type=int, default=1_000_000_000)
    ap.add_argument("--max-seconds", type=float, default=7200.0)
    ap.add_argument("--skip-final-cache-hash-check", action="store_true")
    args = ap.parse_args()
    _validate_name(args.name)

    modes = int(args.prepare_only) + int(args.shard_index is not None) + int(args.finalize)
    if modes != 1:
        raise ValueError("Choose exactly one of --prepare-only, --shard-index N, or --finalize")

    if args.prepare_only:
        result = prepare_plan(args.name, args.mapping_name, shard_size=args.shard_size)
        print(json.dumps(result, indent=2, default=str))
        return
    if args.finalize:
        result = finalize(
            args.name,
            args.mapping_name,
            verify_cache_hashes=not args.skip_final_cache_hash_check,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "shards"}, indent=2, default=str))
        return

    result = execute_shard(
        args.name,
        args.mapping_name,
        args.shard_index,
        shard_size=args.shard_size,
        max_requests=args.max_requests,
        max_bytes=args.max_bytes,
        max_seconds=args.max_seconds,
    )
    print(json.dumps({k: v for k, v in result.items() if k not in {"requests", "failed", "skipped"}}, indent=2))


if __name__ == "__main__":
    main()
