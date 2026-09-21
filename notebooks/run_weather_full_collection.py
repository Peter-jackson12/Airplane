"""Sequential, fail-closed orchestration for full-weather bulk shards.

This wrapper does not change the immutable request plan and does not finalize the
run. It advances shards strictly one at a time, reuses each shard's checkpoint,
and stops immediately when a shard is incomplete or unsuccessful.

Resource limits are expressed as *additional headroom* for the shard being
resumed. They are converted to the cumulative ceilings expected by
execute_shard(), so prior requests/bytes/seconds remain charged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from notebooks.fetch_weather_full_sharded import (
    ROOT,
    SHARD_MANIFEST_SCHEMA_VERSION,
    _validate_bulk_file,
    execute_shard,
    load_plan,
    request_from_row,
    shard_checkpoint_path,
    shard_manifest_path,
)
from notebooks.fetch_weather_sample import digest
from notebooks.fetch_weather_sample_expanded import load_checkpoint

DEFAULT_RETRY_HEADROOM = 10
DEFAULT_ADDITIONAL_MAX_BYTES = 300_000_000
DEFAULT_ADDITIONAL_MAX_SECONDS = 5_400.0


def _shard_rows(plan: pd.DataFrame, shard_index: int) -> pd.DataFrame:
    rows = plan.loc[plan["shard_index"].eq(shard_index)].sort_values("ordinal")
    if rows.empty:
        raise ValueError(f"shard_index={shard_index} is outside the plan")
    return rows


def _expected_requests(plan: pd.DataFrame, shard_index: int) -> dict[str, dict]:
    return {
        req["request_id"]: req
        for req in (
            request_from_row(row)
            for row in _shard_rows(plan, shard_index).itertuples(index=False)
        )
    }


def verify_successful_shard(
    name: str,
    plan_manifest: dict,
    plan: pd.DataFrame,
    shard_index: int,
    *,
    verify_cache_hashes: bool = True,
) -> dict:
    """Strictly revalidate a completed shard without issuing network calls."""
    path = shard_manifest_path(name, shard_index)
    if not path.exists():
        raise FileNotFoundError(f"missing shard manifest: {path}")

    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SHARD_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"shard {shard_index} has unsupported manifest schema")
    if payload.get("plan_sha256") != plan_manifest["plan_sha256"]:
        raise ValueError(f"shard {shard_index} belongs to a different request plan")
    if not payload.get("complete") or not payload.get("successful"):
        raise ValueError(f"shard {shard_index} is not complete and successful")
    if int(payload.get("failed_groups", -1)) != 0:
        raise ValueError(f"shard {shard_index} records failed groups")
    if int(payload.get("skipped_due_to_cap", -1)) != 0:
        raise ValueError(f"shard {shard_index} records cap-skipped groups")

    expected = _expected_requests(plan, shard_index)
    if int(payload.get("request_groups_in_shard", -1)) != len(expected):
        raise ValueError(f"shard {shard_index} request count differs from plan")
    records = payload.get("requests", [])
    if len(records) != len(expected):
        raise ValueError(f"shard {shard_index} completed request list differs from plan")

    seen: set[str] = set()
    for rec in records:
        rid = str(rec.get("request_id"))
        if rid in seen:
            raise ValueError(f"shard {shard_index} duplicates request {rid}")
        if rid not in expected:
            raise ValueError(f"shard {shard_index} contains unknown request {rid}")
        exp = expected[rid]
        if rec.get("network") != exp["network"] or rec.get("month") != exp["month"]:
            raise ValueError(f"shard {shard_index} identity mismatch for {rid}")
        if list(rec.get("stations", [])) != list(exp["stations"]):
            raise ValueError(f"shard {shard_index} station batch mismatch for {rid}")
        if (
            str(rec.get("window_start_utc")) != exp["window_start_utc"].isoformat()
            or str(rec.get("window_end_utc")) != exp["window_end_utc"].isoformat()
        ):
            raise ValueError(f"shard {shard_index} request window mismatch for {rid}")

        cache = ROOT / str(rec["cache_file"])
        if not cache.exists():
            raise ValueError(f"shard {shard_index} cache is missing: {cache}")
        n_rows = _validate_bulk_file(cache, exp)
        if rec.get("rows") is not None and int(rec["rows"]) != n_rows:
            raise ValueError(f"shard {shard_index} row count mismatch for {rid}")
        if verify_cache_hashes and digest(cache) != rec.get("sha256"):
            raise ValueError(f"shard {shard_index} cache hash mismatch for {rid}")
        seen.add(rid)

    if seen != set(expected):
        raise ValueError(f"shard {shard_index} completed request set differs from plan")
    return payload


def checkpoint_progress(name: str, plan: pd.DataFrame, shard_index: int) -> tuple[dict, dict]:
    """Return validated checkpoint progress for one shard."""
    expected_rows = _shard_rows(plan, shard_index)
    expected_keys = {
        f"{row.request_id}|{row.month}" for row in expected_rows.itertuples(index=False)
    }
    checkpoint = load_checkpoint(shard_checkpoint_path(name, shard_index))
    actual_keys = set(checkpoint.get("groups", {}))
    unexpected = sorted(actual_keys - expected_keys)
    if unexpected:
        raise ValueError(
            f"shard {shard_index} checkpoint contains groups outside the plan: {unexpected}"
        )

    fetched = 0
    permanent_failed = 0
    for key, rec in checkpoint.get("groups", {}).items():
        status = rec.get("status") if isinstance(rec, dict) else None
        if status == "fetched":
            fetched += 1
        elif status == "failed" and rec.get("permanent"):
            permanent_failed += 1

    planned = int(len(expected_rows))
    pending = planned - fetched - permanent_failed
    if pending < 0:
        raise ValueError(f"shard {shard_index} checkpoint counts exceed the plan")

    summary = {
        "shard_index": int(shard_index),
        "planned_groups": planned,
        "fetched_groups_in_checkpoint": int(fetched),
        "permanent_failed_groups_in_checkpoint": int(permanent_failed),
        "pending_groups": int(pending),
        "cumulative_http_attempts": int(checkpoint["requests_used"]),
        "cumulative_bytes_reserved": int(checkpoint["bytes_used"]),
        "cumulative_bytes_measured": int(checkpoint.get("bytes_measured", 0)),
        "cumulative_seconds": float(checkpoint["seconds_used"]),
    }
    return checkpoint, summary


def cumulative_caps(
    checkpoint: dict,
    *,
    pending_groups: int,
    retry_headroom: int,
    additional_max_bytes: int,
    additional_max_seconds: float,
) -> dict:
    """Translate per-invocation headroom into execute_shard cumulative caps."""
    if pending_groups < 0:
        raise ValueError("pending_groups must be >= 0")
    if retry_headroom < 0:
        raise ValueError("retry_headroom must be >= 0")
    if additional_max_bytes <= 0:
        raise ValueError("additional_max_bytes must be > 0")
    if additional_max_seconds <= 0:
        raise ValueError("additional_max_seconds must be > 0")

    return {
        "max_requests": int(checkpoint["requests_used"]) + int(pending_groups) + int(retry_headroom),
        "max_bytes": int(checkpoint["bytes_used"]) + int(additional_max_bytes),
        "max_seconds": float(checkpoint["seconds_used"]) + float(additional_max_seconds),
    }


def run_shards(
    name: str,
    mapping_name: str,
    *,
    start_shard: int,
    end_shard: int | None,
    shard_size: int,
    max_stations_per_request: int,
    retry_headroom: int,
    additional_max_bytes: int,
    additional_max_seconds: float,
    status_only: bool = False,
) -> int:
    plan_manifest, plan = load_plan(
        name,
        mapping_name,
        shard_size=shard_size,
        max_stations_per_request=max_stations_per_request,
    )
    shard_count = int(plan_manifest["denominators"]["shard_count"])
    if shard_count <= 0:
        raise ValueError("request plan has no shards")
    if start_shard < 0 or start_shard >= shard_count:
        raise ValueError(f"start_shard must be in [0, {shard_count - 1}]")
    if end_shard is None:
        end_shard = shard_count - 1
    if end_shard < start_shard or end_shard >= shard_count:
        raise ValueError(
            f"end_shard must be in [{start_shard}, {shard_count - 1}]"
        )

    # Never leap over an incomplete earlier shard. This makes a resumed command
    # fail closed if local evidence has gaps or corruption.
    for idx in range(start_shard):
        verify_successful_shard(name, plan_manifest, plan, idx)

    for idx in range(start_shard, end_shard + 1):
        manifest_file = shard_manifest_path(name, idx)
        if manifest_file.exists():
            existing = json.loads(manifest_file.read_text())
            if existing.get("schema_version") != SHARD_MANIFEST_SCHEMA_VERSION:
                raise ValueError(f"shard {idx} has unsupported manifest schema")
            if existing.get("plan_sha256") != plan_manifest["plan_sha256"]:
                raise ValueError(f"shard {idx} manifest belongs to a different request plan")
            if existing.get("successful"):
                verify_successful_shard(name, plan_manifest, plan, idx)
                print(
                    json.dumps(
                        {
                            "event": "skip_successful_shard",
                            "shard_index": idx,
                            "request_groups": existing["request_groups_in_shard"],
                        }
                    ),
                    flush=True,
                )
                continue

        checkpoint, progress = checkpoint_progress(name, plan, idx)
        print(json.dumps({"event": "shard_preflight", **progress}), flush=True)

        if progress["permanent_failed_groups_in_checkpoint"]:
            print(
                json.dumps(
                    {
                        "event": "stop_permanent_failure",
                        "shard_index": idx,
                        "permanent_failed_groups": progress[
                            "permanent_failed_groups_in_checkpoint"
                        ],
                    }
                ),
                flush=True,
            )
            return 2

        if status_only:
            continue

        caps = cumulative_caps(
            checkpoint,
            pending_groups=progress["pending_groups"],
            retry_headroom=retry_headroom,
            additional_max_bytes=additional_max_bytes,
            additional_max_seconds=additional_max_seconds,
        )
        print(
            json.dumps(
                {"event": "execute_shard", "shard_index": idx, "caps": caps}
            ),
            flush=True,
        )
        result = execute_shard(
            name,
            mapping_name,
            idx,
            shard_size=shard_size,
            max_stations_per_request=max_stations_per_request,
            max_requests=caps["max_requests"],
            max_bytes=caps["max_bytes"],
            max_seconds=caps["max_seconds"],
        )
        print(
            json.dumps(
                {
                    "event": "shard_result",
                    "shard_index": idx,
                    "complete": bool(result["complete"]),
                    "successful": bool(result["successful"]),
                    "fetched_groups": int(result["fetched_groups"]),
                    "failed_groups": int(result["failed_groups"]),
                    "skipped_due_to_cap": int(result["skipped_due_to_cap"]),
                    "cap_that_stopped_collection": result[
                        "cap_that_stopped_collection"
                    ],
                    "cumulative_http_attempts": int(
                        result["cumulative_http_attempts"]
                    ),
                    "cumulative_bytes_downloaded": int(
                        result["cumulative_bytes_downloaded"]
                    ),
                    "cumulative_bytes_reserved_against_budget": int(
                        result["cumulative_bytes_reserved_against_budget"]
                    ),
                    "cumulative_active_fetch_seconds": float(
                        result["cumulative_active_fetch_seconds"]
                    ),
                }
            ),
            flush=True,
        )

        if not result["successful"]:
            print(
                json.dumps(
                    {
                        "event": "stop_unsuccessful_shard",
                        "shard_index": idx,
                        "message": "Later shards were not started.",
                    }
                ),
                flush=True,
            )
            return 2

        verify_successful_shard(name, plan_manifest, plan, idx)

    print(
        json.dumps(
            {
                "event": "range_complete",
                "start_shard": start_shard,
                "end_shard": end_shard,
                "status_only": bool(status_only),
            }
        ),
        flush=True,
    )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mapping-name", required=True)
    ap.add_argument("--start-shard", type=int, default=0)
    ap.add_argument("--end-shard", type=int)
    ap.add_argument("--shard-size", type=int, default=50)
    ap.add_argument("--max-stations-per-request", type=int, default=20)
    ap.add_argument("--retry-headroom", type=int, default=DEFAULT_RETRY_HEADROOM)
    ap.add_argument(
        "--additional-max-bytes",
        type=int,
        default=DEFAULT_ADDITIONAL_MAX_BYTES,
    )
    ap.add_argument(
        "--additional-max-seconds",
        type=float,
        default=DEFAULT_ADDITIONAL_MAX_SECONDS,
    )
    ap.add_argument("--status-only", action="store_true")
    args = ap.parse_args()

    code = run_shards(
        args.name,
        args.mapping_name,
        start_shard=args.start_shard,
        end_shard=args.end_shard,
        shard_size=args.shard_size,
        max_stations_per_request=args.max_stations_per_request,
        retry_headroom=args.retry_headroom,
        additional_max_bytes=args.additional_max_bytes,
        additional_max_seconds=args.additional_max_seconds,
        status_only=args.status_only,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
