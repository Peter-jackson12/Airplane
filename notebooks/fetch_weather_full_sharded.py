"""Full adopted-row IEM weather collection via bounded bulk requests.

The 21/300-row validator intentionally requests one station/day-sized window.
That is a good correctness probe but a poor full-archive transport plan. IEM's
ASOS backend supports multiple stations per request, documents a 1 second
per-IP throttle, and enforces a 1,000 station-year practical request limit.

For the full adopted pool this module materializes the *actual network/month
bulk requests* before any network call:

* only confirmed_period rows with a resolved prediction_at create demand;
* every required [prediction_at-6h, prediction_at+2h] interval contributes its
  station to each UTC calendar month it touches;
* requests are grouped by IEM network/month and capped at a small station
  batch size (default 20), far below the provider's station-year ceiling;
* request groups are deterministically ordered and assigned to small
  checkpoint shards (default 50 requests);
* every shard has its own checkpoint, cache directory and local manifest;
* the existing checkpoint-v3 retry/crash/budget engine is reused, with a
  bulk-response byte reservation cap supplied explicitly;
* finalization succeeds only when every materialized request is complete with
  zero permanent failures and every cache hash still matches.

Existing 21/300/stratafix sample caches remain untouched. They are partial
station windows and are never treated as proof that a whole network/month bulk
request is already complete.

The row-level plan/checkpoints/shard manifests/raw IEM CSVs live under
data/weather_probe/<name>/ and are Git-ignored. Small plan/final manifests live
under output/. No target/delay/actual-outcome column is read. --prepare-only
makes no network request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from notebooks.fetch_weather_sample import (
    LOOKAHEAD_HOURS,
    LOOKBACK_HOURS,
    SUBPROCESS_TERMINATION_GRACE_SECONDS,
    digest,
    http_get_once,
)
from notebooks.fetch_weather_sample_expanded import (
    BACKOFF_BASE_SECONDS,
    MAX_ATTEMPTS_PER_GROUP,
    MAX_CONSECUTIVE_FAILURES,
    checkpoint_has_prior_progress,
    compute_plan_fingerprint,
    execute_with_caps,
    load_checkpoint,
    record_budget_limits,
    save_checkpoint,
    verify_plan_fingerprint,
)
from notebooks.join_weather_sample import WEATHER_FIELDS
from notebooks.scope_weather_collection_refined import ATTRIBUTION_RUN, load_adopted_pool

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
PROVIDER_HELP = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help"
PROVIDER_CONTRACT_CHECKED_ON = "2026-09-21"
PROVIDER_DOCUMENTED_THROTTLE_SECONDS = 1.0
PROVIDER_DOCUMENTED_STATION_YEAR_LIMIT = 1000.0

PLAN_SCHEMA_VERSION = 2
SHARD_MANIFEST_SCHEMA_VERSION = 2
FINAL_MANIFEST_SCHEMA_VERSION = 2
DEFAULT_SHARD_SIZE = 50
DEFAULT_MAX_STATIONS_PER_REQUEST = 20
BULK_SUCCESS_PAUSE_SECONDS = 1.25
BULK_ATTEMPT_TIMEOUT_SECONDS = 120.0
MAX_BULK_RESPONSE_BYTES = 20_000_000

PLAN_COLUMNS = [
    "ordinal",
    "shard_index",
    "request_id",
    "network",
    "month",
    "batch_index",
    "stations_json",
    "station_count",
    "window_start_utc",
    "window_end_utc",
    "station_years",
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
    if path.is_absolute():
        return _as_root_relative(path)
    return path.as_posix().replace("\\", "/")


def run_dir(name: str) -> Path:
    return ROOT / "data" / "weather_probe" / name


def plan_path(name: str) -> Path:
    return run_dir(name) / "bulk_request_plan.csv"


def plan_manifest_path(name: str) -> Path:
    return ROOT / "output" / f"{name}_full_weather_plan_manifest.json"


def final_manifest_path(name: str) -> Path:
    return ROOT / "output" / f"{name}_full_weather_fetch_manifest.json"


def shard_checkpoint_path(name: str, shard_index: int) -> Path:
    return run_dir(name) / "checkpoints" / f"shard_{shard_index:04d}.json"


def shard_manifest_path(name: str, shard_index: int) -> Path:
    return run_dir(name) / "shards" / f"shard_{shard_index:04d}_manifest.json"


def shard_cache_dir(name: str, shard_index: int) -> Path:
    return run_dir(name) / "cache" / f"shard_{shard_index:04d}"


def _month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{month}-01T00:00:00Z")
    end = start + pd.offsets.MonthBegin(1)
    return start, end


def _request_id(network: str, month: str, stations: list[str]) -> str:
    payload = f"{network}|{month}|{','.join(stations)}"
    suffix = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"{network}_{month.replace('-', '')}_{suffix}"


def _station_batches(stations: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("max_stations_per_request must be > 0")
    return [stations[i:i + size] for i in range(0, len(stations), size)]


def build_bulk_month_plan(
    pool: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
    max_stations_per_request: int = DEFAULT_MAX_STATIONS_PER_REQUEST,
) -> tuple[pd.DataFrame, dict]:
    """Materialize provider-facing network/month/station-batch requests.

    Every row-level weather window is contained by the union of the month
    requests created for its station. The transport deliberately fetches a
    whole UTC month once a station is needed in that month: this trades some
    extra bytes for dramatically fewer provider requests while preserving the
    same downstream point-in-time join contract.
    """
    if shard_size <= 0:
        raise ValueError("shard_size must be > 0")
    if max_stations_per_request <= 0:
        raise ValueError("max_stations_per_request must be > 0")

    required_pool = {
        "row_tier",
        "prediction_at",
        "Origin_Airport",
        "Destination_Airport",
    }
    missing_pool = required_pool - set(pool.columns)
    if missing_pool:
        raise ValueError(f"pool is missing required columns: {sorted(missing_pool)}")
    required_mapping = {"iata", "candidate_sid", "matched_network", "verification_tier"}
    missing_mapping = required_mapping - set(mapping.columns)
    if missing_mapping:
        raise ValueError(f"mapping is missing required columns: {sorted(missing_mapping)}")

    lookup = mapping.set_index("iata")
    mapping_eligible = pool["row_tier"].eq("confirmed_period")
    utc_resolved = mapping_eligible & pool["prediction_at"].notna()
    safe = pool.loc[
        utc_resolved,
        ["Origin_Airport", "Destination_Airport", "prediction_at"],
    ].copy()
    safe["prediction_at"] = pd.to_datetime(safe["prediction_at"], utc=True)

    def role_events(airport_col: str) -> pd.DataFrame:
        events = pd.DataFrame(
            {
                "station": safe[airport_col].map(lookup["candidate_sid"]),
                "network": safe[airport_col].map(lookup["matched_network"]),
                "prediction_at": safe["prediction_at"],
            }
        )
        if events[["station", "network"]].isna().any().any():
            raise ValueError("confirmed_period rows must have station and matched_network")
        return events

    events = pd.concat(
        [role_events("Origin_Airport"), role_events("Destination_Airport")],
        ignore_index=True,
    )
    events["window_start"] = events["prediction_at"] - pd.Timedelta(hours=LOOKBACK_HOURS)
    events["window_end"] = events["prediction_at"] + pd.Timedelta(hours=LOOKAHEAD_HOURS)
    events["start_month"] = events["window_start"].dt.strftime("%Y-%m")
    events["end_month"] = events["window_end"].dt.strftime("%Y-%m")

    start_pairs = events[["network", "station", "start_month"]].rename(
        columns={"start_month": "month"}
    )
    end_pairs = events.loc[
        events["end_month"].ne(events["start_month"]),
        ["network", "station", "end_month"],
    ].rename(columns={"end_month": "month"})
    station_month = (
        pd.concat([start_pairs, end_pairs], ignore_index=True)
        .drop_duplicates()
        .sort_values(["month", "network", "station"], kind="mergesort")
        .reset_index(drop=True)
    )

    rows: list[dict] = []
    for (month, network), sub in station_month.groupby(["month", "network"], sort=True):
        stations = sorted(sub["station"].astype(str).unique().tolist())
        start, end = _month_bounds(str(month))
        days = (end - start).total_seconds() / 86400.0
        for batch_index, batch in enumerate(_station_batches(stations, max_stations_per_request)):
            station_years = len(batch) * days / 365.2425
            if station_years >= PROVIDER_DOCUMENTED_STATION_YEAR_LIMIT:
                raise ValueError(
                    f"planned request exceeds provider station-year limit: "
                    f"{network} {month} {station_years:.2f}"
                )
            rows.append(
                {
                    "request_id": _request_id(str(network), str(month), batch),
                    "network": str(network),
                    "month": str(month),
                    "batch_index": int(batch_index),
                    "stations_json": json.dumps(batch, separators=(",", ":")),
                    "station_count": int(len(batch)),
                    "window_start_utc": start,
                    "window_end_utc": end,
                    "station_years": float(station_years),
                }
            )

    plan = pd.DataFrame(rows)
    if len(plan):
        plan = plan.sort_values(
            ["station_count", "month", "network", "batch_index"],
            ascending=[False, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        if plan["request_id"].duplicated().any():
            raise ValueError("bulk request_id collision")
        plan.insert(0, "ordinal", range(len(plan)))
        plan.insert(1, "shard_index", plan["ordinal"] // shard_size)
        plan = plan[PLAN_COLUMNS]
    else:
        plan = pd.DataFrame(columns=PLAN_COLUMNS)

    denominators = {
        "total_adopted_rows": int(len(pool)),
        "mapping_eligible_rows_confirmed_period_both_ends": int(mapping_eligible.sum()),
        "utc_time_resolved_rows": int(utc_resolved.sum()),
        "mapping_eligible_but_utc_unresolved_rows": int((mapping_eligible & ~utc_resolved).sum()),
        "not_mapping_eligible_rows": int((~mapping_eligible).sum()),
        "distinct_stations": int(station_month["station"].nunique()),
        "distinct_networks": int(station_month["network"].nunique()),
        "station_month_pairs": int(len(station_month)),
        "bulk_request_groups": int(len(plan)),
        "shard_size": int(shard_size),
        "shard_count": int(plan["shard_index"].max() + 1) if len(plan) else 0,
        "max_stations_per_request": int(max_stations_per_request),
        "max_station_years_in_one_request": (
            float(plan["station_years"].max()) if len(plan) else 0.0
        ),
        "total_station_years_requested": (
            float(plan["station_years"].sum()) if len(plan) else 0.0
        ),
    }
    return plan, denominators


def request_from_row(row) -> dict:
    return {
        "request_id": str(row.request_id),
        "network": str(row.network),
        "month": str(row.month),
        "batch_index": int(row.batch_index),
        "stations": list(json.loads(row.stations_json)),
        "station_count": int(row.station_count),
        "window_start_utc": pd.Timestamp(row.window_start_utc),
        "window_end_utc": pd.Timestamp(row.window_end_utc),
        "station_years": float(row.station_years),
    }


def requests_for_shard(
    plan: pd.DataFrame, shard_index: int
) -> tuple[list[tuple[str, str]], dict, dict[str, dict]]:
    shard = plan.loc[plan["shard_index"].eq(shard_index)].sort_values("ordinal")
    if shard.empty:
        raise ValueError(f"shard_index={shard_index} is outside the plan")

    planned: list[tuple[str, str]] = []
    groups: dict[tuple[str, str], list[pd.Timestamp]] = {}
    requests: dict[str, dict] = {}
    for row in shard.itertuples(index=False):
        req = request_from_row(row)
        rid = req["request_id"]
        if rid in requests:
            raise ValueError(f"duplicate request_id in shard: {rid}")
        # execute_with_caps derives windows as min(ts)-LOOKBACK .. max(ts)+LOOKAHEAD.
        # These synthetic anchors reproduce the materialized month bounds exactly.
        anchors = [
            req["window_start_utc"] + pd.Timedelta(hours=LOOKBACK_HOURS),
            req["window_end_utc"] - pd.Timedelta(hours=LOOKAHEAD_HOURS),
        ]
        key = (rid, req["month"])
        groups[key] = anchors
        planned.append(key)
        requests[rid] = req
    return planned, groups, requests


def build_bulk_request_url(req: dict) -> str:
    params = {
        "network": req["network"],
        "station": req["stations"],
        "data": WEATHER_FIELDS,
        "sts": req["window_start_utc"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ets": req["window_end_utc"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tz": "UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "trace": "0.0001",
        "report_type": [3, 4],
    }
    return ENDPOINT + "?" + urlencode(params, doseq=True)


def bulk_cache_path(req: dict, cache_dir: Path) -> Path:
    return cache_dir / f"iem_bulk_{req['network']}_{req['month'].replace('-', '')}_{req['request_id'][-12:]}.csv"


def _validate_bulk_file(path: Path, req: dict) -> int:
    try:
        header = pd.read_csv(
            path,
            comment="#",
            na_values=["M"],
            keep_default_na=False,
            nrows=0,
        )
    except Exception as exc:
        raise ValueError(f"{path} failed to parse ({exc})") from exc
    required = {"station", "valid", *WEATHER_FIELDS}
    missing = required - set(header.columns)
    if missing:
        raise ValueError(f"{path} is missing expected IEM columns: {sorted(missing)}")

    keys = pd.read_csv(
        path,
        comment="#",
        na_values=["M"],
        keep_default_na=False,
        usecols=["station", "valid"],
        dtype={"station": str},
    )
    allowed = set(req["stations"])
    unexpected = sorted(set(keys["station"].dropna().astype(str)) - allowed)
    if unexpected:
        raise ValueError(
            f"{path} returned stations not present in the request: {unexpected}"
        )
    if len(keys):
        parsed = pd.to_datetime(keys["valid"], utc=True, errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"{path} contains unparseable valid timestamps")
    return int(len(keys))


def validate_bulk_cache(req: dict, cache_dir: Path) -> Path | None:
    path = bulk_cache_path(req, cache_dir)
    if not path.exists():
        return None
    _validate_bulk_file(path, req)
    return path


def fetch_bulk_attempt(
    req: dict,
    cache_dir: Path,
    *,
    timeout: float,
    max_bytes_remaining: int,
) -> dict:
    cache = bulk_cache_path(req, cache_dir)
    part = cache.with_name(cache.name + ".part")
    effective_cap = min(MAX_BULK_RESPONSE_BYTES, int(max_bytes_remaining))
    if effective_cap <= 0:
        return {
            "success": False,
            "bytes_received": None,
            "seconds": 0.0,
            "error": "no remaining byte budget for this attempt",
            "url": build_bulk_request_url(req),
        }

    url = build_bulk_request_url(req)
    outcome = http_get_once(url, timeout=timeout, out_path=part, max_bytes=effective_cap)
    if not outcome["success"]:
        part.unlink(missing_ok=True)
        return {
            "success": False,
            "bytes_received": outcome["bytes_received"],
            "seconds": outcome["seconds"],
            "error": outcome["error"],
            "url": url,
        }
    if outcome["bytes_received"] is not None and outcome["bytes_received"] > effective_cap:
        part.unlink(missing_ok=True)
        return {
            "success": False,
            "bytes_received": outcome["bytes_received"],
            "seconds": outcome["seconds"],
            "error": "response exceeds bounded size",
            "url": url,
        }
    try:
        n_rows = _validate_bulk_file(part, req)
    except Exception as exc:
        part.unlink(missing_ok=True)
        return {
            "success": False,
            "bytes_received": outcome["bytes_received"],
            "seconds": outcome["seconds"],
            "error": f"response failed validation: {exc}",
            "url": url,
        }

    cache.parent.mkdir(parents=True, exist_ok=True)
    part.replace(cache)
    return {
        "success": True,
        "bytes_received": outcome["bytes_received"],
        "seconds": outcome["seconds"],
        "url": url,
        "cache_path": cache,
        "cache_file": _as_root_relative(cache),
        "n_rows": n_rows,
        "sha256": digest(cache),
    }


def _source_manifest_row_sha() -> str:
    path = ROOT / "output" / f"{ATTRIBUTION_RUN}_manifest.json"
    return str(json.loads(path.read_text())["row_sha256"])


def _bulk_fetch_policy(plan_manifest: dict) -> dict:
    return {
        "request_unit": "IEM network x UTC month x station batch",
        "data_fields": WEATHER_FIELDS,
        "report_types": [3, 4],
        "missing_representation": "M",
        "trace_representation": "0.0001",
        "max_stations_per_request": int(
            plan_manifest["denominators"]["max_stations_per_request"]
        ),
        "max_response_bytes_per_attempt": MAX_BULK_RESPONSE_BYTES,
        "attempt_timeout_seconds": BULK_ATTEMPT_TIMEOUT_SECONDS,
        "max_attempts_per_group": MAX_ATTEMPTS_PER_GROUP,
        "max_consecutive_failures": MAX_CONSECUTIVE_FAILURES,
        "backoff_base_seconds": BACKOFF_BASE_SECONDS,
        "subprocess_termination_grace_seconds": SUBPROCESS_TERMINATION_GRACE_SECONDS,
        "success_pause_seconds": BULK_SUCCESS_PAUSE_SECONDS,
        "provider_documented_throttle_seconds": PROVIDER_DOCUMENTED_THROTTLE_SECONDS,
        "provider_documented_station_year_limit": PROVIDER_DOCUMENTED_STATION_YEAR_LIMIT,
        "provider_contract_checked_on": PROVIDER_CONTRACT_CHECKED_ON,
        "provider_help": PROVIDER_HELP,
    }


def prepare_plan(
    name: str,
    mapping_name: str,
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
    max_stations_per_request: int = DEFAULT_MAX_STATIONS_PER_REQUEST,
) -> dict:
    _validate_name(name)
    local_plan = plan_path(name)
    out_manifest = plan_manifest_path(name)
    if local_plan.exists() or out_manifest.exists():
        raise FileExistsError(
            "Full-weather plan evidence already exists. Reuse it unchanged or use a fresh --name; "
            "existing plan evidence is never overwritten."
        )

    mapping_path = ROOT / "output" / f"{mapping_name}_mapping_table.csv"
    mapping = pd.read_csv(mapping_path)
    pool = load_adopted_pool(mapping_name)
    plan, denominators = build_bulk_month_plan(
        pool,
        mapping,
        shard_size=shard_size,
        max_stations_per_request=max_stations_per_request,
    )

    local_plan.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(local_plan, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    raw_path = ROOT / "data" / "train.csv"

    manifest = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "name": name,
        "mapping_name": mapping_name,
        "attribution_run": ATTRIBUTION_RUN,
        "attribution_row_sha256": _source_manifest_row_sha(),
        "raw_train_sha256": digest(raw_path),
        "mapping_sha256": digest(mapping_path),
        "code_sha256": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "plan_file": _as_root_relative(local_plan),
        "plan_sha256": digest(local_plan),
        "plan_columns": PLAN_COLUMNS,
        "plan_order": "station_count descending (pilot stress first), then UTC month/network/batch ascending",
        "denominators": denominators,
        "provider_contract": {
            "checked_on": PROVIDER_CONTRACT_CHECKED_ON,
            "help_url": PROVIDER_HELP,
            "documented_per_ip_throttle_seconds": PROVIDER_DOCUMENTED_THROTTLE_SECONDS,
            "documented_station_year_limit_per_request": PROVIDER_DOCUMENTED_STATION_YEAR_LIMIT,
            "planned_success_pause_seconds": BULK_SUCCESS_PAUSE_SECONDS,
        },
        "transport_contract": {
            "request_unit": "network_month_station_batch",
            "whole_month_fetch": True,
            "weather_fields": WEATHER_FIELDS,
            "report_types": [3, 4],
            "missing_representation": "M",
            "trace_representation": "0.0001",
            "max_response_bytes_per_attempt": MAX_BULK_RESPONSE_BYTES,
            "attempt_timeout_seconds": BULK_ATTEMPT_TIMEOUT_SECONDS,
            "legacy_partial_cache_policy": (
                "21/300/stratafix station-window caches are preserved but do not skip any "
                "network/month bulk request; partial overlap is not full bulk coverage"
            ),
            "checkpoint_policy": (
                "one checkpoint per fixed-size request shard; shard_size and plan are immutable "
                "under one run name"
            ),
        },
        "fits_any_model": False,
        "uses_external_data": False,
        "target_columns_used": [],
        "limitations": [
            "prepare-only performs no network request. Actual bytes, elapsed time, 503/422 behavior "
            "and archive completeness remain unmeasured until the pilot shard runs.",
            "The transport intentionally downloads whole UTC months for station-months that contain "
            "at least one required row-level weather window. This can download observations not "
            "ultimately selected by the 90-minute as-of join; it reduces provider request count, not "
            "the prediction-time information boundary.",
            "Provider limits are documentation observed on the checked_on date, not a permanent "
            "guarantee. Re-verify the backend help before a materially later rerun.",
            "Only confirmed_period rows with a resolved prediction_at create fetch demand; all adopted "
            "rows remain represented in the denominator summary.",
        ],
    }
    _json_atomic(out_manifest, manifest)
    return manifest


def load_plan(
    name: str,
    mapping_name: str,
    *,
    shard_size: int | None = None,
    max_stations_per_request: int | None = None,
) -> tuple[dict, pd.DataFrame]:
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

    den = manifest["denominators"]
    if shard_size is not None and int(shard_size) != int(den["shard_size"]):
        raise ValueError(
            f"plan shard_size={den['shard_size']}, requested {shard_size}; "
            "shard boundaries are immutable under one run name"
        )
    if (
        max_stations_per_request is not None
        and int(max_stations_per_request) != int(den["max_stations_per_request"])
    ):
        raise ValueError(
            f"plan max_stations_per_request={den['max_stations_per_request']}, requested "
            f"{max_stations_per_request}; request boundaries are immutable under one run name"
        )

    local_plan = ROOT / manifest["plan_file"]
    if not local_plan.exists():
        raise FileNotFoundError(f"local request plan is missing: {local_plan}")
    if digest(local_plan) != manifest["plan_sha256"]:
        raise ValueError("local request plan hash no longer matches the recorded plan manifest")

    plan = pd.read_csv(
        local_plan,
        parse_dates=["window_start_utc", "window_end_utc"],
    )
    if list(plan.columns) != PLAN_COLUMNS:
        raise ValueError("request plan columns no longer match the recorded schema")
    if len(plan) != int(den["bulk_request_groups"]):
        raise ValueError("request plan row count no longer matches the manifest")
    if len(plan) and plan["ordinal"].tolist() != list(range(len(plan))):
        raise ValueError("request plan ordinals are not contiguous")
    return manifest, plan


def execute_shard(
    name: str,
    mapping_name: str,
    shard_index: int,
    *,
    shard_size: int | None,
    max_stations_per_request: int | None,
    max_requests: int,
    max_bytes: int,
    max_seconds: float,
) -> dict:
    plan_manifest, plan = load_plan(
        name,
        mapping_name,
        shard_size=shard_size,
        max_stations_per_request=max_stations_per_request,
    )
    shard_count = int(plan_manifest["denominators"]["shard_count"])
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count - 1}]")

    planned, groups, requests = requests_for_shard(plan, shard_index)
    checkpoint_file = shard_checkpoint_path(name, shard_index)
    cache_dir = shard_cache_dir(name, shard_index)

    checkpoint = load_checkpoint(checkpoint_file)
    resuming = checkpoint_has_prior_progress(checkpoint)
    if not resuming and cache_dir.exists() and any(cache_dir.glob("*.csv")):
        raise ValueError(
            f"{cache_dir} contains bulk cache files but the shard checkpoint has no prior progress; "
            "do not silently adopt orphan evidence. Restore the matching checkpoint or use a fresh run name."
        )
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    options = _bulk_fetch_policy(plan_manifest)
    options.update(
        {
            "full_plan_schema_version": PLAN_SCHEMA_VERSION,
            "full_plan_sha256": plan_manifest["plan_sha256"],
            "shard_index": int(shard_index),
            "shard_size": int(plan_manifest["denominators"]["shard_size"]),
        }
    )
    fingerprint = compute_plan_fingerprint(
        plan_manifest["plan_sha256"],
        plan_manifest["mapping_sha256"],
        groups,
        options,
    )
    verify_plan_fingerprint(
        checkpoint, fingerprint, name=f"{name}/shard_{shard_index:04d}"
    )
    caps = record_budget_limits(
        checkpoint,
        max_requests=max_requests,
        max_bytes=max_bytes,
        max_seconds=max_seconds,
    )
    save_checkpoint(checkpoint_file, checkpoint)

    def cache_exists(
        request_id: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> Path | None:
        req = requests[request_id]
        if start != req["window_start_utc"] or end != req["window_end_utc"]:
            raise ValueError(f"executor window differs from materialized plan for {request_id}")
        return validate_bulk_cache(req, cache_dir)

    def attempt(
        request_id: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        timeout: float,
        max_bytes_remaining: int,
    ) -> dict:
        req = requests[request_id]
        if start != req["window_start_utc"] or end != req["window_end_utc"]:
            raise ValueError(f"executor window differs from materialized plan for {request_id}")
        return fetch_bulk_attempt(
            req,
            cache_dir,
            timeout=timeout,
            max_bytes_remaining=max_bytes_remaining,
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
        now_fn=time.monotonic,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_file,
        sleep_fn=time.sleep,
        progress_every=10,
        max_attempts_per_group=MAX_ATTEMPTS_PER_GROUP,
        max_consecutive_failures=MAX_CONSECUTIVE_FAILURES,
        default_attempt_timeout=BULK_ATTEMPT_TIMEOUT_SECONDS,
        attempt_overhead_seconds=SUBPROCESS_TERMINATION_GRACE_SECONDS,
        success_pause_seconds=BULK_SUCCESS_PAUSE_SECONDS,
        max_response_bytes_per_attempt=MAX_BULK_RESPONSE_BYTES,
    )

    fetched = []
    for rec in result["fetched"]:
        request_id = str(rec["station"])
        req = requests[request_id]
        item = {
            **req,
            "window_start_utc": req["window_start_utc"].isoformat(),
            "window_end_utc": req["window_end_utc"].isoformat(),
            "cache_file": _normalize_cache_file(rec["cache_file"]),
            "sha256": rec["sha256"],
            "rows": rec.get("rows"),
            "was_already_cached": bool(rec.get("was_already_cached")),
            "attempts": int(rec.get("attempts", 0)),
            "attempts_detail": rec.get("attempts_detail", []),
            "status": "fetched",
        }
        fetched.append(item)

    failed_ids = {str(rec.get("station")) for rec in result["failed"] if rec.get("station")}
    complete = result["cap_hit"] is None and not result["skipped_cap"]
    successful = (
        complete
        and not result["failed"]
        and not failed_ids
        and len(fetched) == len(planned)
    )
    payload = {
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
        "cumulative_bytes_reserved_against_budget": int(
            result["cumulative_bytes_reserved"]
        ),
        "cumulative_active_fetch_seconds": float(result["cumulative_seconds"]),
        "unmeasured_byte_attempts": int(result["unmeasured_byte_attempts"]),
        "cache_hit_groups": int(sum(1 for f in fetched if f["was_already_cached"])),
        "new_fetch_groups": int(sum(1 for f in fetched if not f["was_already_cached"])),
        "checkpoint_file": _as_root_relative(checkpoint_file),
        "cache_directory": _as_root_relative(cache_dir),
        "requests": fetched,
        "failed": result["failed"],
        "skipped": result["skipped_cap"],
    }
    _json_atomic(shard_manifest_path(name, shard_index), payload)
    return payload


def _expected_requests(plan: pd.DataFrame) -> dict[str, dict]:
    return {
        req["request_id"]: req
        for req in (request_from_row(row) for row in plan.itertuples(index=False))
    }


def finalize(name: str, mapping_name: str, *, verify_cache_hashes: bool = True) -> dict:
    plan_manifest, plan = load_plan(name, mapping_name)
    expected = _expected_requests(plan)
    shard_count = int(plan_manifest["denominators"]["shard_count"])

    actual_ids: set[str] = set()
    all_requests: list[dict] = []
    shard_index_rows: list[dict] = []
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
            raise ValueError(
                f"cannot finalize: missing shard manifest {idx}/{shard_count - 1}"
            )
        shard = json.loads(path.read_text())
        if shard.get("schema_version") != SHARD_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"shard {idx} has unsupported manifest schema")
        if shard.get("plan_sha256") != plan_manifest["plan_sha256"]:
            raise ValueError(f"shard {idx} belongs to a different request plan")
        if not shard.get("successful"):
            raise ValueError(
                f"cannot finalize: shard {idx} is not successful "
                f"(cap={shard.get('cap_that_stopped_collection')!r}, "
                f"failed={shard.get('failed_groups')})"
            )

        for rec in shard["requests"]:
            rid = str(rec["request_id"])
            if rid in actual_ids:
                raise ValueError(f"duplicate completed request across shards: {rid}")
            if rid not in expected:
                raise ValueError(f"shard {idx} contains request not present in plan: {rid}")
            exp = expected[rid]
            if rec["network"] != exp["network"] or rec["month"] != exp["month"]:
                raise ValueError(f"shard {idx} request identity differs from plan for {rid}")
            if list(rec["stations"]) != list(exp["stations"]):
                raise ValueError(f"shard {idx} station batch differs from plan for {rid}")
            if (
                str(rec["window_start_utc"]) != exp["window_start_utc"].isoformat()
                or str(rec["window_end_utc"]) != exp["window_end_utc"].isoformat()
            ):
                raise ValueError(f"shard {idx} request window differs from plan for {rid}")

            cache = ROOT / rec["cache_file"]
            if not cache.exists():
                raise ValueError(f"shard {idx} cache file is missing: {cache}")
            _validate_bulk_file(cache, exp)
            if verify_cache_hashes and digest(cache) != rec["sha256"]:
                raise ValueError(f"shard {idx} cache hash mismatch for {rid}")

            actual_ids.add(rid)
            all_requests.append(rec)

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

    if actual_ids != set(expected):
        missing = len(set(expected) - actual_ids)
        extra = len(actual_ids - set(expected))
        raise ValueError(
            f"completed shard requests do not equal plan: missing={missing}, extra={extra}"
        )

    all_requests.sort(key=lambda r: expected[r["request_id"]]["month"] + "|" + expected[r["request_id"]]["network"] + "|" + r["request_id"])
    payload = {
        "schema_version": FINAL_MANIFEST_SCHEMA_VERSION,
        "name": name,
        "mapping_name": mapping_name,
        "plan_manifest": _as_root_relative(plan_manifest_path(name)),
        "plan_manifest_sha256": digest(plan_manifest_path(name)),
        "plan_sha256": plan_manifest["plan_sha256"],
        "bulk_request_groups": int(len(plan)),
        "shard_size": int(plan_manifest["denominators"]["shard_size"]),
        "shard_count": shard_count,
        "all_shards_complete": True,
        "failed_groups": 0,
        "cache_hashes_reverified": bool(verify_cache_hashes),
        **totals,
        "requests": all_requests,
        "shards": shard_index_rows,
        "provider_contract": plan_manifest["provider_contract"],
        "fits_any_model": False,
        "uses_external_data": True,
        "target_columns_used": [],
        "limitations": [
            "This proves completion of the materialized IEM archive transport plan, not historical "
            "publication latency, row-level join coverage, or model performance.",
            "Whole-month bulk files intentionally contain observations that may never be selected by "
            "the later 90-minute as-of join.",
            "Legacy sample caches remain separate evidence and were not used to skip partial bulk windows.",
            "The final manifest request list is directly consumable by the existing load_observations "
            "shape because each entry records cache_file and sha256; row-level full joining remains a "
            "separate stage.",
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
    ap.add_argument(
        "--max-stations-per-request",
        type=int,
        default=DEFAULT_MAX_STATIONS_PER_REQUEST,
    )
    ap.add_argument("--max-requests", type=int, default=150)
    ap.add_argument("--max-bytes", type=int, default=1_000_000_000)
    ap.add_argument("--max-seconds", type=float, default=7200.0)
    ap.add_argument("--skip-final-cache-hash-check", action="store_true")
    args = ap.parse_args()
    _validate_name(args.name)

    modes = int(args.prepare_only) + int(args.shard_index is not None) + int(args.finalize)
    if modes != 1:
        raise ValueError(
            "Choose exactly one of --prepare-only, --shard-index N, or --finalize"
        )

    if args.prepare_only:
        result = prepare_plan(
            args.name,
            args.mapping_name,
            shard_size=args.shard_size,
            max_stations_per_request=args.max_stations_per_request,
        )
        print(json.dumps(result, indent=2, default=str))
        return

    if args.finalize:
        result = finalize(
            args.name,
            args.mapping_name,
            verify_cache_hashes=not args.skip_final_cache_hash_check,
        )
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in {"requests", "shards"}},
                indent=2,
                default=str,
            )
        )
        return

    result = execute_shard(
        args.name,
        args.mapping_name,
        args.shard_index,
        shard_size=args.shard_size,
        max_stations_per_request=args.max_stations_per_request,
        max_requests=args.max_requests,
        max_bytes=args.max_bytes,
        max_seconds=args.max_seconds,
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"requests", "failed", "skipped"}
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
