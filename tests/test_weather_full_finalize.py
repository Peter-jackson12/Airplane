import hashlib
import json

import pandas as pd
import pytest

import notebooks.fetch_weather_full_sharded as full
from notebooks.join_weather_sample import WEATHER_FIELDS


def _write_bulk(path, station, valid):
    row = {field: "" for field in WEATHER_FIELDS}
    row.update({"station": station, "valid": valid, "metar": f"METAR {station}"})
    pd.DataFrame([row], columns=["station", "valid", *WEATHER_FIELDS]).to_csv(
        path, index=False
    )


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_summarize_attempts_classifies_retries_and_interruption():
    records = [
        {
            "request_id": "REQ0",
            "attempts": 2,
            "attempts_detail": [
                {
                    "attempt": 1,
                    "success": False,
                    "bytes_received": None,
                    "seconds": 2.0,
                    "error": "curl exit 22: HTTP 503",
                },
                {
                    "attempt": 2,
                    "success": True,
                    "bytes_received": 100,
                    "seconds": 1.0,
                    "error": None,
                },
            ],
        },
        {
            "request_id": "REQ1",
            "attempts": 2,
            "attempts_detail": [
                {
                    "attempt": 1,
                    "success": False,
                    "bytes_received": None,
                    "seconds": None,
                    "error": "interrupted mid-attempt (process ended before the outcome was recorded)",
                },
                {
                    "attempt": 2,
                    "success": True,
                    "bytes_received": 200,
                    "seconds": 1.0,
                    "error": None,
                },
            ],
        },
    ]

    summary = full._summarize_attempts(records)

    assert summary["http_attempts_from_request_records"] == 4
    assert summary["successful_http_attempts"] == 2
    assert summary["failed_http_attempts"] == 2
    assert summary["retry_attempts"] == 2
    assert summary["unmeasured_http_attempts_from_request_records"] == 2
    assert summary["request_groups_with_network_attempts"] == 2
    assert summary["cache_only_request_groups"] == 0
    assert summary["provider_error_counts"]["http_503"] == 1
    assert summary["provider_error_counts"]["interrupted_unknown"] == 1


def test_summarize_attempts_rejects_declared_attempt_count_drift():
    with pytest.raises(ValueError, match="attempts_detail"):
        full._summarize_attempts(
            [{"request_id": "REQ", "attempts": 2, "attempts_detail": []}]
        )


def test_finalize_seals_complete_evidence_and_reconciles_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(full, "ROOT", tmp_path)

    start = pd.Timestamp("2019-01-01T00:00:00Z")
    end = pd.Timestamp("2019-02-01T00:00:00Z")
    plan = pd.DataFrame(
        [
            {
                "ordinal": 0,
                "shard_index": 0,
                "request_id": "REQ0",
                "network": "AA_ASOS",
                "month": "2019-01",
                "batch_index": 0,
                "stations_json": '["S0"]',
                "station_count": 1,
                "window_start_utc": start,
                "window_end_utc": end,
                "station_years": 0.1,
            },
            {
                "ordinal": 1,
                "shard_index": 0,
                "request_id": "REQ1",
                "network": "AA_ASOS",
                "month": "2019-01",
                "batch_index": 1,
                "stations_json": '["S1"]',
                "station_count": 1,
                "window_start_utc": start,
                "window_end_utc": end,
                "station_years": 0.1,
            },
        ],
        columns=full.PLAN_COLUMNS,
    )
    plan_manifest = {
        "plan_sha256": "plan-sha",
        "mapping_sha256": "mapping-sha",
        "denominators": {"shard_count": 1, "shard_size": 2},
        "provider_contract": {"checked_on": "2026-09-21"},
    }
    monkeypatch.setattr(
        full, "load_plan", lambda *args, **kwargs: (plan_manifest, plan)
    )

    plan_manifest_file = tmp_path / "plan_manifest.json"
    plan_manifest_file.write_text(json.dumps(plan_manifest))
    monkeypatch.setattr(
        full, "plan_manifest_path", lambda name: plan_manifest_file
    )

    final_file = tmp_path / "final.json"
    monkeypatch.setattr(full, "final_manifest_path", lambda name: final_file)

    cache0 = tmp_path / "cache0.csv"
    cache1 = tmp_path / "cache1.csv"
    _write_bulk(cache0, "S0", "2019-01-01 00:00")
    _write_bulk(cache1, "S1", "2019-01-02 00:00")

    req0 = {
        "request_id": "REQ0",
        "network": "AA_ASOS",
        "month": "2019-01",
        "batch_index": 0,
        "stations": ["S0"],
        "station_count": 1,
        "window_start_utc": start.isoformat(),
        "window_end_utc": end.isoformat(),
        "station_years": 0.1,
        "cache_file": "cache0.csv",
        "sha256": _sha(cache0),
        "rows": 1,
        "was_already_cached": False,
        "attempts": 2,
        "attempts_detail": [
            {
                "attempt": 1,
                "success": False,
                "bytes_received": None,
                "seconds": 2.0,
                "error": "curl exit 22: HTTP 503",
            },
            {
                "attempt": 2,
                "success": True,
                "bytes_received": cache0.stat().st_size,
                "seconds": 1.0,
                "error": None,
            },
        ],
        "recovered_from_existing_cache_after_incomplete_checkpoint": False,
        "status": "fetched",
    }
    req1 = {
        "request_id": "REQ1",
        "network": "AA_ASOS",
        "month": "2019-01",
        "batch_index": 1,
        "stations": ["S1"],
        "station_count": 1,
        "window_start_utc": start.isoformat(),
        "window_end_utc": end.isoformat(),
        "station_years": 0.1,
        "cache_file": "cache1.csv",
        "sha256": _sha(cache1),
        "rows": 1,
        "was_already_cached": False,
        "attempts": 2,
        "attempts_detail": [
            {
                "attempt": 1,
                "success": False,
                "bytes_received": None,
                "seconds": None,
                "error": "interrupted mid-attempt (process ended before the outcome was recorded)",
            },
            {
                "attempt": 2,
                "success": True,
                "bytes_received": cache1.stat().st_size,
                "seconds": 1.0,
                "error": None,
            },
        ],
        "recovered_from_existing_cache_after_incomplete_checkpoint": True,
        "status": "fetched",
    }

    shard = {
        "schema_version": full.SHARD_MANIFEST_SCHEMA_VERSION,
        "plan_sha256": "plan-sha",
        "successful": True,
        "request_groups_in_shard": 2,
        "cumulative_http_attempts": 4,
        "cumulative_bytes_downloaded": cache0.stat().st_size + cache1.stat().st_size,
        "cumulative_bytes_reserved_against_budget": (
            cache0.stat().st_size + cache1.stat().st_size + 40_000_000
        ),
        "cumulative_active_fetch_seconds": 10.0,
        "unmeasured_byte_attempts": 2,
        "cache_hit_groups": 0,
        "recovered_incomplete_checkpoint_groups": 1,
        "new_fetch_groups": 2,
        "requests": [req0, req1],
    }
    shard_file = tmp_path / "shard_0000_manifest.json"
    shard_file.write_text(json.dumps(shard))
    monkeypatch.setattr(
        full, "shard_manifest_path", lambda name, idx: shard_file
    )

    payload = full.finalize("baseline_recovery_v2_test", "mapping")

    assert payload["schema_version"] == full.FINAL_MANIFEST_SCHEMA_VERSION
    assert payload["bulk_request_groups"] == 2
    assert payload["cache_file_count"] == 2
    assert payload["total_cache_rows"] == 2
    assert payload["total_cache_bytes_on_disk"] == (
        cache0.stat().st_size + cache1.stat().st_size
    )
    assert payload["cumulative_http_attempts"] == 4
    assert payload["retry_attempts"] == 2
    assert payload["successful_http_attempts"] == 2
    assert payload["failed_http_attempts"] == 2
    assert payload["unmeasured_byte_attempts"] == 2
    assert payload["provider_error_counts"]["http_503"] == 1
    assert payload["provider_error_counts"]["interrupted_unknown"] == 1
    assert payload["recovered_incomplete_checkpoint_groups"] == 1
    assert final_file.exists()
    assert json.loads(final_file.read_text())["total_cache_rows"] == 2


def test_finalize_never_overwrites_existing_final_manifest(tmp_path, monkeypatch):
    final_file = tmp_path / "final.json"
    final_file.write_text('{"existing": true}')
    monkeypatch.setattr(full, "final_manifest_path", lambda name: final_file)
    monkeypatch.setattr(
        full,
        "load_plan",
        lambda *args, **kwargs: pytest.fail("existing final evidence must stop before loading"),
    )

    with pytest.raises(FileExistsError, match="never overwritten"):
        full.finalize("baseline_recovery_v2_test", "mapping")

    assert final_file.read_text() == '{"existing": true}'
