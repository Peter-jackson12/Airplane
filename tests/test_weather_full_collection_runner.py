import json

import pandas as pd
import pytest

import notebooks.run_weather_full_collection as runner
from notebooks.fetch_weather_full_sharded import (
    SHARD_MANIFEST_SCHEMA_VERSION,
    WEATHER_FIELDS,
)
from notebooks.fetch_weather_sample import digest
from notebooks.fetch_weather_sample_expanded import new_checkpoint_state


def _minimal_plan(shards=(0, 1, 2)):
    rows = []
    for idx in shards:
        start = pd.Timestamp("2019-01-01T00:00:00Z") + pd.offsets.MonthBegin(idx)
        end = start + pd.offsets.MonthBegin(1)
        rows.append(
            {
                "ordinal": idx,
                "shard_index": idx,
                "request_id": f"REQ{idx}",
                "network": "AA_ASOS",
                "month": start.strftime("%Y-%m"),
                "batch_index": 0,
                "stations_json": json.dumps([f"S{idx}"], separators=(",", ":")),
                "station_count": 1,
                "window_start_utc": start,
                "window_end_utc": end,
                "station_years": 0.1,
            }
        )
    return pd.DataFrame(rows)


def test_cumulative_caps_add_headroom_to_existing_usage():
    checkpoint = new_checkpoint_state()
    checkpoint["requests_used"] = 7
    checkpoint["bytes_used"] = 123
    checkpoint["seconds_used"] = 4.5

    caps = runner.cumulative_caps(
        checkpoint,
        pending_groups=3,
        retry_headroom=2,
        additional_max_bytes=1000,
        additional_max_seconds=90,
    )

    assert caps == {
        "max_requests": 12,
        "max_bytes": 1123,
        "max_seconds": 94.5,
    }


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"pending_groups": -1}, "pending_groups"),
        ({"retry_headroom": -1}, "retry_headroom"),
        ({"additional_max_bytes": 0}, "additional_max_bytes"),
        ({"additional_max_seconds": 0}, "additional_max_seconds"),
    ],
)
def test_cumulative_caps_reject_invalid_headroom(kwargs, match):
    checkpoint = new_checkpoint_state()
    values = {
        "pending_groups": 1,
        "retry_headroom": 1,
        "additional_max_bytes": 1,
        "additional_max_seconds": 1.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        runner.cumulative_caps(checkpoint, **values)


def test_checkpoint_progress_counts_fetched_permanent_and_pending(tmp_path, monkeypatch):
    checkpoint = new_checkpoint_state()
    checkpoint["requests_used"] = 5
    checkpoint["bytes_used"] = 77
    checkpoint["bytes_measured"] = 55
    checkpoint["seconds_used"] = 12.5
    checkpoint["groups"] = {
        "REQ0|2019-01": {"status": "fetched"},
        "REQ1|2019-02": {"status": "failed", "permanent": True},
    }
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(checkpoint))
    monkeypatch.setattr(runner, "shard_checkpoint_path", lambda name, idx: path)

    plan = _minimal_plan((0, 1, 2)).copy()
    plan["shard_index"] = 0
    _, summary = runner.checkpoint_progress("run", plan, 0)

    assert summary["planned_groups"] == 3
    assert summary["fetched_groups_in_checkpoint"] == 1
    assert summary["permanent_failed_groups_in_checkpoint"] == 1
    assert summary["pending_groups"] == 1
    assert summary["cumulative_http_attempts"] == 5
    assert summary["cumulative_bytes_reserved"] == 77
    assert summary["cumulative_bytes_measured"] == 55
    assert summary["cumulative_seconds"] == 12.5


def test_verify_successful_shard_revalidates_cache_and_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    manifest_path = tmp_path / "shard.json"
    monkeypatch.setattr(
        runner, "shard_manifest_path", lambda name, idx: manifest_path
    )

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
            }
        ]
    )

    cache = tmp_path / "cache.csv"
    row = {field: "" for field in WEATHER_FIELDS}
    row.update({"station": "S0", "valid": "2019-01-15 12:00", "metar": "METAR S0"})
    pd.DataFrame([row], columns=["station", "valid", *WEATHER_FIELDS]).to_csv(
        cache, index=False
    )

    payload = {
        "schema_version": SHARD_MANIFEST_SCHEMA_VERSION,
        "plan_sha256": "planhash",
        "complete": True,
        "successful": True,
        "failed_groups": 0,
        "skipped_due_to_cap": 0,
        "request_groups_in_shard": 1,
        "requests": [
            {
                "request_id": "REQ0",
                "network": "AA_ASOS",
                "month": "2019-01",
                "stations": ["S0"],
                "window_start_utc": start.isoformat(),
                "window_end_utc": end.isoformat(),
                "cache_file": "cache.csv",
                "sha256": digest(cache),
                "rows": 1,
            }
        ],
    }
    manifest_path.write_text(json.dumps(payload))

    checked = runner.verify_successful_shard(
        "run", {"plan_sha256": "planhash"}, plan, 0
    )
    assert checked["successful"] is True

    cache.write_text(cache.read_text() + "\n")
    with pytest.raises(ValueError, match="cache hash mismatch"):
        runner.verify_successful_shard(
            "run", {"plan_sha256": "planhash"}, plan, 0
        )


def test_run_shards_stops_before_later_shards_after_unsuccessful_result(
    tmp_path, monkeypatch
):
    plan = _minimal_plan()
    manifest = {"denominators": {"shard_count": 3}, "plan_sha256": "planhash"}
    monkeypatch.setattr(runner, "load_plan", lambda *args, **kwargs: (manifest, plan))
    monkeypatch.setattr(
        runner,
        "shard_manifest_path",
        lambda name, idx: tmp_path / f"shard_{idx}.json",
    )

    verified = []
    monkeypatch.setattr(
        runner,
        "verify_successful_shard",
        lambda name, pm, p, idx, **kwargs: verified.append(idx) or {"successful": True},
    )

    checkpoint = new_checkpoint_state()
    progress = {
        "shard_index": 1,
        "planned_groups": 1,
        "fetched_groups_in_checkpoint": 0,
        "permanent_failed_groups_in_checkpoint": 0,
        "pending_groups": 1,
        "cumulative_http_attempts": 0,
        "cumulative_bytes_reserved": 0,
        "cumulative_bytes_measured": 0,
        "cumulative_seconds": 0.0,
    }
    monkeypatch.setattr(
        runner, "checkpoint_progress", lambda name, p, idx: (checkpoint, {**progress, "shard_index": idx})
    )

    executed = []

    def fake_execute(name, mapping_name, idx, **kwargs):
        executed.append(idx)
        return {
            "complete": False,
            "successful": False,
            "fetched_groups": 0,
            "failed_groups": 0,
            "skipped_due_to_cap": 1,
            "cap_that_stopped_collection": "synthetic cap",
            "cumulative_http_attempts": 1,
            "cumulative_bytes_downloaded": 1,
            "cumulative_bytes_reserved_against_budget": 1,
            "cumulative_active_fetch_seconds": 1.0,
        }

    monkeypatch.setattr(runner, "execute_shard", fake_execute)

    code = runner.run_shards(
        "run",
        "mapping",
        start_shard=1,
        end_shard=2,
        shard_size=50,
        max_stations_per_request=20,
        retry_headroom=10,
        additional_max_bytes=100,
        additional_max_seconds=100,
    )

    assert code == 2
    assert verified == [0]
    assert executed == [1]


def test_run_shards_skips_already_successful_shard_without_network(
    tmp_path, monkeypatch
):
    plan = _minimal_plan((0, 1))
    manifest = {"denominators": {"shard_count": 2}, "plan_sha256": "planhash"}
    monkeypatch.setattr(runner, "load_plan", lambda *args, **kwargs: (manifest, plan))

    current_manifest = tmp_path / "shard_1.json"
    current_manifest.write_text(
        json.dumps(
            {
                "schema_version": SHARD_MANIFEST_SCHEMA_VERSION,
                "plan_sha256": "planhash",
                "successful": True,
                "request_groups_in_shard": 1,
            }
        )
    )
    monkeypatch.setattr(
        runner,
        "shard_manifest_path",
        lambda name, idx: current_manifest if idx == 1 else tmp_path / "shard_0.json",
    )

    verified = []
    monkeypatch.setattr(
        runner,
        "verify_successful_shard",
        lambda name, pm, p, idx, **kwargs: verified.append(idx) or {"successful": True},
    )
    monkeypatch.setattr(
        runner,
        "execute_shard",
        lambda *args, **kwargs: pytest.fail("successful shard must not hit network"),
    )

    code = runner.run_shards(
        "run",
        "mapping",
        start_shard=1,
        end_shard=1,
        shard_size=50,
        max_stations_per_request=20,
        retry_headroom=10,
        additional_max_bytes=100,
        additional_max_seconds=100,
    )

    assert code == 0
    assert verified == [0, 1]



def test_run_shards_rejects_existing_manifest_from_different_plan(
    tmp_path, monkeypatch
):
    plan = _minimal_plan((0,))
    manifest = {"denominators": {"shard_count": 1}, "plan_sha256": "planhash"}
    monkeypatch.setattr(runner, "load_plan", lambda *args, **kwargs: (manifest, plan))

    current_manifest = tmp_path / "shard_0.json"
    current_manifest.write_text(
        json.dumps(
            {
                "schema_version": SHARD_MANIFEST_SCHEMA_VERSION,
                "plan_sha256": "different-plan",
                "successful": False,
            }
        )
    )
    monkeypatch.setattr(
        runner,
        "shard_manifest_path",
        lambda name, idx: current_manifest,
    )
    monkeypatch.setattr(
        runner,
        "execute_shard",
        lambda *args, **kwargs: pytest.fail("mismatched manifest must stop before network"),
    )

    with pytest.raises(ValueError, match="different request plan"):
        runner.run_shards(
            "run",
            "mapping",
            start_shard=0,
            end_shard=0,
            shard_size=50,
            max_stations_per_request=20,
            retry_headroom=10,
            additional_max_bytes=100,
            additional_max_seconds=100,
        )
