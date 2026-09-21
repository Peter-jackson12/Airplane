import json
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from notebooks.fetch_weather_full_sharded import (
    MAX_BULK_RESPONSE_BYTES,
    WEATHER_FIELDS,
    _validate_bulk_file,
    build_bulk_month_plan,
    build_bulk_request_url,
    requests_for_shard,
)
from notebooks.fetch_weather_sample import LOOKAHEAD_HOURS, LOOKBACK_HOURS
from notebooks.fetch_weather_sample_expanded import execute_with_caps, new_checkpoint_state


def _mapping() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "iata": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "candidate_sid": ["A1", "B1", "C1", "D1", "E1"],
            "matched_network": ["AA_ASOS", "AA_ASOS", "AA_ASOS", "BB_ASOS", "BB_ASOS"],
            "verification_tier": [
                "confirmed_period",
                "confirmed_period",
                "confirmed_period",
                "confirmed_period",
                "confirmed_period",
            ],
        }
    )


def _pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_tier": [
                "confirmed_period",
                "confirmed_period",
                "unconfirmed",
                "confirmed_period",
            ],
            "prediction_at": pd.to_datetime(
                [
                    "2019-01-15T12:00:00Z",
                    "2019-01-31T23:30:00Z",  # +2h crosses into February
                    "2019-01-20T12:00:00Z",
                    None,
                ],
                utc=True,
            ),
            "Origin_Airport": ["AAA", "CCC", "AAA", "EEE"],
            "Destination_Airport": ["BBB", "DDD", "DDD", "AAA"],
        }
    )


def test_bulk_plan_groups_by_network_month_and_station_batches():
    plan, den = build_bulk_month_plan(
        _pool(), _mapping(), shard_size=2, max_stations_per_request=2
    )

    assert den["total_adopted_rows"] == 4
    assert den["mapping_eligible_rows_confirmed_period_both_ends"] == 3
    assert den["utc_time_resolved_rows"] == 2
    assert den["mapping_eligible_but_utc_unresolved_rows"] == 1
    assert den["not_mapping_eligible_rows"] == 1
    assert den["max_stations_per_request"] == 2
    assert den["bulk_request_groups"] == len(plan)
    assert den["shard_count"] == (len(plan) + 1) // 2
    assert plan.station_count.max() <= 2
    assert plan.station_years.max() < 1000

    requests = [
        (r.network, r.month, tuple(json.loads(r.stations_json)))
        for r in plan.itertuples(index=False)
    ]
    # First resolved row needs A1/B1 in January.
    assert ("AA_ASOS", "2019-01", ("A1", "B1")) in requests
    # Second resolved row's +2h window crosses month boundary, so both of its
    # stations are requested in January and February.
    assert any(
        net == "AA_ASOS" and month == "2019-02" and "C1" in stations
        for net, month, stations in requests
    )
    assert any(
        net == "BB_ASOS" and month == "2019-02" and "D1" in stations
        for net, month, stations in requests
    )


def test_bulk_plan_is_deterministic_under_row_reordering():
    a, den_a = build_bulk_month_plan(
        _pool(), _mapping(), shard_size=2, max_stations_per_request=2
    )
    b, den_b = build_bulk_month_plan(
        _pool().sample(frac=1, random_state=7),
        _mapping().sample(frac=1, random_state=11),
        shard_size=2,
        max_stations_per_request=2,
    )
    pd.testing.assert_frame_equal(a, b)
    assert den_a == den_b
    assert a.station_count.tolist() == sorted(a.station_count.tolist(), reverse=True)


def test_bulk_plan_covers_every_required_row_window():
    pool = _pool()
    mapping = _mapping().set_index("iata")
    plan, _ = build_bulk_month_plan(
        pool, mapping.reset_index(), shard_size=3, max_stations_per_request=2
    )

    memberships = {}
    for row in plan.itertuples(index=False):
        for station in json.loads(row.stations_json):
            memberships.setdefault((row.network, station), []).append(
                (pd.Timestamp(row.window_start_utc), pd.Timestamp(row.window_end_utc))
            )

    resolved = pool.loc[
        pool.row_tier.eq("confirmed_period") & pool.prediction_at.notna()
    ]
    for row in resolved.itertuples(index=False):
        for airport in (row.Origin_Airport, row.Destination_Airport):
            station = mapping.loc[airport, "candidate_sid"]
            network = mapping.loc[airport, "matched_network"]
            need_start = row.prediction_at - pd.Timedelta(hours=LOOKBACK_HOURS)
            need_end = row.prediction_at + pd.Timedelta(hours=LOOKAHEAD_HOURS)
            windows = memberships[(network, station)]
            assert any(start <= need_start < end for start, end in windows)
            assert any(start < need_end <= end for start, end in windows)


def test_requests_for_shard_reconstruct_exact_month_bounds_for_executor():
    plan, _ = build_bulk_month_plan(
        _pool(), _mapping(), shard_size=2, max_stations_per_request=2
    )
    planned, groups, requests = requests_for_shard(plan, shard_index=0)

    assert len(planned) == 2
    for request_id, month in planned:
        req = requests[request_id]
        timestamps = groups[(request_id, month)]
        assert min(timestamps) - pd.Timedelta(hours=LOOKBACK_HOURS) == req["window_start_utc"]
        assert max(timestamps) + pd.Timedelta(hours=LOOKAHEAD_HOURS) == req["window_end_utc"]


def test_bulk_request_url_uses_one_network_multiple_stations_and_only_join_fields():
    start = pd.Timestamp("2019-01-01T00:00:00Z")
    end = pd.Timestamp("2019-02-01T00:00:00Z")
    req = {
        "network": "AA_ASOS",
        "stations": ["A1", "B1"],
        "window_start_utc": start,
        "window_end_utc": end,
    }
    query = parse_qs(urlparse(build_bulk_request_url(req)).query)

    assert query["network"] == ["AA_ASOS"]
    assert query["station"] == ["A1", "B1"]
    assert query["data"] == WEATHER_FIELDS
    assert query["report_type"] == ["3", "4"]
    assert query["sts"] == ["2019-01-01T00:00:00Z"]
    assert query["ets"] == ["2019-02-01T00:00:00Z"]
    assert query["tz"] == ["UTC"]
    assert query["missing"] == ["M"]
    assert query["trace"] == ["0.0001"]


def _write_bulk_csv(path, stations):
    rows = []
    for idx, station in enumerate(stations):
        row = {field: "" for field in WEATHER_FIELDS}
        row.update(
            {
                "station": station,
                "valid": f"2019-01-01 0{idx}:00",
                "metar": f"METAR {station}",
            }
        )
        rows.append(row)
    pd.DataFrame(rows, columns=["station", "valid", *WEATHER_FIELDS]).to_csv(path, index=False)


def test_bulk_cache_validation_accepts_only_requested_station_ids(tmp_path):
    path = tmp_path / "bulk.csv"
    req = {"stations": ["A1", "B1"]}
    _write_bulk_csv(path, ["A1", "B1"])
    assert _validate_bulk_file(path, req) == 2

    _write_bulk_csv(path, ["A1", "WRONG"])
    with pytest.raises(ValueError, match="not present in the request"):
        _validate_bulk_file(path, req)


def test_bulk_plan_rejects_invalid_batch_or_shard_sizes():
    with pytest.raises(ValueError, match="shard_size"):
        build_bulk_month_plan(_pool(), _mapping(), shard_size=0)
    with pytest.raises(ValueError, match="max_stations_per_request"):
        build_bulk_month_plan(_pool(), _mapping(), max_stations_per_request=0)


def test_executor_custom_response_cap_controls_crash_reservation():
    groups = {
        ("REQ", "2019-01"): [
            pd.Timestamp("2019-01-01T06:00:00Z"),
            pd.Timestamp("2019-01-31T22:00:00Z"),
        ]
    }
    checkpoint = new_checkpoint_state()

    def crash(*args, **kwargs):
        raise RuntimeError("simulated bulk-attempt crash")

    with pytest.raises(RuntimeError, match="simulated bulk-attempt crash"):
        execute_with_caps(
            [("REQ", "2019-01")],
            groups,
            max_requests=1,
            max_bytes=MAX_BULK_RESPONSE_BYTES * 2,
            max_seconds=1000,
            cache_exists_fn=lambda *args: None,
            attempt_fn=crash,
            digest_fn=lambda p: "x",
            now_fn=lambda: 0.0,
            checkpoint=checkpoint,
            max_response_bytes_per_attempt=12345,
        )

    assert checkpoint["requests_used"] == 1
    assert checkpoint["bytes_used"] == 12345
    assert checkpoint["groups"]["REQ|2019-01"]["in_flight"]["bytes_reserved"] == 12345


def test_executor_rejects_nonpositive_custom_response_cap():
    checkpoint = new_checkpoint_state()
    with pytest.raises(ValueError, match="max_response_bytes_per_attempt"):
        execute_with_caps(
            [],
            {},
            max_requests=1,
            max_bytes=1,
            max_seconds=1,
            cache_exists_fn=lambda *args: None,
            attempt_fn=lambda *args, **kwargs: None,
            digest_fn=lambda p: "x",
            now_fn=lambda: 0.0,
            checkpoint=checkpoint,
            max_response_bytes_per_attempt=0,
        )


def test_executor_failed_group_keeps_request_identity():
    checkpoint = new_checkpoint_state()

    def fail_once(*args, **kwargs):
        return {
            "success": False,
            "bytes_received": 1,
            "seconds": 0.0,
            "error": "synthetic failure",
        }

    result = execute_with_caps(
        [("REQ", "2019-01")],
        {
            ("REQ", "2019-01"): [
                pd.Timestamp("2019-01-01T06:00:00Z"),
                pd.Timestamp("2019-01-31T22:00:00Z"),
            ]
        },
        max_requests=1,
        max_bytes=100,
        max_seconds=100,
        cache_exists_fn=lambda *args: None,
        attempt_fn=fail_once,
        digest_fn=lambda p: "x",
        now_fn=lambda: 0.0,
        checkpoint=checkpoint,
        max_attempts_per_group=1,
        max_response_bytes_per_attempt=10,
    )

    assert result["failed"][0]["station"] == "REQ"
    assert result["failed"][0]["day"] == "2019-01"


def test_executor_attempt_pause_applies_after_failure_and_counts_against_time_budget():
    checkpoint = new_checkpoint_state()
    slept = []

    result = execute_with_caps(
        [("REQ", "2019-01")],
        {
            ("REQ", "2019-01"): [
                pd.Timestamp("2019-01-01T06:00:00Z"),
                pd.Timestamp("2019-01-31T22:00:00Z"),
            ]
        },
        max_requests=1,
        max_bytes=100,
        max_seconds=100,
        cache_exists_fn=lambda *args: None,
        attempt_fn=lambda *args, **kwargs: {
            "success": False,
            "bytes_received": 1,
            "seconds": 0.0,
            "error": "synthetic failure",
        },
        digest_fn=lambda p: "x",
        now_fn=lambda: 0.0,
        checkpoint=checkpoint,
        sleep_fn=slept.append,
        max_attempts_per_group=1,
        attempt_pause_seconds=1.25,
        max_response_bytes_per_attempt=10,
    )

    assert slept == [1.25]
    assert checkpoint["seconds_used"] == pytest.approx(1.25)
    assert result["failed"][0]["station"] == "REQ"
