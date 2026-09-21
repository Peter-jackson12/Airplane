import pandas as pd
import pytest

from notebooks.fetch_weather_full_sharded import (
    build_station_day_plan,
    groups_from_plan_shard,
)
from notebooks.fetch_weather_sample import LOOKAHEAD_HOURS, LOOKBACK_HOURS, cache_path_for
from notebooks.fetch_weather_sample_expanded import build_station_day_groups


def _pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_tier": [
                "confirmed_period",
                "confirmed_period",
                "confirmed_period",
                "unconfirmed",
                "confirmed_period",
            ],
            "prediction_at": pd.to_datetime(
                [
                    "2018-01-02T10:00:00Z",
                    "2018-01-02T14:00:00Z",
                    "2018-01-01T09:00:00Z",
                    "2018-01-01T12:00:00Z",
                    None,
                ],
                utc=True,
            ),
            "origin_station": ["ATL", "ATL", "ORD", "XXX", "DEN"],
            "destination_station": ["ORD", "ORD", "DEN", "YYY", "ATL"],
        }
    )


def test_full_plan_matches_current_station_day_fetch_windows_exactly():
    pool = _pool()
    plan, denominators = build_station_day_plan(pool, shard_size=2)

    eligible = pool.loc[
        pool.row_tier.eq("confirmed_period") & pool.prediction_at.notna(),
        ["origin_station", "destination_station", "prediction_at"],
    ]
    existing_groups = build_station_day_groups(eligible)

    assert len(plan) == len(existing_groups)
    assert denominators["station_day_request_groups"] == len(existing_groups)
    assert denominators["mapping_eligible_rows_confirmed_period_both_ends"] == 4
    assert denominators["utc_time_resolved_rows"] == 3
    assert denominators["mapping_eligible_but_utc_unresolved_rows"] == 1
    assert denominators["not_mapping_eligible_rows"] == 1

    by_key = {(r.station, r.day): r for r in plan.itertuples(index=False)}
    for key, timestamps in existing_groups.items():
        row = by_key[key]
        assert pd.Timestamp(row.min_prediction_at) == min(timestamps)
        assert pd.Timestamp(row.max_prediction_at) == max(timestamps)
        assert pd.Timestamp(row.window_start_utc) == min(timestamps) - pd.Timedelta(hours=LOOKBACK_HOURS)
        assert pd.Timestamp(row.window_end_utc) == max(timestamps) + pd.Timedelta(hours=LOOKAHEAD_HOURS)
        assert row.n_prediction_refs == len(timestamps)


def test_full_plan_is_deterministic_day_then_station_and_fixed_shards():
    plan_a, den_a = build_station_day_plan(_pool(), shard_size=2)
    plan_b, den_b = build_station_day_plan(_pool().sample(frac=1, random_state=7), shard_size=2)

    cols = [
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
    pd.testing.assert_frame_equal(plan_a[cols], plan_b[cols])
    assert den_a == den_b
    assert list(zip(plan_a.day, plan_a.station)) == sorted(zip(plan_a.day, plan_a.station))
    assert plan_a.shard_index.tolist() == [i // 2 for i in range(len(plan_a))]


def test_groups_reconstructed_from_plan_reproduce_recorded_windows():
    plan, _ = build_station_day_plan(_pool(), shard_size=2)
    planned, groups = groups_from_plan_shard(plan, shard_index=0)
    shard = plan.loc[plan.shard_index.eq(0)].sort_values("ordinal")

    assert planned == list(zip(shard.station, shard.day))
    for row in shard.itertuples(index=False):
        timestamps = groups[(row.station, row.day)]
        assert min(timestamps) - pd.Timedelta(hours=LOOKBACK_HOURS) == pd.Timestamp(row.window_start_utc)
        assert max(timestamps) + pd.Timedelta(hours=LOOKAHEAD_HOURS) == pd.Timestamp(row.window_end_utc)


def test_groups_reject_tampered_window():
    plan, _ = build_station_day_plan(_pool(), shard_size=2)
    plan.loc[plan.index[0], "window_end_utc"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="plan window no longer matches"):
        groups_from_plan_shard(plan, shard_index=0)


def test_plan_rejects_nonpositive_shard_size():
    with pytest.raises(ValueError, match="shard_size"):
        build_station_day_plan(_pool(), shard_size=0)


def test_plan_rejects_missing_station_on_collectible_row():
    pool = _pool()
    pool.loc[0, "origin_station"] = None
    with pytest.raises(ValueError, match="both mapped stations"):
        build_station_day_plan(pool, shard_size=2)


def test_cache_path_can_be_scoped_to_a_shard_directory(tmp_path):
    start = pd.Timestamp("2019-01-01T00:00:00Z")
    end = pd.Timestamp("2019-01-01T08:00:00Z")
    path = cache_path_for("ATL", start, end, cache_dir=tmp_path / "shard_00000")
    assert path.parent == tmp_path / "shard_00000"
    assert path.name.startswith("iem_ATL_")
