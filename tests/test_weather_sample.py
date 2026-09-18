import json

import pandas as pd
import pytest

from notebooks.select_weather_sample import (
    NEAR_MIDNIGHT_MINUTES, is_dst_transition_departure, load_airport_timezones,
    minutes_from_midnight_distance)
from notebooks.fetch_weather_sample import cache_path_for, compute_prediction_at, validate_cached_window
from notebooks.join_weather_sample import load_observations


# ---- validate_cached_window: cache schema/station validation before reuse ----

def test_validate_cached_window_returns_none_when_nothing_cached(tmp_path, monkeypatch):
    from notebooks import fetch_weather_sample as mod
    monkeypatch.setattr(mod, 'CACHE_DIR', tmp_path)
    start, end = pd.Timestamp('2019-01-01T00:00Z'), pd.Timestamp('2019-01-01T08:00Z')
    assert validate_cached_window('ATL', start, end) is None


def test_validate_cached_window_accepts_a_well_formed_cache_file(tmp_path, monkeypatch):
    from notebooks import fetch_weather_sample as mod
    monkeypatch.setattr(mod, 'CACHE_DIR', tmp_path)
    start, end = pd.Timestamp('2019-01-01T00:00Z'), pd.Timestamp('2019-01-01T08:00Z')
    cache_path_for('ATL', start, end).write_text('station,valid,tmpf\nATL,2019-01-01 00:52,50\n')
    result = validate_cached_window('ATL', start, end)
    assert result == cache_path_for('ATL', start, end)


def test_validate_cached_window_rejects_a_file_missing_required_columns(tmp_path, monkeypatch):
    from notebooks import fetch_weather_sample as mod
    monkeypatch.setattr(mod, 'CACHE_DIR', tmp_path)
    start, end = pd.Timestamp('2019-01-01T00:00Z'), pd.Timestamp('2019-01-01T08:00Z')
    cache_path_for('ATL', start, end).write_text('not,the,expected,schema\n1,2,3,4\n')
    with pytest.raises(ValueError, match='expected archive schema'):
        validate_cached_window('ATL', start, end)


def test_validate_cached_window_rejects_a_file_from_the_wrong_station(tmp_path, monkeypatch):
    from notebooks import fetch_weather_sample as mod
    monkeypatch.setattr(mod, 'CACHE_DIR', tmp_path)
    start, end = pd.Timestamp('2019-01-01T00:00Z'), pd.Timestamp('2019-01-01T08:00Z')
    # same cache path a mismatched request would compute -- simulates a foreign/corrupted file
    # sitting at the expected path, which must never be silently trusted as this station's evidence
    path = cache_path_for('ATL', start, end)
    path.write_text('station,valid,tmpf\nORD,2019-01-01 00:52,50\n')
    with pytest.raises(ValueError, match='does not match requested station'):
        validate_cached_window('ATL', start, end)


def test_select_weather_sample_never_reads_a_target_or_outcome_column():
    """The only two train.csv reads in the selection script must stay limited
    to airports/scheduled times; a target/outcome column here would let it
    leak into sample selection."""
    import re

    from notebooks import select_weather_sample as mod
    import inspect
    source = inspect.getsource(mod.main)
    usecols_lists = re.findall(r"usecols=\[(.*?)\]", source, re.S)
    assert usecols_lists, 'expected at least one usecols= read_csv call'
    forbidden = {'Delay', 'ArrDelay', 'DepDelay', 'Not_Delayed', 'Delayed',
                'Actual_Departure_Time', 'Actual_Arrival_Time'}
    for cols_text in usecols_lists:
        columns = {c.strip().strip("'\"") for c in cols_text.split(',')}
        assert not (forbidden & columns), columns


def test_minutes_from_midnight_distance_wraps_around_zero():
    result = minutes_from_midnight_distance(pd.Series([0, 30, 2330, 1200, 2400]))
    # 1200 (noon) is 720 minutes from midnight either direction; 2400 means next midnight (0)
    assert result.tolist() == [0, 30, 30, 720, 0]


def test_near_midnight_threshold_matches_the_documented_window():
    boundary = minutes_from_midnight_distance(pd.Series([NEAR_MIDNIGHT_MINUTES * 100 // 60]))
    assert boundary.iloc[0] <= NEAR_MIDNIGHT_MINUTES


def test_dst_transition_flags_only_the_exact_clock_discontinuity():
    # spring-forward 2018-03-11: 02:00-02:59 does not exist locally
    assert is_dst_transition_departure('2018-03-11', 'America/Denver', 230)
    assert not is_dst_transition_departure('2018-03-11', 'America/Denver', 130)
    assert not is_dst_transition_departure('2018-03-11', 'America/Denver', 300)
    # fall-back 2019-11-03: 01:00-01:59 occurs twice locally
    assert is_dst_transition_departure('2019-11-03', 'America/Los_Angeles', 140)
    assert not is_dst_transition_departure('2019-11-03', 'America/Los_Angeles', 200)


def test_phoenix_never_flags_a_dst_transition_even_on_the_transition_date():
    """America/Phoenix does not observe DST, so it has no clock discontinuity."""
    assert not is_dst_transition_departure('2018-03-11', 'America/Phoenix', 230)
    assert not is_dst_transition_departure('2019-11-03', 'America/Phoenix', 140)


def test_dst_transition_requires_a_transition_date():
    assert not is_dst_transition_departure('2018-06-15', 'America/Denver', 230)


def test_missing_departure_time_is_not_a_transition_case():
    assert not is_dst_transition_departure('2018-03-11', 'America/Denver', float('nan'))


def test_load_airport_timezones_rejects_a_changed_source_hash(tmp_path, monkeypatch):
    from notebooks import select_weather_sample as mod
    fake = tmp_path / 'airports.json'
    fake.write_text(json.dumps({'KATL': {'iata': 'ATL', 'tz': 'America/New_York'}}))
    monkeypatch.setattr(mod, 'MWGG_CACHE', fake)
    monkeypatch.setattr(mod, 'MWGG_EXPECTED_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='hash changed'):
        load_airport_timezones()


def test_load_airport_timezones_rejects_an_iata_mismatch(tmp_path, monkeypatch):
    from notebooks import select_weather_sample as mod
    import hashlib
    fake = tmp_path / 'airports.json'
    data = {meta['icao']: {'iata': 'WRONG', 'tz': 'America/New_York'} for meta in mod.AIRPORTS.values()}
    fake.write_text(json.dumps(data))
    monkeypatch.setattr(mod, 'MWGG_CACHE', fake)
    monkeypatch.setattr(mod, 'MWGG_EXPECTED_SHA256', hashlib.sha256(fake.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='IATA mismatch'):
        load_airport_timezones()


def test_compute_prediction_at_is_60_minutes_before_scheduled_departure():
    selection = pd.DataFrame({
        'attributed_date': ['2019-06-15'], 'origin_timezone': ['America/New_York'],
        'Estimated_Departure_Time': [1200.0]})
    prediction_at = compute_prediction_at(selection)
    departure_utc = pd.Timestamp('2019-06-15 16:00:00+00:00')  # noon EDT = 16:00 UTC
    assert prediction_at.iloc[0] == departure_utc - pd.Timedelta(minutes=60)


def test_compute_prediction_at_leaves_ambiguous_local_time_as_nat():
    """The DST fall-back hour occurs twice; the module's own policy is NaT,
    not an arbitrary guess, and this must propagate rather than crash."""
    selection = pd.DataFrame({
        'attributed_date': ['2019-11-03'], 'origin_timezone': ['America/Los_Angeles'],
        'Estimated_Departure_Time': [140.0]})
    prediction_at = compute_prediction_at(selection)
    assert prediction_at.isna().all()


def test_compute_prediction_at_handles_mixed_timezones_in_one_call():
    selection = pd.DataFrame({
        'attributed_date': ['2019-06-15', '2019-06-15'],
        'origin_timezone': ['America/New_York', 'America/Los_Angeles'],
        'Estimated_Departure_Time': [1200.0, 1200.0]})
    prediction_at = compute_prediction_at(selection)
    assert prediction_at.iloc[0] != prediction_at.iloc[1]
    assert prediction_at.notna().all()


WEATHER_FIELD_DEFAULTS = {f: None for f in
                          ['tmpf', 'dwpf', 'relh', 'sknt', 'gust', 'vsby', 'p01i',
                           'skyc1', 'wxcodes', 'snowdepth']}


def build_manifest(tmp_path, rows):
    """load_observations resolves paths as mod.ROOT / entry['cache_file'], so
    point ROOT at tmp_path and use a plain relative filename."""
    path = tmp_path / 'obs.csv'
    pd.DataFrame([{**WEATHER_FIELD_DEFAULTS, **row} for row in rows]).to_csv(path, index=False)
    import hashlib
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'requests': [{'cache_file': 'obs.csv', 'sha256': sha}]}


def test_load_observations_collapses_identical_refetched_rows(tmp_path, monkeypatch):
    from notebooks import join_weather_sample as mod
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    rows = [{'station': 'ATL', 'valid': '2019-01-01 00:52', 'metar': 'KATL X'},
            {'station': 'ATL', 'valid': '2019-01-01 00:52', 'metar': 'KATL X'}]
    obs, dropped = load_observations(build_manifest(tmp_path, rows))
    assert len(obs) == 1
    assert dropped == 1


def test_load_observations_rejects_conflicting_reports_at_the_same_instant(tmp_path, monkeypatch):
    from notebooks import join_weather_sample as mod
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    rows = [{'station': 'ATL', 'valid': '2019-01-01 00:52', 'metar': 'KATL X'},
            {'station': 'ATL', 'valid': '2019-01-01 00:52', 'metar': 'KATL Y'}]
    with pytest.raises(ValueError, match='Conflicting reports'):
        load_observations(build_manifest(tmp_path, rows))
