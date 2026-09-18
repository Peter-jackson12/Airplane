from pathlib import Path

import pandas as pd
import pytest

from notebooks.fetch_weather_sample_expanded import build_station_day_groups, execute_with_caps
from notebooks.join_weather_sample import load_observations


# ---- build_station_day_groups: interval/window merging ----

def test_station_day_groups_merge_same_station_and_day_across_rows():
    collectible = pd.DataFrame({
        'origin_station': ['ATL', 'ATL', 'ORD'],
        'destination_station': ['ORD', 'ORD', 'DEN'],
        'prediction_at': pd.to_datetime(
            ['2019-06-15 10:00Z', '2019-06-15 14:00Z', '2019-06-15 09:00Z'], utc=True),
    })
    groups = build_station_day_groups(collectible)
    assert set(groups) == {('ATL', '2019-06-15'), ('ORD', '2019-06-15'), ('DEN', '2019-06-15')}
    assert len(groups[('ORD', '2019-06-15')]) == 3  # destination of rows 1-2, origin of row 3


# ---- execute_with_caps: request/byte/time limits, progress preservation ----

def fake_now():
    fake_now.t += 1
    return fake_now.t


def make_fetch_fn(bytes_per_request=1000):
    def fetch_fn(station, start, end):
        p = Path(f'/fake/{station}_{start.isoformat()}.csv')
        return p, f'https://example/{station}', 5
    return fetch_fn


def test_execute_with_caps_stops_at_max_requests_and_reports_skips():
    fake_now.t = 0
    planned = [('A', '2019-01-01'), ('B', '2019-01-01'), ('C', '2019-01-01')]
    groups = {k: [pd.Timestamp('2019-01-01T00:00Z')] for k in planned}
    result = execute_with_caps(
        planned, groups, max_requests=2, max_bytes=10**9, max_seconds=10**9,
        cache_exists_fn=lambda p: False, fetch_fn=make_fetch_fn(), digest_fn=lambda p: 'deadbeef',
        now_fn=fake_now, size_fn=lambda p: 1000)
    assert result['new_requests'] == 2
    assert len(result['fetched']) == 2
    assert len(result['skipped_cap']) == 1
    assert result['skipped_cap'][0]['reason'].startswith('max_requests')
    assert result['cap_hit'] is not None


def test_execute_with_caps_does_not_count_cache_hits_against_request_cap():
    fake_now.t = 0
    planned = [('A', '2019-01-01'), ('B', '2019-01-01')]
    groups = {k: [pd.Timestamp('2019-01-01T00:00Z')] for k in planned}
    result = execute_with_caps(
        planned, groups, max_requests=0, max_bytes=10**9, max_seconds=10**9,
        cache_exists_fn=lambda p: True, fetch_fn=make_fetch_fn(), digest_fn=lambda p: 'deadbeef',
        now_fn=fake_now)
    # both groups were "already cached", so a request cap of 0 must not block them
    assert result['new_requests'] == 0
    assert len(result['fetched']) == 2
    assert len(result['skipped_cap']) == 0


def test_execute_with_caps_records_failures_without_losing_other_progress():
    fake_now.t = 0
    planned = [('A', '2019-01-01'), ('BAD', '2019-01-01'), ('C', '2019-01-01')]
    groups = {k: [pd.Timestamp('2019-01-01T00:00Z')] for k in planned}

    def fetch_fn(station, start, end):
        if station == 'BAD':
            raise RuntimeError('simulated network failure')
        return Path(f'/fake/{station}.csv'), f'https://example/{station}', 1

    result = execute_with_caps(
        planned, groups, max_requests=10**9, max_bytes=10**9, max_seconds=10**9,
        cache_exists_fn=lambda p: False, fetch_fn=fetch_fn, digest_fn=lambda p: 'deadbeef',
        now_fn=fake_now, size_fn=lambda p: 1000)
    assert len(result['fetched']) == 2
    assert len(result['failed']) == 1
    assert result['failed'][0]['station'] == 'BAD'


# ---- load_observations: duplicate handling, including the A,A,B,B interleaved case ----

WEATHER_DEFAULTS = {f: None for f in
                    ['tmpf', 'dwpf', 'relh', 'sknt', 'gust', 'vsby', 'p01i', 'skyc1', 'wxcodes', 'snowdepth']}


def manifest_for(tmp_path, rows):
    path = tmp_path / 'obs.csv'
    pd.DataFrame([{**WEATHER_DEFAULTS, **r} for r in rows]).to_csv(path, index=False)
    import hashlib
    return {'requests': [{'cache_file': 'obs.csv',
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]}


def test_interleaved_duplicate_and_conflicting_reports_are_both_detected(tmp_path, monkeypatch):
    """A,A,B,B: two stations each with one exactly-duplicated report, and the
    two stations' reports must not be confused with each other or averaged."""
    from notebooks import join_weather_sample as mod
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    rows = [
        {'station': 'A', 'valid': '2019-01-01 00:52', 'metar': 'A-X'},
        {'station': 'A', 'valid': '2019-01-01 00:52', 'metar': 'A-X'},  # exact re-fetch of A, fine
        {'station': 'B', 'valid': '2019-01-01 00:52', 'metar': 'B-X'},
        {'station': 'B', 'valid': '2019-01-01 00:52', 'metar': 'B-X'},  # exact re-fetch of B, fine
    ]
    obs, dropped = load_observations(manifest_for(tmp_path, rows))
    assert len(obs) == 2  # one row per station survives
    assert dropped == 2
    assert set(obs.station) == {'A', 'B'}


def test_interleaved_conflicting_reports_are_rejected_not_averaged(tmp_path, monkeypatch):
    from notebooks import join_weather_sample as mod
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    rows = [
        {'station': 'A', 'valid': '2019-01-01 00:52', 'metar': 'A-X'},
        {'station': 'A', 'valid': '2019-01-01 00:52', 'metar': 'A-Y'},  # genuine conflict for A
        {'station': 'B', 'valid': '2019-01-01 00:52', 'metar': 'B-X'},
        {'station': 'B', 'valid': '2019-01-01 00:52', 'metar': 'B-X'},  # B is a harmless exact re-fetch
    ]
    with pytest.raises(ValueError, match='Conflicting reports'):
        load_observations(manifest_for(tmp_path, rows))


def test_load_observations_detects_a_corrupted_cache_file(tmp_path, monkeypatch):
    from notebooks import join_weather_sample as mod
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    manifest = manifest_for(tmp_path, [{'station': 'A', 'valid': '2019-01-01 00:52', 'metar': 'A-X'}])
    (tmp_path / 'obs.csv').write_text('station,valid,metar\nA,2019-01-01 00:52,TAMPERED\n')
    with pytest.raises(ValueError, match='content changed'):
        load_observations(manifest)
