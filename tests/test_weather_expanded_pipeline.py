from pathlib import Path

import pandas as pd
import pytest

from notebooks.fetch_weather_sample_expanded import (build_station_day_groups, execute_with_caps,
                                                      load_checkpoint, new_checkpoint_state,
                                                      save_checkpoint)
from notebooks.join_weather_sample import load_observations
from notebooks.join_weather_sample_expanded import build_requests
from src.weather import join_weather_asof


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


# ---- execute_with_caps: attempt-level request/byte/time budget, resume ----

def fake_clock(step=1.0):
    state = {'t': 0.0}

    def now():
        state['t'] += step
        return state['t']
    return now


def make_attempt_fn(outcomes_by_station):
    """outcomes_by_station[station] is a list of outcome dicts consumed one
    per call (one per HTTP attempt for that station); the last entry repeats
    once exhausted."""
    calls = []

    def attempt_fn(station, start, end, *, timeout, max_bytes_remaining):
        calls.append((station, start, end, timeout, max_bytes_remaining))
        seq = outcomes_by_station[station]
        idx = min(sum(1 for c in calls if c[0] == station) - 1, len(seq) - 1)
        outcome = dict(seq[idx])
        outcome.setdefault('url', f'https://example/{station}')
        return outcome

    attempt_fn.calls = calls
    return attempt_fn


def success(bytes_received=1000, seconds=1.0, n_rows=5):
    return {'success': True, 'bytes_received': bytes_received, 'seconds': seconds,
           'cache_file': 'fake.csv', 'sha256': 'deadbeef', 'n_rows': n_rows}


def failure(bytes_received=0, error='simulated failure'):
    return {'success': False, 'bytes_received': bytes_received, 'error': error}


def run(planned, groups, attempt_fn, *, checkpoint=None, checkpoint_path=None, clock=None, **caps):
    checkpoint = checkpoint if checkpoint is not None else new_checkpoint_state()

    def default_cache_exists(station, start, end):
        # mimics a real filesystem: a group this SAME checkpoint already recorded as fetched has a
        # file that persists across a resumed invocation, without a test needing a real tmp_path disk
        for rec in checkpoint['groups'].values():
            if rec.get('status') == 'fetched' and rec.get('station') == station:
                return Path(rec.get('cache_file') or f'/fake/{station}.csv')
        return None

    result = execute_with_caps(
        planned, groups, cache_exists_fn=caps.pop('cache_exists_fn', None) or default_cache_exists,
        attempt_fn=attempt_fn, digest_fn=lambda p: 'deadbeef', now_fn=clock or fake_clock(),
        checkpoint=checkpoint, checkpoint_path=checkpoint_path,
        sleep_fn=caps.pop('sleep_fn', lambda s: None), **caps)
    return result, checkpoint


def one_ts_groups(planned):
    return {k: [pd.Timestamp('2019-01-01T00:00Z')] for k in planned}


def test_retries_then_succeeds_charges_every_attempt():
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    attempt_fn = make_attempt_fn({'A': [failure(bytes_received=100), failure(bytes_received=50), success()]})
    result, checkpoint = run(planned, groups, attempt_fn,
                             max_requests=10, max_bytes=10**9, max_seconds=10**9)
    assert len(result['fetched']) == 1
    assert result['fetched'][0]['attempts'] == 3
    # every HTTP try -- 2 failures + 1 success -- charges the request budget, not just the final success
    assert checkpoint['requests_used'] == 3
    assert checkpoint['bytes_used'] == 100 + 50 + 1000


def test_repeated_failure_exhausts_retries_and_is_permanent():
    planned = [('BAD', '2019-01-01')]
    groups = one_ts_groups(planned)
    attempt_fn = make_attempt_fn({'BAD': [failure(), failure(), failure()]})
    result, checkpoint = run(planned, groups, attempt_fn,
                             max_requests=10**9, max_bytes=10**9, max_seconds=10**9)
    assert len(result['fetched']) == 0
    assert len(result['failed']) == 1
    assert result['failed'][0]['permanent'] is True
    assert checkpoint['requests_used'] == 3  # 3 attempts, all charged, none succeeded


def test_partial_response_bytes_counted_even_though_attempt_failed():
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    # a response that was cut off mid-transfer: some bytes DID arrive before the failure
    attempt_fn = make_attempt_fn({'A': [failure(bytes_received=734), failure(bytes_received=734),
                                        failure(bytes_received=734)]})
    result, checkpoint = run(planned, groups, attempt_fn,
                             max_requests=10**9, max_bytes=10**9, max_seconds=10**9)
    assert checkpoint['bytes_used'] == 734 * 3
    assert checkpoint['unmeasured_byte_attempts'] == 0


def test_unmeasurable_bytes_are_never_assumed_to_be_zero():
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    attempt_fn = make_attempt_fn({'A': [failure(bytes_received=None), failure(bytes_received=None),
                                        failure(bytes_received=None)]})
    result, checkpoint = run(planned, groups, attempt_fn,
                             max_requests=10**9, max_bytes=10**9, max_seconds=10**9)
    assert checkpoint['bytes_used'] == 0
    assert checkpoint['unmeasured_byte_attempts'] == 3  # tracked as "unknown", not folded into a 0 total


def test_size_exceeded_response_is_charged_as_a_failed_attempt():
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    oversized = failure(bytes_received=5_000_000, error='response exceeds bounded size')
    attempt_fn = make_attempt_fn({'A': [oversized, oversized, oversized]})
    result, checkpoint = run(planned, groups, attempt_fn,
                             max_requests=10**9, max_bytes=10**9, max_seconds=10**9)
    assert len(result['failed']) == 1
    assert checkpoint['bytes_used'] == 5_000_000 * 3


def test_stops_at_max_requests_and_reports_skips_without_attempting():
    planned = [('A', '2019-01-01'), ('B', '2019-01-01'), ('C', '2019-01-01')]
    groups = one_ts_groups(planned)
    attempt_fn = make_attempt_fn({s: [success()] for s in ('A', 'B', 'C')})
    result, checkpoint = run(planned, groups, attempt_fn,
                             max_requests=2, max_bytes=10**9, max_seconds=10**9)
    assert result['new_requests'] == 2
    assert len(result['fetched']) == 2
    assert len(result['skipped_cap']) == 1
    assert result['skipped_cap'][0]['reason'].startswith('max_requests')
    assert result['cap_hit'] is not None
    assert len(attempt_fn.calls) == 2  # the third group's attempt_fn is never even called


def test_stops_at_max_bytes_mid_run():
    planned = [('A', '2019-01-01'), ('B', '2019-01-01')]
    groups = one_ts_groups(planned)
    attempt_fn = make_attempt_fn({'A': [success(bytes_received=1000)], 'B': [success(bytes_received=900)]})
    result, checkpoint = run(planned, groups, attempt_fn,
                             max_requests=10**9, max_bytes=1000, max_seconds=10**9)
    assert len(result['fetched']) == 1
    assert len(result['skipped_cap']) == 1
    assert result['skipped_cap'][0]['reason'].startswith('max_bytes')
    assert len(attempt_fn.calls) == 1  # B is never attempted once the byte budget is already spent


def test_remaining_byte_budget_is_passed_to_each_attempt():
    planned = [('A', '2019-01-01'), ('B', '2019-01-01')]
    groups = one_ts_groups(planned)
    attempt_fn = make_attempt_fn({'A': [success(bytes_received=400)], 'B': [success(bytes_received=100)]})
    run(planned, groups, attempt_fn, max_requests=10**9, max_bytes=1000, max_seconds=10**9)
    seen = {call[0]: call[4] for call in attempt_fn.calls}  # (station -> max_bytes_remaining)
    assert seen['A'] == 1000  # nothing spent yet
    assert seen['B'] == 600   # 1000 - 400 already spent by A


def test_exhausted_time_budget_stops_before_any_new_attempt():
    planned = [('A', '2019-01-01'), ('B', '2019-01-01')]
    groups = one_ts_groups(planned)
    attempt_fn = make_attempt_fn({'A': [success()], 'B': [success()]})
    checkpoint = new_checkpoint_state()
    checkpoint['seconds_used'] = 100.0  # already at budget before this invocation even starts
    result, checkpoint = run(planned, groups, attempt_fn, checkpoint=checkpoint,
                             max_requests=10**9, max_bytes=10**9, max_seconds=100)
    assert len(result['fetched']) == 0
    assert len(result['skipped_cap']) == 2
    assert all(s['reason'].startswith('max_seconds') for s in result['skipped_cap'])
    assert attempt_fn.calls == []  # no attempt is ever made once the time budget is already spent


def test_attempt_timeout_is_bounded_by_remaining_time_not_the_default():
    planned = [('A', '2019-01-01'), ('B', '2019-01-01')]
    groups = one_ts_groups(planned)
    state = {'t': 0.0}

    def now():
        return state['t']

    seen_timeouts = []

    def attempt_fn(station, start, end, *, timeout, max_bytes_remaining):
        seen_timeouts.append(timeout)
        state['t'] += timeout  # this attempt consumes exactly its allotted timeout
        return success()

    run(planned, groups, attempt_fn, clock=now, max_requests=10**9, max_bytes=10**9, max_seconds=20,
       default_attempt_timeout=15)
    # A gets the full default timeout (15s fits within the 20s budget); B only has 5s left afterwards,
    # so its attempt must be capped at 5s, never the 15s default that would overshoot the budget
    assert seen_timeouts == [15, 5]


def test_does_not_count_cache_hits_against_request_cap():
    planned = [('A', '2019-01-01'), ('B', '2019-01-01')]
    groups = one_ts_groups(planned)
    attempt_fn = make_attempt_fn({})
    result, checkpoint = run(planned, groups, attempt_fn,
                             cache_exists_fn=lambda s, a, b: Path(f'/fake/{s}.csv'),
                             max_requests=0, max_bytes=10**9, max_seconds=10**9)
    assert result['new_requests'] == 0
    assert len(result['fetched']) == 2
    assert len(result['skipped_cap']) == 0
    assert attempt_fn.calls == []  # cache-only work never calls attempt_fn at all


def test_interrupted_then_resumed_run_does_not_reset_cumulative_budget(tmp_path):
    checkpoint_path = tmp_path / 'ckpt.json'
    planned = [('A', '2019-01-01'), ('B', '2019-01-01'), ('C', '2019-01-01')]
    groups = one_ts_groups(planned)
    attempt_fn = make_attempt_fn({s: [success()] for s in ('A', 'B', 'C')})

    first, checkpoint1 = run(planned, groups, attempt_fn, checkpoint_path=checkpoint_path,
                             max_requests=1, max_bytes=10**9, max_seconds=10**9)
    save_checkpoint(checkpoint_path, checkpoint1)
    assert len(first['fetched']) == 1
    assert first['cumulative_requests'] == 1

    # simulate a fresh process: reload the checkpoint from disk instead of reusing the in-memory dict
    resumed_checkpoint = load_checkpoint(checkpoint_path)
    attempt_fn2 = make_attempt_fn({s: [success()] for s in ('A', 'B', 'C')})
    second, checkpoint2 = run(planned, groups, attempt_fn2, checkpoint=resumed_checkpoint,
                              checkpoint_path=checkpoint_path,
                              max_requests=10**9, max_bytes=10**9, max_seconds=10**9)
    assert len(second['fetched']) == 3  # A (from checkpoint) + B, C (newly fetched)
    assert second['cumulative_requests'] == 1 + 2  # A's earlier attempt is NOT re-charged
    assert attempt_fn2.calls == [c for c in attempt_fn2.calls if c[0] in ('B', 'C')]  # A never re-attempted


def test_group_cut_off_by_cap_is_retried_first_on_resume_via_cache():
    checkpoint_path = None
    planned = [('A', '2019-01-01'), ('B', '2019-01-01')]
    groups = one_ts_groups(planned)
    attempt_fn = make_attempt_fn({'A': [success()]})
    first, checkpoint = run(planned, groups, attempt_fn, max_requests=1, max_bytes=10**9, max_seconds=10**9)
    assert len(first['skipped_cap']) == 1  # B was cut off before any attempt

    # on resume, B is now satisfiable purely from cache (e.g. another process fetched it meanwhile)
    attempt_fn2 = make_attempt_fn({})
    second, checkpoint = run(planned, groups, attempt_fn2, checkpoint=checkpoint,
                             cache_exists_fn=lambda s, a, b: Path(f'/fake/{s}.csv') if s == 'B' else None,
                             max_requests=10**9, max_bytes=10**9, max_seconds=10**9)
    assert len(second['fetched']) == 2
    assert attempt_fn2.calls == []  # B was served from cache, never re-attempted over the network


def test_newly_discovered_cache_with_a_recorded_hash_mismatch_raises():
    """A group whose checkpoint entry is mid-retry (no success yet, so no
    trusted record) but for which a cache file now exists with a hash that
    does not match a hash this checkpoint separately recorded for that key
    must not be silently substituted as if it were the run's own evidence."""
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    checkpoint = new_checkpoint_state()
    checkpoint['groups']['A|2019-01-01'] = {'status': 'fetched', 'sha256': 'expected-hash'}
    # force the "fetched, trust directly" fast path to be bypassed so the cache-discovery branch (with
    # its hash cross-check) is what actually runs, by clearing status back to a non-trusted state
    checkpoint['groups']['A|2019-01-01']['status'] = 'in_progress'
    with pytest.raises(ValueError, match='does not match'):
        execute_with_caps(planned, groups, max_requests=10, max_bytes=10**9, max_seconds=10**9,
                          cache_exists_fn=lambda s, a, b: Path('/fake/A.csv'), attempt_fn=make_attempt_fn({}),
                          digest_fn=lambda p: 'actual-hash-differs', now_fn=fake_clock(), checkpoint=checkpoint)


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


def test_load_observations_handles_a_totally_failed_collection_without_crashing():
    """If every station/day group failed (or nothing was ever collectible),
    fetch_manifest['requests'] is empty; pd.concat([]) on zero frames used to
    raise, turning "no data available" into a hard crash instead of a
    reasoned empty result that downstream code can join against and get an
    honest, fully-unmatched outcome."""
    obs, dropped = load_observations({'requests': []})
    assert len(obs) == 0
    assert dropped == 0
    assert {'station', 'observed_at', 'available_at'}.issubset(obs.columns)


# ---- build_requests (join_weather_sample_expanded): not-collectible rows must never be eligible ----

def test_build_requests_masks_station_for_not_collectible_rows():
    selection = pd.DataFrame({
        'ID': ['R1', 'R2'],
        'prediction_at': pd.to_datetime(['2019-06-15T09:00Z', '2019-06-15T09:00Z'], utc=True),
        'Origin_Airport': ['ATL', 'ATL'], 'Destination_Airport': ['ORD', 'ORD'],
        'collectible': [True, False],  # R2 is not_collectible (e.g. tz_conflict) despite same airports
    })
    mapping = pd.DataFrame({'iata': ['ATL', 'ORD'], 'candidate_sid': ['ATL', 'ORD']}).set_index('iata')
    origin_req, dest_req = build_requests(selection, mapping)
    assert origin_req.loc[origin_req.ID.eq('R1'), 'station'].iloc[0] == 'ATL'
    assert pd.isna(origin_req.loc[origin_req.ID.eq('R2'), 'station'].iloc[0])
    assert pd.isna(dest_req.loc[dest_req.ID.eq('R2'), 'station'].iloc[0])


def test_not_collectible_row_never_joins_even_when_a_matching_observation_exists():
    """The real risk this closes: a not-collectible row (R2) shares the exact
    same station/day as a collectible row (R1) that WAS fetched. Before the
    fix, R2's station was filled in unconditionally and only an assertion
    after the join caught the accidental match; now R2 is never eligible for
    the asof merge at all."""
    selection = pd.DataFrame({
        'ID': ['R1', 'R2'],
        'prediction_at': pd.to_datetime(['2019-06-15T09:00Z', '2019-06-15T09:00Z'], utc=True),
        'Origin_Airport': ['ATL', 'ATL'], 'Destination_Airport': ['ORD', 'ORD'],
        'collectible': [True, False],
    })
    mapping = pd.DataFrame({'iata': ['ATL', 'ORD'], 'candidate_sid': ['ATL', 'ORD']}).set_index('iata')
    origin_req, _ = build_requests(selection, mapping)
    obs = pd.DataFrame({
        'station': ['ATL'], 'observed_at': pd.to_datetime(['2019-06-15T08:30Z'], utc=True),
        'available_at': pd.to_datetime(['2019-06-15T08:40Z'], utc=True), 'tmpf': [50.0]})
    joined = join_weather_asof(origin_req[['station', 'prediction_at']], obs, max_age='90min')
    assert joined.loc[origin_req.ID.eq('R1').values, 'observed_at'].notna().all()
    assert joined.loc[origin_req.ID.eq('R2').values, 'observed_at'].isna().all()


def test_build_requests_with_no_resolvable_prediction_at_produces_reasoned_missing_not_an_error():
    """Every row unresolved (e.g. all fell in a DST gap/overlap) must still
    produce a clean, fully-missing join result -- not an exception -- and the
    same for an empty observation set (every collection attempt failed)."""
    selection = pd.DataFrame({
        'ID': ['R1', 'R2'],
        'prediction_at': pd.to_datetime([pd.NaT, pd.NaT], utc=True),
        'Origin_Airport': ['ATL', 'ORD'], 'Destination_Airport': ['ORD', 'ATL'],
        'collectible': [True, True],
    })
    mapping = pd.DataFrame({'iata': ['ATL', 'ORD'], 'candidate_sid': ['ATL', 'ORD']}).set_index('iata')
    origin_req, dest_req = build_requests(selection, mapping)
    empty_obs, dropped = load_observations({'requests': []})
    joined = join_weather_asof(origin_req[['station', 'prediction_at']], empty_obs, max_age='90min')
    assert len(joined) == 2
    assert joined.observed_at.isna().all()
    assert dropped == 0
