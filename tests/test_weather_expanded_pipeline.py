import json
from pathlib import Path

import pandas as pd
import pytest

from notebooks.fetch_weather_sample_expanded import (CHECKPOINT_SCHEMA_VERSION, LOOKAHEAD_HOURS,
                                                      LOOKBACK_HOURS, build_station_day_groups,
                                                      compute_fetch_plan_fingerprint,
                                                      compute_plan_fingerprint, execute_with_caps,
                                                      fetch_plan_options, load_checkpoint,
                                                      new_checkpoint_state, record_budget_limits,
                                                      recover_interrupted_attempts, save_checkpoint,
                                                      validate_checkpoint_schema, verify_plan_fingerprint)
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


def test_unmeasurable_bytes_are_conservatively_charged_not_assumed_zero():
    """The budget-enforcement field (bytes_used) must never treat an
    unmeasurable attempt as free: each one is charged the same
    min(per-request cap, remaining budget) reservation a measured attempt
    would settle to, and that charge is never refunded. bytes_measured (the
    separate, real-usage-only field) stays 0 since nothing was ever actually
    measured -- the two fields are never conflated."""
    from notebooks.fetch_weather_sample import MAX_RESPONSE_BYTES
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    attempt_fn = make_attempt_fn({'A': [failure(bytes_received=None), failure(bytes_received=None),
                                        failure(bytes_received=None)]})
    result, checkpoint = run(planned, groups, attempt_fn,
                             max_requests=10**9, max_bytes=10**9, max_seconds=10**9)
    assert checkpoint['bytes_used'] == 3 * MAX_RESPONSE_BYTES
    assert checkpoint['bytes_measured'] == 0
    assert checkpoint['unmeasured_byte_attempts'] == 3  # tracked as "unknown", not folded into a 0 total


def test_next_attempt_sees_reduced_budget_after_an_unmeasured_failure():
    """A single unmeasured failure must eat into the remaining byte budget
    seen by the NEXT attempt -- not leave it at the full amount, which would
    let repeated unmeasured attempts spend an unbounded number of bytes
    while reporting 0 against the cap."""
    from notebooks.fetch_weather_sample import MAX_RESPONSE_BYTES
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    seen = []

    def attempt_fn(station, start, end, *, timeout, max_bytes_remaining):
        seen.append(max_bytes_remaining)
        if len(seen) == 1:
            return failure(bytes_received=None)
        return success(bytes_received=100)

    run(planned, groups, attempt_fn, max_requests=10, max_bytes=3_000_000, max_seconds=10**9)
    assert seen[0] == 3_000_000  # nothing spent yet
    assert seen[1] == 3_000_000 - MAX_RESPONSE_BYTES  # the unmeasured failure's reservation already spent


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


def test_cached_group_is_still_served_after_an_earlier_cap_hit_in_the_same_run():
    """Even once a cap has been hit and is blocking NEW network attempts,
    validated cache processing for a LATER group in the same invocation must
    still go through -- cache-only work and new network work are budgeted
    separately, so a cap on one must never block the other. D (right after
    A/B exhaust the request cap) triggers and records the cap hit; C (cached)
    comes after D in iteration order and must still succeed."""
    planned = [('A', '2019-01-01'), ('B', '2019-01-01'), ('D', '2019-01-01'), ('C', '2019-01-01')]
    groups = one_ts_groups(planned)
    attempt_fn = make_attempt_fn({'A': [success()], 'B': [success()], 'D': [success()]})
    result, checkpoint = run(
        planned, groups, attempt_fn,
        cache_exists_fn=lambda s, a, b: Path(f'/fake/{s}.csv') if s == 'C' else None,
        max_requests=2, max_bytes=10**9, max_seconds=10**9)
    assert result['cap_hit'] is not None  # the request cap was hit while processing D
    assert result['skipped_cap'] == [{'station': 'D', 'day': '2019-01-01',
                                      'reason': 'max_requests=2 reached'}]
    fetched_stations = {f['station'] for f in result['fetched']}
    assert fetched_stations == {'A', 'B', 'C'}  # C still served from cache despite D's cap hit


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

    # on resume, B is now satisfiable purely from cache (e.g. another process fetched it meanwhile);
    # A's own already-fetched cache file is also still there and unchanged, matching a real filesystem
    attempt_fn2 = make_attempt_fn({})
    second, checkpoint = run(planned, groups, attempt_fn2, checkpoint=checkpoint,
                             cache_exists_fn=lambda s, a, b: Path(f'/fake/{s}.csv'),
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


# ---- resumed "fetched" checkpoints are RE-VALIDATED, not trusted blindly ----

def _window_for(planned_key):
    station, day = planned_key
    ts = [pd.Timestamp(f'{day}T00:00Z')]
    start = min(ts) - pd.Timedelta(hours=LOOKBACK_HOURS)
    end = max(ts) + pd.Timedelta(hours=LOOKAHEAD_HOURS)
    return start, end


def test_resumed_fetched_checkpoint_raises_if_cache_file_deleted():
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    start, end = _window_for(planned[0])
    checkpoint = new_checkpoint_state()
    checkpoint['groups']['A|2019-01-01'] = {
        'station': 'A', 'day': '2019-01-01', 'status': 'fetched', 'sha256': 'abc',
        'window_start_utc': start.isoformat(), 'window_end_utc': end.isoformat(),
        'cache_file': '/fake/gone.csv'}
    with pytest.raises(ValueError, match='missing or fails schema validation'):
        execute_with_caps(planned, groups, max_requests=10, max_bytes=10**9, max_seconds=10**9,
                          cache_exists_fn=lambda s, a, b: None,  # simulates a deleted/invalid cache file
                          attempt_fn=make_attempt_fn({}), digest_fn=lambda p: 'abc',
                          now_fn=fake_clock(), checkpoint=checkpoint)


def test_resumed_fetched_checkpoint_raises_if_content_tampered():
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    start, end = _window_for(planned[0])
    checkpoint = new_checkpoint_state()
    checkpoint['groups']['A|2019-01-01'] = {
        'station': 'A', 'day': '2019-01-01', 'status': 'fetched', 'sha256': 'recorded-hash',
        'window_start_utc': start.isoformat(), 'window_end_utc': end.isoformat(),
        'cache_file': '/fake/A.csv'}
    with pytest.raises(ValueError, match='no longer matches'):
        execute_with_caps(planned, groups, max_requests=10, max_bytes=10**9, max_seconds=10**9,
                          cache_exists_fn=lambda s, a, b: Path('/fake/A.csv'),  # file still there, still parses
                          attempt_fn=make_attempt_fn({}), digest_fn=lambda p: 'tampered-hash',
                          now_fn=fake_clock(), checkpoint=checkpoint)


def test_resumed_fetched_checkpoint_raises_if_query_window_changed():
    """The SAME station/day key but a different query start/end (e.g. the
    underlying selection changed) must never be treated as the same
    completed work, even though the cache file itself is intact."""
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)  # implies a DIFFERENT window than the one recorded below
    checkpoint = new_checkpoint_state()
    checkpoint['groups']['A|2019-01-01'] = {
        'station': 'A', 'day': '2019-01-01', 'status': 'fetched', 'sha256': 'abc',
        'window_start_utc': '2019-01-01T00:00:00+00:00', 'window_end_utc': '2019-01-01T01:00:00+00:00',
        'cache_file': '/fake/A.csv'}
    with pytest.raises(ValueError, match='different window'):
        execute_with_caps(planned, groups, max_requests=10, max_bytes=10**9, max_seconds=10**9,
                          cache_exists_fn=lambda s, a, b: Path('/fake/A.csv'), attempt_fn=make_attempt_fn({}),
                          digest_fn=lambda p: 'abc', now_fn=fake_clock(), checkpoint=checkpoint)


def test_resumed_fetched_checkpoint_with_matching_window_and_hash_is_trusted():
    """The negative control: an intact, unchanged cache under the SAME window
    resumes cleanly without any error and without spending any budget."""
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    start, end = _window_for(planned[0])
    checkpoint = new_checkpoint_state()
    checkpoint['groups']['A|2019-01-01'] = {
        'station': 'A', 'day': '2019-01-01', 'status': 'fetched', 'sha256': 'abc',
        'window_start_utc': start.isoformat(), 'window_end_utc': end.isoformat(),
        'cache_file': '/fake/A.csv'}
    result = execute_with_caps(planned, groups, max_requests=10, max_bytes=10**9, max_seconds=10**9,
                               cache_exists_fn=lambda s, a, b: Path('/fake/A.csv'), attempt_fn=make_attempt_fn({}),
                               digest_fn=lambda p: 'abc', now_fn=fake_clock(), checkpoint=checkpoint)
    assert len(result['fetched']) == 1
    assert result['cumulative_requests'] == 0


# ---- plan fingerprint: ties a checkpoint to its input selection/mapping/options ----

def test_plan_fingerprint_changes_when_selection_or_mapping_or_options_change():
    groups = one_ts_groups([('A', '2019-01-01')])
    base = compute_plan_fingerprint('sel-A', 'map-A', groups, {'k': 1})
    assert compute_plan_fingerprint('sel-B', 'map-A', groups, {'k': 1}) != base
    assert compute_plan_fingerprint('sel-A', 'map-B', groups, {'k': 1}) != base
    assert compute_plan_fingerprint('sel-A', 'map-A', groups, {'k': 2}) != base
    assert compute_plan_fingerprint('sel-A', 'map-A', groups, {'k': 1}) == base


def test_plan_fingerprint_changes_when_the_normalized_request_plan_changes():
    """Same selection/mapping/options hash, but a different actual request
    window for the same station/day, must still change the fingerprint."""
    groups_a = {('A', '2019-01-01'): [pd.Timestamp('2019-01-01T09:00Z')]}
    groups_b = {('A', '2019-01-01'): [pd.Timestamp('2019-01-01T15:00Z')]}
    fp_a = compute_plan_fingerprint('sel', 'map', groups_a, {})
    fp_b = compute_plan_fingerprint('sel', 'map', groups_b, {})
    assert fp_a != fp_b


def test_verify_plan_fingerprint_raises_on_mismatch_and_records_when_absent():
    checkpoint = new_checkpoint_state()
    verify_plan_fingerprint(checkpoint, 'fp1', name='some_run')
    assert checkpoint['plan_fingerprint'] == 'fp1'
    verify_plan_fingerprint(checkpoint, 'fp1', name='some_run')  # same plan again: no error
    with pytest.raises(ValueError, match='different input selection/mapping/request plan'):
        verify_plan_fingerprint(checkpoint, 'fp2', name='some_run')
    assert checkpoint['plan_fingerprint'] == 'fp1'  # a rejected mismatch never overwrites the recorded one


def test_production_fetch_plan_fingerprint_excludes_resumable_total_budget_ceilings():
    groups = one_ts_groups([('A', '2019-01-01')])
    options = fetch_plan_options()
    assert {'max_requests', 'max_bytes', 'max_seconds'}.isdisjoint(options)
    assert compute_fetch_plan_fingerprint('sel', 'map', groups) == compute_plan_fingerprint(
        'sel', 'map', groups, options)


def test_budget_limit_history_records_cap_changes_without_resetting_cumulative_usage():
    checkpoint = new_checkpoint_state()
    checkpoint['requests_used'] = 7
    checkpoint['bytes_used'] = 1234
    checkpoint['seconds_used'] = 56.0

    first = record_budget_limits(checkpoint, max_requests=10, max_bytes=2000, max_seconds=60.0)
    record_budget_limits(checkpoint, max_requests=10, max_bytes=2000, max_seconds=60.0)
    second = record_budget_limits(checkpoint, max_requests=20, max_bytes=4000, max_seconds=120.0)

    assert checkpoint['budget_limit_history'] == [first, second]
    assert checkpoint['requests_used'] == 7
    assert checkpoint['bytes_used'] == 1234
    assert checkpoint['seconds_used'] == 56.0


# ---- budget/interrupt boundaries: reservation-before-call, conservative crash recovery ----

def test_request_budget_is_reserved_before_the_network_call_not_after():
    """A crash that happens mid-attempt (simulated here by an attempt_fn that
    raises) must still leave the request charged in the checkpoint, since it
    was reserved and persisted BEFORE attempt_fn was ever called."""
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    checkpoint = new_checkpoint_state()

    def crashing_attempt_fn(station, start, end, *, timeout, max_bytes_remaining):
        raise RuntimeError('simulated process crash mid-attempt')

    with pytest.raises(RuntimeError):
        execute_with_caps(planned, groups, max_requests=10, max_bytes=10**9, max_seconds=10**9,
                          cache_exists_fn=lambda s, a, b: None, attempt_fn=crashing_attempt_fn,
                          digest_fn=lambda p: 'x', now_fn=fake_clock(), checkpoint=checkpoint)
    assert checkpoint['requests_used'] == 1  # charged before the call, survives the crash
    assert checkpoint['groups']['A|2019-01-01']['in_flight'] is not None
    assert checkpoint['bytes_used'] > 0  # the byte reservation is charged before the call too
    assert checkpoint['groups']['A|2019-01-01']['in_flight']['bytes_reserved'] == checkpoint['bytes_used']


def test_recover_interrupted_attempts_charges_conservative_time_and_marks_unmeasured():
    state = new_checkpoint_state()
    state['groups']['A|2019-01-01'] = {'status': 'in_progress', 'attempts_detail': [],
                                       'in_flight': {'attempt': 1, 'timeout': 12.5}}
    recover_interrupted_attempts(state)
    rec = state['groups']['A|2019-01-01']
    assert rec['status'] == 'in_progress'
    assert 'in_flight' not in rec
    assert len(rec['attempts_detail']) == 1
    assert rec['attempts_detail'][0]['success'] is False
    assert rec['attempts_detail'][0]['bytes_received'] is None
    assert state['unmeasured_byte_attempts'] == 1
    assert state['seconds_used'] == 12.5  # conservative upper bound, not left at 0


def test_recover_interrupted_attempts_includes_attempt_overhead_seconds_when_present():
    """A real interrupted attempt's in_flight marker also carries
    attempt_overhead_seconds (the worst-case subprocess-termination grace
    reserved on top of the timeout, see execute_with_caps). Recovery must
    charge timeout + overhead, not timeout alone -- the real wall-clock
    worst case an attempt could have taken."""
    state = new_checkpoint_state()
    state['groups']['A|2019-01-01'] = {
        'status': 'in_progress', 'attempts_detail': [],
        'in_flight': {'attempt': 1, 'timeout': 12.5, 'attempt_overhead_seconds': 5.0,
                      'bytes_reserved': 2_000_000}}
    recover_interrupted_attempts(state)
    assert state['seconds_used'] == 17.5  # 12.5 timeout + 5.0 worst-case termination grace
    assert state['unmeasured_byte_attempts'] == 1
    # bytes_used is untouched by recovery -- the reservation was already applied at persist time
    assert state['bytes_used'] == 0


def test_recover_interrupted_attempts_never_refunds_the_byte_reservation():
    """The reservation for an interrupted attempt is added to bytes_used at
    the SAME persist point as the in_flight marker (before the network
    call), so recovery must leave bytes_used exactly as it already is --
    never adding to it (double-charging) and never subtracting from it
    (refunding a byte cost that is, in fact, unknown)."""
    state = new_checkpoint_state()
    state['bytes_used'] = 2_000_000  # as if the reservation was already persisted before the crash
    state['groups']['A|2019-01-01'] = {
        'status': 'in_progress', 'attempts_detail': [],
        'in_flight': {'attempt': 1, 'timeout': 10.0, 'attempt_overhead_seconds': 5.0,
                      'bytes_reserved': 2_000_000}}
    recover_interrupted_attempts(state)
    assert state['bytes_used'] == 2_000_000  # unchanged: neither refunded nor double-charged


def test_recover_interrupted_attempts_is_a_noop_when_nothing_is_in_flight():
    state = new_checkpoint_state()
    state['groups']['A|2019-01-01'] = {'status': 'fetched', 'sha256': 'x'}
    before = json.loads(json.dumps(state, default=str))
    recover_interrupted_attempts(state)
    assert state == before


def test_load_checkpoint_recovers_an_in_flight_attempt_from_disk(tmp_path):
    """End-to-end: a checkpoint saved mid-attempt (as save_checkpoint would
    leave it if the process died right after the pre-call persist) is
    recovered on load, and the recovered request is never re-issued as free
    on the next run -- the group simply gets one more real attempt."""
    checkpoint_path = tmp_path / 'ckpt.json'
    crashed = new_checkpoint_state()
    crashed['requests_used'] = 1
    crashed['bytes_used'] = 500_000
    crashed['groups']['A|2019-01-01'] = {
        'status': 'in_progress', 'attempts_detail': [],
        'in_flight': {'attempt': 1, 'timeout': 15.0, 'attempt_overhead_seconds': 0.0,
                      'bytes_reserved': 500_000}}
    save_checkpoint(checkpoint_path, crashed)

    resumed = load_checkpoint(checkpoint_path)
    assert resumed['requests_used'] == 1  # the crashed reservation is preserved, not lost or reset
    assert resumed['seconds_used'] == 15.0
    assert resumed['unmeasured_byte_attempts'] == 1

    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    attempt_fn = make_attempt_fn({'A': [success()]})
    result, checkpoint = run(planned, groups, attempt_fn, checkpoint=resumed,
                             checkpoint_path=checkpoint_path, max_requests=10**9, max_bytes=10**9,
                             max_seconds=10**9)
    assert len(result['fetched']) == 1
    assert result['fetched'][0]['attempts'] == 2  # the recovered interrupted attempt + this real one
    assert result['cumulative_requests'] == 1 + 1  # the crashed attempt's charge is kept, plus the new one


def test_resumed_run_cannot_exceed_max_bytes_after_an_unmeasured_interruption(tmp_path):
    """End-to-end: a process crashes mid-attempt near-exhausting the byte
    budget; on resume, the cumulative byte cap must reflect that reservation
    -- the next attempt must NOT be handed the full original max_bytes as if
    the interrupted attempt used nothing."""
    checkpoint_path = tmp_path / 'ckpt.json'
    planned = [('A', '2019-01-01'), ('B', '2019-01-01')]
    groups = one_ts_groups(planned)
    checkpoint = new_checkpoint_state()

    def crashing_attempt_fn(station, start, end, *, timeout, max_bytes_remaining):
        raise RuntimeError('simulated process crash mid-attempt')

    max_bytes = 3_000_000  # bigger than one attempt's cap, so B still gets SOME (but reduced) room
    with pytest.raises(RuntimeError):
        execute_with_caps([planned[0]], groups, max_requests=10, max_bytes=max_bytes, max_seconds=10**9,
                          cache_exists_fn=lambda s, a, b: None, attempt_fn=crashing_attempt_fn,
                          digest_fn=lambda p: 'x', now_fn=fake_clock(), checkpoint=checkpoint,
                          checkpoint_path=checkpoint_path)
    reserved_after_crash = checkpoint['bytes_used']
    assert reserved_after_crash > 0

    resumed = load_checkpoint(checkpoint_path)
    assert resumed['bytes_used'] == reserved_after_crash  # survives the crash, not reset to 0
    assert resumed['unmeasured_byte_attempts'] == 1

    seen = {}

    def attempt_fn_b(station, start, end, *, timeout, max_bytes_remaining):
        seen['max_bytes_remaining'] = max_bytes_remaining
        return success(bytes_received=100)

    execute_with_caps([('B', '2019-01-01')], groups, max_requests=10, max_bytes=max_bytes,
                      max_seconds=10**9, cache_exists_fn=lambda s, a, b: None, attempt_fn=attempt_fn_b,
                      digest_fn=lambda p: 'x', now_fn=fake_clock(), checkpoint=resumed)
    assert seen['max_bytes_remaining'] == max_bytes - reserved_after_crash
    assert seen['max_bytes_remaining'] < max_bytes  # the crash's reservation already ate into the budget


def test_repeated_interrupt_and_resume_charges_each_reservation_exactly_once(tmp_path):
    """Interrupting and resuming the SAME group's attempts twice in a row
    must charge exactly two reservations -- never double-charging the first
    one on resume, and never dropping the second."""
    checkpoint_path = tmp_path / 'ckpt.json'
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    checkpoint = new_checkpoint_state()

    def crashing_attempt_fn(station, start, end, *, timeout, max_bytes_remaining):
        raise RuntimeError('simulated process crash mid-attempt')

    for _ in range(2):
        with pytest.raises(RuntimeError):
            execute_with_caps(planned, groups, max_requests=10, max_bytes=10**9, max_seconds=10**9,
                              cache_exists_fn=lambda s, a, b: None, attempt_fn=crashing_attempt_fn,
                              digest_fn=lambda p: 'x', now_fn=fake_clock(), checkpoint=checkpoint,
                              checkpoint_path=checkpoint_path)
        checkpoint = load_checkpoint(checkpoint_path)

    assert checkpoint['requests_used'] == 2
    assert checkpoint['unmeasured_byte_attempts'] == 2
    from notebooks.fetch_weather_sample import MAX_RESPONSE_BYTES
    assert checkpoint['bytes_used'] == 2 * MAX_RESPONSE_BYTES


def test_timeout_is_never_inflated_above_remaining_seconds_even_when_below_one():
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    seen = []

    def attempt_fn(station, start, end, *, timeout, max_bytes_remaining):
        seen.append(timeout)
        return success()

    checkpoint = new_checkpoint_state()
    checkpoint['seconds_used'] = 19.7  # only 0.3s left of a 20s budget
    run(planned, groups, attempt_fn, checkpoint=checkpoint, max_requests=10**9, max_bytes=10**9, max_seconds=20)
    assert seen == [pytest.approx(0.3)]  # never bumped up to a 1.0s floor


def test_attempt_overhead_reserves_worst_case_subprocess_grace_before_starting():
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    attempt_fn = make_attempt_fn({'A': [success()]})
    checkpoint = new_checkpoint_state()
    checkpoint['seconds_used'] = 17.0  # 3.0s left of a 20s budget
    result, checkpoint = run(planned, groups, attempt_fn, checkpoint=checkpoint,
                             max_requests=10**9, max_bytes=10**9, max_seconds=20,
                             attempt_overhead_seconds=5.0)  # bigger than what remains
    assert result['fetched'] == []
    assert result['skipped_cap'][0]['reason'].startswith('max_seconds')
    assert attempt_fn.calls == []  # never attempted: not enough room for the worst-case overhead


def test_attempt_overhead_zero_preserves_prior_exact_timeout_behavior():
    """Regression guard for the default (attempt_overhead_seconds=0): the
    pre-existing bound-by-remaining-time behavior is unchanged."""
    planned = [('A', '2019-01-01'), ('B', '2019-01-01')]
    groups = one_ts_groups(planned)
    state = {'t': 0.0}

    def now():
        return state['t']

    seen_timeouts = []

    def attempt_fn(station, start, end, *, timeout, max_bytes_remaining):
        seen_timeouts.append(timeout)
        state['t'] += timeout
        return success()

    run(planned, groups, attempt_fn, clock=now, max_requests=10**9, max_bytes=10**9, max_seconds=20,
       default_attempt_timeout=15)
    assert seen_timeouts == [15, 5]


def test_success_pause_applies_after_a_new_fetch_and_counts_against_time_budget():
    planned = [('A', '2019-01-01'), ('B', '2019-01-01')]
    groups = one_ts_groups(planned)
    attempt_fn = make_attempt_fn({'A': [success()], 'B': [success()]})
    sleeps = []
    result, checkpoint = run(planned, groups, attempt_fn, sleep_fn=lambda s: sleeps.append(s),
                             max_requests=10**9, max_bytes=10**9, max_seconds=10**9,
                             success_pause_seconds=3.0)
    assert sleeps == [3.0, 3.0]
    assert checkpoint['seconds_used'] >= 6.0  # both pauses counted, not free


def test_success_pause_not_applied_to_cache_hits():
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    attempt_fn = make_attempt_fn({})
    sleeps = []
    result, checkpoint = run(planned, groups, attempt_fn, sleep_fn=lambda s: sleeps.append(s),
                             cache_exists_fn=lambda s, a, b: Path(f'/fake/{s}.csv'),
                             max_requests=10**9, max_bytes=10**9, max_seconds=10**9, success_pause_seconds=3.0)
    assert sleeps == []


def test_successful_fetch_record_preserves_per_attempt_log():
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    attempt_fn = make_attempt_fn({'A': [failure(), success()]})
    result, checkpoint = run(planned, groups, attempt_fn, max_requests=10**9, max_bytes=10**9, max_seconds=10**9)
    detail = result['fetched'][0]['attempts_detail']
    assert len(detail) == 2
    assert detail[0]['success'] is False
    assert detail[1]['success'] is True


# ---- fetch_attempt: the in-transit size limit is min(per-request cap, remaining budget) ----

def test_fetch_attempt_uses_min_of_per_request_cap_and_remaining_budget(tmp_path, monkeypatch):
    from notebooks import fetch_weather_sample as mod
    seen = {}

    def fake_http_get_once(url, *, timeout, out_path, max_bytes=None):
        seen['max_bytes'] = max_bytes
        out_path.write_text('station,valid\nXXX,2019-01-01 00:00\n')
        return {'success': True, 'bytes_received': out_path.stat().st_size, 'seconds': 0.1,
               'returncode': 0, 'error': None, 'timed_out': False}

    monkeypatch.setattr(mod, 'http_get_once', fake_http_get_once)
    monkeypatch.setattr(mod, 'CACHE_DIR', tmp_path)
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    mod.fetch_attempt('XXX', pd.Timestamp('2019-01-01', tz='UTC'), pd.Timestamp('2019-01-02', tz='UTC'),
                      timeout=5, max_bytes_remaining=100)  # far smaller than MAX_RESPONSE_BYTES
    assert seen['max_bytes'] == 100  # min(MAX_RESPONSE_BYTES, 100) == 100, not the flat per-request constant


def test_fetch_attempt_rejects_response_exceeding_the_tightened_remaining_budget(tmp_path, monkeypatch):
    """A response bigger than the remaining OVERALL budget but smaller than
    the flat per-request constant must still be rejected -- the two caps are
    no longer checked inconsistently (curl cutoff vs. post-hoc validation)."""
    from notebooks import fetch_weather_sample as mod

    def fake_http_get_once(url, *, timeout, out_path, max_bytes=None):
        out_path.write_text('x' * 150)
        return {'success': True, 'bytes_received': 150, 'seconds': 0.1,
               'returncode': 0, 'error': None, 'timed_out': False}

    monkeypatch.setattr(mod, 'http_get_once', fake_http_get_once)
    monkeypatch.setattr(mod, 'CACHE_DIR', tmp_path)
    outcome = mod.fetch_attempt('XXX', pd.Timestamp('2019-01-01', tz='UTC'), pd.Timestamp('2019-01-02', tz='UTC'),
                                timeout=5, max_bytes_remaining=100)
    assert outcome['success'] is False
    assert 'exceeds bounded size' in outcome['error']


def test_fetch_attempt_refuses_to_start_with_no_remaining_byte_budget(tmp_path, monkeypatch):
    from notebooks import fetch_weather_sample as mod
    called = []
    monkeypatch.setattr(mod, 'http_get_once', lambda *a, **k: called.append(1))
    monkeypatch.setattr(mod, 'CACHE_DIR', tmp_path)
    outcome = mod.fetch_attempt('XXX', pd.Timestamp('2019-01-01', tz='UTC'), pd.Timestamp('2019-01-02', tz='UTC'),
                                timeout=5, max_bytes_remaining=0)
    assert outcome['success'] is False
    assert called == []  # no network call is made once the byte budget is already exhausted


# ---- checkpoint schema gate: an older-format checkpoint is rejected before any network call ----

def _old_v1_completed_checkpoint() -> dict:
    """Shape of a pre-reservation-accounting checkpoint (schema_version=1)
    for an already-completed group: no bytes_measured, no plan_fingerprint,
    no unmeasured_byte_attempts -- fields the current accounting depends on."""
    return {'schema_version': 1, 'requests_used': 3, 'bytes_used': 4500, 'groups': {
        'A|2019-01-01': {'station': 'A', 'day': '2019-01-01', 'status': 'fetched', 'sha256': 'abc',
                         'window_start_utc': '2019-01-01T00:00:00+00:00',
                         'window_end_utc': '2019-01-01T01:00:00+00:00', 'cache_file': '/fake/A.csv'}},
           'attempts_log': [], 'cap_hit': None}


def _old_v1_in_flight_checkpoint() -> dict:
    """An old in_flight marker never recorded bytes_reserved/
    attempt_overhead_seconds -- the exact fields the new accounting requires."""
    state = _old_v1_completed_checkpoint()
    state['groups']['A|2019-01-01'] = {'status': 'in_progress', 'attempts_detail': [],
                                       'in_flight': {'attempt': 1, 'timeout': 15.0}}
    return state


def _old_v1_unmeasured_failure_checkpoint() -> dict:
    """An old failed-group record with no byte/time accounting recorded on
    its attempts at all."""
    state = _old_v1_completed_checkpoint()
    state['groups']['A|2019-01-01'] = {
        'station': 'A', 'day': '2019-01-01', 'attempts': 3,
        'attempts_detail': [{'attempt': i, 'success': False, 'error': 'boom'} for i in range(1, 4)],
        'errors': ['boom'] * 3, 'permanent': True, 'reason': 'max_attempts_per_group exhausted',
        'status': 'failed'}
    return state


@pytest.mark.parametrize('build_old_checkpoint', [
    _old_v1_completed_checkpoint, _old_v1_in_flight_checkpoint, _old_v1_unmeasured_failure_checkpoint])
def test_old_schema_checkpoint_is_explicitly_rejected(tmp_path, build_old_checkpoint):
    checkpoint_path = tmp_path / 'ckpt.json'
    original_text = json.dumps(build_old_checkpoint(), indent=2)
    checkpoint_path.write_text(original_text)

    with pytest.raises(ValueError, match='schema_version'):
        load_checkpoint(checkpoint_path)

    # the original file must be preserved byte-for-byte; a rejected checkpoint is never rewritten,
    # migrated, or reset, and load_checkpoint makes no network call of any kind
    assert checkpoint_path.read_text() == original_text


def test_v2_checkpoint_is_explicitly_rejected_after_fingerprint_semantics_change(tmp_path):
    checkpoint_path = tmp_path / 'ckpt.json'
    state = new_checkpoint_state()
    state['schema_version'] = 2
    state.pop('budget_limit_history')
    original_text = json.dumps(state, indent=2)
    checkpoint_path.write_text(original_text)

    with pytest.raises(ValueError, match='schema_version'):
        load_checkpoint(checkpoint_path)

    assert checkpoint_path.read_text() == original_text


def test_current_schema_checkpoint_missing_a_required_top_level_field_is_rejected(tmp_path):
    """A file that already CLAIMS schema_version=CHECKPOINT_SCHEMA_VERSION but is missing a required
    accounting field (e.g. truncated or hand-edited) must not be silently defaulted."""
    checkpoint_path = tmp_path / 'ckpt.json'
    state = new_checkpoint_state()
    del state['bytes_measured']
    original_text = json.dumps(state, indent=2)
    checkpoint_path.write_text(original_text)
    with pytest.raises(ValueError, match='missing required accounting field'):
        load_checkpoint(checkpoint_path)
    assert checkpoint_path.read_text() == original_text


def test_current_schema_checkpoint_with_incomplete_in_flight_is_rejected(tmp_path):
    """An in_flight marker under the CURRENT schema must carry
    bytes_reserved/attempt_overhead_seconds; a current-version checkpoint
    missing them is not silently completed with defaults."""
    checkpoint_path = tmp_path / 'ckpt.json'
    state = new_checkpoint_state()
    state['groups']['A|2019-01-01'] = {'status': 'in_progress', 'attempts_detail': [],
                                       'in_flight': {'attempt': 1, 'timeout': 15.0}}
    original_text = json.dumps(state, indent=2)
    checkpoint_path.write_text(original_text)
    with pytest.raises(ValueError, match='missing required accounting field'):
        load_checkpoint(checkpoint_path)
    assert checkpoint_path.read_text() == original_text


def test_current_schema_complete_checkpoint_loads_cleanly():
    """Negative control: a fully-populated current-schema checkpoint (as
    save_checkpoint actually writes) passes validation without raising."""
    state = new_checkpoint_state()
    assert state['schema_version'] == CHECKPOINT_SCHEMA_VERSION
    validate_checkpoint_schema(state, Path('irrelevant'))  # must not raise


def test_new_schema_resume_after_two_interrupts_reserves_and_charges_grace_without_duplication_or_loss(tmp_path):
    """End-to-end through disk (save_checkpoint/load_checkpoint) under the CURRENT schema: two
    separate interrupted-then-resumed cycles for the SAME group must each contribute exactly one
    reservation and one worst-case overhead charge -- never double-counted on resume, never dropped."""
    checkpoint_path = tmp_path / 'ckpt.json'
    planned = [('A', '2019-01-01')]
    groups = one_ts_groups(planned)
    checkpoint = new_checkpoint_state()

    def crashing_attempt_fn(station, start, end, *, timeout, max_bytes_remaining):
        raise RuntimeError('simulated crash mid-attempt')

    for _ in range(2):
        with pytest.raises(RuntimeError):
            execute_with_caps(planned, groups, max_requests=10, max_bytes=10**9, max_seconds=10**9,
                              cache_exists_fn=lambda s, a, b: None, attempt_fn=crashing_attempt_fn,
                              digest_fn=lambda p: 'x', now_fn=fake_clock(), checkpoint=checkpoint,
                              checkpoint_path=checkpoint_path, attempt_overhead_seconds=2.0)
        checkpoint = load_checkpoint(checkpoint_path)  # goes through the full v2 validation + recovery

    assert checkpoint['requests_used'] == 2  # exactly one reservation per crash, none lost or doubled
    assert checkpoint['unmeasured_byte_attempts'] == 2
    from notebooks.fetch_weather_sample import MAX_RESPONSE_BYTES
    assert checkpoint['bytes_used'] == 2 * MAX_RESPONSE_BYTES
    assert checkpoint['seconds_used'] == pytest.approx(2 * (15.0 + 2.0))  # each: default timeout + overhead

    # finally resolve the group for real; the two recovered interrupted attempts are preserved
    # alongside the successful one, not discarded
    attempt_fn = make_attempt_fn({'A': [success()]})
    result, checkpoint = run(planned, groups, attempt_fn, checkpoint=checkpoint,
                             checkpoint_path=checkpoint_path, max_requests=10**9, max_bytes=10**9,
                             max_seconds=10**9)
    assert len(result['fetched']) == 1
    assert result['fetched'][0]['attempts'] == 3  # 2 recovered-interrupted + 1 real success
