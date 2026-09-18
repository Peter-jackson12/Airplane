"""Fetch real, bounded IEM ASOS observations for the stratified expanded
sample, reusing fetch_weather_sample.py's generic per-(station, UTC-day)
downloader and src.weather unmodified. Generalizes station resolution to the
full mapping table (map_weather_stations.py) instead of the original 8-airport
constant, since the expanded sample deliberately spans many more airports.

Hard resource caps for THIS step (not a performance target -- a resource
boundary for this validation round): at most --max-requests HTTP ATTEMPTS
(successes, retries and failures all charge one unit each), --max-bytes of
measured response bytes, and --max-seconds of active fetch time. Already
validated cached (station, day) windows never touch these caps -- cache-only
work and new network work are accounted separately.

Progress is checkpointed to disk after every single attempt
(<name>_fetch_checkpoint.json, written atomically via a temp file + rename).
Re-running with the SAME --name resumes the same logical run: cumulative
requests/bytes/seconds carry over from the checkpoint rather than resetting,
already-fetched or permanently-failed groups are skipped without spending any
more budget, and a group that was merely cut off by a cap (never attempted,
or mid-retry when the cap hit) is retried on the next run. Nothing already
fetched is discarded or silently re-requested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from notebooks.fetch_weather_sample import (LOOKAHEAD_HOURS, LOOKBACK_HOURS, compute_prediction_at,
                                            digest, fetch_attempt, validate_cached_window)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 15.0
MAX_ATTEMPTS_PER_GROUP = 3
MAX_CONSECUTIVE_FAILURES = 8  # a persistent block (e.g. sustained rate-limit) should stop, not spin
BACKOFF_BASE_SECONDS = 10.0
ATTEMPT_LOG_LIMIT = 500  # cap the checkpoint's attempt log so it can't grow without bound


def build_station_day_groups(collectible: pd.DataFrame) -> dict[tuple[str, str], list[pd.Timestamp]]:
    """One group per (station, UTC calendar day of prediction_at), covering
    both the origin_station and destination_station role of every row."""
    groups: dict[tuple[str, str], list[pd.Timestamp]] = {}
    for _, row in collectible.iterrows():
        for station in (row.origin_station, row.destination_station):
            key = (station, row.prediction_at.strftime('%Y-%m-%d'))
            groups.setdefault(key, []).append(row.prediction_at)
    return groups


def new_checkpoint_state() -> dict:
    return {'schema_version': CHECKPOINT_SCHEMA_VERSION, 'requests_used': 0, 'bytes_used': 0,
           'unmeasured_byte_attempts': 0, 'seconds_used': 0.0, 'groups': {}, 'attempts_log': [],
           'cap_hit': None}


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return new_checkpoint_state()
    state = json.loads(path.read_text())
    if state.get('schema_version') != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f'{path} has an incompatible checkpoint schema; use a fresh --name')
    return state


def save_checkpoint(path: Path, state: dict) -> None:
    """Atomic write: a crash between these two lines leaves either the old
    checkpoint or the new one intact, never a half-written file."""
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(path)


def execute_with_caps(planned, groups, *, max_requests, max_bytes, max_seconds,
                      cache_exists_fn, attempt_fn, digest_fn, now_fn, checkpoint,
                      checkpoint_path=None, sleep_fn=lambda s: None, progress_every=10,
                      max_attempts_per_group=MAX_ATTEMPTS_PER_GROUP,
                      max_consecutive_failures=MAX_CONSECUTIVE_FAILURES,
                      default_attempt_timeout=DEFAULT_ATTEMPT_TIMEOUT_SECONDS):
    """Resumable, budget-aware fetch loop.

    `checkpoint` is a mutable dict carrying CUMULATIVE state across resumed
    runs of the same logical --name: requests_used/bytes_used/seconds_used
    are never reset just because this process restarted. `cache_exists_fn(
    station, start, end) -> path-or-None` looks up and validates an existing
    cache file for exactly this request; a cache hit costs nothing. Every
    other HTTP try -- success, failure, or retry -- calls `attempt_fn(
    station, start, end, timeout=..., max_bytes_remaining=...) -> {success,
    bytes_received, seconds, error, ...}` exactly once and is charged one
    request unit plus whatever bytes/seconds it actually used (bytes may be
    None for a genuinely unmeasurable attempt; such attempts are counted
    separately in unmeasured_byte_attempts, never assumed to be 0). Each
    attempt's timeout is bounded by the remaining time budget, and time spent
    in the loop's own backoff sleeps also counts against that budget. The
    checkpoint is persisted after every attempt so a crash mid-run loses at
    most the in-flight attempt, not prior progress.
    """
    fetched, skipped_cap, failed = [], [], []
    session_start = now_fn()
    session_requests = 0
    session_bytes = 0
    # cap_hit is transient PER INVOCATION, unlike requests_used/bytes_used/seconds_used: a prior run's
    # cap (e.g. an old, smaller --max-requests) must not keep blocking new attempts once this run is
    # given a larger budget. checkpoint['cap_hit'] below is overwritten purely for reporting.
    cap_hit = None
    consecutive_failures = 0

    def persist():
        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, checkpoint)

    def remaining_seconds() -> float:
        return max_seconds - checkpoint['seconds_used']

    for i, (station, day) in enumerate(planned):
        key = f'{station}|{day}'
        if i % progress_every == 0:
            print(f'[{i}/{len(planned)}] cumulative_requests={checkpoint["requests_used"]} '
                 f'cumulative_bytes={checkpoint["bytes_used"]} '
                 f'cumulative_seconds={checkpoint["seconds_used"]:.0f}', flush=True)

        prior = checkpoint['groups'].get(key)
        if prior and prior.get('status') == 'fetched':
            fetched.append({**prior, 'station': station, 'day': day})
            continue
        if prior and prior.get('status') == 'failed' and prior.get('permanent'):
            failed.append({**prior, 'station': station, 'day': day})
            continue

        timestamps = groups[(station, day)]
        window_start = min(timestamps) - pd.Timedelta(hours=LOOKBACK_HOURS)
        window_end = max(timestamps) + pd.Timedelta(hours=LOOKAHEAD_HOURS)

        # A cache hit found here has NO prior record in this checkpoint -- it is either evidence left
        # over from a different run (the original 21-row sample, an interrupted prior attempt under a
        # different --name, etc.) or a foreign/corrupted file. cache_exists_fn already validated its
        # schema; a recorded hash (if this exact key was ever fetched under THIS run before, e.g. the
        # checkpoint was reset but the file survived) is still cross-checked so stale/foreign evidence
        # is never silently substituted for what this run believes it already fetched.
        cached = cache_exists_fn(station, window_start, window_end)
        if cached is not None:
            recorded_sha = prior.get('sha256') if prior else None
            actual_sha = digest_fn(cached)
            if recorded_sha and recorded_sha != actual_sha:
                raise ValueError(
                    f'{cached} content ({actual_sha}) does not match the checkpoint\'s recorded '
                    f'evidence ({recorded_sha}) for {key}; existing evidence is not overwritten '
                    'automatically -- resolve the mismatch before resuming')
            record = {'station': station, 'day': day, 'window_start_utc': window_start.isoformat(),
                     'window_end_utc': window_end.isoformat(), 'cache_file': str(cached),
                     'sha256': actual_sha, 'rows': prior.get('rows') if prior else None,
                     'was_already_cached': True, 'attempts': 0, 'status': 'fetched'}
            checkpoint['groups'][key] = record
            fetched.append(record)
            persist()
            continue  # a validated cache hit spends none of the request/byte/time budget

        if cap_hit:
            skipped_cap.append({'station': station, 'day': day, 'reason': cap_hit})
            continue

        attempts_this_group = list(prior.get('attempts_detail', [])) if prior else []
        group_outcome = None
        while len(attempts_this_group) < max_attempts_per_group:
            if checkpoint['requests_used'] >= max_requests:
                cap_hit = f'max_requests={max_requests} reached'
                break
            if checkpoint['bytes_used'] >= max_bytes:
                cap_hit = f'max_bytes={max_bytes} reached'
                break
            remaining = remaining_seconds()
            if remaining <= 0:
                cap_hit = f'max_seconds={max_seconds} reached'
                break

            timeout = max(1.0, min(default_attempt_timeout, remaining))
            t0 = now_fn()
            outcome = attempt_fn(station, window_start, window_end, timeout=timeout,
                                 max_bytes_remaining=max_bytes - checkpoint['bytes_used'])
            elapsed = now_fn() - t0
            checkpoint['requests_used'] += 1  # every attempt -- success, retry, or failure -- charges the budget
            checkpoint['seconds_used'] += elapsed
            session_requests += 1
            if outcome.get('bytes_received') is not None:
                checkpoint['bytes_used'] += outcome['bytes_received']
                session_bytes += outcome['bytes_received']
            else:
                checkpoint['unmeasured_byte_attempts'] += 1
            attempts_this_group.append({
                'attempt': len(attempts_this_group) + 1, 'success': bool(outcome.get('success')),
                'bytes_received': outcome.get('bytes_received'), 'seconds': elapsed,
                'error': outcome.get('error')})
            checkpoint['attempts_log'] = (checkpoint['attempts_log'] + [
                {'station': station, 'day': day, **attempts_this_group[-1]}])[-ATTEMPT_LOG_LIMIT:]
            checkpoint['groups'][key] = {'status': 'in_progress', 'attempts_detail': attempts_this_group}
            persist()

            if outcome.get('success'):
                group_outcome = outcome
                consecutive_failures = 0
                break
            consecutive_failures += 1
            print(f'  FAILED {station} {day} (attempt {len(attempts_this_group)}): {outcome.get("error")}',
                 flush=True)
            if consecutive_failures >= max_consecutive_failures:
                cap_hit = (f'{max_consecutive_failures} consecutive failures '
                          '(likely a sustained rate limit or outage); stopping early')
                break
            remaining = remaining_seconds()
            if remaining <= 0:
                cap_hit = f'max_seconds={max_seconds} reached'
                break
            if len(attempts_this_group) < max_attempts_per_group:
                backoff = min(BACKOFF_BASE_SECONDS * len(attempts_this_group), remaining)
                if backoff > 0:
                    sleep_fn(backoff)
                    checkpoint['seconds_used'] += backoff
                    persist()

        if group_outcome is not None:
            record = {'station': station, 'day': day, 'window_start_utc': window_start.isoformat(),
                     'window_end_utc': window_end.isoformat(), 'url': group_outcome.get('url'),
                     'cache_file': group_outcome.get('cache_file'), 'sha256': group_outcome.get('sha256'),
                     'rows': group_outcome.get('n_rows'), 'was_already_cached': False,
                     'attempts': len(attempts_this_group), 'status': 'fetched'}
            checkpoint['groups'][key] = record
            fetched.append(record)
        elif not attempts_this_group and cap_hit:
            # cap hit before this group ever got a single attempt -- not a failure, retry on resume
            checkpoint['groups'].pop(key, None)
            skipped_cap.append({'station': station, 'day': day, 'reason': cap_hit})
        else:
            permanent = len(attempts_this_group) >= max_attempts_per_group and not cap_hit
            record = {'station': station, 'day': day, 'attempts': len(attempts_this_group),
                     'attempts_detail': attempts_this_group,
                     'errors': [a['error'] for a in attempts_this_group], 'permanent': permanent,
                     'reason': cap_hit if cap_hit and not permanent else 'max_attempts_per_group exhausted',
                     'status': 'failed'}
            checkpoint['groups'][key] = record
            failed.append(record)
        persist()

    checkpoint['cap_hit'] = cap_hit
    persist()
    return {
        'fetched': fetched, 'skipped_cap': skipped_cap, 'failed': failed,
        'new_requests': session_requests, 'new_bytes': session_bytes,
        'cumulative_requests': checkpoint['requests_used'], 'cumulative_bytes': checkpoint['bytes_used'],
        'cumulative_seconds': checkpoint['seconds_used'],
        'unmeasured_byte_attempts': checkpoint['unmeasured_byte_attempts'],
        'elapsed_seconds': now_fn() - session_start, 'cap_hit': cap_hit,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--mapping-name', required=True)
    ap.add_argument('--max-requests', type=int, default=600)
    ap.add_argument('--max-bytes', type=int, default=200_000_000)
    ap.add_argument('--max-seconds', type=float, default=3600.0)
    args = ap.parse_args()
    out = ROOT / 'output'
    selection_path = out / f'{args.name}_selection.csv'
    if not selection_path.exists():
        raise FileNotFoundError(f'Run select_weather_sample_expanded.py --name {args.name} first')

    selection = pd.read_csv(selection_path)
    mapping = pd.read_csv(out / f'{args.mapping_name}_mapping_table.csv').set_index('iata')

    selection['prediction_at'] = compute_prediction_at(selection)
    unresolved = selection.loc[selection.prediction_at.isna()]
    if len(unresolved):
        print('rows with no valid prediction_at (DST ambiguous/nonexistent local time, '
              'left unmatched by policy):', unresolved.ID.tolist(), flush=True)

    collectible = selection.loc[selection.collectible & selection.prediction_at.notna()].copy()
    collectible['origin_station'] = collectible.Origin_Airport.map(mapping.candidate_sid)
    collectible['destination_station'] = collectible.Destination_Airport.map(mapping.candidate_sid)

    groups = build_station_day_groups(collectible)
    planned = sorted(groups)
    print(f'planned station-day groups: {len(planned)} (cap: {args.max_requests} attempts)', flush=True)

    checkpoint_path = out / f'{args.name}_fetch_checkpoint.json'
    checkpoint = load_checkpoint(checkpoint_path)
    resuming = checkpoint['requests_used'] > 0 or checkpoint['groups']
    if resuming:
        print(f'resuming from checkpoint: cumulative_requests={checkpoint["requests_used"]} '
             f'cumulative_bytes={checkpoint["bytes_used"]} '
             f'cumulative_seconds={checkpoint["seconds_used"]:.0f}', flush=True)

    result = execute_with_caps(
        planned, groups, max_requests=args.max_requests, max_bytes=args.max_bytes,
        max_seconds=args.max_seconds, cache_exists_fn=validate_cached_window,
        attempt_fn=fetch_attempt, digest_fn=digest, now_fn=time.monotonic,
        checkpoint=checkpoint, checkpoint_path=checkpoint_path, sleep_fn=time.sleep)
    fetched, skipped_cap, failed = result['fetched'], result['skipped_cap'], result['failed']

    selection.to_csv(out / f'{args.name}_selection_with_prediction_at.csv', index=False)
    manifest = {
        'name': args.name, 'mapping_name': args.mapping_name,
        'code_sha256': hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'caps': {'max_requests': args.max_requests, 'max_bytes': args.max_bytes,
                'max_seconds': args.max_seconds},
        'resumed_from_checkpoint': resuming,
        'planned_station_day_groups': len(planned),
        'fetched_groups': len(fetched), 'skipped_due_to_cap': len(skipped_cap),
        'failed_groups': len(failed),
        'this_run_http_attempts': result['new_requests'], 'this_run_bytes_downloaded': result['new_bytes'],
        'this_run_elapsed_seconds': result['elapsed_seconds'],
        'cumulative_http_attempts': result['cumulative_requests'],
        'cumulative_bytes_downloaded': result['cumulative_bytes'],
        'cumulative_active_fetch_seconds': result['cumulative_seconds'],
        'unmeasured_byte_attempts': result['unmeasured_byte_attempts'],
        'cache_hit_groups': sum(1 for f in fetched if f.get('was_already_cached')),
        'new_fetch_groups': sum(1 for f in fetched if not f.get('was_already_cached')),
        'cap_that_stopped_collection': result['cap_hit'],
        'checkpoint_file': str(checkpoint_path.relative_to(ROOT)),
        'requests': fetched, 'skipped': skipped_cap, 'failed': failed,
        'fits_any_model': False, 'uses_external_data': True, 'target_columns_used': [],
        'limitations': [
            'not_collectible rows (mapping unconfirmed/tz_conflict/current-only) are never fetched; '
            'they remain in the selection with their reason and are reported, not dropped.',
            'this_run_* counts only what THIS process invocation attempted; cumulative_* is the '
            'total for the whole logical run (across any prior resumes under the same --name), '
            'read from the checkpoint and never reset by a restart.',
            'cumulative_http_attempts counts every HTTP try -- successes, retries, and failures -- '
            'not just successful new fetches, and a cache hit costs none of it.',
            'unmeasured_byte_attempts counts attempts whose byte count could not be determined at '
            'all (e.g. the response file was never created); these are reported separately and are '
            'NOT assumed to be 0 bytes, so cumulative_bytes_downloaded is a lower bound when this is > 0.',
            'A cap hit preserves everything fetched so far in the checkpoint; re-running with the '
            'SAME --name resumes without losing or re-spending prior budget, and any group that was '
            'cut off mid-retry or never attempted is retried first.',
            'Cache reuse validates the cached file\'s schema/station and, when the checkpoint already '
            'recorded a hash for that group, its content hash too; a mismatch raises rather than '
            'silently overwriting or trusting unverified existing evidence.',
        ],
    }
    (out / f'{args.name}_fetch_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest.items() if k not in {'requests', 'skipped', 'failed', 'limitations'}},
                     indent=2, default=str))


if __name__ == '__main__':
    main()
