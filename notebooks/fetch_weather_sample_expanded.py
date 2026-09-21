"""Fetch real, bounded IEM ASOS observations for the stratified expanded
sample, reusing fetch_weather_sample.py's generic per-(station, UTC-day)
downloader and src.weather unmodified. Generalizes station resolution to the
full mapping table (map_weather_stations.py) instead of the original 8-airport
constant, since the expanded sample deliberately spans many more airports.

Hard resource caps for THIS step (not a performance target -- a resource
boundary for this validation round): at most --max-requests HTTP ATTEMPTS
(successes, retries and failures all charge one unit each), --max-bytes
conservatively enforced against every attempt's response (a measured outcome
counts its real bytes; an unmeasurable one is charged its full per-request
cap rather than 0, so the cumulative cap can never be silently bypassed), and
--max-seconds of active fetch time. Already validated cached (station, day)
windows never touch these caps -- cache-only work and new network work are
accounted separately.

Progress is checkpointed to disk after every single attempt
(<name>_fetch_checkpoint.json, written atomically via a temp file + rename).
Re-running with the SAME --name resumes the same logical run: cumulative
requests/bytes/seconds carry over from the checkpoint rather than resetting,
already-fetched or permanently-failed groups are skipped without spending any
more budget, and a group that was merely cut off by a cap (never attempted,
or mid-retry when the cap hit) is retried on the next run. The three run-level
resource ceilings (--max-requests/--max-bytes/--max-seconds) may change between
resumed invocations: increasing them can grant additional headroom without
changing the fetch plan, while lowering them never erases cumulative usage.
Every distinct ceiling set is recorded in the checkpoint/manifest budget
history. Nothing already fetched is discarded or silently re-requested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from notebooks.fetch_weather_sample import (LOOKAHEAD_HOURS, LOOKBACK_HOURS, MAX_RESPONSE_BYTES,
                                            SUBPROCESS_TERMINATION_GRACE_SECONDS, compute_prediction_at,
                                            digest, fetch_attempt, validate_cached_window)

ROOT = Path(__file__).resolve().parents[1]
# v1 checkpoints lacked the reservation accounting fields that v2 made explicit.
# v2 then folded mutable total run ceilings (max requests/bytes/seconds) into plan_fingerprint,
# even though execute_with_caps intentionally supports resuming the SAME logical run with a larger
# ceiling after a cap hit. That made the CLI path self-contradictory: the only change that could
# grant more budget also changed the fingerprint and was rejected before resume.
#
# v3 separates those concerns. plan_fingerprint covers immutable input/request/fetch-policy identity,
# while budget_limit_history records each distinct invocation-level ceiling set. Because a v2 hash
# cannot be decomposed safely to prove that only its ceilings changed, v2 is rejected rather than
# guessed or auto-migrated under the new semantics.
CHECKPOINT_SCHEMA_VERSION = 3
REQUIRED_CHECKPOINT_FIELDS = ('schema_version', 'requests_used', 'bytes_used', 'bytes_measured',
                             'unmeasured_byte_attempts', 'seconds_used', 'groups', 'attempts_log',
                             'cap_hit', 'plan_fingerprint', 'budget_limit_history')
REQUIRED_IN_FLIGHT_FIELDS = ('attempt', 'timeout', 'attempt_overhead_seconds', 'bytes_reserved')
DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 15.0
MAX_ATTEMPTS_PER_GROUP = 3
MAX_CONSECUTIVE_FAILURES = 8  # a persistent block (e.g. sustained rate-limit) should stop, not spin
BACKOFF_BASE_SECONDS = 10.0
ATTEMPT_LOG_LIMIT = 500  # caps only the rolling cross-group activity log; each group's OWN
                        # attempts_detail (the accounting/audit basis) is never truncated by this
SUCCESS_PAUSE_SECONDS = 3.0  # be polite to the free public archive between successful requests,
                            # matching fetch_weather_sample.py's fetch_window; charged against the time budget


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
           'bytes_measured': 0, 'unmeasured_byte_attempts': 0, 'seconds_used': 0.0, 'groups': {},
           'attempts_log': [], 'cap_hit': None, 'plan_fingerprint': None,
           'budget_limit_history': []}


def checkpoint_has_prior_progress(checkpoint: dict) -> bool:
    """Return a stable boolean for whether this logical run is being resumed.

    Do not return checkpoint['groups'] directly: an empty dict is a mutable
    object, so keeping it in the manifest and then filling the checkpoint
    during execute_with_caps would serialize the whole groups mapping instead
    of the intended True/False audit flag.
    """
    return bool(checkpoint['requests_used'] > 0 or checkpoint['groups'])


def recover_interrupted_attempts(state: dict) -> None:
    """A crash between the pre-call reservation persist and the post-call
    resolution persist (see execute_with_caps) leaves a group's checkpoint
    entry holding an 'in_flight' marker for an attempt whose outcome was
    never recorded. requests_used AND bytes_used already count it -- both
    were persisted BEFORE the network call was made, bytes_used with a
    conservative reservation (min(MAX_RESPONSE_BYTES, remaining budget) at
    the time of that call, stored as in_flight['bytes_reserved']) rather than
    0 -- so resuming must not drop either charge, must not re-issue the
    attempt as a free retry, and must not silently assume it used 0
    bytes/seconds. It is resolved here into a single 'interrupted, outcome
    unknown' attempt entry. bytes_used is left exactly as it already is (the
    reservation stands permanently uncharged-back, since the real byte usage
    can never be recovered and a strict upper bound is safer than treating it
    as free). The time budget is conservatively charged the full timeout that
    attempt was allotted PLUS the worst-case subprocess-termination grace it
    reserved on top of that timeout (in_flight['attempt_overhead_seconds']) --
    the real elapsed time cannot be recovered, and the worst case is the only
    safe upper bound.
    """
    for key, rec in list(state.get('groups', {}).items()):
        in_flight = rec.get('in_flight') if isinstance(rec, dict) else None
        if not in_flight:
            continue
        attempts = list(rec.get('attempts_detail', []))
        attempts.append({'attempt': in_flight.get('attempt', len(attempts) + 1), 'success': False,
                         'bytes_received': None, 'seconds': None,
                         'bytes_charged_conservatively': in_flight.get('bytes_reserved'),
                         'error': 'interrupted mid-attempt (process ended before the outcome was '
                                  'recorded); request and byte budget were already reserved before '
                                  'the network call and are not re-issued or refunded on resume'})
        state['unmeasured_byte_attempts'] = state.get('unmeasured_byte_attempts', 0) + 1
        reserved_timeout = in_flight.get('timeout')
        if reserved_timeout:
            overhead = in_flight.get('attempt_overhead_seconds') or 0.0
            state['seconds_used'] = state.get('seconds_used', 0.0) + reserved_timeout + overhead
        # bytes_used is NOT touched here: the reservation was already added to it at the same persist
        # point as this in_flight marker, before the network call, so it survives the crash unchanged.
        state['groups'][key] = {'status': 'in_progress', 'attempts_detail': attempts}


def validate_checkpoint_schema(state: dict, path: Path) -> None:
    """A checkpoint that already claims to BE schema_version=CHECKPOINT_SCHEMA_VERSION must still
    carry every accounting field this version's recovery/budget logic depends on. A missing field
    is never silently defaulted here (that would let a truncated or hand-edited checkpoint resume
    from a wrong accounting baseline without any warning) -- it raises instead."""
    missing_top = [f for f in REQUIRED_CHECKPOINT_FIELDS if f not in state]
    if missing_top:
        raise ValueError(
            f'{path} claims schema_version={CHECKPOINT_SCHEMA_VERSION} but is missing required '
            f'accounting field(s) {missing_top}; refusing to silently fill in defaults for a schema '
            'that is supposed to already be complete. Resolve the file or use a fresh --name.')
    for key, rec in state['groups'].items():
        in_flight = rec.get('in_flight') if isinstance(rec, dict) else None
        if not in_flight:
            continue
        missing = [f for f in REQUIRED_IN_FLIGHT_FIELDS if f not in in_flight]
        if missing:
            raise ValueError(
                f'{path}: group {key!r} has an in_flight attempt missing required accounting '
                f'field(s) {missing}; refusing to silently default them. Resolve the file or use a '
                'fresh --name.')


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return new_checkpoint_state()
    state = json.loads(path.read_text())
    schema_version = state.get('schema_version')
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f'{path} has checkpoint schema_version={schema_version!r}, but this code requires '
            f'schema_version={CHECKPOINT_SCHEMA_VERSION} (the checkpoint semantics changed: v3 keeps '
            'mutable run-level resource ceilings out of plan_fingerprint and records them separately in '
            'budget_limit_history; it also retains the v2 reservation accounting fields). '
            'An older checkpoint is never auto-migrated, and its cumulative counters are never reset or '
            'assumed continued under a reused name -- resolve it explicitly (e.g. inspect it manually) '
            'or start a fresh --name for a new logical run. No network call is made and the original '
            'checkpoint/cache files are left untouched.')
    validate_checkpoint_schema(state, path)
    recover_interrupted_attempts(state)
    return state


def save_checkpoint(path: Path, state: dict) -> None:
    """Atomic write: a crash between these two lines leaves either the old
    checkpoint or the new one intact, never a half-written file."""
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(path)


def compute_plan_fingerprint(selection_sha256: str, mapping_sha256: str,
                             groups: dict[tuple[str, str], list[pd.Timestamp]], options: dict) -> str:
    """Fingerprints immutable logical-plan inputs.

    `options` is deliberately generic so tests can exercise the hashing rule,
    but the production caller uses fetch_plan_options() below. Run-level
    cumulative resource ceilings do NOT belong here: they can be raised or
    lowered on resume without changing which observations are requested.
    """
    normalized_groups = sorted(
        (station, day, min(ts).isoformat(), max(ts).isoformat())
        for (station, day), ts in groups.items())
    payload = {'selection_sha256': selection_sha256, 'mapping_sha256': mapping_sha256,
              'groups': normalized_groups, 'options': options}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def fetch_plan_options() -> dict:
    """Immutable fetch-policy settings that make checkpointed work comparable.

    Total invocation ceilings (--max-requests/--max-bytes/--max-seconds) are
    intentionally absent. They only bound how much of this same plan one
    invocation may advance.
    """
    return {
        'lookback_hours': LOOKBACK_HOURS,
        'lookahead_hours': LOOKAHEAD_HOURS,
        'max_response_bytes_per_attempt': MAX_RESPONSE_BYTES,
        'default_attempt_timeout_seconds': DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
        'max_attempts_per_group': MAX_ATTEMPTS_PER_GROUP,
        'max_consecutive_failures': MAX_CONSECUTIVE_FAILURES,
        'backoff_base_seconds': BACKOFF_BASE_SECONDS,
        'subprocess_termination_grace_seconds': SUBPROCESS_TERMINATION_GRACE_SECONDS,
        'success_pause_seconds': SUCCESS_PAUSE_SECONDS,
    }


def compute_fetch_plan_fingerprint(selection_sha256: str, mapping_sha256: str,
                                   groups: dict[tuple[str, str], list[pd.Timestamp]]) -> str:
    return compute_plan_fingerprint(selection_sha256, mapping_sha256, groups, fetch_plan_options())


def record_budget_limits(checkpoint: dict, *, max_requests: int, max_bytes: int,
                         max_seconds: float) -> dict:
    """Record changed invocation ceilings without touching cumulative usage."""
    caps = {'max_requests': max_requests, 'max_bytes': max_bytes, 'max_seconds': max_seconds}
    history = checkpoint['budget_limit_history']
    if not history or history[-1] != caps:
        history.append(caps)
    return caps


def verify_plan_fingerprint(checkpoint: dict, fingerprint: str, *, name: str) -> None:
    """Raises if this checkpoint was already built from a DIFFERENT plan than
    the one about to run under the same --name; records the fingerprint the
    first time a fresh checkpoint sees one. A mismatch stops explicitly --
    it never silently overwrites or extends the prior evidence with
    incompatible new work."""
    prior = checkpoint.get('plan_fingerprint')
    if prior is not None and prior != fingerprint:
        raise ValueError(
            f'{name} checkpoint was built from a different input selection/mapping/request plan '
            f'(recorded fingerprint {prior}, current {fingerprint}). Resuming under the same --name '
            'would silently mix incompatible evidence. Use a fresh --name for the changed plan, or '
            'restore the exact selection/mapping/options this checkpoint was created from.')
    checkpoint['plan_fingerprint'] = fingerprint


def execute_with_caps(planned, groups, *, max_requests, max_bytes, max_seconds,
                      cache_exists_fn, attempt_fn, digest_fn, now_fn, checkpoint,
                      checkpoint_path=None, sleep_fn=lambda s: None, progress_every=10,
                      max_attempts_per_group=MAX_ATTEMPTS_PER_GROUP,
                      max_consecutive_failures=MAX_CONSECUTIVE_FAILURES,
                      default_attempt_timeout=DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
                      attempt_overhead_seconds=0.0, success_pause_seconds=0.0):
    """Resumable, budget-aware fetch loop.

    `checkpoint` is a mutable dict carrying CUMULATIVE state across resumed
    runs of the same logical --name: requests_used/bytes_used/bytes_measured/
    seconds_used are never reset just because this process restarted.
    `cache_exists_fn(station, start, end) -> path-or-None` looks up and
    validates an existing cache file for exactly this request; a cache hit
    costs nothing. Every other HTTP try -- success, failure, or retry --
    calls `attempt_fn(station, start, end, timeout=..., max_bytes_remaining=
    ...) -> {success, bytes_received, seconds, error, ...}` exactly once and
    is charged one request unit plus whatever bytes/seconds it actually used.
    Bytes are accounted in TWO separate fields: bytes_used is a CONSERVATIVE
    reservation -- min(a fixed per-request cap, the remaining budget) is
    added to it BEFORE the call, a measured outcome (bytes_received is not
    None) replaces that reservation with the real count, and a genuinely
    unmeasurable outcome (bytes_received is None) leaves the reservation
    standing rather than refunding it to 0 -- so bytes_used is always an
    upper bound the max_bytes cap can actually rely on. bytes_measured is the
    separate, real-usage-only total; unmeasured_byte_attempts counts how many
    attempts never got a real measurement. The request-budget unit AND the
    byte reservation are both reserved and persisted BEFORE the network call,
    not after it returns, so a crash mid-attempt cannot make that attempt
    vanish from accounting, be re-issued for free, or have its byte cost
    silently assumed to be zero on resume (see recover_interrupted_attempts).
    Each attempt's timeout is bounded by the
    remaining time budget MINUS attempt_overhead_seconds (the worst-case extra
    wall time an attempt can take beyond its own timeout, e.g. a subprocess's
    own termination grace period) so a single attempt's worst case can never
    overshoot max_seconds; it is never inflated above what is actually left,
    even when that is under a second. Time spent in the loop's own backoff
    sleeps, and in success_pause_seconds after a genuinely new (non-cached)
    successful fetch, also counts against that budget. The checkpoint is
    persisted after every state change so a crash loses at most the
    in-flight attempt, not prior progress. This function makes no claim of
    "exactly once" delivery for an attempt whose true remote outcome could
    not be observed (e.g. an interrupted attempt) -- only that it is never
    dropped from budget accounting and never silently re-run for free.
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
                 f'cumulative_bytes_reserved={checkpoint["bytes_used"]} '
                 f'cumulative_bytes_measured={checkpoint.get("bytes_measured", 0)} '
                 f'cumulative_seconds={checkpoint["seconds_used"]:.0f}', flush=True)

        timestamps = groups[(station, day)]
        window_start = min(timestamps) - pd.Timedelta(hours=LOOKBACK_HOURS)
        window_end = max(timestamps) + pd.Timedelta(hours=LOOKAHEAD_HOURS)
        current_window = (window_start.isoformat(), window_end.isoformat())

        prior = checkpoint['groups'].get(key)
        if prior and prior.get('status') == 'fetched':
            # A resumed "fetched" checkpoint is re-validated, not trusted blindly: the SAME station/day
            # under a DIFFERENT query window is different work, not the same completed request, and the
            # recorded cache evidence must still exist on disk with unchanged content.
            prior_window = (prior.get('window_start_utc'), prior.get('window_end_utc'))
            if prior_window[0] is not None and prior_window != current_window:
                raise ValueError(
                    f'{key} was already fetched for window {prior_window[0]}..{prior_window[1]}, but '
                    f'the current plan asks for a different window {current_window[0]}..{current_window[1]} '
                    '-- this is a different request under the same station/day, not the same completed '
                    'work. Resolve the plan mismatch (e.g. a fresh --name) before resuming; existing '
                    'evidence is never silently overwritten or re-collected.')
            revalidated = cache_exists_fn(station, window_start, window_end)
            if revalidated is None:
                raise ValueError(
                    f'checkpoint records {key} as already fetched, but its cache file is now missing or '
                    'fails schema validation; resolve (restore the file, or start a fresh --name) before '
                    'resuming -- a missing cache is never silently re-fetched under the same recorded '
                    'evidence.')
            actual_sha = digest_fn(revalidated)
            if prior.get('sha256') and actual_sha != prior['sha256']:
                raise ValueError(
                    f'{revalidated} content ({actual_sha}) no longer matches the checkpoint\'s recorded '
                    f'evidence ({prior["sha256"]}) for {key}; the cache file appears to have changed '
                    'since it was fetched. Existing evidence is never silently trusted again after it '
                    'changes -- resolve the mismatch before resuming.')
            fetched.append({**prior, 'station': station, 'day': day, 'cache_file': str(revalidated)})
            continue
        if prior and prior.get('status') == 'failed' and prior.get('permanent'):
            failed.append({**prior, 'station': station, 'day': day})
            continue

        # A cache hit found here has NO prior 'fetched' record in this checkpoint -- it is either
        # evidence left over from a different run (the original 21-row sample, an interrupted prior
        # attempt under a different --name, etc.) or a foreign/corrupted file. cache_exists_fn already
        # validated its schema; a recorded hash (if this exact key was ever fetched under THIS run
        # before, e.g. the checkpoint was reset but the file survived) is still cross-checked so
        # stale/foreign evidence is never silently substituted for what this run believes it already
        # fetched.
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
            usable = remaining - attempt_overhead_seconds
            if usable <= 0:
                cap_hit = (f'max_seconds={max_seconds} reached (remaining {remaining:.1f}s does not '
                          f'leave room for the {attempt_overhead_seconds}s worst-case per-attempt '
                          'overhead)')
                break
            timeout = min(default_attempt_timeout, usable)  # never inflated above what is actually left

            # This attempt can transfer at most effective_cap bytes (the SAME bound fetch_attempt itself
            # enforces via curl's --max-filesize), so that amount -- not 0 -- is reserved against bytes_used
            # BEFORE the call. A crash mid-attempt therefore leaves bytes_used already conservatively
            # charged; a resolved-but-unmeasurable outcome (bytes_received is None) keeps that charge
            # standing instead of being refunded, so the cumulative byte cap can never be bypassed by
            # attempts whose real usage is unknown. Only a MEASURED outcome replaces the reservation with
            # the actual bytes transferred.
            bytes_remaining_before = max_bytes - checkpoint['bytes_used']
            effective_cap = min(MAX_RESPONSE_BYTES, bytes_remaining_before)

            # Reserve THIS attempt's request AND byte budget and persist BEFORE the network call: if the
            # process dies mid-attempt, both reservations survive on disk and are never re-issued as a
            # free request, nor refunded as if 0 bytes were used, when this run is resumed
            # (recover_interrupted_attempts resolves it).
            checkpoint['requests_used'] += 1
            checkpoint['bytes_used'] += effective_cap
            checkpoint['groups'][key] = {'status': 'in_progress', 'attempts_detail': attempts_this_group,
                                         'in_flight': {'attempt': len(attempts_this_group) + 1,
                                                      'timeout': timeout,
                                                      'attempt_overhead_seconds': attempt_overhead_seconds,
                                                      'bytes_reserved': effective_cap}}
            persist()

            t0 = now_fn()
            outcome = attempt_fn(station, window_start, window_end, timeout=timeout,
                                 max_bytes_remaining=bytes_remaining_before)
            elapsed = now_fn() - t0
            checkpoint['seconds_used'] += elapsed
            session_requests += 1
            if outcome.get('bytes_received') is not None:
                # Settle: replace the conservative reservation with the ACTUAL measured amount (which may
                # be smaller OR larger than effective_cap, e.g. the bounded-size-exceeded failure path).
                checkpoint['bytes_used'] += outcome['bytes_received'] - effective_cap
                checkpoint['bytes_measured'] = checkpoint.get('bytes_measured', 0) + outcome['bytes_received']
                session_bytes += outcome['bytes_received']
            else:
                # Genuinely unmeasurable: the reservation already charged above is left standing, never
                # assumed to be 0 -- this is what makes the cumulative max_bytes cap actually enforceable.
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
                # Be polite to the provider between successful requests; this pause is bounded by
                # whatever time budget remains and is charged against it, not free.
                pause = min(success_pause_seconds, max(remaining_seconds(), 0.0))
                if pause > 0:
                    sleep_fn(pause)
                    checkpoint['seconds_used'] += pause
                    persist()
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
                     'attempts': len(attempts_this_group), 'attempts_detail': attempts_this_group,
                     'status': 'fetched'}
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
        'cumulative_requests': checkpoint['requests_used'],
        # cumulative_bytes_reserved is the CONSERVATIVE value the max_bytes cap is actually enforced
        # against (measured bytes replace their attempt's reservation; an unmeasurable attempt's
        # reservation stands permanently) -- it is always >= real bytes downloaded. cumulative_bytes_measured
        # is the separate, real-usage-only figure; the two are never conflated.
        'cumulative_bytes_reserved': checkpoint['bytes_used'],
        'cumulative_bytes_measured': checkpoint.get('bytes_measured', 0),
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
    mapping_path = out / f'{args.mapping_name}_mapping_table.csv'
    mapping = pd.read_csv(mapping_path).set_index('iata')

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
    resuming = checkpoint_has_prior_progress(checkpoint)
    if resuming:
        print(f'resuming from checkpoint: cumulative_requests={checkpoint["requests_used"]} '
             f'cumulative_bytes_reserved={checkpoint["bytes_used"]} '
             f'cumulative_bytes_measured={checkpoint.get("bytes_measured", 0)} '
             f'cumulative_seconds={checkpoint["seconds_used"]:.0f}', flush=True)

    plan_fingerprint = compute_fetch_plan_fingerprint(
        digest(selection_path), digest(mapping_path), groups)
    verify_plan_fingerprint(checkpoint, plan_fingerprint, name=args.name)
    current_caps = record_budget_limits(
        checkpoint, max_requests=args.max_requests, max_bytes=args.max_bytes, max_seconds=args.max_seconds)
    # Persist plan identity and this invocation's budget before any network call.
    save_checkpoint(checkpoint_path, checkpoint)

    result = execute_with_caps(
        planned, groups, max_requests=args.max_requests, max_bytes=args.max_bytes,
        max_seconds=args.max_seconds, cache_exists_fn=validate_cached_window,
        attempt_fn=fetch_attempt, digest_fn=digest, now_fn=time.monotonic,
        checkpoint=checkpoint, checkpoint_path=checkpoint_path, sleep_fn=time.sleep,
        attempt_overhead_seconds=SUBPROCESS_TERMINATION_GRACE_SECONDS,
        success_pause_seconds=SUCCESS_PAUSE_SECONDS)
    fetched, skipped_cap, failed = result['fetched'], result['skipped_cap'], result['failed']

    selection.to_csv(out / f'{args.name}_selection_with_prediction_at.csv', index=False)
    manifest = {
        'name': args.name, 'mapping_name': args.mapping_name,
        'code_sha256': hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'caps': {**current_caps,
                'subprocess_termination_grace_seconds': SUBPROCESS_TERMINATION_GRACE_SECONDS,
                'success_pause_seconds': SUCCESS_PAUSE_SECONDS},
        'budget_limit_history': checkpoint['budget_limit_history'],
        'plan_fingerprint': plan_fingerprint,
        'resumed_from_checkpoint': resuming,
        'planned_station_day_groups': len(planned),
        'fetched_groups': len(fetched), 'skipped_due_to_cap': len(skipped_cap),
        'failed_groups': len(failed),
        'this_run_http_attempts': result['new_requests'], 'this_run_bytes_downloaded': result['new_bytes'],
        'this_run_elapsed_seconds': result['elapsed_seconds'],
        'cumulative_http_attempts': result['cumulative_requests'],
        'cumulative_bytes_downloaded': result['cumulative_bytes_measured'],
        'cumulative_bytes_reserved_against_budget': result['cumulative_bytes_reserved'],
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
            'cumulative_bytes_downloaded is the REAL measured total (only attempts with a known '
            'bytes_received count toward it); cumulative_bytes_reserved_against_budget is the separate, '
            'CONSERVATIVE figure the max_bytes cap is actually enforced against -- every attempt reserves '
            'min(a fixed per-request cap, the remaining budget) before the network call, a measured '
            'outcome replaces that reservation with the real byte count, and an unmeasurable outcome '
            '(bytes_received unknown, e.g. the response file was never created or the process was '
            'interrupted mid-attempt) leaves the reservation standing rather than refunding it to 0. '
            'unmeasured_byte_attempts counts how many attempts fall in that last category. The two '
            'cumulative_bytes_* fields are therefore never conflated: reserved is always >= measured, '
            'and only measured claims to reflect bytes actually transferred.',
            'A cap hit preserves everything fetched so far in the checkpoint; re-running with the '
            'SAME --name resumes without losing or re-spending prior budget, and any group that was '
            'cut off mid-retry or never attempted is retried first.',
            'Cache reuse validates the cached file\'s schema/station and, when the checkpoint already '
            'recorded a hash for that group, its content hash too; a mismatch raises rather than '
            'silently overwriting or trusting unverified existing evidence.',
            'A group already marked fetched in the checkpoint is RE-validated on every resume, not '
            'trusted blindly: a deleted/corrupted cache file, tampered content, or a changed query '
            'window for the same station/day all raise explicitly instead of silently re-fetching or '
            'keeping stale evidence.',
            'plan_fingerprint ties this checkpoint to the exact input selection file, mapping table, '
            'normalized per-(station, day) request windows, and immutable fetch-policy options '
            '(padding, per-attempt response/timeout/retry policy, grace/pause seconds). The cumulative '
            'run-level ceilings max_requests/max_bytes/max_seconds are intentionally NOT part of that '
            'identity: they may change on resume, while budget_limit_history records every distinct '
            'ceiling set and cumulative usage is never reset. Input/request/fixed-policy changes still '
            'raise rather than silently mixing incompatible evidence.',
            'The request AND byte budget for each HTTP attempt are reserved and persisted BEFORE the '
            'network call, not after; an attempt interrupted by a process crash is never re-issued as '
            'a free retry on resume, and its unknown byte/time cost is charged conservatively (the '
            'reserved min(per-request cap, remaining budget) bytes and the reserved timeout plus '
            'subprocess-termination-grace seconds added to elapsed time) rather than assumed to be zero.',
            'Exact-once delivery of a network request is never claimed: an interrupted attempt\'s true '
            'remote outcome may be unknown (it could have succeeded or failed on the server side before '
            'the process died); only the budget accounting and evidence-preservation guarantees above '
            'are made.',
            'Each attempt\'s bounded timeout reserves subprocess_termination_grace_seconds on top of '
            'the nominal timeout (matching fetch_attempt\'s underlying subprocess kill grace), so a '
            'single attempt\'s worst-case wall time cannot overshoot max_seconds; when the remaining '
            'budget cannot cover even that reserved grace, collection stops rather than attempting a '
            'request it cannot safely bound.',
            'success_pause_seconds is applied (and charged against max_seconds) after every genuinely '
            'new network fetch, not after a cache hit, so this path is polite to the provider between '
            'successive real requests the same way fetch_weather_sample.py\'s fetch_window already is.',
            'Per-attempt logs (attempts_detail) are kept on the group\'s own record for both failed AND '
            'successful groups; only the separate rolling attempts_log (capped at '
            f'{ATTEMPT_LOG_LIMIT} entries, a cross-group activity feed) is truncated -- truncating it '
            'never erases the per-group evidence that cumulative_* is accounted from.',
        ],
    }
    (out / f'{args.name}_fetch_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest.items() if k not in {'requests', 'skipped', 'failed', 'limitations'}},
                     indent=2, default=str))


if __name__ == '__main__':
    main()
