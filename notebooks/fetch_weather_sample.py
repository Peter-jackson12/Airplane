"""Fetch real, small, bounded IEM ASOS observations for exactly the
station/day windows the weather-sample selection needs. No target/delay
column exists in the selection file this reads, so none can leak in here.

Each request is scoped to one station and roughly one UTC day around the
row(s) that need it -- the same bounded-probe pattern as
notebooks/weather_feasibility.py's original connectivity check, just
parameterized over the selected sample instead of one fixed demo day.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from notebooks.select_weather_sample import AIRPORTS
from src.weather import local_hhmm_to_utc

ROOT = Path(__file__).resolve().parents[1]


class FetchError(RuntimeError):
    pass


def http_get_once(url: str, *, timeout: float, out_path: Path, max_bytes: int | None = None) -> dict:
    """Exactly ONE curl attempt (no retry, no sleep) via subprocess, not
    urllib -- confirmed by direct comparison on 2026-09-18 that identical
    requests urllib.request.urlopen hung on indefinitely (past its own
    `timeout=` and a process-wide socket.setdefaulttimeout backstop) were
    completed by curl in a few seconds.

    The response body is written to `out_path` (via curl's own -o) rather
    than captured in memory, so bytes actually received are measurable from
    disk even when curl is killed mid-transfer by a timeout or --max-filesize
    -- capturing stdout would lose that partial data along with the process.

    Returns a dict: success, bytes_received (int, or None only when out_path
    was never created at all -- never assumed to be 0 for a genuinely
    unmeasurable attempt), seconds (wall time for this attempt), returncode,
    error, timed_out. Callers own retry/backoff; this function never retries.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    cmd = ['curl', '-sS', '--max-time', str(timeout), '--fail',
          '-H', 'User-Agent: Airplane-course-weather-sample/1.0', '-o', str(out_path)]
    if max_bytes:
        cmd += ['--max-filesize', str(int(max_bytes))]
    cmd.append(url)
    start = time.monotonic()
    timed_out = False
    returncode = None
    error = None
    try:
        result = subprocess.run(cmd, capture_output=True,
                               timeout=timeout + SUBPROCESS_TERMINATION_GRACE_SECONDS, check=False)
        returncode = result.returncode
        if returncode != 0:
            error = f'curl exit {returncode}: {result.stderr.decode(errors="replace")[:500]}'
    except subprocess.TimeoutExpired:
        timed_out = True
        error = f'curl exceeded {timeout + SUBPROCESS_TERMINATION_GRACE_SECONDS}s wall-clock timeout for {url}'
    seconds = time.monotonic() - start
    bytes_received = out_path.stat().st_size if out_path.exists() else None
    return {'success': returncode == 0 and not timed_out, 'bytes_received': bytes_received,
           'seconds': seconds, 'returncode': returncode, 'error': error, 'timed_out': timed_out}


CACHE_DIR = ROOT / 'data/weather_probe/sample'
PREDICTION_OFFSET_MINUTES = 60  # fixed contract: 60 minutes before scheduled departure
LOOKBACK_HOURS = 6  # window padding before the earliest prediction_at in a group
LOOKAHEAD_HOURS = 2  # window padding after the latest prediction_at in a group
MAX_RESPONSE_BYTES = 2_000_000
# subprocess.run's own timeout is `timeout + this` (below), so curl gets a short grace window to shut
# down cleanly instead of racing a hard kill at exactly `timeout`. A caller bounding an attempt against a
# remaining time budget must reserve this on top of `timeout`, since real wall-clock time for one attempt
# can exceed `timeout` by up to this much in the worst case.
SUBPROCESS_TERMINATION_GRACE_SECONDS = 5.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_path_for(station: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    tag = f"{station}_{start.strftime('%Y%m%dT%H%M')}_{end.strftime('%Y%m%dT%H%M')}"
    return CACHE_DIR / f'iem_{tag}.csv'


def build_request_url(station: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    params = {'station': station, 'data': 'all',
             'sts': start.strftime('%Y-%m-%dT%H:%M:%SZ'), 'ets': end.strftime('%Y-%m-%dT%H:%M:%SZ'),
             'tz': 'UTC', 'format': 'onlycomma', 'latlon': 'yes',
             'missing': 'M', 'trace': 'T', 'report_type': [3, 4]}
    return 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?' + urlencode(params, doseq=True)


def validate_cached_window(station: str, start: pd.Timestamp, end: pd.Timestamp) -> Path | None:
    """Returns the cache path if a schema-valid cache file already exists for
    exactly this (station, window) request, else None (never partially/blindly
    trusted). A file that exists but fails to parse or does not match the
    expected archive schema/station is a hard error, not a silent re-fetch --
    existing evidence must not be quietly overwritten or bypassed."""
    cache = cache_path_for(station, start, end)
    if not cache.exists():
        return None
    try:
        head = pd.read_csv(cache, comment='#', na_values=['M'], keep_default_na=False, nrows=5)
    except Exception as exc:
        raise ValueError(f'{cache} exists but failed to parse ({exc}); resolve before reuse') from exc
    if not {'station', 'valid'}.issubset(head.columns):
        raise ValueError(f'{cache} exists but does not match the expected archive schema')
    if len(head) and (head.station != station).any():
        raise ValueError(f'{cache} station column does not match requested station {station}')
    return cache


def fetch_attempt(station: str, start: pd.Timestamp, end: pd.Timestamp, *, timeout: float,
                  max_bytes_remaining: int) -> dict:
    """One bounded HTTP attempt for a (station, window) request; never
    retries internally. The caller owns retry/backoff/budget accounting so
    every attempt -- successful or not -- can be charged against the request
    budget, and partial bytes from a failed attempt are still reported.

    The in-transit size limit is min(MAX_RESPONSE_BYTES, max_bytes_remaining):
    a single attempt is never allowed to exceed the fixed per-request cap even
    when the caller's overall byte budget still has room left, and it is
    tightened further once the overall budget has less than that much left.
    Both curl's own --max-filesize cutoff and the post-download validation use
    this SAME effective cap, so a response cannot pass one and fail the other.
    """
    url = build_request_url(station, start, end)
    cache = cache_path_for(station, start, end)
    part = cache.with_name(cache.name + '.part')
    effective_cap = (MAX_RESPONSE_BYTES if max_bytes_remaining is None
                     else min(MAX_RESPONSE_BYTES, max_bytes_remaining))
    if effective_cap <= 0:
        return {'success': False, 'bytes_received': None, 'seconds': 0.0,
               'error': 'no remaining byte budget for this attempt', 'url': url}
    outcome = http_get_once(url, timeout=timeout, out_path=part, max_bytes=effective_cap)
    if not outcome['success']:
        if part.exists():
            part.unlink()
        return {'success': False, 'bytes_received': outcome['bytes_received'], 'seconds': outcome['seconds'],
               'error': outcome['error'], 'url': url}
    if outcome['bytes_received'] is not None and outcome['bytes_received'] > effective_cap:
        part.unlink(missing_ok=True)
        return {'success': False, 'bytes_received': outcome['bytes_received'], 'seconds': outcome['seconds'],
               'error': 'response exceeds bounded size', 'url': url}
    try:
        parsed = pd.read_csv(part, comment='#', na_values=['M'], keep_default_na=False)
        if not {'station', 'valid'}.issubset(parsed.columns):
            raise ValueError('unexpected archive schema; response not cached')
    except Exception as exc:
        part.unlink(missing_ok=True)
        return {'success': False, 'bytes_received': outcome['bytes_received'], 'seconds': outcome['seconds'],
               'error': f'response failed validation: {exc}', 'url': url}
    part.replace(cache)
    return {'success': True, 'bytes_received': outcome['bytes_received'], 'seconds': outcome['seconds'],
           'url': url, 'cache_path': cache, 'cache_file': str(cache.relative_to(ROOT)),
           'n_rows': len(parsed), 'sha256': digest(cache)}


def compute_prediction_at(selection: pd.DataFrame) -> pd.Series:
    """departure_utc - 60min, using each row's own origin timezone.
    local_hhmm_to_utc takes one timezone per call, so this groups by zone."""
    dates = pd.to_datetime(selection.attributed_date)
    parts = []
    for tz, idx in selection.groupby('origin_timezone').groups.items():
        departure_utc = local_hhmm_to_utc(dates.loc[idx], selection.loc[idx, 'Estimated_Departure_Time'], tz)
        parts.append(departure_utc)
    departure_utc = pd.concat(parts).reindex(selection.index)
    return departure_utc - pd.Timedelta(minutes=PREDICTION_OFFSET_MINUTES)


def fetch_window(station: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[Path, str, int]:
    """Small, uncapped fetch for the original 21-row sample only (no request/
    byte/time budget here -- that resource accounting lives in
    execute_with_caps for the expanded sample). Retries up to 3 attempts with
    linear backoff, each attempt charged nothing but its own wall time since
    this path has no shared budget to protect."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = build_request_url(station, start, end)
    cache = validate_cached_window(station, start, end)
    if cache is None:
        outcome = None
        for attempt in range(3):
            outcome = fetch_attempt(station, start, end, timeout=15, max_bytes_remaining=MAX_RESPONSE_BYTES)
            if outcome['success']:
                break
            if attempt == 2:
                raise FetchError(outcome['error'])
            time.sleep(10 * (attempt + 1))
        cache = outcome['cache_path']
        time.sleep(3)  # be polite to the free public archive between requests
    obs = pd.read_csv(cache, comment='#', na_values=['M'], keep_default_na=False)
    return cache, url, len(obs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    args = ap.parse_args()
    out = ROOT / 'output'
    selection_path = out / f'{args.name}_selection.csv'
    if not selection_path.exists():
        raise FileNotFoundError(f'Run select_weather_sample.py --name {args.name} first')

    selection = pd.read_csv(selection_path)
    selection['prediction_at'] = compute_prediction_at(selection)
    unresolved = selection.loc[selection.prediction_at.isna()]
    if len(unresolved):
        # Expected for a row whose scheduled LOCAL departure falls in the DST
        # fall-back hour that occurs twice: local_hhmm_to_utc's explicit,
        # unmodified policy is ambiguous -> NaT, not an arbitrary guess. Such
        # rows carry no valid prediction_at and are left unmatched downstream,
        # not silently dropped or resolved here.
        print('rows with no valid prediction_at (DST ambiguous/nonexistent local time, '
              'left unmatched by policy):', unresolved.ID.tolist(), flush=True)

    # one fetch group per (station, UTC calendar day of prediction_at), for
    # both the origin and destination role of every selected row that HAS a
    # valid prediction_at
    groups: dict[tuple[str, str], list[pd.Timestamp]] = {}
    for _, row in selection.dropna(subset=['prediction_at']).iterrows():
        for airport in (row.Origin_Airport, row.Destination_Airport):
            station = AIRPORTS[airport]['station']
            key = (station, row.prediction_at.strftime('%Y-%m-%d'))
            groups.setdefault(key, []).append(row.prediction_at)

    fetched = []
    for (station, day), timestamps in sorted(groups.items()):
        start = min(timestamps) - pd.Timedelta(hours=LOOKBACK_HOURS)
        end = max(timestamps) + pd.Timedelta(hours=LOOKAHEAD_HOURS)
        cache, url, n_rows = fetch_window(station, start, end)
        fetched.append({'station': station, 'day': day, 'window_start_utc': start.isoformat(),
                        'window_end_utc': end.isoformat(), 'url': url,
                        'cache_file': str(cache.relative_to(ROOT)), 'sha256': digest(cache),
                        'rows': n_rows, 'n_requests_in_group': len(timestamps)})
        print(fetched[-1], flush=True)

    selection.to_csv(out / f'{args.name}_selection_with_prediction_at.csv', index=False)
    manifest = {
        'name': args.name, 'prediction_offset_minutes': PREDICTION_OFFSET_MINUTES,
        'lookback_hours': LOOKBACK_HOURS, 'lookahead_hours': LOOKAHEAD_HOURS,
        'source': 'IEM ASOS (mesonet.agron.iastate.edu/request/asos.py)',
        'fits_any_model': False, 'uses_external_data': True, 'target_columns_used': [],
        'stations_fetched': sorted({f['station'] for f in fetched}),
        'requests': fetched, 'total_requests': len(fetched),
        'total_obs_rows_downloaded': sum(f['rows'] for f in fetched),
        'limitations': [
            'IEM valid time is not the archived original receipt/publication timestamp; '
            'available_at is a documented assumption applied downstream, not a verified fact.',
            'This is a bounded probe for the 21-row sample only, not full-archive weather collection.',
        ],
    }
    (out / f'{args.name}_fetch_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'requests'}, indent=2))


if __name__ == '__main__':
    main()
