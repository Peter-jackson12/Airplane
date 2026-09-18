"""Fetch real, bounded IEM ASOS observations for the stratified expanded
sample, reusing fetch_weather_sample.py's generic per-(station, UTC-day)
downloader and src.weather unmodified. Generalizes station resolution to the
full mapping table (map_weather_stations.py) instead of the original 8-airport
constant, since the expanded sample deliberately spans many more airports.

Hard resource caps for THIS step (not a performance target -- a resource
boundary for this validation round): at most --max-requests new HTTP
requests, --max-bytes new bytes downloaded, and --max-seconds of online time.
Already-cached (station, day) windows do not count against these caps. On
reaching a cap, already-fetched data and a manifest of what was and was not
attempted are saved; nothing already cached is discarded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from notebooks.fetch_weather_sample import compute_prediction_at, fetch_window

ROOT = Path(__file__).resolve().parents[1]
LOOKBACK_HOURS = 6
LOOKAHEAD_HOURS = 2


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_station_day_groups(collectible: pd.DataFrame) -> dict[tuple[str, str], list[pd.Timestamp]]:
    """One group per (station, UTC calendar day of prediction_at), covering
    both the origin_station and destination_station role of every row."""
    groups: dict[tuple[str, str], list[pd.Timestamp]] = {}
    for _, row in collectible.iterrows():
        for station in (row.origin_station, row.destination_station):
            key = (station, row.prediction_at.strftime('%Y-%m-%d'))
            groups.setdefault(key, []).append(row.prediction_at)
    return groups


def execute_with_caps(planned, groups, *, max_requests, max_bytes, max_seconds,
                      cache_exists_fn, fetch_fn, digest_fn, now_fn, progress_every=10,
                      size_fn=lambda p: p.stat().st_size):
    """Runs the capped fetch loop; `fetch_fn(station, start, end) -> (cache_path,
    url, n_rows)` and `cache_exists_fn(cache_path) -> bool` are injected so this
    is testable without real HTTP or disk state."""
    fetched, skipped_cap, failed = [], [], []
    new_requests = 0
    new_bytes = 0
    start_time = now_fn()
    cap_hit = None
    consecutive_failures = 0
    max_consecutive_failures = 8  # a persistent block (e.g. sustained rate-limit) should stop, not spin
    for i, (station, day) in enumerate(planned):
        if i % progress_every == 0:
            print(f'[{i}/{len(planned)}] new_requests={new_requests} new_bytes={new_bytes} '
                 f'failed={len(failed)} elapsed={now_fn() - start_time:.0f}s', flush=True)
        if cap_hit:
            skipped_cap.append({'station': station, 'day': day, 'reason': cap_hit})
            continue
        timestamps = groups[(station, day)]
        window_start = min(timestamps) - pd.Timedelta(hours=LOOKBACK_HOURS)
        window_end = max(timestamps) + pd.Timedelta(hours=LOOKAHEAD_HOURS)
        tag = f"{station}_{window_start.strftime('%Y%m%dT%H%M')}_{window_end.strftime('%Y%m%dT%H%M')}"
        cache_path = ROOT / 'data/weather_probe/sample' / f'iem_{tag}.csv'
        was_cached = cache_exists_fn(cache_path)
        if not was_cached:
            if new_requests >= max_requests:
                cap_hit = f'max_requests={max_requests} reached'
            elif new_bytes >= max_bytes:
                cap_hit = f'max_bytes={max_bytes} reached'
            elif now_fn() - start_time >= max_seconds:
                cap_hit = f'max_seconds={max_seconds} reached'
            if cap_hit:
                skipped_cap.append({'station': station, 'day': day, 'reason': cap_hit})
                continue
            print(f'  fetching {station} {day} ...', flush=True)
        try:
            cache, url, n_rows = fetch_fn(station, window_start, window_end)
        except Exception as exc:  # noqa: BLE001 -- record and continue; do not lose progress
            failed.append({'station': station, 'day': day, 'error': str(exc)})
            consecutive_failures += 1
            print(f'  FAILED {station} {day}: {exc}', flush=True)
            if consecutive_failures >= max_consecutive_failures:
                cap_hit = (f'{max_consecutive_failures} consecutive failures '
                          '(likely a sustained rate limit or outage); stopping early')
            continue
        consecutive_failures = 0
        if not was_cached:
            new_requests += 1
            new_bytes += size_fn(cache)
        cache_rel = str(cache.relative_to(ROOT)) if cache.is_absolute() else str(cache)
        fetched.append({'station': station, 'day': day, 'window_start_utc': window_start.isoformat(),
                        'window_end_utc': window_end.isoformat(), 'url': url,
                        'cache_file': cache_rel, 'sha256': digest_fn(cache),
                        'rows': n_rows, 'was_already_cached': was_cached})
    return {'fetched': fetched, 'skipped_cap': skipped_cap, 'failed': failed,
            'new_requests': new_requests, 'new_bytes': new_bytes,
            'elapsed_seconds': now_fn() - start_time, 'cap_hit': cap_hit}


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
    print(f'planned station-day groups: {len(planned)} (cap: {args.max_requests} new requests)', flush=True)

    result = execute_with_caps(
        planned, groups, max_requests=args.max_requests, max_bytes=args.max_bytes,
        max_seconds=args.max_seconds, cache_exists_fn=lambda p: p.exists(),
        fetch_fn=fetch_window, digest_fn=digest, now_fn=time.monotonic)
    fetched, skipped_cap, failed = result['fetched'], result['skipped_cap'], result['failed']

    selection.to_csv(out / f'{args.name}_selection_with_prediction_at.csv', index=False)
    manifest = {
        'name': args.name, 'mapping_name': args.mapping_name,
        'code_sha256': hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'caps': {'max_requests': args.max_requests, 'max_bytes': args.max_bytes,
                'max_seconds': args.max_seconds},
        'planned_station_day_groups': len(planned),
        'fetched_groups': len(fetched), 'skipped_due_to_cap': len(skipped_cap),
        'failed_groups': len(failed),
        'new_requests_made': result['new_requests'], 'new_bytes_downloaded': result['new_bytes'],
        'elapsed_seconds': result['elapsed_seconds'], 'cap_that_stopped_collection': result['cap_hit'],
        'requests': fetched, 'skipped': skipped_cap, 'failed': failed,
        'fits_any_model': False, 'uses_external_data': True, 'target_columns_used': [],
        'limitations': [
            'not_collectible rows (mapping unconfirmed/tz_conflict/current-only) are never fetched; '
            'they remain in the selection with their reason and are reported, not dropped.',
            'A cap hit preserves everything fetched so far and lists every skipped group so the run '
            'can be resumed with a fresh --name later; it does not silently truncate the sample.',
        ],
    }
    (out / f'{args.name}_fetch_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest.items() if k not in {'requests', 'skipped', 'failed', 'limitations'}},
                     indent=2, default=str))


if __name__ == '__main__':
    main()
