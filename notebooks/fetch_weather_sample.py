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
import io
import json
import time
import urllib.error
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from notebooks.select_weather_sample import AIRPORTS
from src.weather import local_hhmm_to_utc

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / 'data/weather_probe/sample'
PREDICTION_OFFSET_MINUTES = 60  # fixed contract: 60 minutes before scheduled departure
LOOKBACK_HOURS = 6  # window padding before the earliest prediction_at in a group
LOOKAHEAD_HOURS = 2  # window padding after the latest prediction_at in a group


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{station}_{start.strftime('%Y%m%dT%H%M')}_{end.strftime('%Y%m%dT%H%M')}"
    cache = CACHE_DIR / f'iem_{tag}.csv'
    params = {'station': station, 'data': 'all',
              'sts': start.strftime('%Y-%m-%dT%H:%M:%SZ'), 'ets': end.strftime('%Y-%m-%dT%H:%M:%SZ'),
              'tz': 'UTC', 'format': 'onlycomma', 'latlon': 'yes',
              'missing': 'M', 'trace': 'T', 'report_type': [3, 4]}
    url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?' + urlencode(params, doseq=True)
    if not cache.exists():
        for attempt in range(5):
            try:
                req = Request(url, headers={'User-Agent': 'Airplane-course-weather-sample/1.0'})
                with urlopen(req, timeout=60) as r:
                    body = r.read(2_000_001)
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError('unreachable')
        if len(body) > 2_000_000:
            raise ValueError('Response exceeds bounded size')
        parsed = pd.read_csv(io.BytesIO(body), comment='#', na_values=['M'], keep_default_na=False)
        if not {'station', 'valid'}.issubset(parsed):
            raise ValueError('Unexpected archive schema; response not cached')
        cache.write_bytes(body)
        time.sleep(2)  # be polite to the free public archive between requests
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
