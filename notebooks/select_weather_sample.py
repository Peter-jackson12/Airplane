"""Select a small, target-blind weather validation sample.

Selection uses ONLY the row-level date attribution's ADOPTED rows
(status == complete_single_candidate_year, from
baseline_recovery_v2_row_date_attribution_20260918) plus non-target source
columns (airports, scheduled departure/arrival time). No target, delay,
actual-time, or weather value is read or used to pick rows. The rule below
is deterministic (bucket + smallest-ID tiebreak), not random, so it is
reproducible without a seed.

Airports are restricted to a small, individually-verified set so the
airport -> IANA timezone -> weather-station mapping can be checked by hand
against a cited source, instead of trusting an automated mapping for all
374 airports in the file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTION_RUN = 'baseline_recovery_v2_row_date_attribution_20260918'
MWGG_URL = 'https://raw.githubusercontent.com/mwgg/Airports/master/airports.json'
MWGG_CACHE = ROOT / 'data/weather_probe/airports_mwgg.json'
MWGG_EXPECTED_SHA256 = 'f369eaa1c2944280d9678a96d5b477cedea0417f3c12a333c39834bf1c035739'

# Chosen for IANA-timezone spread (5 distinct zones; DEN/PHX both "Mountain"
# label but PHX does not observe DST) and confirmed live IEM ASOS coverage
# (checked 2026-09-18: plain 3-letter station id works for all but ANC,
# which IEM only serves under its ICAO id PANC, not the IATA code ANC).
AIRPORTS = {
    'ATL': {'icao': 'KATL', 'station': 'ATL'},
    'ORD': {'icao': 'KORD', 'station': 'ORD'},
    'DEN': {'icao': 'KDEN', 'station': 'DEN'},
    'LAX': {'icao': 'KLAX', 'station': 'LAX'},
    'PHX': {'icao': 'KPHX', 'station': 'PHX'},
    'ANC': {'icao': 'PANC', 'station': 'PANC'},
    'LAS': {'icao': 'KLAS', 'station': 'LAS'},
    'AUS': {'icao': 'KAUS', 'station': 'AUS'},
}

# Actual US DST transition Sundays covering the two candidate years, and the
# LOCAL wall-clock hour each discontinuity affects (spring-forward 02:00-03:00
# does not exist; fall-back 01:00-02:00 occurs twice). America/Phoenix does
# not observe DST, so it never has this discontinuity even on these dates.
DST_TRANSITION_DATES = {
    '2018-03-11': ('spring', 200, 300), '2019-03-10': ('spring', 200, 300),
    '2018-11-04': ('fall', 100, 200), '2019-11-03': ('fall', 100, 200),
}
NO_DST_TIMEZONES = {'America/Phoenix'}
NEAR_MIDNIGHT_MINUTES = 60  # local HHMM within this many minutes of 00:00


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_airport_timezones() -> pd.DataFrame:
    """Re-derive the tz mapping from the cached open dataset, so the mapping
    is checked against its source every run instead of hand-typed and trusted."""
    if not MWGG_CACHE.exists():
        raise FileNotFoundError(
            f'{MWGG_CACHE} missing. Fetch it once with: '
            f'curl -sS "{MWGG_URL}" -o "{MWGG_CACHE}"')
    actual_sha = digest(MWGG_CACHE)
    if actual_sha != MWGG_EXPECTED_SHA256:
        raise ValueError(
            f'{MWGG_CACHE} hash changed ({actual_sha}); re-verify before using it '
            'as a timezone source')
    data = json.loads(MWGG_CACHE.read_text(encoding='utf-8'))
    rows = []
    for iata, meta in AIRPORTS.items():
        entry = data.get(meta['icao'])
        if entry is None or entry.get('iata') != iata:
            raise ValueError(f'{iata}: ICAO {meta["icao"]} not found or IATA mismatch in source')
        if not entry.get('tz'):
            raise ValueError(f'{iata}: source has no timezone')
        rows.append({'iata': iata, 'icao': meta['icao'], 'iem_station': meta['station'],
                     'iana_timezone': entry['tz'], 'source_city': entry.get('city'),
                     'source_state': entry.get('state')})
    return pd.DataFrame(rows)


def minutes_from_midnight_distance(hhmm: pd.Series) -> pd.Series:
    value = pd.to_numeric(hhmm, errors='coerce')
    minutes = (value.floordiv(100) * 60 + value.mod(100)).where(value.lt(2400), 0)
    return pd.concat([minutes, 1440 - minutes], axis=1).min(axis=1)


def is_dst_transition_departure(attributed_date: str, origin_timezone: str, departure_hhmm: float) -> bool:
    """True only if the scheduled LOCAL departure falls inside the actual clock
    discontinuity for that origin's timezone on a real US DST-transition Sunday
    (spring-forward 02:00-03:00 does not exist; fall-back 01:00-02:00 repeats).
    Flagging the whole calendar day would pull in every unrelated flight, and a
    non-DST-observing origin (Phoenix) never has this discontinuity at all."""
    spec = DST_TRANSITION_DATES.get(attributed_date)
    if spec is None or origin_timezone in NO_DST_TIMEZONES or pd.isna(departure_hhmm):
        return False
    _, lo, hi = spec
    return lo <= departure_hhmm < hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    args = ap.parse_args()
    if not args.name.startswith('baseline_recovery_v2') or Path(args.name).name != args.name:
        raise ValueError('Invalid output name')
    out = ROOT / 'output'
    if list(out.glob(args.name + '_selection*')):
        raise FileExistsError('Use a fresh name; existing evidence is preserved')

    tz_map = load_airport_timezones()
    tz_map.to_csv(out / f'{args.name}_airport_stations.csv', index=False)

    manifest = json.loads((out / f'{ATTRIBUTION_RUN}_manifest.json').read_text())
    row_path = ROOT / manifest['row_evidence']
    if digest(row_path) != manifest['row_sha256']:
        raise ValueError(f'{row_path} does not match {ATTRIBUTION_RUN}\'s recorded hash')
    attribution = pd.read_csv(row_path, low_memory=False)[
        ['ID', 'row_position', 'status', 'date_attributed', 'attributed_year', 'attributed_date']]
    adopted = attribution.loc[attribution.status.eq('complete_single_candidate_year')].copy()
    assert adopted.date_attributed.all()

    source = ROOT / 'data/train.csv'
    raw = pd.read_csv(source, usecols=[
        'ID', 'Origin_Airport', 'Destination_Airport',
        'Estimated_Departure_Time', 'Estimated_Arrival_Time'])
    pool = adopted.merge(raw, on='ID', how='left', validate='one_to_one')
    if pool[['Origin_Airport', 'Destination_Airport']].isna().any().any():
        raise ValueError('adopted rows must have both airports observed (complete fingerprint)')

    allowed = set(AIRPORTS)
    pool = pool.loc[pool.Origin_Airport.isin(allowed) & pool.Destination_Airport.isin(allowed)].copy()
    pool['origin_timezone'] = pool.Origin_Airport.map(tz_map.set_index('iata').iana_timezone)
    pool['year'] = pool.attributed_date.str.slice(0, 4)
    pool['quarter'] = pool.attributed_date.str.slice(5, 7).astype(int).sub(1).floordiv(3).add(1)
    pool['midnight_distance_minutes'] = minutes_from_midnight_distance(pool.Estimated_Departure_Time)
    pool['near_midnight'] = pool.midnight_distance_minutes.le(NEAR_MIDNIGHT_MINUTES)
    pool['dst_transition_date'] = pool.apply(
        lambda row: is_dst_transition_departure(
            row.attributed_date, row.origin_timezone, row.Estimated_Departure_Time), axis=1)
    pool = pool.sort_values('ID').reset_index(drop=True)

    picks = {}  # ID -> set of reasons

    def add(frame, reason):
        for row_id in frame.ID:
            picks.setdefault(row_id, set()).add(reason)

    # 1) year x quarter x timezone coverage: round-robin through airports
    #    (alphabetical) across (year, quarter) buckets, smallest ID per bucket.
    airport_cycle = sorted(AIRPORTS)
    i = 0
    for year in sorted(pool.year.unique()):
        for quarter in range(1, 5):
            bucket = pool.loc[pool.year.eq(year) & pool.quarter.eq(quarter)]
            if bucket.empty:
                continue
            for _ in range(len(airport_cycle)):
                airport = airport_cycle[i % len(airport_cycle)]
                i += 1
                candidate = bucket.loc[bucket.Origin_Airport.eq(airport)]
                if not candidate.empty:
                    add(candidate.head(1), 'year_quarter_timezone_coverage')
                    break

    # 2) near-midnight local departures, one per (year, origin_timezone), capped.
    midnight_pool = pool.loc[pool.near_midnight]
    for (_, _), bucket in midnight_pool.groupby(['year', 'origin_timezone']):
        add(bucket.head(1), 'near_midnight_departure')

    # 3) actual US DST-transition-Sunday cases near the clock discontinuity,
    #    one per (transition date, origin timezone), capped.
    dst_pool = pool.loc[pool.dst_transition_date]
    for (_, _), bucket in dst_pool.groupby(['attributed_date', 'origin_timezone']):
        add(bucket.head(1), 'dst_transition_date')

    selected_ids = sorted(picks)
    selection = pool.loc[pool.ID.isin(selected_ids)].copy()
    selection['selection_reasons'] = selection.ID.map(lambda i: '+'.join(sorted(picks[i])))
    selection = selection.sort_values('ID').reset_index(drop=True)

    cols = ['ID', 'row_position', 'attributed_year', 'attributed_date', 'year', 'quarter',
            'Origin_Airport', 'Destination_Airport', 'origin_timezone',
            'Estimated_Departure_Time', 'Estimated_Arrival_Time',
            'midnight_distance_minutes', 'near_midnight', 'dst_transition_date',
            'selection_reasons']
    selection[cols].to_csv(out / f'{args.name}_selection.csv', index=False)

    stations_needed = sorted(set(selection.Origin_Airport) | set(selection.Destination_Airport))
    dates_needed = sorted(selection.attributed_date.unique())
    manifest_out = {
        'name': args.name,
        'source_sha256': digest(source),
        'attribution_run': ATTRIBUTION_RUN,
        'attribution_row_sha256': manifest['row_sha256'],
        'code_sha256': hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'fits_any_model': False, 'uses_external_data': True, 'target_columns_used': [],
        'timezone_source_url': MWGG_URL, 'timezone_source_sha256': MWGG_EXPECTED_SHA256,
        'eligible_pool_rows': int(len(pool)), 'selected_rows': int(len(selection)),
        'selection_reason_counts': {r: int(selection.selection_reasons.str.contains(r).sum())
                                    for r in ['year_quarter_timezone_coverage',
                                             'near_midnight_departure', 'dst_transition_date']},
        'airports_in_sample': stations_needed,
        'iem_stations_needed': sorted({AIRPORTS[a]['station'] for a in stations_needed}),
        'dates_needed_local_calendar': dates_needed,
        'years_represented': sorted(selection.year.unique().tolist()),
        'limitations': [
            f'Selection pool is restricted to {len(AIRPORTS)} individually-verified airports; it is '
            'not a representative sample of the full 706,759 adopted rows or of the other airports.',
            'attributed_date/attributed_year come from the 2018/2019 BTS candidate comparison; '
            'this sample inherits that scope and does not confirm a year outside those two.',
            'No target, delay, actual-time, or weather value was read to build this selection.',
        ],
    }
    (out / f'{args.name}_selection_manifest.json').write_text(json.dumps(manifest_out, indent=2))
    print(json.dumps(manifest_out, indent=2))
    print(selection[cols].to_string(index=False))


if __name__ == '__main__':
    main()
