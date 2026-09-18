"""Join the fetched real IEM ASOS observations onto the 21-row weather sample
using src/weather.py's UNMODIFIED join_weather_asof/local_hhmm_to_utc, and
verify every boundary condition the task requires.

available_at is NOT a recorded historical receipt time -- IEM's `valid` field
is the observation's nominal time, not when it was actually published. This
script assumes a fixed publication latency (documented below) and reports the
result as a "past join under an assumed latency", not a verified point-in-time
replay. Nothing here fits a model or reads Delay/target/actual-time columns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from notebooks.select_weather_sample import AIRPORTS
from src.weather import join_weather_asof, local_hhmm_to_utc

ROOT = Path(__file__).resolve().parents[1]
MAX_OBS_AGE = '90min'
ASSUMED_PUBLICATION_LATENCY_MINUTES = 10  # documented assumption; not a verified archive fact
WEATHER_FIELDS = ['tmpf', 'dwpf', 'relh', 'sknt', 'gust', 'vsby', 'p01i', 'skyc1', 'wxcodes', 'snowdepth', 'metar']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_observations(fetch_manifest: dict) -> pd.DataFrame:
    frames = []
    for entry in fetch_manifest['requests']:
        path = ROOT / entry['cache_file']
        if digest(path) != entry['sha256']:
            raise ValueError(f'{path} content changed since it was fetched')
        frames.append(pd.read_csv(path, comment='#', na_values=['M'], keep_default_na=False))
    obs = pd.concat(frames, ignore_index=True)
    before = len(obs)
    # Overlapping fetch windows (adjacent dates share a lookback/lookahead
    # buffer) can re-download the exact same METAR line twice. An identical
    # (station, valid, metar) repeat is a re-fetch artifact, not a correction;
    # a differing metar for the same (station, valid) would be a real
    # duplicate-report conflict and is NOT silently resolved here.
    conflicting = obs.duplicated(['station', 'valid'], keep=False) & ~obs.duplicated(
        ['station', 'valid', 'metar'], keep=False)
    if conflicting.any():
        raise ValueError('Conflicting reports at the same station/time; resolve before joining')
    obs = obs.drop_duplicates(['station', 'valid']).reset_index(drop=True)
    obs['observed_at'] = pd.to_datetime(obs.pop('valid'), utc=True)
    obs['available_at'] = obs.observed_at + pd.Timedelta(minutes=ASSUMED_PUBLICATION_LATENCY_MINUTES)
    return obs[['station', 'observed_at', 'available_at'] + WEATHER_FIELDS], before - len(obs)


def run_join(requests: pd.DataFrame, obs: pd.DataFrame, prefix: str) -> pd.DataFrame:
    joined = join_weather_asof(requests[['station', 'prediction_at']], obs, max_age=MAX_OBS_AGE)
    joined.index = requests.index
    return joined.add_prefix(f'{prefix}_')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    args = ap.parse_args()
    out = ROOT / 'output'
    local = ROOT / 'data/weather_probe' / f'{args.name}_joined'
    if local.exists() or (out / f'{args.name}_join_summary.csv').exists():
        raise FileExistsError('Use a fresh name; existing evidence is preserved')

    selection = pd.read_csv(out / f'{args.name}_selection_with_prediction_at.csv')
    selection['prediction_at'] = pd.to_datetime(selection.prediction_at, utc=True)
    fetch_manifest = json.loads((out / f'{args.name}_fetch_manifest.json').read_text())
    obs, dropped_exact_duplicates = load_observations(fetch_manifest)

    forbidden = {'Delay', 'delay', 'Not_Delayed', 'Delayed', 'ArrDelay', 'DepDelay',
                'Actual_Departure_Time', 'Actual_Arrival_Time'}
    if forbidden & set(selection.columns):
        raise ValueError('Target/actual-outcome columns must not reach the weather join')

    origin_req = selection[['ID', 'prediction_at']].copy()
    origin_req['station'] = selection.Origin_Airport.map(lambda a: AIRPORTS[a]['station'])
    dest_req = selection[['ID', 'prediction_at']].copy()
    dest_req['station'] = selection.Destination_Airport.map(lambda a: AIRPORTS[a]['station'])

    origin = run_join(origin_req, obs, 'origin')
    dest = run_join(dest_req, obs, 'destination')
    # both roles were asked at the SAME prediction_at (already in `selection`);
    # confirm that instead of carrying two more copies of the column through
    assert origin.origin_prediction_at.equals(dest.destination_prediction_at.rename('origin_prediction_at'))
    origin = origin.drop(columns='origin_prediction_at')
    dest = dest.drop(columns='destination_prediction_at')
    result = pd.concat([selection.reset_index(drop=True),
                        origin.reset_index(drop=True), dest.reset_index(drop=True)], axis=1)

    # ---- required invariants ----
    assert result.ID.tolist() == selection.ID.tolist(), 'row identity/order must be preserved'
    assert len(result) == len(selection), 'no rows added or dropped'

    for role, airport_col in (('origin', 'Origin_Airport'), ('destination', 'Destination_Airport')):
        matched = result[f'{role}_observed_at'].notna()
        assert (result.loc[matched, f'{role}_station']
                == result.loc[matched, airport_col].map(lambda a: AIRPORTS[a]['station'])).all(), \
            f'{role} weather must come from the {role} airport\'s own station, never the other one'
        avail = result.loc[matched, f'{role}_available_at']
        obsd = result.loc[matched, f'{role}_observed_at']
        pred = result.loc[matched, 'prediction_at']
        assert (pd.to_datetime(obsd) <= pd.to_datetime(pred)).all(), 'observed_at <= prediction_at required'
        assert (pd.to_datetime(avail) <= pd.to_datetime(pred)).all(), 'available_at <= prediction_at required'
        age = result.loc[matched, f'{role}_weather_age_minutes']
        assert (age <= 90).all(), 'matched observations must respect the max_age cutoff'
        stale_but_present = result.loc[~matched, f'{role}_weather_age_minutes']
        assert stale_but_present.isna().all(), 'unmatched rows must not carry a stale age value'

    unresolved = result.prediction_at.isna()
    assert unresolved.sum() == int(selection.prediction_at.isna().sum())
    assert result.loc[unresolved, ['origin_tmpf', 'destination_tmpf']].isna().all().all(), \
        'a row with no valid prediction_at (DST ambiguous/nonexistent local time) must join nothing'

    row_path = local
    local.mkdir(parents=True)
    result.to_csv(row_path / 'joined_sample.csv', index=False)

    summary_rows = []
    for role in ('origin', 'destination'):
        matched = result[f'{role}_observed_at'].notna()
        summary_rows.append({
            'role': role, 'requests': len(result), 'matched': int(matched.sum()),
            'unmatched_no_prediction_at': int(unresolved.sum()),
            'unmatched_stale_or_no_report': int((~matched & ~unresolved).sum()),
            'match_rate': float(matched.mean()),
            'median_age_minutes': float(result.loc[matched, f'{role}_weather_age_minutes'].median())
            if matched.any() else None,
            'max_age_minutes': float(result.loc[matched, f'{role}_weather_age_minutes'].max())
            if matched.any() else None,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / f'{args.name}_join_summary.csv', index=False)

    coverage = (result.assign(matched_both=result.origin_observed_at.notna() & result.destination_observed_at.notna())
                .groupby(['year', 'Origin_Airport'])
                .agg(rows=('ID', 'size'), matched_both=('matched_both', 'sum')))
    coverage.to_csv(out / f'{args.name}_join_coverage_by_airport_year.csv')

    manifest = {
        'name': args.name,
        'code_sha256': hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'fits_any_model': False, 'uses_external_data': True, 'target_columns_used': [],
        'prediction_offset_minutes': 60, 'max_obs_age': MAX_OBS_AGE,
        'assumed_publication_latency_minutes': ASSUMED_PUBLICATION_LATENCY_MINUTES,
        'assumed_publication_latency_is_verified': False,
        'rows': int(len(result)), 'dropped_exact_duplicate_reports': int(dropped_exact_duplicates),
        'unresolved_no_prediction_at': int(unresolved.sum()),
        'unresolved_ids': result.loc[unresolved, 'ID'].tolist(),
        'row_evidence': str((row_path / 'joined_sample.csv').relative_to(ROOT)),
        'row_sha256': digest(row_path / 'joined_sample.csv'),
        'join_summary': summary_rows,
        'characterization': 'a past combination under an assumed publication latency, not a '
                            'verified operational point-in-time replay',
        'limitations': [
            'This validates the join contract on 21 rows across 8 airports; it is not full-archive '
            'weather collection and is not a performance result.',
            'available_at uses a fixed 10-minute assumed latency; the archive does not record actual '
            'historical receipt/publication time, so this cannot be verified against real dissemination logs.',
            'Sample rows come only from the complete_single_candidate_year (adopted) status; the '
            '291,308 missing-key single-candidate rows and other held statuses are untouched.',
            'Any future performance comparison must evaluate weather-on vs weather-off on the SAME '
            'rows, and must not generalize the ~70.68% adopted share to full-dataset performance.',
        ],
    }
    (out / f'{args.name}_join_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k not in {'limitations'}}, indent=2))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
