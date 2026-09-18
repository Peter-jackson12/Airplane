"""Join the expanded stratified sample's fetched observations using
src/weather.py UNMODIFIED, at four fixed publication-latency ASSUMPTIONS
(0/10/30/60 minutes) against the SAME fetched observations and the SAME
90-minute max age, to show how sensitive the match rate/age is to a latency
assumption nobody has verified against a real historical receipt log.

Reuses notebooks.join_weather_sample.load_observations unchanged. Station
resolution comes from the mapping table (map_weather_stations.py), not the
original 8-airport constant, since this sample spans far more airports.
Rows whose mapping is not confirmed_period (tz_conflict / unconfirmed /
confirmed_current_only) are kept in the output with no weather at all --
never silently dropped, never silently mapped to a nearby station.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from notebooks.join_weather_sample import load_observations
from src.weather import join_weather_asof

ROOT = Path(__file__).resolve().parents[1]
MAX_OBS_AGE = '90min'
LATENCY_SCENARIOS_MINUTES = [0, 10, 30, 60]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_join(requests: pd.DataFrame, obs: pd.DataFrame, prefix: str) -> pd.DataFrame:
    joined = join_weather_asof(requests[['station', 'prediction_at']], obs, max_age=MAX_OBS_AGE)
    joined.index = requests.index
    return joined.add_prefix(f'{prefix}_')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--mapping-name', required=True)
    args = ap.parse_args()
    out = ROOT / 'output'
    local = ROOT / 'data/weather_probe' / f'{args.name}_joined'
    if local.exists() or (out / f'{args.name}_join_summary.csv').exists():
        raise FileExistsError('Use a fresh name; existing evidence is preserved')

    selection = pd.read_csv(out / f'{args.name}_selection_with_prediction_at.csv')
    selection['prediction_at'] = pd.to_datetime(selection.prediction_at, utc=True)
    mapping = pd.read_csv(out / f'{args.mapping_name}_mapping_table.csv').set_index('iata')
    fetch_manifest = json.loads((out / f'{args.name}_fetch_manifest.json').read_text())

    forbidden = {'Delay', 'delay', 'Not_Delayed', 'Delayed', 'ArrDelay', 'DepDelay',
                'Actual_Departure_Time', 'Actual_Arrival_Time'}
    if forbidden & set(selection.columns):
        raise ValueError('Target/actual-outcome columns must not reach the weather join')

    raw_obs, dropped_exact_duplicates = load_observations(fetch_manifest)

    origin_req = selection[['ID', 'prediction_at']].copy()
    origin_req['station'] = selection.Origin_Airport.map(mapping.candidate_sid)
    dest_req = selection[['ID', 'prediction_at']].copy()
    dest_req['station'] = selection.Destination_Airport.map(mapping.candidate_sid)

    local.mkdir(parents=True)
    sensitivity_rows = []
    base_cols = None
    per_scenario_frames = {}
    for latency in LATENCY_SCENARIOS_MINUTES:
        obs = raw_obs.copy()
        obs['available_at'] = obs.observed_at + pd.Timedelta(minutes=latency)

        origin = run_join(origin_req, obs, 'origin').drop(columns='origin_prediction_at')
        dest = run_join(dest_req, obs, 'destination').drop(columns='destination_prediction_at')
        result = pd.concat([selection.reset_index(drop=True),
                            origin.reset_index(drop=True), dest.reset_index(drop=True)], axis=1)

        assert result.ID.tolist() == selection.ID.tolist(), 'row identity/order must be preserved'
        assert len(result) == len(selection), 'no rows added or dropped'
        for role, airport_col in (('origin', 'Origin_Airport'), ('destination', 'Destination_Airport')):
            matched = result[f'{role}_observed_at'].notna()
            expected_station = result[airport_col].map(mapping.candidate_sid)
            assert (result.loc[matched, f'{role}_station'] == expected_station[matched]).all(), \
                f'{role} weather must come from the {role} airport\'s own mapped station'
            obsd = pd.to_datetime(result.loc[matched, f'{role}_observed_at'])
            avail = pd.to_datetime(result.loc[matched, f'{role}_available_at'])
            pred = pd.to_datetime(result.loc[matched, 'prediction_at'])
            assert (obsd <= pred).all(), 'observed_at <= prediction_at required'
            assert (avail <= pred).all(), 'available_at <= prediction_at required'
            age = result.loc[matched, f'{role}_weather_age_minutes']
            assert (age >= 0).all() and (age <= 90).all(), 'matched observations must be within [0, 90] minutes'
            not_collectible = ~result.collectible
            assert result.loc[not_collectible, f'{role}_observed_at'].isna().all(), \
                'not_collectible rows must never receive weather'
            unresolved = result.prediction_at.isna()
            assert result.loc[unresolved, f'{role}_observed_at'].isna().all(), \
                'rows with no prediction_at must never receive weather'

        result.to_csv(local / f'joined_latency{latency}min.csv', index=False)
        per_scenario_frames[latency] = result

        for role in ('origin', 'destination'):
            matched = result[f'{role}_observed_at'].notna()
            eligible = result.collectible & result.prediction_at.notna()
            sensitivity_rows.append({
                'latency_minutes': latency, 'role': role,
                'eligible_rows': int(eligible.sum()), 'matched_rows': int(matched.sum()),
                'match_rate_of_eligible': float(matched.sum() / eligible.sum()) if eligible.sum() else None,
                'median_age_minutes': float(result.loc[matched, f'{role}_weather_age_minutes'].median())
                if matched.any() else None,
                'max_age_minutes': float(result.loc[matched, f'{role}_weather_age_minutes'].max())
                if matched.any() else None,
            })

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(out / f'{args.name}_latency_sensitivity.csv', index=False)

    # headline summary uses the same 10-minute assumption as the original 21-row sample,
    # for a like-for-like comparison; the other three latencies are the sensitivity check
    headline = per_scenario_frames[10]
    total_not_collectible = int((~headline.collectible).sum())
    total_unresolved = int(headline.prediction_at.isna().sum())
    summary_rows = []
    for role in ('origin', 'destination'):
        matched = headline[f'{role}_observed_at'].notna()
        summary_rows.append({'role': role, 'requests': len(headline), 'matched': int(matched.sum()),
                             'not_collectible_mapping': total_not_collectible,
                             'unresolved_no_prediction_at': total_unresolved,
                             'unmatched_stale_or_no_report': int(
                                 (~matched & headline.collectible & headline.prediction_at.notna()).sum()),
                             'match_rate_of_all_selected': float(matched.mean())})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / f'{args.name}_join_summary.csv', index=False)

    coverage = (headline.assign(matched_both=headline.origin_observed_at.notna()
                                & headline.destination_observed_at.notna())
                .groupby(['attributed_year', 'season', 'region'], dropna=False)
                .agg(rows=('ID', 'size'), collectible=('collectible', 'sum'),
                     matched_both=('matched_both', 'sum')))
    coverage.to_csv(out / f'{args.name}_join_coverage_by_year_season_region.csv')

    manifest = {
        'name': args.name, 'mapping_name': args.mapping_name,
        'code_sha256': hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'fits_any_model': False, 'uses_external_data': True, 'target_columns_used': [],
        'max_obs_age': MAX_OBS_AGE, 'latency_scenarios_minutes': LATENCY_SCENARIOS_MINUTES,
        'headline_scenario_minutes': 10,
        'rows': int(len(headline)), 'dropped_exact_duplicate_reports': int(dropped_exact_duplicates),
        'not_collectible_rows': total_not_collectible, 'unresolved_no_prediction_at': total_unresolved,
        'row_evidence_by_scenario': {str(lat): str((local / f'joined_latency{lat}min.csv').relative_to(ROOT))
                                     for lat in LATENCY_SCENARIOS_MINUTES},
        'row_sha256_by_scenario': {str(lat): digest(local / f'joined_latency{lat}min.csv')
                                   for lat in LATENCY_SCENARIOS_MINUTES},
        'latency_sensitivity_table': str((out / f'{args.name}_latency_sensitivity.csv').relative_to(ROOT)),
        'characterization': 'a past combination under an ASSUMED publication latency, not a verified '
                            'operational point-in-time replay; none of the four latency values (0/10/30/60 '
                            'minutes) is a measured historical receipt time',
        'limitations': [
            'This validates representativeness and the join contract on 300 stratified rows (271 '
            'collectible); it is not full-archive weather collection and is not a performance result.',
            'not_collectible rows (mapping unconfirmed/tz_conflict/current-only) never receive weather '
            'in any scenario -- they are not excluded from the row count or denominators.',
            'Any future performance comparison must evaluate weather-on vs weather-off on the SAME '
            'rows and must not generalize the adopted-population share to full-dataset performance.',
        ],
    }
    (out / f'{args.name}_join_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'limitations'}, indent=2, default=str))
    print(sensitivity.to_string(index=False))


if __name__ == '__main__':
    main()
