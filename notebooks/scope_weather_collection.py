"""Size a full weather collection over the ADOPTED (complete_single_candidate_year)
population BEFORE downloading anything at scale.

Reads only: the row-level date attribution (year/date, non-target), source
airport/scheduled-time columns, and the airport-station mapping table from
map_weather_stations.py. No target, delay, actual-time, or weather value is
read. Prediction_at reuses src.weather.local_hhmm_to_utc unmodified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTION_RUN = 'baseline_recovery_v2_row_date_attribution_20260918'
LOOKBACK_HOURS = 6
LOOKAHEAD_HOURS = 2

# US Census Bureau 4-region grouping (https://www2.census.gov/geo/pdfs/maps-data/
# maps/reference/us_regdiv.pdf), by full state name as mwgg records it. Alaska,
# Hawaii and the Pacific/Caribbean territories are called out separately rather
# than folded into "West"/"South", since they are not part of that Census grouping.
CENSUS_REGION = {
    'Connecticut': 'Northeast', 'Maine': 'Northeast', 'Massachusetts': 'Northeast',
    'New Hampshire': 'Northeast', 'Rhode Island': 'Northeast', 'Vermont': 'Northeast',
    'New Jersey': 'Northeast', 'New York': 'Northeast', 'Pennsylvania': 'Northeast',
    'Illinois': 'Midwest', 'Indiana': 'Midwest', 'Michigan': 'Midwest', 'Ohio': 'Midwest',
    'Wisconsin': 'Midwest', 'Iowa': 'Midwest', 'Kansas': 'Midwest', 'Minnesota': 'Midwest',
    'Missouri': 'Midwest', 'Nebraska': 'Midwest', 'North Dakota': 'Midwest', 'South Dakota': 'Midwest',
    'Delaware': 'South', 'District of Columbia': 'South', 'Florida': 'South', 'Georgia': 'South',
    'Maryland': 'South', 'North Carolina': 'South', 'South Carolina': 'South', 'Virginia': 'South',
    'West Virginia': 'South', 'Alabama': 'South', 'Kentucky': 'South', 'Mississippi': 'South',
    'Tennessee': 'South', 'Arkansas': 'South', 'Louisiana': 'South', 'Oklahoma': 'South', 'Texas': 'South',
    'Arizona': 'West', 'Colorado': 'West', 'Idaho': 'West', 'Montana': 'West', 'Nevada': 'West',
    'New Mexico': 'West', 'Utah': 'West', 'Wyoming': 'West', 'California': 'West', 'Oregon': 'West',
    'Washington': 'West',
    'Alaska': 'Alaska', 'Hawaii': 'Hawaii',
}
NON_CENSUS_TERRITORY = 'Non-contiguous territory'
SEASON_BY_MONTH = {12: 'winter', 1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
                   6: 'summer', 7: 'summer', 8: 'summer', 9: 'fall', 10: 'fall', 11: 'fall'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def region_for(state: str, country: str) -> str:
    if country != 'US':
        return NON_CENSUS_TERRITORY
    return CENSUS_REGION.get(state, 'Unknown')


def compute_prediction_at(dates: pd.Series, hhmm: pd.Series, timezones: pd.Series) -> pd.Series:
    from src.weather import local_hhmm_to_utc
    parts = []
    frame = pd.DataFrame({'date': dates.values, 'hhmm': hhmm.values, 'tz': timezones.values},
                         index=dates.index)
    for tz, sub in frame.groupby('tz'):
        parts.append(local_hhmm_to_utc(sub.date, sub.hhmm, tz))
    return pd.concat(parts).reindex(dates.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--mapping-name', required=True,
                    help='name used for notebooks/map_weather_stations.py output')
    ap.add_argument('--measured-bytes-per-request', type=float, default=None,
                    help='observed avg response bytes per fetch_weather_sample.py request, for extrapolation')
    ap.add_argument('--measured-seconds-per-request', type=float, default=None)
    args = ap.parse_args()
    if not args.name.startswith('baseline_recovery_v2') or Path(args.name).name != args.name:
        raise ValueError('Invalid output name')
    out = ROOT / 'output'
    if list(out.glob(args.name + '_scope*')):
        raise FileExistsError('Use a fresh name; existing evidence is preserved')

    manifest = json.loads((out / f'{ATTRIBUTION_RUN}_manifest.json').read_text())
    row_path = ROOT / manifest['row_evidence']
    if digest(row_path) != manifest['row_sha256']:
        raise ValueError(f'{row_path} does not match {ATTRIBUTION_RUN}\'s recorded hash')
    attribution = pd.read_csv(row_path, low_memory=False)[
        ['ID', 'row_position', 'status', 'attributed_date']]
    adopted = attribution.loc[attribution.status.eq('complete_single_candidate_year')].copy()

    raw = pd.read_csv(ROOT / 'data/train.csv', usecols=[
        'ID', 'Origin_Airport', 'Destination_Airport',
        'Estimated_Departure_Time', 'Estimated_Arrival_Time'])
    pool = adopted.merge(raw, on='ID', how='left', validate='one_to_one')
    if len(pool) != len(adopted):
        raise ValueError('merge changed row count')
    if pool[['Origin_Airport', 'Destination_Airport']].isna().any().any():
        raise ValueError('adopted rows must have both airports observed')

    mapping = pd.read_csv(out / f'{args.mapping_name}_mapping_table.csv')
    mapping_lookup = mapping.set_index('iata')

    pool['origin_tier'] = pool.Origin_Airport.map(mapping_lookup.verification_tier)
    pool['dest_tier'] = pool.Destination_Airport.map(mapping_lookup.verification_tier)
    if pool[['origin_tier', 'dest_tier']].isna().any().any():
        missing = sorted(set(pool.loc[pool.origin_tier.isna(), 'Origin_Airport'])
                         | set(pool.loc[pool.dest_tier.isna(), 'Destination_Airport']))
        raise ValueError(f'airports missing from mapping table: {missing}')

    TIER_RANK = {'confirmed_period': 0, 'confirmed_current_only': 1,
                'tz_conflict_needs_resolution': 2, 'unconfirmed': 3}
    pool['row_tier'] = pool.apply(
        lambda r: max((r.origin_tier, r.dest_tier), key=lambda t: TIER_RANK[t]), axis=1)

    pool['year'] = pool.attributed_date.str.slice(0, 4).astype(int)
    pool['month'] = pool.attributed_date.str.slice(5, 7).astype(int)
    pool['season'] = pool.month.map(SEASON_BY_MONTH)
    pool['origin_state'] = pool.Origin_Airport.map(mapping_lookup.state)
    pool['origin_country'] = pool.Origin_Airport.map(mapping_lookup.country)
    pool['region'] = pool.apply(lambda r: region_for(r.origin_state, r.origin_country), axis=1)
    pool['origin_timezone'] = pool.Origin_Airport.map(mapping_lookup.mwgg_tz)

    pool['prediction_at'] = compute_prediction_at(
        pd.to_datetime(pool.attributed_date), pool.Estimated_Departure_Time, pool.origin_timezone
    ) - pd.Timedelta(minutes=60)
    dst_unresolved = pool.prediction_at.isna()

    # ---- descriptive coverage: year x season x region x timezone ----
    coverage = (pool.groupby(['year', 'season', 'region', 'origin_timezone', 'row_tier'], dropna=False)
                .size().rename('rows').reset_index())
    coverage.to_csv(out / f'{args.name}_scope_coverage.csv', index=False)

    tier_totals = pool.row_tier.value_counts().to_dict()
    safe = pool.loc[pool.row_tier.eq('confirmed_period') & ~dst_unresolved]

    # ---- required (station, UTC day) groups for a hypothetical full collection,
    # restricted to the safely-mapped (confirmed_period) rows only ----
    groups: dict[tuple[str, str], list[pd.Timestamp]] = {}
    origin_sid = pool.Origin_Airport.map(mapping_lookup.candidate_sid)
    dest_sid = pool.Destination_Airport.map(mapping_lookup.candidate_sid)
    for sid, pred in list(zip(origin_sid[safe.index], safe.prediction_at)) + \
                      list(zip(dest_sid[safe.index], safe.prediction_at)):
        groups.setdefault((sid, pred.strftime('%Y-%m-%d')), []).append(pred)
    station_days = len(groups)
    distinct_stations = len({k[0] for k in groups})

    n_requests = station_days  # one bounded request per (station, day) group, as in fetch_weather_sample.py
    measured_bytes = args.measured_bytes_per_request
    measured_seconds = args.measured_seconds_per_request
    estimate = {
        'basis': 'measured 21-row/40-request probe (2026-09-18)' if measured_bytes else 'not provided',
        'station_days_for_confirmed_period_rows_only': station_days,
        'distinct_stations': distinct_stations,
        'requests_at_one_per_station_day': n_requests,
    }
    if measured_bytes:
        estimate['extrapolated_bytes'] = measured_bytes * n_requests
        estimate['extrapolated_bytes_is_measured'] = False
    if measured_seconds:
        estimate['extrapolated_seconds_sequential_with_politeness_sleep'] = measured_seconds * n_requests
        estimate['extrapolated_seconds_is_measured'] = False

    manifest_out = {
        'name': args.name,
        'attribution_run': ATTRIBUTION_RUN, 'attribution_row_sha256': manifest['row_sha256'],
        'mapping_name': args.mapping_name,
        'code_sha256': hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'fits_any_model': False, 'uses_external_data': False, 'target_columns_used': [],
        'adopted_rows': int(len(pool)),
        'row_tier_counts': tier_totals,
        'prediction_at_unresolved_dst': int(dst_unresolved.sum()),
        'safe_rows_confirmed_period_both_ends': int(len(safe)),
        'season_definition': 'meteorological: Dec/Jan/Feb=winter, Mar/Apr/May=spring, Jun/Jul/Aug=summer, Sep/Oct/Nov=fall',
        'region_definition': 'US Census Bureau 4-region (Northeast/Midwest/South/West) by origin airport state; '
                             'Alaska and Hawaii kept separate; non-US territories grouped as '
                             f'"{NON_CENSUS_TERRITORY}"',
        'full_collection_estimate_confirmed_period_only': estimate,
        'coverage_table': str((out / f'{args.name}_scope_coverage.csv').relative_to(ROOT)),
        'limitations': [
            'This sizing covers only confirmed_period-tier rows (both origin and destination). '
            'unconfirmed/tz_conflict/confirmed_current_only rows are excluded from the station-day '
            'estimate and reported separately in row_tier_counts, not silently dropped from the '
            'adopted-row denominator.',
            'The byte/time estimate is a linear extrapolation from the 21-row probe\'s measured average '
            'when --measured-* is supplied; it is NOT a new measurement at this scale.',
            'DST-unresolved rows (ambiguous/nonexistent local departure) have no prediction_at and are '
            'excluded from station-day sizing, matching the row-level join policy already in place.',
        ],
    }
    (out / f'{args.name}_scope_manifest.json').write_text(json.dumps(manifest_out, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest_out.items() if k != 'limitations'}, indent=2, default=str))


if __name__ == '__main__':
    main()
