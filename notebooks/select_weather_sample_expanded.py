"""Deterministic stratified expanded weather-validation sample over the FULL
adopted (complete_single_candidate_year) population -- a representativeness
check, kept separate from the original 21-row boundary-regression sample
(baseline_recovery_v2_weather_sample_20260918), which this script never
touches or recomputes.

Strata: year x season x region/timezone-bucket x origin-airport size tier.
Selection within each non-empty stratum is by a stable hash of ID (not
insertion order, not a mutable RNG), so it is reproducible without a seed
file. Rows whose airports are NOT in the confirmed_period mapping tier are
still eligible for selection -- they are marked not-collectible with a
reason instead of being silently excluded from the sampling frame.

No target, delay, actual-time, or weather value is read anywhere here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from notebooks.scope_weather_collection import (CENSUS_REGION, NON_CENSUS_TERRITORY,
                                                 SEASON_BY_MONTH, region_for)

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTION_RUN = 'baseline_recovery_v2_row_date_attribution_20260918'
PER_STRATUM_CAP_PASSES = 2  # up to this many rows per stratum, added one pass at a time


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_rank(ids: pd.Series) -> pd.Series:
    """Deterministic, seed-free tiebreaker: a stable hash of the ID string,
    not row order and not a mutable RNG draw."""
    return ids.map(lambda s: hashlib.sha256(s.encode()).hexdigest())


def region_timezone_bucket(region: str, timezone: str) -> str:
    if region in {'Alaska', 'Hawaii', NON_CENSUS_TERRITORY}:
        return region
    # For the four mainland Census regions, fold in the timezone so e.g. the
    # South's Eastern-time and Central-time airports stratify separately.
    short_tz = timezone.rsplit('/', 1)[-1] if isinstance(timezone, str) else 'unknown'
    return f'{region}:{short_tz}'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--mapping-name', required=True)
    ap.add_argument('--max-rows', type=int, default=300)
    args = ap.parse_args()
    if not args.name.startswith('baseline_recovery_v2') or Path(args.name).name != args.name:
        raise ValueError('Invalid output name')
    out = ROOT / 'output'
    if list(out.glob(args.name + '_selection*')):
        raise FileExistsError('Use a fresh name; existing evidence is preserved')

    manifest = json.loads((out / f'{ATTRIBUTION_RUN}_manifest.json').read_text())
    row_path = ROOT / manifest['row_evidence']
    if digest(row_path) != manifest['row_sha256']:
        raise ValueError(f'{row_path} does not match {ATTRIBUTION_RUN}\'s recorded hash')
    attribution = pd.read_csv(row_path, low_memory=False)[
        ['ID', 'row_position', 'status', 'attributed_date']]
    adopted = attribution.loc[attribution.status.eq('complete_single_candidate_year')].copy()

    source = ROOT / 'data/train.csv'
    raw = pd.read_csv(source, usecols=[
        'ID', 'Origin_Airport', 'Destination_Airport', 'Estimated_Departure_Time'])
    pool = adopted.merge(raw, on='ID', how='left', validate='one_to_one')
    if len(pool) != len(adopted):
        raise ValueError('merge changed row count')
    if pool[['Origin_Airport', 'Destination_Airport']].isna().any().any():
        raise ValueError('adopted rows must have both airports observed')

    mapping = pd.read_csv(out / f'{args.mapping_name}_mapping_table.csv')
    lookup = mapping.set_index('iata')
    TIER_RANK = {'confirmed_period': 0, 'confirmed_current_only': 1,
                'tz_conflict_needs_resolution': 2, 'unconfirmed': 3}

    pool['origin_tier'] = pool.Origin_Airport.map(lookup.verification_tier)
    pool['dest_tier'] = pool.Destination_Airport.map(lookup.verification_tier)
    if pool[['origin_tier', 'dest_tier']].isna().any().any():
        missing = sorted(set(pool.loc[pool.origin_tier.isna(), 'Origin_Airport'])
                         | set(pool.loc[pool.dest_tier.isna(), 'Destination_Airport']))
        raise ValueError(f'airports missing from mapping table: {missing}')
    pool['row_tier'] = pool.apply(
        lambda r: max((r.origin_tier, r.dest_tier), key=lambda t: TIER_RANK[t]), axis=1)
    pool['collectible'] = pool.row_tier.eq('confirmed_period')
    pool['not_collectible_reason'] = pool.apply(
        lambda r: '' if r.collectible else f'origin={r.origin_tier},destination={r.dest_tier}', axis=1)

    pool['year'] = pool.attributed_date.str.slice(0, 4).astype(int)
    month = pool.attributed_date.str.slice(5, 7).astype(int)
    pool['season'] = month.map(SEASON_BY_MONTH)
    pool['origin_state'] = pool.Origin_Airport.map(lookup.state)
    pool['origin_country'] = pool.Origin_Airport.map(lookup.country)
    pool['region'] = pool.apply(lambda r: region_for(r.origin_state, r.origin_country), axis=1)
    pool['origin_timezone'] = pool.Origin_Airport.map(lookup.mwgg_tz)
    pool['region_timezone_bucket'] = pool.apply(
        lambda r: region_timezone_bucket(r.region, r.origin_timezone), axis=1)

    # Airport size tier: tertile of how often this airport is the ORIGIN
    # within the adopted population itself (a traffic proxy from the same
    # non-target source columns, not an external popularity ranking).
    origin_counts = pool.Origin_Airport.value_counts()
    size_rank = origin_counts.rank(pct=True)
    pool['origin_size_tier'] = pool.Origin_Airport.map(
        lambda a: 'large' if size_rank[a] > 2 / 3 else ('medium' if size_rank[a] > 1 / 3 else 'small'))

    pool['stratum'] = pool.apply(
        lambda r: (r.year, r.season, r.region_timezone_bucket, r.origin_size_tier), axis=1)
    pool['stratum_label'] = pool.apply(
        lambda r: f'{r.year}|{r.season}|{r.region_timezone_bucket}|{r.origin_size_tier}', axis=1)
    pool['hash_rank'] = stable_rank(pool.ID)
    pool = pool.sort_values(['stratum_label', 'hash_rank']).reset_index(drop=True)

    strata = sorted(pool.stratum_label.unique())
    picked_ids: list[str] = []
    picked_set: set[str] = set()
    for pass_no in range(PER_STRATUM_CAP_PASSES):
        if len(picked_ids) >= args.max_rows:
            break
        for stratum in strata:
            if len(picked_ids) >= args.max_rows:
                break
            bucket = pool.loc[pool.stratum_label.eq(stratum) & ~pool.ID.isin(picked_set)]
            if len(bucket) > pass_no:
                new_id = bucket.iloc[pass_no].ID
                picked_ids.append(new_id)
                picked_set.add(new_id)

    selection = pool.loc[pool.ID.isin(picked_set)].copy().sort_values('ID').reset_index(drop=True)
    selection = selection.rename(columns={'year': 'attributed_year'})

    cols = ['ID', 'row_position', 'attributed_year', 'attributed_date', 'season',
            'region', 'origin_timezone', 'region_timezone_bucket', 'origin_size_tier', 'stratum_label',
            'Origin_Airport', 'Destination_Airport', 'Estimated_Departure_Time',
            'origin_tier', 'dest_tier', 'row_tier', 'collectible', 'not_collectible_reason']
    selection[cols].to_csv(out / f'{args.name}_selection.csv', index=False)

    strata_frame = pd.DataFrame({'stratum': strata})
    strata_frame['non_empty_population_rows'] = strata_frame.stratum.map(
        lambda s: int(pool.stratum_label.eq(s).sum()))
    strata_frame['selected_rows'] = strata_frame.stratum.map(
        lambda s: int(selection.stratum_label.eq(s).sum()))
    strata_frame.to_csv(out / f'{args.name}_strata.csv', index=False)

    population_composition = pool.groupby(
        ['year', 'season', 'region', 'origin_size_tier'], dropna=False).size().rename('rows').reset_index()
    sample_composition = selection.groupby(
        ['attributed_year', 'season', 'region', 'origin_size_tier'], dropna=False).size().rename(
        'rows').reset_index()
    population_composition.to_csv(out / f'{args.name}_population_composition.csv', index=False)
    sample_composition.to_csv(out / f'{args.name}_sample_composition.csv', index=False)

    manifest_out = {
        'name': args.name, 'attribution_run': ATTRIBUTION_RUN,
        'attribution_row_sha256': manifest['row_sha256'], 'mapping_name': args.mapping_name,
        'code_sha256': hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'fits_any_model': False, 'uses_external_data': False, 'target_columns_used': [],
        'selection_rule': 'stratify by (year, season, region/timezone bucket, origin-airport size '
                          'tertile); within each non-empty stratum, sort by sha256(ID) and take up '
                          f'to {PER_STRATUM_CAP_PASSES} rows per stratum, breadth-first across strata',
        'max_rows_cap': args.max_rows, 'selected_rows': int(len(selection)),
        'strata_total': len(strata), 'strata_with_at_least_one_row_selected': int(strata_frame.selected_rows.gt(0).sum()),
        'collectible_rows': int(selection.collectible.sum()),
        'not_collectible_rows': int((~selection.collectible).sum()),
        'not_collectible_breakdown': selection.loc[~selection.collectible, 'row_tier'].value_counts().to_dict(),
        'strata_table': str((out / f'{args.name}_strata.csv').relative_to(ROOT)),
        'population_composition': str((out / f'{args.name}_population_composition.csv').relative_to(ROOT)),
        'sample_composition': str((out / f'{args.name}_sample_composition.csv').relative_to(ROOT)),
        'limitations': [
            'This sample is for representativeness/coverage checking; the original 21-row sample '
            'remains the boundary-condition regression sample and is not replaced.',
            'not_collectible rows are kept in the selection and denominator with an explicit reason; '
            'they are never weather-joined and never silently dropped.',
            'Airport size tier is a tertile of in-sample origin frequency within the adopted '
            'population itself, not an external traffic ranking.',
            'Overlap between strata is not possible by construction (each row has exactly one '
            'year/season/region-timezone/size-tier combination), but a stratum can still be very '
            'small (or a single airport), so oversampling relative to the population is expected.',
        ],
    }
    (out / f'{args.name}_selection_manifest.json').write_text(json.dumps(manifest_out, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest_out.items() if k != 'limitations'}, indent=2, default=str))


if __name__ == '__main__':
    main()
