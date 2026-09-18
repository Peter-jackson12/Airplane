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
from collections import deque
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


def round_robin_merge(sequences: list[list]) -> list:
    """Deterministic round-robin flatten: one item from each non-empty
    sequence in turn (in the given sequence order), repeating until every
    sequence is exhausted, preserving each sequence's own internal order.
    No randomness, no dependence on wall-clock or insertion order beyond
    what the caller already fixed by sorting each sequence's key."""
    queues = deque(deque(s) for s in sequences if s)
    order = []
    while queues:
        q = queues.popleft()
        order.append(q.popleft())
        if q:
            queues.append(q)
    return order


def _group_by(items, key):
    groups: dict = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    return groups


def broad_inclusion_order(strata: list[tuple]) -> list[tuple]:
    """Deterministic ordering of (year, season, region_timezone_bucket,
    origin_size_tier) strata that prioritizes touching as many DISTINCT
    year/season/region/timezone combinations as possible before repeating
    any one of them, via nested round-robin merging (year > season >
    region_timezone_bucket > size_tier, each level sorted for determinism).

    This matters because the row cap (300) is smaller than the number of
    strata (341): whichever strata land LATEST in the picking order are the
    ones a fixed budget cuts off entirely. A plain alphabetical sort of the
    full stratum label sorts by year FIRST, so it exhausts one entire year
    (every season/region/tier combination in it) before touching the other
    year at all -- if the budget runs out partway through the second year,
    every stratum that happens to be alphabetically last within it (e.g. one
    particular season across several regions) is cut, not a cross-section
    spread fairly across dimensions. Round-robin merging instead interleaves
    across each dimension at every level, so a budget shortfall trims a bit
    from every year/season/region/tier combination instead of erasing entire
    slices of them.
    """
    by_year = _group_by(strata, key=lambda s: s[0])
    year_orders = []
    for year in sorted(by_year):
        by_season = _group_by(by_year[year], key=lambda s: s[1])
        season_orders = []
        for season in sorted(by_season):
            by_region = _group_by(by_season[season], key=lambda s: s[2])
            region_orders = [sorted(by_region[region], key=lambda s: s[3]) for region in sorted(by_region)]
            season_orders.append(round_robin_merge(region_orders))
        year_orders.append(round_robin_merge(season_orders))
    return round_robin_merge(year_orders)


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

    label_by_dims = pool[['stratum_label', 'year', 'season', 'region_timezone_bucket',
                          'origin_size_tier']].drop_duplicates().set_index('stratum_label')
    dims_by_label = {label: tuple(row) for label, row in label_by_dims.iterrows()}
    ordered_dims = broad_inclusion_order(list(dims_by_label.values()))
    label_by_dims_tuple = {dims: label for label, dims in dims_by_label.items()}
    strata = [label_by_dims_tuple[d] for d in ordered_dims]  # broad year/season/region/tz order, not alphabetical

    picked_ids: list[str] = []
    picked_set: set[str] = set()
    for pass_no in range(PER_STRATUM_CAP_PASSES):
        if len(picked_ids) >= args.max_rows:
            break
        for stratum in strata:
            if len(picked_ids) >= args.max_rows:
                break
            bucket = pool.loc[pool.stratum_label.eq(stratum) & ~pool.ID.isin(picked_set)]
            if len(bucket):
                # bucket already excludes rows picked in earlier passes (including THIS stratum's
                # earlier pass), so its own rank-0 row is always the correct next-ranked pick;
                # indexing by pass_no here was a bug -- once a pass removed rank 0, bucket re-indexes
                # from 0, so iloc[pass_no] silently skipped the row that was actually next in rank.
                new_id = bucket.iloc[0].ID
                picked_ids.append(new_id)
                picked_set.add(new_id)

    selection = pool.loc[pool.ID.isin(picked_set)].copy().sort_values('ID').reset_index(drop=True)
    selection = selection.rename(columns={'year': 'attributed_year'})

    cols = ['ID', 'row_position', 'attributed_year', 'attributed_date', 'season',
            'region', 'origin_timezone', 'region_timezone_bucket', 'origin_size_tier', 'stratum_label',
            'Origin_Airport', 'Destination_Airport', 'Estimated_Departure_Time',
            'origin_tier', 'dest_tier', 'row_tier', 'collectible', 'not_collectible_reason']
    selection[cols].to_csv(out / f'{args.name}_selection.csv', index=False)

    strata_frame = pd.DataFrame({'stratum': strata, 'picking_order_rank': range(len(strata))})
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

    excluded_strata = strata_frame.loc[strata_frame.selected_rows.eq(0)]
    excluded_population_rows = int(excluded_strata.non_empty_population_rows.sum())

    manifest_out = {
        'name': args.name, 'attribution_run': ATTRIBUTION_RUN,
        'attribution_row_sha256': manifest['row_sha256'], 'mapping_name': args.mapping_name,
        'code_sha256': hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'fits_any_model': False, 'uses_external_data': False, 'target_columns_used': [],
        'selection_rule': 'stratify by (year, season, region/timezone bucket, origin-airport size '
                          'tertile); strata are visited in a broad_inclusion_order (nested round-robin '
                          'across year > season > region/timezone > size-tier, NOT alphabetical) so a '
                          'row cap smaller than the stratum count trims breadth fairly across all four '
                          f'dimensions; within each non-empty stratum, sort by sha256(ID) and take up '
                          f'to {PER_STRATUM_CAP_PASSES} rows per stratum, breadth-first across strata '
                          'in that order',
        'max_rows_cap': args.max_rows, 'selected_rows': int(len(selection)),
        'strata_total': len(strata),
        'strata_with_at_least_one_row_selected': int(strata_frame.selected_rows.gt(0).sum()),
        'strata_with_zero_rows_selected': int(len(excluded_strata)),
        'excluded_strata_population_rows': excluded_population_rows,
        'excluded_strata_population_share_of_pool': (
            float(excluded_population_rows / len(pool)) if len(pool) else None),
        'excluded_strata_labels': sorted(excluded_strata.stratum.tolist()),
        'collectible_rows': int(selection.collectible.sum()),
        'not_collectible_rows': int((~selection.collectible).sum()),
        'not_collectible_breakdown': selection.loc[~selection.collectible, 'row_tier'].value_counts().to_dict(),
        'strata_table': str((out / f'{args.name}_strata.csv').relative_to(ROOT)),
        'population_composition': str((out / f'{args.name}_population_composition.csv').relative_to(ROOT)),
        'sample_composition': str((out / f'{args.name}_sample_composition.csv').relative_to(ROOT)),
        'limitations': [
            'This sample is for representativeness/coverage checking; the original 21-row sample '
            'remains the boundary-condition regression sample and is not replaced.',
            'The 300-row cap cannot include all 341 strata (a fixed constraint, not a bug): '
            'strata_with_zero_rows_selected/excluded_strata_labels name exactly which strata got no '
            'row, and excluded_strata_population_share_of_pool is their share of the adopted pool -- '
            'this selection was NOT tuned by looking at outcome, target, or weather-join rates.',
            'Selection order (broad_inclusion_order) is a deterministic function of the strata set '
            'alone (year/season/region-timezone/size-tier), never of row count, collectibility, or any '
            'downstream result, so it cannot be read as favoring "easier to collect" strata.',
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
