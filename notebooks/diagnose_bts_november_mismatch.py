"""Diagnose the November rows that found no exact BTS match.

Diagnostic only: no target columns, no imputation, no model fitting, no year
assignment. Relaxed keys locate which field disagrees; they are candidate
evidence and are never treated as confirmed matches. Row evidence stays in
data/ and is excluded from Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
YEARS = [2007, 2013, 2018, 2019]

BASE = ['Month', 'Day_of_Month', 'Tail_Number', 'Origin_Airport',
        'Destination_Airport', 'Distance', 'Carrier_ID(DOT)',
        'Estimated_Departure_Time', 'Estimated_Arrival_Time']
CONTEXT = ['ID', 'Airline', 'Carrier_Code(IATA)']
TEXT = {'Tail_Number', 'Origin_Airport', 'Destination_Airport'}
RENAMES = {'DayofMonth': 'Day_of_Month', 'Origin': 'Origin_Airport',
           'Dest': 'Destination_Airport',
           'DOT_ID_Reporting_Airline': 'Carrier_ID(DOT)'}

DAY = ['Month', 'Day_of_Month']
ROUTE = ['Origin_Airport', 'Destination_Airport']
TIMES = ['Estimated_Departure_Time', 'Estimated_Arrival_Time']

LADDER = [
    ('L0_full', BASE),
    ('L1_drop_distance', [k for k in BASE if k != 'Distance']),
    ('L2_drop_dot', [k for k in BASE if k != 'Carrier_ID(DOT)']),
    ('L3_drop_distance_dot', [k for k in BASE if k not in {'Distance', 'Carrier_ID(DOT)'}]),
    ('L4_drop_arrival_time', [k for k in BASE if k != 'Estimated_Arrival_Time']),
    ('L5_drop_both_times', [k for k in BASE if k not in set(TIMES)]),
    ('L6_day_tail_route', DAY + ['Tail_Number'] + ROUTE),
    ('L7_day_tail', DAY + ['Tail_Number']),
    ('L8_tail_only', ['Tail_Number']),
    ('L9_day_route_times', DAY + ROUTE + TIMES),
    ('L10_day_route_dep', DAY + ROUTE + ['Estimated_Departure_Time']),
    ('L11_day_route', DAY + ROUTE),
    ('L12_tail_route_times_any_day', ['Month', 'Tail_Number'] + ROUTE + TIMES),
    ('L13_dot_only', ['Carrier_ID(DOT)']),
]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            sha.update(chunk)
    return sha.hexdigest()


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in BASE:
        if column not in frame.columns:
            continue
        if column in TEXT:
            frame[column] = frame[column].astype('string').str.strip().str.upper()
        else:
            frame[column] = pd.to_numeric(frame[column], errors='raise').astype('Int64')
    return frame


def match_counts(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]):
    counts = right.dropna(subset=keys).groupby(keys, dropna=False).size().rename('n')
    result = left[keys].merge(counts, on=keys, how='left', validate='many_to_one')
    assert len(result) == len(left)
    return result.n.fillna(0).astype(int).to_numpy()


def read_bts(year: int) -> tuple[pd.DataFrame, Path]:
    path = ROOT / ('data/bts/On_Time_Reporting_Carrier_On_Time_Performance_1987'
                   f'_present_{year}_11.zip')
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f'Invalid ZIP: {path}')
        members = [n for n in archive.namelist() if n.endswith('.csv')]
        if len(members) != 1:
            raise ValueError('Expected exactly one CSV')
        usecols = ['Year', 'Month', 'DayofMonth', 'Tail_Number', 'Origin', 'Dest',
                   'Distance', 'DOT_ID_Reporting_Airline', 'CRSDepTime', 'CRSArrTime',
                   'Reporting_Airline', 'Flight_Number_Reporting_Airline']
        with archive.open(members[0]) as stream:
            bts = pd.read_csv(stream, usecols=usecols)
    assert bts.Year.eq(year).all() and bts.Month.eq(11).all()
    bts = bts.rename(columns=RENAMES)
    bts = bts.rename(columns=dict(zip(['CRSDepTime', 'CRSArrTime'], TIMES)))
    return normalize(bts), path


def field_disagreement(unmatched: pd.DataFrame, bts: pd.DataFrame, year: int) -> pd.DataFrame:
    """For unmatched rows, join on day+tail+route and report which fields differ."""
    keys = DAY + ['Tail_Number'] + ROUTE
    right = bts.dropna(subset=keys)[keys + ['Distance', 'Carrier_ID(DOT)'] + TIMES
                                    + ['Reporting_Airline']]
    joined = unmatched[['ID'] + keys + ['Distance', 'Carrier_ID(DOT)'] + TIMES].merge(
        right, on=keys, how='inner', suffixes=('_src', '_bts'))
    if joined.empty:
        return joined
    for field in ['Distance', 'Carrier_ID(DOT)'] + TIMES:
        joined[f'{field}_delta'] = (joined[f'{field}_bts'].astype('Int64')
                                    - joined[f'{field}_src'].astype('Int64'))
    joined['year'] = year
    # keep the BTS candidate whose scheduled departure is closest
    joined['abs_dep_delta'] = joined['Estimated_Departure_Time_delta'].abs()
    joined = joined.sort_values('abs_dep_delta').groupby('ID', as_index=False).first()
    return joined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='baseline_recovery_v2_bts_november_mismatch_20260917')
    args = ap.parse_args()
    if not args.name.startswith('baseline_recovery_v2') or Path(args.name).name != args.name:
        raise ValueError('Invalid output name')
    out = ROOT / 'output'
    local = ROOT / 'data/bts' / args.name
    if local.exists() or list(out.glob(args.name + '_*')):
        raise FileExistsError('Use a fresh name; existing evidence is preserved')

    source = ROOT / 'data/train.csv'
    raw = pd.read_csv(source, usecols=CONTEXT + BASE)
    november = normalize(raw.loc[raw.Month.eq(11)]).reset_index(drop=True)
    complete = november.dropna(subset=BASE).copy().reset_index(drop=True)
    incomplete = november.loc[november[BASE].isna().any(axis=1)].copy().reset_index(drop=True)

    prior = pd.read_csv(ROOT / 'data/bts/baseline_recovery_v2_bts_november_20260917/matches.csv.gz')
    prior_cols = [f'{y}_scheduled_strict' for y in YEARS]
    prior_any = prior.set_index('ID')[prior_cols].gt(0).any(axis=1)

    local.mkdir()
    ladder_rows, missing_rows = [], []
    per_year_hit = {}
    strict_hit = pd.Series(False, index=complete.index)

    cache = {}
    for year in YEARS:
        bts, path = read_bts(year)
        cache[year] = bts
        hit = match_counts(complete, bts, BASE) > 0
        per_year_hit[year] = hit
        strict_hit = strict_hit | pd.Series(hit, index=complete.index)
        print(f'{year} strict scheduled matches: {int(hit.sum())}', flush=True)

    unmatched_mask = ~strict_hit
    unmatched = complete.loc[unmatched_mask].copy().reset_index(drop=True)
    print(f'unmatched rows (all four years): {len(unmatched)}', flush=True)

    reproduced = bool((pd.Series(strict_hit.to_numpy(), index=complete.ID)
                       == prior_any.reindex(complete.ID)).all())
    print(f'prior row evidence reproduced: {reproduced}', flush=True)

    deltas = []
    for year in YEARS:
        bts = cache[year]
        for level, keys in LADDER:
            counts = match_counts(unmatched, bts, keys)
            ladder_rows.append({'year': year, 'level': level, 'keys': '+'.join(keys),
                                'unmatched_rows': len(unmatched),
                                'candidate_rows': int((counts > 0).sum()),
                                'candidate_rate': float((counts > 0).mean())})
        if year in (2018, 2019):
            deltas.append(field_disagreement(unmatched, bts, year))
        # missing-key rows, reported separately from complete-fingerprint results
        for level, keys in LADDER:
            usable = incomplete.dropna(subset=keys)
            if usable.empty:
                missing_rows.append({'year': year, 'level': level,
                                     'incomplete_rows': len(incomplete),
                                     'rows_with_these_keys': 0,
                                     'candidate_rows': 0, 'candidate_rate': float('nan')})
                continue
            counts = match_counts(usable, bts, keys)
            missing_rows.append({'year': year, 'level': level,
                                 'incomplete_rows': len(incomplete),
                                 'rows_with_these_keys': int(len(usable)),
                                 'candidate_rows': int((counts > 0).sum()),
                                 'candidate_rate': float((counts > 0).mean())})
    print('ladder complete', flush=True)

    ladder = pd.DataFrame(ladder_rows)
    ladder.to_csv(out / f'{args.name}_ladder.csv', index=False)
    pd.DataFrame(missing_rows).to_csv(out / f'{args.name}_missing_key_ladder.csv', index=False)

    delta = pd.concat(deltas, ignore_index=True) if deltas else pd.DataFrame()
    if not delta.empty:
        delta = delta.sort_values('abs_dep_delta').groupby('ID', as_index=False).first()
        profile = []
        for field in ['Distance', 'Carrier_ID(DOT)'] + TIMES:
            d = delta[f'{field}_delta']
            profile.append({'field': field, 'rows_with_candidate': int(d.notna().sum()),
                            'equal': int(d.eq(0).sum()), 'differ': int(d.ne(0).sum()),
                            'median_abs_delta': float(d.abs().median()) if d.notna().any() else float('nan'),
                            'max_abs_delta': float(d.abs().max()) if d.notna().any() else float('nan')})
        pd.DataFrame(profile).to_csv(out / f'{args.name}_field_profile.csv', index=False)
        delta.to_csv(out / f'{args.name}_field_deltas_sample.csv', index=False)

    # group breakdowns over the whole eligible set, matched vs not
    eligible = complete.copy()
    eligible['matched'] = strict_hit.to_numpy()
    groups = []
    eligible['route'] = eligible.Origin_Airport + '->' + eligible.Destination_Airport
    for dimension in ['Carrier_ID(DOT)', 'Airline', 'Carrier_Code(IATA)', 'route',
                      'Day_of_Month', 'Tail_Number', 'Origin_Airport', 'Destination_Airport']:
        # keep the missing-value group explicit: pandas 3 drops NA groups that
        # pandas 2 kept as the literal string 'nan'
        label = eligible[dimension].astype('object').where(
            eligible[dimension].notna(), '(missing)').astype(str)
        table = eligible.groupby(label).matched.agg(['size', 'sum'])
        for value, row in table.iterrows():
            groups.append({'dimension': dimension, 'value': value,
                           'eligible_rows': int(row['size']),
                           'matched_rows': int(row['sum']),
                           'unmatched_rows': int(row['size'] - row['sum']),
                           'unmatched_rate': float(1 - row['sum'] / row['size'])})
    pd.DataFrame(groups).to_csv(out / f'{args.name}_groups.csv', index=False)

    # candidate-year counts for the multi-candidate rows
    cand = pd.DataFrame({y: per_year_hit[y] for y in YEARS})
    n_cand = cand.sum(axis=1)
    multi = complete.loc[n_cand.gt(1)].copy()
    multi['candidate_years'] = [','.join(str(y) for y in YEARS if cand.loc[i, y])
                                for i in multi.index]
    multi.to_csv(out / f'{args.name}_multi_candidate_rows.csv', index=False)

    row_path = local / 'unmatched_rows.csv.gz'
    unmatched.to_csv(row_path, index=False, compression='gzip')

    manifest = {
        'name': args.name,
        'source_sha256': digest(source),
        'code_sha256': hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'fits_any_model': False,
        'uses_external_data': True,
        'target_columns_used': [],
        'november_rows': int(len(november)),
        'complete_rows': int(len(complete)),
        'incomplete_rows': int(len(incomplete)),
        'unmatched_rows': int(len(unmatched)),
        'multi_candidate_rows': int(n_cand.gt(1).sum()),
        'prior_row_evidence_reproduced': reproduced,
        'prior_row_evidence': 'data/bts/baseline_recovery_v2_bts_november_20260917/matches.csv.gz',
        'row_evidence': str(row_path.relative_to(ROOT)),
        'row_sha256': digest(row_path),
        'limitations': [
            'Relaxed keys are diagnostic candidates, not confirmed matches.',
            'A relaxed-key candidate does not assign a year to any row.',
            'Only November and the four candidate years are covered.',
            'Missing-key rows are reported separately and never merged with complete rows.',
            'A failed exact match can reflect reporting scope, column transformation, or another year.',
        ],
    }
    (out / f'{args.name}_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'limitations'}, indent=2))


if __name__ == '__main__':
    main()
