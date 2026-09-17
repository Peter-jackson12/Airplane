"""Assess whether ambiguous and missing-key November rows can be resolved.

Diagnostic only: no target columns, no imputation, no model fitting. Nothing
here assigns a year to any row; rows that stay ambiguous are reported as
ambiguous. Missing-key rows are reported separately from complete-fingerprint
rows and the two sets are never pooled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
YEARS = [2018, 2019]

BASE = ['Month', 'Day_of_Month', 'Tail_Number', 'Origin_Airport',
        'Destination_Airport', 'Distance', 'Carrier_ID(DOT)',
        'Estimated_Departure_Time', 'Estimated_Arrival_Time']
# every non-target source column we may use to break a tie
EXTRA = ['Origin_Airport_ID', 'Origin_State', 'Destination_Airport_ID',
         'Destination_State', 'Cancelled', 'Diverted']
TEXT = {'Tail_Number', 'Origin_Airport', 'Destination_Airport'}
RENAMES = {'DayofMonth': 'Day_of_Month', 'Origin': 'Origin_Airport',
           'Dest': 'Destination_Airport',
           'DOT_ID_Reporting_Airline': 'Carrier_ID(DOT)',
           'CRSDepTime': 'Estimated_Departure_Time',
           'CRSArrTime': 'Estimated_Arrival_Time',
           'OriginAirportID': 'Origin_Airport_ID', 'OriginState': 'Origin_State',
           'DestAirportID': 'Destination_Airport_ID', 'DestState': 'Destination_State'}


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


def read_bts(year: int) -> pd.DataFrame:
    path = ROOT / ('data/bts/On_Time_Reporting_Carrier_On_Time_Performance_1987'
                   f'_present_{year}_11.zip')
    usecols = ['Year', 'Month', 'DayofMonth', 'Tail_Number', 'Origin', 'Dest',
               'Distance', 'DOT_ID_Reporting_Airline', 'CRSDepTime', 'CRSArrTime',
               'OriginAirportID', 'OriginState', 'DestAirportID', 'DestState',
               'Cancelled', 'Diverted', 'Flight_Number_Reporting_Airline',
               'Reporting_Airline']
    with zipfile.ZipFile(path) as archive:
        member = [n for n in archive.namelist() if n.endswith('.csv')][0]
        with archive.open(member) as stream:
            bts = pd.read_csv(stream, usecols=usecols)
    assert bts.Year.eq(year).all() and bts.Month.eq(11).all()
    return normalize(bts.rename(columns=RENAMES))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='baseline_recovery_v2_bts_november_recovery_20260917')
    args = ap.parse_args()
    if not args.name.startswith('baseline_recovery_v2') or Path(args.name).name != args.name:
        raise ValueError('Invalid output name')
    out = ROOT / 'output'
    if list(out.glob(args.name + '_*')):
        raise FileExistsError('Use a fresh name; existing evidence is preserved')

    source = ROOT / 'data/train.csv'
    raw = pd.read_csv(source, usecols=['ID'] + BASE + EXTRA + ['Airline', 'Carrier_Code(IATA)'])
    november = normalize(raw.loc[raw.Month.eq(11)]).reset_index(drop=True)
    complete = november.dropna(subset=BASE).reset_index(drop=True)
    incomplete = november.loc[november[BASE].isna().any(axis=1)].reset_index(drop=True)

    bts = {year: read_bts(year) for year in YEARS}
    print('bts loaded', flush=True)

    # ---- Part A: rows matching more than one candidate year on the full key ----
    hits = {}
    for year, frame in bts.items():
        counts = frame.dropna(subset=BASE).groupby(BASE, dropna=False).size().rename('n')
        merged = complete[BASE].merge(counts, on=BASE, how='left', validate='many_to_one')
        hits[year] = merged.n.fillna(0).astype(int).to_numpy()
    n_years = pd.DataFrame(hits).gt(0).sum(axis=1)
    multi = complete.loc[n_years.gt(1)].copy().reset_index(drop=True)
    print(f'multi-candidate rows: {len(multi)}', flush=True)

    tie_rows = []
    for year, frame in bts.items():
        cand = multi[['ID'] + BASE].merge(frame, on=BASE, how='left')
        for field in EXTRA:
            cand = cand.rename(columns={field: f'{field}__{year}'})
        tie_rows.append(cand.set_index('ID')[[f'{f}__{year}' for f in EXTRA]
                                             + ['Flight_Number_Reporting_Airline']]
                        .rename(columns={'Flight_Number_Reporting_Airline': f'flight_no__{year}'}))
    tie = pd.concat(tie_rows, axis=1)
    tie = multi.set_index('ID').join(tie)
    breaks = []
    for field in EXTRA:
        same_as_source = [(tie[f'{field}__{y}'].astype(str) == tie[field].astype(str)) for y in YEARS]
        differ_between_years = tie[f'{field}__{YEARS[0]}'].astype(str) != tie[f'{field}__{YEARS[1]}'].astype(str)
        breaks.append({'field': field,
                       'rows': int(len(tie)),
                       'differs_between_candidate_years': int(differ_between_years.sum()),
                       'matches_source_in_2018': int(same_as_source[0].sum()),
                       'matches_source_in_2019': int(same_as_source[1].sum()),
                       'can_break_tie': bool(differ_between_years.any())})
    # flight number is NOT present in the source; recorded to show why it cannot be used
    breaks.append({'field': 'Flight_Number_Reporting_Airline (not in source)',
                   'rows': int(len(tie)),
                   'differs_between_candidate_years': int((tie[f'flight_no__{YEARS[0]}']
                                                           != tie[f'flight_no__{YEARS[1]}']).sum()),
                   'matches_source_in_2018': 0, 'matches_source_in_2019': 0,
                   'can_break_tie': False})
    pd.DataFrame(breaks).to_csv(out / f'{args.name}_tiebreak_fields.csv', index=False)
    tie.reset_index().to_csv(out / f'{args.name}_multi_candidate_detail.csv', index=False)
    print('part A done', flush=True)

    # ---- Part B: missing-key rows, matched on the keys each row actually has ----
    present = incomplete[BASE].notna()
    pattern = present.apply(lambda r: '+'.join(c for c in BASE if r[c]), axis=1)
    incomplete = incomplete.assign(key_pattern=pattern)
    per_row = []
    pattern_rows = []
    for keys_text, chunk in incomplete.groupby('key_pattern'):
        keys = keys_text.split('+')
        counts_by_year = {}
        for year, frame in bts.items():
            counts = frame.dropna(subset=keys).groupby(keys, dropna=False).size().rename('n')
            merged = chunk[keys].merge(counts, on=keys, how='left', validate='many_to_one')
            counts_by_year[year] = merged.n.fillna(0).astype(int).to_numpy()
        table = pd.DataFrame(counts_by_year, index=chunk.index)
        total = table.sum(axis=1)
        years_hit = table.gt(0).sum(axis=1)
        per_row.append(pd.DataFrame({'ID': chunk.ID.to_numpy(), 'key_pattern': keys_text,
                                     'missing_keys': '+'.join(c for c in BASE if c not in keys),
                                     'candidates_2018': table[2018].to_numpy(),
                                     'candidates_2019': table[2019].to_numpy(),
                                     'candidates_total': total.to_numpy(),
                                     'candidate_years': years_hit.to_numpy()}))
        pattern_rows.append({'key_pattern': keys_text,
                             'missing_keys': '+'.join(c for c in BASE if c not in keys),
                             'n_keys': len(keys), 'rows': int(len(chunk)),
                             'no_candidate': int(total.eq(0).sum()),
                             'unique_single_bts_row': int(total.eq(1).sum()),
                             'unique_row_one_year_only': int((total.eq(1) & years_hit.eq(1)).sum()),
                             'many_candidates': int(total.gt(1).sum())})
    rows = pd.concat(per_row, ignore_index=True)
    patterns = pd.DataFrame(pattern_rows).sort_values('rows', ascending=False)
    patterns.to_csv(out / f'{args.name}_missing_key_patterns.csv', index=False)
    local = ROOT / 'data/bts' / args.name
    local.mkdir(parents=True, exist_ok=True)
    row_path = local / 'missing_key_rows.csv.gz'
    rows.to_csv(row_path, index=False, compression='gzip')

    totals = {'incomplete_rows': int(len(incomplete)),
              'no_candidate': int(patterns.no_candidate.sum()),
              'unique_single_bts_row': int(patterns.unique_single_bts_row.sum()),
              'many_candidates': int(patterns.many_candidates.sum())}
    print(json.dumps(totals, indent=2), flush=True)

    manifest = {
        'name': args.name,
        'source_sha256': digest(source),
        'code_sha256': hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'fits_any_model': False, 'uses_external_data': True, 'target_columns_used': [],
        'years_compared': YEARS,
        'complete_rows': int(len(complete)),
        'multi_candidate_rows': int(len(multi)),
        'incomplete_rows': totals['incomplete_rows'],
        'incomplete_no_candidate': totals['no_candidate'],
        'incomplete_unique_single_bts_row': totals['unique_single_bts_row'],
        'incomplete_many_candidates': totals['many_candidates'],
        'row_evidence': str(row_path.relative_to(ROOT)),
        'limitations': [
            'A unique relaxed-key candidate is not a confirmed match and assigns no year.',
            'Missing-key results are never pooled with complete-fingerprint results.',
            'Only November and the 2018/2019 reporting-carrier files are covered.',
            'Rows that stay ambiguous are left unresolved by design.',
            'Carriers absent from the reporting-carrier file cannot match at any key level.',
        ],
    }
    (out / f'{args.name}_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('done')


if __name__ == '__main__':
    main()
