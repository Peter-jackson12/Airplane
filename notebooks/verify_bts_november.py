"""Compare complete November inputs with four user-supplied BTS ZIPs.

Diagnostic only: no target columns, imputation, model fitting, or year assignment.
Exact matches use tail, route, month/day, distance, DOT ID and both times.
Scheduled and actual BTS times are tested separately; relaxed matches omit DOT
only and are never treated as confirmed matches. Row evidence stays in data/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['Month', 'Day_of_Month', 'Tail_Number', 'Origin_Airport',
        'Destination_Airport', 'Distance', 'Carrier_ID(DOT)',
        'Estimated_Departure_Time', 'Estimated_Arrival_Time']
TEXT = {'Tail_Number', 'Origin_Airport', 'Destination_Airport'}
RENAMES = {'DayofMonth': 'Day_of_Month', 'Origin': 'Origin_Airport',
           'Dest': 'Destination_Airport',
           'DOT_ID_Reporting_Airline': 'Carrier_ID(DOT)'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalize(frame):
    frame = frame.copy()
    for column in KEYS:
        if column in TEXT:
            frame[column] = frame[column].astype('string').str.strip().str.upper()
        else:
            frame[column] = pd.to_numeric(frame[column], errors='raise').astype('Int64')
    return frame


def match_counts(left, right, keys):
    counts = right.dropna(subset=keys).groupby(keys, dropna=False).size().rename('matches')
    result = left[keys].merge(counts, on=keys, how='left', validate='many_to_one')
    assert len(result) == len(left)
    return result.matches.fillna(0).astype(int).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='baseline_recovery_v2_bts_november_20260917')
    args = ap.parse_args()
    if not args.name.startswith('baseline_recovery_v2') or Path(args.name).name != args.name:
        raise ValueError('Invalid output name')
    out = ROOT / 'output'
    local = ROOT / 'data/bts' / args.name
    if local.exists() or list(out.glob(args.name + '_*')):
        raise FileExistsError('Use a fresh name; existing evidence is preserved')
    source = ROOT / 'data/train.csv'
    raw = pd.read_csv(source, usecols=['ID'] + KEYS)
    november = normalize(raw.loc[raw.Month.eq(11)]).reset_index(drop=True)
    complete = november.dropna(subset=KEYS).copy().reset_index(drop=True)
    local.mkdir()
    evidence = complete.copy()
    files, summary, groups = [], [], []
    fingerprints = pd.read_csv(ROOT / 'output/weather_review/year_verification_fingerprints.csv')
    fingerprint_ids = set(fingerprints.loc[fingerprints.Month.eq(11), 'ID'])
    for year in [2007, 2013, 2018, 2019]:
        path = ROOT / f'data/bts/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_11.zip'
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f'Invalid ZIP: {path}')
            members = [n for n in archive.namelist() if n.endswith('.csv')]
            if len(members) != 1:
                raise ValueError('Expected exactly one CSV')
            usecols = ['Year', 'Month', 'DayofMonth', 'Tail_Number', 'Origin', 'Dest',
                       'Distance', 'DOT_ID_Reporting_Airline', 'CRSDepTime', 'CRSArrTime',
                       'DepTime', 'ArrTime']
            with archive.open(members[0]) as stream:
                bts = pd.read_csv(stream, usecols=usecols)
        assert bts.Year.eq(year).all() and bts.Month.eq(11).all()
        files.append({'file': str(path.relative_to(ROOT)), 'sha256': digest(path),
                      'bytes': path.stat().st_size, 'rows': len(bts), 'crc_ok': True,
                      'url': 'https://transtats.bts.gov/PREZIP/' + path.name})
        bts = bts.rename(columns=RENAMES)
        for mode, times in [('scheduled', ['CRSDepTime', 'CRSArrTime']),
                            ('actual', ['DepTime', 'ArrTime'])]:
            right = normalize(bts.rename(columns=dict(zip(times, KEYS[-2:]))))
            for policy, keys in [('strict', KEYS), ('without_dot', [k for k in KEYS if k != 'Carrier_ID(DOT)'])]:
                column = f'{year}_{mode}_{policy}'
                evidence[column] = match_counts(complete, right, keys)
                matched = evidence[column].gt(0)
                fp = evidence.ID.isin(fingerprint_ids)
                summary.append({'year': year, 'time_mode': mode, 'policy': policy,
                                'eligible_rows': len(complete), 'matched_rows': int(matched.sum()),
                                'match_rate': float(matched.mean()),
                                'multiple_bts_rows': int(evidence[column].gt(1).sum()),
                                'fingerprint_rows': int(fp.sum()),
                                'fingerprint_matched': int((matched & fp).sum())})
                if policy == 'strict':
                    for dimension in ['Carrier_ID(DOT)', 'route']:
                        grouping = (complete['Origin_Airport'] + '->' + complete['Destination_Airport']
                                    if dimension == 'route' else complete[dimension].astype(str))
                        table = pd.DataFrame({'value': grouping, 'matched': matched}).groupby('value').matched.agg(['size', 'sum'])
                        for value, row in table.iterrows():
                            groups.append({'year': year, 'time_mode': mode, 'dimension': dimension,
                                           'value': value, 'eligible_rows': int(row['size']),
                                           'matched_rows': int(row['sum']),
                                           'match_rate': float(row['sum'] / row['size'])})
        print(year, summary[-4:], flush=True)
    cross = {}
    for mode in ['scheduled', 'actual']:
        columns = [f'{y}_{mode}_strict' for y in [2007, 2013, 2018, 2019]]
        number = evidence[columns].gt(0).sum(axis=1)
        cross[mode] = {'no_candidate': int(number.eq(0).sum()),
                       'one_candidate_year': int(number.eq(1).sum()),
                       'multiple_candidate_years': int(number.gt(1).sum())}
    row_path = local / 'matches.csv.gz'
    evidence.to_csv(row_path, index=False, compression='gzip')
    pd.DataFrame(summary).to_csv(out / f'{args.name}_summary.csv', index=False)
    pd.DataFrame(groups).to_csv(out / f'{args.name}_groups.csv', index=False)
    manifest = {'name': args.name, 'source_sha256': digest(source),
                'code_sha256': digest(Path(__file__)), 'fits_any_model': False,
                'uses_external_data': True, 'target_columns_used': [],
                'november_rows': len(november), 'complete_rows': len(complete),
                'incomplete_rows_excluded': len(november) - len(complete),
                'keys': KEYS, 'files': files, 'cross_year': cross,
                'row_evidence': str(row_path.relative_to(ROOT)), 'row_sha256': digest(row_path),
                'limitations': ['Only November and four candidate years are covered.',
                    'Missing-key rows are excluded, not counted as failed matches.',
                    'A failed exact match can reflect source coverage or field transformations.',
                    'without_dot is diagnostic only and does not assign years.',
                    'Tail numbers retain their N prefix; no undocumented transformation.',
                    'Other months and event weather remain unverified.']}
    (out / f'{args.name}_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
