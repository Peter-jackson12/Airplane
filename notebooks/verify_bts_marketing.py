"""Compare one month of complete source inputs with the BTS Marketing Carrier files.

The Marketing Carrier table records both the selling and the operating carrier,
so it covers regional operators that the Reporting Carrier table omits. Pass
--month to pick the month; the source month and the BTS files must agree.

Diagnostic only: no target columns, no imputation, no model fitting, no year
assignment. Missing-key rows are reported separately and never pooled with
complete-fingerprint rows. Row evidence stays in data/ and is excluded from Git.
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

KEYS = ['Month', 'Day_of_Month', 'Tail_Number', 'Origin_Airport',
        'Destination_Airport', 'Distance', 'Carrier_ID(DOT)',
        'Estimated_Departure_Time', 'Estimated_Arrival_Time']
TEXT = {'Tail_Number', 'Origin_Airport', 'Destination_Airport'}
# the source's Carrier_ID(DOT) is the OPERATING carrier; Carrier_Code(IATA) the MARKETING one
RENAMES = {'DayofMonth': 'Day_of_Month', 'Origin': 'Origin_Airport', 'Dest': 'Destination_Airport',
           'DOT_ID_Operating_Airline': 'Carrier_ID(DOT)',
           'CRSDepTime': 'Estimated_Departure_Time', 'CRSArrTime': 'Estimated_Arrival_Time'}
USECOLS = ['Year', 'Month', 'DayofMonth', 'Tail_Number', 'Origin', 'Dest', 'Distance',
           'DOT_ID_Operating_Airline', 'CRSDepTime', 'CRSArrTime',
           'IATA_Code_Marketing_Airline', 'DOT_ID_Marketing_Airline']

NON_REPORTING = {19687, 20427, 20046, 21167, 20500, 20237, 20445, 20263, 20225}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            sha.update(chunk)
    return sha.hexdigest()


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in KEYS:
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


def read_marketing(year: int, month: int = 11) -> tuple[pd.DataFrame, Path]:
    path = ROOT / ('data/bts/On_Time_Marketing_Carrier_On_Time_Performance_Beginning'
                   f'_January_2018_{year}_{month}.zip')
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f'Invalid ZIP: {path}')
        members = [n for n in archive.namelist() if n.endswith('.csv')]
        if len(members) != 1:
            raise ValueError('Expected exactly one CSV')
        with archive.open(members[0]) as stream:
            frame = pd.read_csv(stream, usecols=USECOLS)
    assert frame.Year.eq(year).all() and frame.Month.eq(month).all()
    return normalize(frame.rename(columns=RENAMES)), path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--month', type=int, required=True, choices=range(1, 13),
                    metavar='1..12', help='calendar month to compare')
    ap.add_argument('--name', required=True)
    args = ap.parse_args()
    month = args.month
    if not args.name.startswith('baseline_recovery_v2') or Path(args.name).name != args.name:
        raise ValueError('Invalid output name')
    out = ROOT / 'output'
    local = ROOT / 'data/bts' / args.name
    if local.exists() or list(out.glob(args.name + '_*')):
        raise FileExistsError('Use a fresh name; existing evidence is preserved')

    source = ROOT / 'data/train.csv'
    raw = pd.read_csv(source, usecols=['ID'] + KEYS + ['Airline', 'Carrier_Code(IATA)'])
    selected = normalize(raw.loc[raw.Month.eq(month)]).reset_index(drop=True)
    if selected.empty:
        raise ValueError(f'No source rows for month {month}')
    complete = selected.dropna(subset=KEYS).copy().reset_index(drop=True)
    incomplete = selected.loc[selected[KEYS].isna().any(axis=1)].copy().reset_index(drop=True)

    # the Reporting Carrier comparison only covers November, so the recovery
    # crosstab is produced for that month alone
    prior_path = ROOT / 'data/bts/baseline_recovery_v2_bts_november_20260917/matches.csv.gz'
    reporting_hit = None
    if month == 11 and prior_path.exists():
        prior = pd.read_csv(prior_path)
        prior_cols = [f'{y}_scheduled_strict' for y in [2007, 2013, 2018, 2019]]
        reporting_hit = prior.set_index('ID')[prior_cols].gt(0).any(axis=1)

    local.mkdir()
    evidence = complete[['ID'] + KEYS + ['Carrier_Code(IATA)']].copy()
    files, summary = [], []
    marketing_ok = {}
    cache = {}
    for year in YEARS:
        frame, path = read_marketing(year, month)
        cache[year] = frame
        files.append({'file': str(path.relative_to(ROOT)), 'sha256': digest(path),
                      'bytes': path.stat().st_size, 'rows': int(len(frame)), 'crc_ok': True,
                      'operating_carriers': int(frame['Carrier_ID(DOT)'].nunique())})
        counts = match_counts(complete, frame, KEYS)
        evidence[f'{year}_marketing'] = counts
        # does the source's IATA code appear among the marketing codes of the matched rows?
        pairs = (frame.dropna(subset=KEYS)
                 .groupby(KEYS)['IATA_Code_Marketing_Airline']
                 .agg(lambda s: set(s.dropna())).rename('codes'))
        joined = complete[KEYS + ['Carrier_Code(IATA)']].merge(pairs, on=KEYS, how='left')
        marketing_ok[year] = [
            (isinstance(c, set) and (code in c)) if pd.notna(code) else None
            for c, code in zip(joined.codes, joined['Carrier_Code(IATA)'])]
        matched = counts > 0
        summary.append({'month': month, 'year': year, 'source': 'marketing',
                        'eligible_rows': len(complete),
                        'matched_rows': int(matched.sum()), 'match_rate': float(matched.mean()),
                        'multiple_bts_rows': int((counts > 1).sum())})
        print(year, summary[-1], flush=True)

    hit = evidence[[f'{y}_marketing' for y in YEARS]].gt(0)
    any_hit = hit.any(axis=1)
    n_years = hit.sum(axis=1)
    print(f'marketing union matched: {int(any_hit.sum())} of {len(complete)}', flush=True)

    # ---- did the previously unmatched rows recover? (November only) ----
    prior_hit = None
    if reporting_hit is not None:
        prior_hit = reporting_hit.reindex(evidence.ID).to_numpy()
        recovery = pd.crosstab(pd.Series(prior_hit, name='reporting_matched'),
                               pd.Series(any_hit.to_numpy(), name='marketing_matched'))
        recovery.to_csv(out / f'{args.name}_recovery_crosstab.csv')
        print(recovery.to_string(), flush=True)

    # ---- marketing-code agreement, an independent check of the column semantics ----
    code_rows = []
    for year in YEARS:
        flags = pd.Series(marketing_ok[year], dtype='object')
        checked = flags.notna() & pd.Series(hit[f'{year}_marketing'].to_numpy())
        agree = flags.where(checked).fillna(False).astype(bool) & checked
        code_rows.append({'year': year, 'matched_rows': int(hit[f'{year}_marketing'].sum()),
                          'rows_with_source_code': int(checked.sum()),
                          'source_code_in_marketing_codes': int(agree.sum()),
                          'agreement_rate': float(agree.sum() / checked.sum()) if checked.sum() else float('nan')})
    pd.DataFrame(code_rows).to_csv(out / f'{args.name}_marketing_code_agreement.csv', index=False)

    # ---- group breakdown, matched vs not ----
    eligible = complete.copy()
    eligible['matched'] = any_hit.to_numpy()
    eligible['route'] = eligible.Origin_Airport + '->' + eligible.Destination_Airport
    eligible['carrier_group'] = eligible['Carrier_ID(DOT)'].map(
        lambda v: 'previously-absent operator' if int(v) in NON_REPORTING else 'reporting carrier')
    groups = []
    for dimension in ['Carrier_ID(DOT)', 'Airline', 'carrier_group', 'route', 'Day_of_Month']:
        label = eligible[dimension].astype('object').where(
            eligible[dimension].notna(), '(missing)').astype(str)
        table = eligible.groupby(label).matched.agg(['size', 'sum'])
        for value, row in table.iterrows():
            groups.append({'dimension': dimension, 'value': value,
                           'eligible_rows': int(row['size']), 'matched_rows': int(row['sum']),
                           'unmatched_rows': int(row['size'] - row['sum']),
                           'unmatched_rate': float(1 - row['sum'] / row['size'])})
    pd.DataFrame(groups).to_csv(out / f'{args.name}_groups.csv', index=False)

    # ---- rows still ambiguous or still unmatched, kept separate and unresolved ----
    still_unmatched = eligible.loc[~any_hit.to_numpy()]
    multi_year = eligible.loc[n_years.gt(1).to_numpy()]
    still_unmatched.to_csv(out / f'{args.name}_still_unmatched_rows.csv', index=False)
    multi_year[['ID'] + KEYS + ['Airline', 'Carrier_Code(IATA)']].to_csv(
        out / f'{args.name}_multi_candidate_rows.csv', index=False)

    # ---- missing-key rows, reported separately, on the keys each row actually has ----
    present = incomplete[KEYS].notna()
    pattern = present.apply(lambda r: '+'.join(c for c in KEYS if r[c]), axis=1)
    incomplete = incomplete.assign(key_pattern=pattern)
    pattern_rows = []
    for keys_text, chunk in incomplete.groupby('key_pattern'):
        keys = keys_text.split('+')
        per_year = {}
        for year in YEARS:
            per_year[year] = match_counts(chunk, cache[year], keys)
        table = pd.DataFrame(per_year, index=chunk.index)
        total = table.sum(axis=1)
        pattern_rows.append({'key_pattern': keys_text,
                             'missing_keys': '+'.join(c for c in KEYS if c not in keys),
                             'rows': int(len(chunk)), 'no_candidate': int(total.eq(0).sum()),
                             'unique_single_bts_row': int(total.eq(1).sum()),
                             'many_candidates': int(total.gt(1).sum())})
    patterns = pd.DataFrame(pattern_rows).sort_values('rows', ascending=False)
    patterns.to_csv(out / f'{args.name}_missing_key_patterns.csv', index=False)

    pd.DataFrame(summary).to_csv(out / f'{args.name}_summary.csv', index=False)
    row_path = local / 'matches.csv.gz'
    evidence.to_csv(row_path, index=False, compression='gzip')

    manifest = {
        'name': args.name, 'source_sha256': digest(source),
        'code_sha256': hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'fits_any_model': False, 'uses_external_data': True, 'target_columns_used': [],
        'bts_table': 'Marketing Carrier On-Time Performance (Beginning January 2018)',
        'month': month,
        'source_rows_in_month': int(len(selected)), 'complete_rows': int(len(complete)),
        'incomplete_rows': int(len(incomplete)),
        'matched_rows': int(any_hit.sum()),
        'match_rate': float(any_hit.mean()),
        'one_candidate_year': int(n_years.eq(1).sum()),
        'multiple_candidate_years': int(n_years.gt(1).sum()),
        'still_unmatched': int((~any_hit).sum()),
        'reporting_unmatched_now_matched': (
            None if prior_hit is None
            else int(((~pd.Series(prior_hit)) & any_hit.to_numpy()).sum())),
        'incomplete_no_candidate': int(patterns.no_candidate.sum()),
        'incomplete_unique_single_bts_row': int(patterns.unique_single_bts_row.sum()),
        'incomplete_many_candidates': int(patterns.many_candidates.sum()),
        'keys': KEYS, 'files': files,
        'row_evidence': str(row_path.relative_to(ROOT)), 'row_sha256': digest(row_path),
        'limitations': [
            'Only the requested month, in 2018 and 2019, is covered; no other month or year is tested.',
            'A match confirms the row exists in that year, not that the file is that year.',
            'Marketing rows repeat one operated flight per selling carrier, so counts above one are codeshare records, not ambiguity.',
            'Missing-key rows are reported separately and never pooled with complete rows.',
            'Rows matching both years stay unresolved by design.',
        ],
    }
    (out / f'{args.name}_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in {'limitations', 'files', 'keys'}}, indent=2))


if __name__ == '__main__':
    main()
