"""Combine the per-month Marketing Carrier comparisons into one summary.

Reads only the row evidence that the per-month runs already wrote, so it fits
no model, reads no target column and re-reads no BTS archive. Rows matching
both candidate years stay unresolved and are excluded from composition shares
rather than assigned to a year.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
YEARS = [2018, 2019]
# Thanksgiving falls on 22 Nov 2018 and 28 Nov 2019; both are the fourth Thursday.
THANKSGIVING = {2018: (11, 22), 2019: (11, 28)}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            sha.update(chunk)
    return sha.hexdigest()


def load(run: str) -> pd.DataFrame:
    path = ROOT / 'data/bts' / run / 'matches.csv.gz'
    frame = pd.read_csv(path)
    for year in YEARS:
        frame[f'hit_{year}'] = frame[f'{year}_marketing'].gt(0)
    frame['candidate_years'] = frame[[f'hit_{y}' for y in YEARS]].sum(axis=1)
    frame['resolved_year'] = pd.NA
    for year in YEARS:
        only = frame[f'hit_{year}'] & frame.candidate_years.eq(1)
        frame.loc[only, 'resolved_year'] = year
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True,
                    help='per-month run names already produced by verify_bts_marketing.py')
    ap.add_argument('--name', required=True)
    args = ap.parse_args()
    if not args.name.startswith('baseline_recovery_v2') or Path(args.name).name != args.name:
        raise ValueError('Invalid output name')
    out = ROOT / 'output'
    if list(out.glob(args.name + '_*')):
        raise FileExistsError('Use a fresh name; existing evidence is preserved')

    months, daily, sources = [], [], []
    for run in args.runs:
        manifest_path = out / f'{run}_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        frame = load(run)
        month = int(manifest['month'])
        resolved = frame.resolved_year.notna()
        counts = {y: int(frame.resolved_year.eq(y).sum()) for y in YEARS}
        denominator = sum(counts.values())
        months.append({
            'month': month, 'run': run,
            'complete_rows': int(manifest['complete_rows']),
            'matched_rows': int(manifest['matched_rows']),
            'match_rate': float(manifest['match_rate']),
            'unmatched_rows': int(manifest['still_unmatched']),
            'resolved_to_one_year': int(resolved.sum()),
            'unresolved_both_years': int(manifest['multiple_candidate_years']),
            'rows_2018': counts[2018], 'rows_2019': counts[2019],
            'share_2018': counts[2018] / denominator if denominator else float('nan'),
            'missing_key_rows': int(manifest['incomplete_rows']),
            'missing_key_no_candidate': int(manifest['incomplete_no_candidate']),
            'missing_key_unique': int(manifest['incomplete_unique_single_bts_row']),
            'missing_key_multiple': int(manifest['incomplete_many_candidates']),
        })
        sources.append({'run': run, 'manifest_sha256': digest(manifest_path),
                        'row_evidence_sha256': digest(ROOT / 'data/bts' / run / 'matches.csv.gz')})

        table = frame.loc[resolved].groupby(['Day_of_Month', 'resolved_year']).size().unstack(fill_value=0)
        for day, row in table.iterrows():
            a, b = int(row.get(2018, 0)), int(row.get(2019, 0))
            daily.append({'month': month, 'day': int(day), 'rows_2018': a, 'rows_2019': b,
                          'share_2018': a / (a + b) if (a + b) else float('nan')})

    month_frame = pd.DataFrame(months).sort_values('month')
    daily_frame = pd.DataFrame(daily).sort_values(['month', 'day'])
    month_frame.to_csv(out / f'{args.name}_months.csv', index=False)
    daily_frame.to_csv(out / f'{args.name}_daily_composition.csv', index=False)

    # Falsifiable check: the internal calendar analysis inferred two weekday
    # alignments from dips at 11/22 and 11/28. If those are the two Thanksgivings,
    # each holiday must depress its OWN year's share on that day, in opposite
    # directions. This compares each date against that month's median share.
    holiday = []
    if 11 in set(month_frame.month):
        november = daily_frame[daily_frame.month.eq(11)]
        median = float(november.share_2018.median())
        for year, (m, day) in THANKSGIVING.items():
            row = november[november.day.eq(day)]
            if row.empty:
                continue
            share = float(row.share_2018.iloc[0])
            holiday.append({
                'date': f'{m:02d}-{day:02d}', 'thanksgiving_year': year,
                'share_2018': share, 'november_median_share_2018': median,
                'delta_vs_median': share - median,
                'direction': 'depresses 2018' if share < median else 'depresses 2019',
                'matches_prediction': (share < median) if year == 2018 else (share > median),
            })
        pd.DataFrame(holiday).to_csv(out / f'{args.name}_thanksgiving_check.csv', index=False)

    totals = {
        'months_compared': sorted(int(m) for m in month_frame.month),
        'complete_rows': int(month_frame.complete_rows.sum()),
        'matched_rows': int(month_frame.matched_rows.sum()),
        'unmatched_rows': int(month_frame.unmatched_rows.sum()),
        'resolved_to_one_year': int(month_frame.resolved_to_one_year.sum()),
        'unresolved_both_years': int(month_frame.unresolved_both_years.sum()),
        'missing_key_rows': int(month_frame.missing_key_rows.sum()),
        'missing_key_no_candidate': int(month_frame.missing_key_no_candidate.sum()),
        'missing_key_unique': int(month_frame.missing_key_unique.sum()),
        'missing_key_multiple': int(month_frame.missing_key_multiple.sum()),
        'share_2018_min_month': float(month_frame.share_2018.min()),
        'share_2018_max_month': float(month_frame.share_2018.max()),
    }
    manifest = {
        'name': args.name,
        'code_sha256': hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'fits_any_model': False, 'uses_external_data': False, 'target_columns_used': [],
        'sources': sources, 'totals': totals,
        'thanksgiving_check': holiday,
        'limitations': [
            'Only the compared months are covered; the remaining months are untested.',
            'Shares are computed over rows resolved to exactly one year; rows matching both are excluded, not assigned.',
            'A per-month share describes this file\'s composition, not any airline\'s traffic.',
            'Missing-key rows are counted separately and never pooled with complete rows.',
            'Matching a year means the flight existed that year, not that the file contains no other year.',
        ],
    }
    (out / f'{args.name}_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(month_frame.to_string(index=False))
    print()
    print(json.dumps(totals, indent=2))
    if holiday:
        print()
        print(pd.DataFrame(holiday).to_string(index=False))


if __name__ == '__main__':
    main()
