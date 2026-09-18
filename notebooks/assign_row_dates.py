"""Fix an explicit, row-level date-attribution rule and apply it to the months
already compared against BTS Marketing Carrier data.

This is a FOLLOW-UP over the stored results of notebooks/verify_bts_marketing.py,
not a re-implementation of that comparison. For each --run given, the
complete-fingerprint verdict (candidate BTS rows per year) is read from the
run's own matches.csv.gz and cross-checked against its manifest.json; it is
not recomputed from a fresh comparison. The only genuinely new computation is
the missing-key candidate count PER ROW (the existing runs only saved
per-pattern aggregates in *_missing_key_patterns.csv), and a diagnostic that
hides keys on complete rows to see how the missing-key logic behaves when the
true answer is already known.

No target or delay column is read anywhere in this script. data/train.csv is
only read, never written. Months without a --run entry (no verified Marketing
Carrier comparison yet) are reported as not-checked, not silently skipped and
not inferred from the checked months.

Attribution rule
-----------------
A row's calendar date (month, day) is already known for every row in this
file: Month and Day_of_Month have zero missing values across the full
1,000,000-row source (see output/label_coverage/schema.csv). What this script
decides is the YEAR, and, for rows with a missing key, how much to trust a
match found on the remaining keys.

Adopt a year (date_attributed = True) only for:
  * complete_single_candidate_year -- every one of the 9 keys is observed and
    the row matches BTS rows in exactly one of the two candidate years.

Hold (date_attributed = False) for everything else, including:
  * complete_multi_candidate_year   -- matches both 2018 and 2019.
  * complete_checked_no_candidate   -- all 9 keys observed, no BTS match in
    either year (0 rows in the three checked months; kept for months where
    it may not be 0).
  * missing_key_single_candidate_year -- a unique match on the keys the row
    actually has. This is NOT treated as equivalent to a full-fingerprint
    match: a smaller key set can pick out a "unique" candidate that a fuller
    key set would have split or ruled out. See the masking diagnostic below.
  * missing_key_multi_candidate_year / missing_key_checked_no_candidate.
  * not_checked_month_unavailable -- the row's month has no verified
    Marketing Carrier comparison (BTS ZIPs missing, or ZIPs present but not
    yet compared with verify_bts_marketing.py).

Explicitly not done, per the project's own prior findings:
  * The target, delay outcome, model prediction, or weather fit is never used
    to pick a year -- none of those columns are read.
  * A multi-candidate or missing-key row is never assigned by monthly year
    share, "closest year", or arbitrary choice.
  * "Unique among 2018/2019" is never reported as "unique among all possible
    years" -- only those two BTS files exist to compare against.
  * A row with no candidate is never treated as evidence it belongs to some
    other, uncompared year.
  * The three checked months' composition is never propagated to any other
    month; unchecked months stay unchecked.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import zipfile
from pathlib import Path

import pandas as pd

from notebooks.verify_bts_marketing import KEYS, RENAMES, USECOLS, YEARS, digest, normalize

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/train.csv'

# The only three of the nine keys that are ever missing in the source (schema.csv:
# Month, Day_of_Month, Tail_Number, Origin_Airport, Destination_Airport, Distance
# all have 0 missing across the full file). Every real missing-key pattern is a
# non-empty subset of this list.
VOLATILE_KEYS = ['Carrier_ID(DOT)', 'Estimated_Departure_Time', 'Estimated_Arrival_Time']
assert set(VOLATILE_KEYS) <= set(KEYS)

FLIGHT_USECOLS = USECOLS + ['Flight_Number_Operating_Airline']

STATUS_COMPLETE_SINGLE = 'complete_single_candidate_year'
STATUS_COMPLETE_MULTI = 'complete_multi_candidate_year'
STATUS_COMPLETE_NONE = 'complete_checked_no_candidate'
STATUS_MISSING_SINGLE = 'missing_key_single_candidate_year'
STATUS_MISSING_MULTI = 'missing_key_multi_candidate_year'
STATUS_MISSING_NONE = 'missing_key_checked_no_candidate'
STATUS_NOT_CHECKED = 'not_checked_month_unavailable'
ALL_STATUSES = [STATUS_COMPLETE_SINGLE, STATUS_COMPLETE_MULTI, STATUS_COMPLETE_NONE,
                 STATUS_MISSING_SINGLE, STATUS_MISSING_MULTI, STATUS_MISSING_NONE,
                 STATUS_NOT_CHECKED]
ADOPTED_STATUSES = {STATUS_COMPLETE_SINGLE}


def zip_path(year: int, month: int) -> Path:
    return ROOT / ('data/bts/On_Time_Marketing_Carrier_On_Time_Performance_Beginning'
                   f'_January_2018_{year}_{month}.zip')


def read_marketing_with_flight(year: int, month: int) -> tuple[pd.DataFrame, Path]:
    """Same loader as verify_bts_marketing.read_marketing, plus the operating
    flight number, kept separate so the already-verified script is untouched."""
    path = zip_path(year, month)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f'Invalid ZIP: {path}')
        members = [n for n in archive.namelist() if n.endswith('.csv')]
        if len(members) != 1:
            raise ValueError('Expected exactly one CSV')
        with archive.open(members[0]) as stream:
            # 2018-08 has one row with an invalid UTF-8 byte sequence outside the
            # 9 match keys (see verify_bts_marketing.read_marketing); replacing it
            # is required to read the file at all.
            frame = pd.read_csv(stream, usecols=FLIGHT_USECOLS, encoding_errors='replace')
    assert frame.Year.eq(year).all() and frame.Month.eq(month).all()
    return normalize(frame.rename(columns=RENAMES)), path


def group_stats(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = frame.dropna(subset=keys).groupby(keys, dropna=False)
    size = g.size().rename('n_bts_rows')
    n_flight = g['Flight_Number_Operating_Airline'].nunique().rename('n_distinct_flight_numbers')
    return pd.concat([size, n_flight], axis=1)


def load_run(run: str) -> tuple[dict, pd.DataFrame]:
    manifest_path = ROOT / 'output' / f'{run}_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'No manifest for run {run!r}; run verify_bts_marketing.py first')
    manifest = json.loads(manifest_path.read_text())
    if not manifest['bts_table'].startswith('Marketing Carrier'):
        raise ValueError(f'{run} is not a Marketing Carrier comparison')
    row_path = ROOT / manifest['row_evidence']
    if digest(row_path) != manifest['row_sha256']:
        raise ValueError(f'{row_path} does not match the hash recorded in {manifest_path}')
    complete = normalize(pd.read_csv(row_path))
    return manifest, complete


def required_zip_status(month: int) -> list[dict]:
    return [{'file': str(zip_path(year, month).relative_to(ROOT)),
             'downloaded': zip_path(year, month).exists()} for year in YEARS]


def classify_complete(complete: pd.DataFrame) -> pd.DataFrame:
    complete = complete.copy()
    hit = {year: complete[f'{year}_marketing'].gt(0) for year in YEARS}
    n_years = sum(hit.values())
    complete['candidate_years_count'] = n_years
    status = pd.Series(STATUS_COMPLETE_MULTI, index=complete.index)
    status[n_years.eq(0)] = STATUS_COMPLETE_NONE
    status[n_years.eq(1)] = STATUS_COMPLETE_SINGLE
    complete['status'] = status
    complete['attributed_year'] = pd.NA
    for year in YEARS:
        only = hit[year] & n_years.eq(1)
        complete.loc[only, 'attributed_year'] = year
    complete['attributed_year'] = complete['attributed_year'].astype('Int64')
    return complete


def attach_flight_stats(complete: pd.DataFrame, flight_frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    complete = complete.copy()
    for year, frame in flight_frames.items():
        stats = group_stats(frame, KEYS)
        merged = complete[KEYS].merge(stats, on=KEYS, how='left', validate='many_to_one')
        n_rows = merged['n_bts_rows'].fillna(0).astype(int).to_numpy()
        expected = complete[f'{year}_marketing'].to_numpy()
        if not (n_rows == expected).all():
            raise ValueError(
                f'Re-reading the {year} Marketing ZIP for the flight-number check '
                f'disagrees with the stored {year}_marketing counts; the ZIP may '
                'have changed since verify_bts_marketing.py was run.')
        complete[f'{year}_n_distinct_flight_numbers'] = merged['n_distinct_flight_numbers'].to_numpy()
    both = complete[[f'{y}_n_distinct_flight_numbers' for y in YEARS]]
    complete['flight_is_unique'] = both.max(axis=1, skipna=True).le(1).astype('boolean')
    complete.loc[complete['candidate_years_count'].eq(0), 'flight_is_unique'] = pd.NA
    return complete


def missing_key_candidates(incomplete: pd.DataFrame, flight_frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    present = incomplete[KEYS].notna()
    pattern = present.apply(lambda r: '+'.join(c for c in KEYS if r[c]), axis=1)
    incomplete = incomplete.assign(key_pattern=pattern)
    chunks = []
    for keys_text, chunk in incomplete.groupby('key_pattern'):
        keys = keys_text.split('+')
        per_year_n, per_year_flight = {}, {}
        for year, frame in flight_frames.items():
            stats = group_stats(frame, keys)
            merged = chunk[keys].merge(stats, on=keys, how='left', validate='many_to_one')
            per_year_n[year] = merged['n_bts_rows'].fillna(0).astype(int).to_numpy()
            per_year_flight[year] = merged['n_distinct_flight_numbers'].to_numpy()
        total = sum(per_year_n.values())
        years_hit = sum((n > 0).astype(int) for n in per_year_n.values())
        result = chunk[['ID', 'row_position', 'Month', 'Day_of_Month']].copy()
        result['key_pattern'] = keys_text
        result['missing_keys'] = '+'.join(c for c in KEYS if c not in keys)
        for year in YEARS:
            result[f'{year}_marketing'] = per_year_n[year]
            result[f'{year}_n_distinct_flight_numbers'] = per_year_flight[year]
        result['candidate_bts_rows_total'] = total
        result['candidate_years_count'] = years_hit
        chunks.append(result)
    return pd.concat(chunks, ignore_index=True)


def classify_missing(rows: pd.DataFrame) -> pd.DataFrame:
    """Status follows the total raw BTS row count on the reduced key set, matching
    the precedent set by verify_bts_marketing.py's own missing-key aggregation
    (unique_single_bts_row = total.eq(1)). candidate_years_count is kept as a
    separate diagnostic column, but is NOT used for the single/multi split: with
    fewer keys, more than one matching BTS row in the same year is not safely
    assumed to be a codeshare duplicate of the same flight (unlike the
    complete-fingerprint case, where all 9 keys pin the physical flight)."""
    rows = rows.copy()
    status = pd.Series(STATUS_MISSING_MULTI, index=rows.index)
    status[rows.candidate_bts_rows_total.eq(0)] = STATUS_MISSING_NONE
    status[rows.candidate_bts_rows_total.eq(1)] = STATUS_MISSING_SINGLE
    rows['status'] = status
    both = rows[[f'{y}_n_distinct_flight_numbers' for y in YEARS]]
    rows['flight_is_unique'] = both.max(axis=1, skipna=True).le(1).astype('boolean')
    rows.loc[rows.candidate_bts_rows_total.eq(0), 'flight_is_unique'] = pd.NA
    # missing-key rows are never adopted, so no attributed_year is stored --
    # only the diagnostic fields above, which the report layer may read but
    # must not present as a confirmed date.
    return rows


def masking_diagnostic(truth_by_month: dict[int, pd.DataFrame],
                        flight_frames: dict[tuple[int, int], pd.DataFrame]) -> pd.DataFrame:
    """Hide subsets of the keys that are ever actually missing on rows whose
    TRUE year is already known (complete_single_candidate_year), and re-match
    on the reduced key set. This measures how often a missing-key row's
    'unique candidate' would have been wrong or newly ambiguous.

    Limitation: this uses complete rows with keys artificially removed. Real
    missing-key rows are not a random sample of complete rows (their carrier,
    route, or era may differ systematically), so this does not establish the
    accuracy of any actual missing-key row -- it only bounds the risk of
    trusting a 'unique candidate' found on a reduced key set.
    """
    records = []
    for r in (1, 2, 3):
        for combo in itertools.combinations(VOLATILE_KEYS, r):
            reduced = [k for k in KEYS if k not in combo]
            correct = wrong = multi = none = 0
            for month, truth in truth_by_month.items():
                if truth.empty:
                    continue
                counts = {}
                for year in YEARS:
                    frame = flight_frames[(month, year)]
                    g = frame.dropna(subset=reduced).groupby(reduced, dropna=False).size()
                    merged = truth[reduced].merge(g.rename('n'), on=reduced, how='left',
                                                  validate='many_to_one')
                    counts[year] = merged['n'].fillna(0).astype(int).to_numpy()
                n_years = (counts[YEARS[0]] > 0).astype(int) + (counts[YEARS[1]] > 0).astype(int)
                truth_year = truth['attributed_year'].to_numpy()
                for y0, n_year, ty in zip(counts[YEARS[0]], n_years, truth_year):
                    if n_year == 0:
                        none += 1
                    elif n_year == 1:
                        picked = YEARS[0] if y0 > 0 else YEARS[1]
                        if picked == ty:
                            correct += 1
                        else:
                            wrong += 1
                    else:
                        multi += 1
            records.append({
                'masked_keys': '+'.join(combo), 'n_keys_masked': r,
                'n_rows_tested': correct + wrong + multi + none,
                'resolved_single_year_correct': correct,
                'resolved_single_year_wrong': wrong,
                'became_multi_year': multi,
                'became_no_candidate': none,
            })
    return pd.DataFrame(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+',
                    default=['baseline_recovery_v2_bts_marketing_m02_20260917',
                             'baseline_recovery_v2_bts_marketing_m07_20260917',
                             'baseline_recovery_v2_bts_marketing_m11_20260917'],
                    help='verify_bts_marketing.py run names already produced for each covered month')
    ap.add_argument('--name', required=True)
    args = ap.parse_args()
    if not args.name.startswith('baseline_recovery_v2') or Path(args.name).name != args.name:
        raise ValueError('Invalid output name')
    out = ROOT / 'output'
    local = ROOT / 'data/bts' / args.name
    if local.exists() or list(out.glob(args.name + '_*')):
        raise FileExistsError('Use a fresh name; existing evidence is preserved')

    source_hash = digest(SOURCE)
    raw = pd.read_csv(SOURCE, usecols=['ID'] + KEYS + ['Airline', 'Carrier_Code(IATA)'])
    raw = raw.reset_index().rename(columns={'index': 'row_position'})
    if raw.ID.duplicated().any():
        raise ValueError('Duplicate ID in source; row_position mapping would be ambiguous')
    raw_n = normalize(raw)
    id_to_pos = raw_n.set_index('ID')['row_position']

    covered_months: dict[int, str] = {}
    all_rows = []
    complete_truth_by_month: dict[int, pd.DataFrame] = {}
    flight_frames: dict[tuple[int, int], pd.DataFrame] = {}
    month_totals = []

    for run in args.runs:
        manifest, complete = load_run(run)
        month = int(manifest['month'])
        if manifest['source_sha256'] != source_hash:
            raise ValueError(f'{run} was verified against a different data/train.csv')
        if month in covered_months:
            raise ValueError(f'Month {month} covered by two runs: {covered_months[month]} and {run}')
        covered_months[month] = run

        complete = complete.merge(id_to_pos.rename('row_position'), left_on='ID', right_index=True,
                                  how='left')
        if complete.row_position.isna().any():
            raise ValueError(f'{run}: some IDs in matches.csv.gz are not in data/train.csv')
        complete = classify_complete(complete)

        month_mask = raw_n.Month.eq(month)
        incomplete_source = raw_n.loc[month_mask & raw_n[KEYS].isna().any(axis=1)].copy()
        expected_total = int(month_mask.sum())
        if len(complete) + len(incomplete_source) != expected_total:
            raise ValueError(f'Month {month}: complete+incomplete does not cover all source rows')
        if len(complete) != int(manifest['complete_rows']):
            raise ValueError(f'Month {month}: complete row count drifted from {run}\'s manifest')
        if len(incomplete_source) != int(manifest['incomplete_rows']):
            raise ValueError(f'Month {month}: incomplete row count drifted from {run}\'s manifest')

        frames_this_month = {}
        for year in YEARS:
            frame, path = read_marketing_with_flight(year, month)
            recorded = next(f for f in manifest['files'] if f['file'].endswith(f'_{year}_{month}.zip'))
            if digest(path) != recorded['sha256']:
                raise ValueError(f'{path} content changed since {run} was produced')
            frames_this_month[year] = frame
            flight_frames[(month, year)] = frame

        complete = attach_flight_stats(complete, frames_this_month)
        complete_truth_by_month[month] = complete.loc[complete.status.eq(STATUS_COMPLETE_SINGLE),
                                                       KEYS + ['attributed_year']].reset_index(drop=True)

        missing = missing_key_candidates(incomplete_source, frames_this_month)
        missing = classify_missing(missing)

        # ---- regression check: row-level status counts must reproduce the
        # aggregate numbers verify_bts_marketing.py already stored for this run ----
        no_cand = int((missing.status == STATUS_MISSING_NONE).sum())
        single_cand = int((missing.status == STATUS_MISSING_SINGLE).sum())
        multi_cand = int((missing.status == STATUS_MISSING_MULTI).sum())
        if no_cand != int(manifest['incomplete_no_candidate']):
            raise ValueError(f'Month {month}: no-candidate count drifted from {run}\'s manifest')
        if single_cand != int(manifest['incomplete_unique_single_bts_row']):
            raise ValueError(f'Month {month}: unique-candidate count drifted from {run}\'s manifest')
        if multi_cand != int(manifest['incomplete_many_candidates']):
            raise ValueError(f'Month {month}: many-candidates count drifted from {run}\'s manifest')

        # ---- regression check against the numbers this task was handed ----
        one_year = int((complete.candidate_years_count == 1).sum())
        two_year = int((complete.candidate_years_count == 2).sum())
        zero_year = int((complete.candidate_years_count == 0).sum())
        rows_2018 = int((complete.attributed_year == 2018).sum())
        rows_2019 = int((complete.attributed_year == 2019).sum())
        month_totals.append({
            'month': month, 'run': run, 'complete_rows': len(complete),
            'complete_single_year': one_year, 'complete_multi_year': two_year,
            'complete_no_candidate': zero_year, 'rows_2018': rows_2018, 'rows_2019': rows_2019,
            'incomplete_rows': len(incomplete_source),
            'missing_no_candidate': int((missing.status == STATUS_MISSING_NONE).sum()),
            'missing_single_candidate': int((missing.status == STATUS_MISSING_SINGLE).sum()),
            'missing_multi_candidate': int((missing.status == STATUS_MISSING_MULTI).sum()),
        })

        complete_out = pd.DataFrame({
            'ID': complete.ID, 'row_position': complete.row_position.astype(int),
            'month': month, 'day_of_month': complete.Day_of_Month.astype(int),
            'key_pattern': '+'.join(KEYS), 'missing_keys': '',
            'checked_years': '2018,2019', 'checked_series': 'marketing_carrier',
            'candidate_bts_rows_2018': complete['2018_marketing'].astype(int),
            'candidate_bts_rows_2019': complete['2019_marketing'].astype(int),
            'candidate_bts_rows_total': (complete['2018_marketing'] + complete['2019_marketing']).astype(int),
            'candidate_years_count': complete.candidate_years_count.astype(int),
            'distinct_flight_numbers_2018': complete['2018_n_distinct_flight_numbers'],
            'distinct_flight_numbers_2019': complete['2019_n_distinct_flight_numbers'],
            'date_is_unique': True,
            'flight_is_unique': complete.flight_is_unique,
            'status': complete.status,
            'date_attributed': complete.status.eq(STATUS_COMPLETE_SINGLE),
            'attributed_year': complete.attributed_year,
            'evidence_run': run,
        })
        complete_out['attributed_date'] = pd.NA
        adopted = complete_out.date_attributed
        complete_out.loc[adopted, 'attributed_date'] = (
            complete_out.loc[adopted, 'attributed_year'].astype(str) + '-'
            + f'{month:02d}-' + complete_out.loc[adopted, 'day_of_month'].map(lambda d: f'{d:02d}'))

        missing_out = pd.DataFrame({
            'ID': missing.ID, 'row_position': missing.row_position.astype(int),
            'month': month, 'day_of_month': missing.Day_of_Month.astype(int),
            'key_pattern': missing.key_pattern, 'missing_keys': missing.missing_keys,
            'checked_years': '2018,2019', 'checked_series': 'marketing_carrier',
            'candidate_bts_rows_2018': missing['2018_marketing'].astype(int),
            'candidate_bts_rows_2019': missing['2019_marketing'].astype(int),
            'candidate_bts_rows_total': missing.candidate_bts_rows_total.astype(int),
            'candidate_years_count': missing.candidate_years_count.astype(int),
            'distinct_flight_numbers_2018': missing['2018_n_distinct_flight_numbers'],
            'distinct_flight_numbers_2019': missing['2019_n_distinct_flight_numbers'],
            'date_is_unique': True,
            'flight_is_unique': missing.flight_is_unique,
            'status': missing.status,
            'date_attributed': False,
            'attributed_year': pd.NA,
            'attributed_date': pd.NA,
            'evidence_run': run,
        })
        all_rows.append(complete_out)
        all_rows.append(missing_out)
        print(f'month {month} ({run}): {month_totals[-1]}', flush=True)

    # ---- months without a verified comparison: not checked, not inferred ----
    for month in range(1, 13):
        if month in covered_months:
            continue
        month_rows = raw_n.loc[raw_n.Month.eq(month), ['ID', 'row_position', 'Day_of_Month']]
        zip_status = required_zip_status(month)
        block = pd.DataFrame({
            'ID': month_rows.ID, 'row_position': month_rows.row_position.astype(int),
            'month': month, 'day_of_month': month_rows.Day_of_Month.astype(int),
            'key_pattern': pd.NA, 'missing_keys': pd.NA,
            'checked_years': pd.NA, 'checked_series': pd.NA,
            'candidate_bts_rows_2018': pd.NA, 'candidate_bts_rows_2019': pd.NA,
            'candidate_bts_rows_total': pd.NA, 'candidate_years_count': pd.NA,
            'distinct_flight_numbers_2018': pd.NA, 'distinct_flight_numbers_2019': pd.NA,
            'date_is_unique': True, 'flight_is_unique': pd.NA,
            'status': STATUS_NOT_CHECKED, 'date_attributed': False,
            'attributed_year': pd.NA, 'attributed_date': pd.NA,
            'evidence_run': pd.NA,
        })
        all_rows.append(block)
        print(f'month {month}: not checked -- required files: {zip_status}', flush=True)

    result = pd.concat(all_rows, ignore_index=True)
    if len(result) != 1_000_000:
        raise ValueError(f'Expected 1,000,000 rows, got {len(result)}')
    if result.ID.duplicated().any():
        raise ValueError('Duplicate ID in the assembled row-level output')
    if sorted(result.row_position.tolist()) != list(range(1_000_000)):
        raise ValueError('row_position is not a permutation of the original row order')
    if result.groupby('status').size().sum() != len(result):
        raise ValueError('status assignment does not partition all rows')

    local.mkdir(parents=True)
    row_path = local / 'row_dates.csv.gz'
    result.to_csv(row_path, index=False, compression='gzip')

    status_summary = (result.groupby(['month', 'status'], dropna=False).size()
                      .rename('rows').reset_index())
    status_summary.to_csv(out / f'{args.name}_status_summary.csv', index=False)

    diagnostic = masking_diagnostic(complete_truth_by_month, flight_frames)
    diagnostic.to_csv(out / f'{args.name}_masking_diagnostic.csv', index=False)

    months_missing = [m for m in range(1, 13) if m not in covered_months]
    manifest = {
        'name': args.name,
        'source_sha256': source_hash,
        'code_sha256': hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'fits_any_model': False, 'uses_external_data': True, 'target_columns_used': [],
        'input_runs': args.runs,
        'months_checked': sorted(covered_months),
        'months_not_checked': months_missing,
        'required_zips_for_unchecked_months': {
            str(m): required_zip_status(m) for m in months_missing},
        'adopted_statuses': sorted(ADOPTED_STATUSES),
        'all_statuses': ALL_STATUSES,
        'month_totals': month_totals,
        'total_rows': int(len(result)),
        'row_evidence': str(row_path.relative_to(ROOT)), 'row_sha256': digest(row_path),
        'status_summary': str((out / f'{args.name}_status_summary.csv').relative_to(ROOT)),
        'masking_diagnostic': str((out / f'{args.name}_masking_diagnostic.csv').relative_to(ROOT)),
        'limitations': [
            'Only months_checked have any row attributed a date; every other row keeps status '
            'not_checked_month_unavailable regardless of what the checked months looked like.',
            'missing_key_single_candidate_year is never attributed a date, by design -- it is not '
            'evidence-equivalent to a complete-fingerprint match. See masking_diagnostic for why.',
            'The masking diagnostic reuses complete rows with keys hidden; it bounds the risk of '
            'trusting a missing-key unique candidate but does not measure the accuracy of any real '
            'missing-key row, whose carrier/route/era may differ systematically from complete rows.',
            'A two-candidate-year row is "unique among 2018 and 2019", not unique among all possible years.',
            'No target, delay, model prediction, or weather column is read by this script.',
            'This script does not modify data/train.csv and does not fill or infer missing keys.',
        ],
    }
    (out / f'{args.name}_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in {'limitations', 'required_zips_for_unchecked_months'}}, indent=2, default=str))


if __name__ == '__main__':
    main()
