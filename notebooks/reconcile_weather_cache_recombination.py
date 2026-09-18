"""Re-combine the ALREADY-COLLECTED 21-row and 300-row weather samples with
the CURRENT (post-33bf481-fix) join code, using ONLY their existing local
caches -- no new selection, no new fetch, no network call of any kind -- and
compare the result cell-by-cell against the ORIGINAL saved join outputs.

Why this exists: notebooks/diagnose_weather_expanded_cache.py re-aggregates
the 300-row run's ALREADY-JOINED CSVs; it never re-runs join_weather_asof or
build_requests. That alone does not demonstrate that the fixed src/weather.py
and join_weather_sample_expanded.build_requests produce the SAME real-data
result as before -- only that the old result, read back, aggregates a certain
way. This script actually calls the current join code again, from the same
cached observations, and diffs the fresh output against the original.

Reads only: the 21-row and 300-row runs' selection_with_prediction_at.csv,
fetch_manifest.json, cached observation CSVs, the mapping table the 300-row
run used, and the ORIGINAL joined_sample.csv / joined_latency{N}min.csv this
compares against. No target/delay/actual-time column is read anywhere.
A network guard monkeypatches subprocess.run and urllib.request.urlopen for
the duration of the re-join so that if any code path unexpectedly attempted a
real network call, this run fails loudly instead of silently succeeding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from notebooks.fetch_weather_sample import compute_prediction_at
from notebooks.fetch_weather_sample_expanded import (LOOKAHEAD_HOURS, LOOKBACK_HOURS,
                                                      MAX_ATTEMPTS_PER_GROUP, build_station_day_groups,
                                                      compute_plan_fingerprint)
from notebooks.join_weather_sample import load_observations
from notebooks.join_weather_sample import run_join as run_join_sample21
from notebooks.join_weather_sample_expanded import build_requests
from notebooks.join_weather_sample_expanded import run_join as run_join_expanded300
from notebooks.select_weather_sample import AIRPORTS

ROOT = Path(__file__).resolve().parents[1]
SAMPLE21_RUN = 'baseline_recovery_v2_weather_sample_20260918'
EXPANDED300_RUN = 'baseline_recovery_v2_weather_expanded_20260918'
EXPANDED300_MAPPING_RUN = 'baseline_recovery_v2_weather_scope_20260918'
LATENCIES_MINUTES = [0, 10, 30, 60]

# Every code file this reconciliation actually imports and executes, beyond this script itself --
# so the manifest's code fingerprint scope is explicit rather than silently limited to __file__.
CODE_FILES_USED_IN_RECOMBINATION = (
    'notebooks/reconcile_weather_cache_recombination.py', 'notebooks/join_weather_sample.py',
    'notebooks/join_weather_sample_expanded.py', 'notebooks/select_weather_sample.py',
    'notebooks/fetch_weather_sample_expanded.py', 'notebooks/fetch_weather_sample.py', 'src/weather.py')

DATETIME_SUFFIXES = ('_observed_at', '_available_at')
NUMERIC_SUFFIXES = ('_weather_age_minutes', '_tmpf', '_dwpf', '_relh', '_sknt', '_gust', '_vsby',
                    '_p01i', '_snowdepth')
# The ONLY approved semantic change for this comparison: a not_collectible row's station column
# flipping from a (wrongly-filled, pre-2nd-verification) value to correctly-missing. Anything else --
# including this SAME change on a collectible row -- is an unexpected regression.
STATION_COLUMN_ROLE = {'origin_station': 'origin', 'destination_station': 'destination'}
ROLE_WEATHER_SUFFIXES = ('_observed_at', '_available_at', '_tmpf', '_dwpf', '_relh', '_sknt', '_gust',
                        '_vsby', '_p01i', '_skyc1', '_wxcodes', '_snowdepth', '_metar',
                        '_weather_age_minutes')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NetworkCallAttempted(AssertionError):
    pass


def network_guard():
    """Any attempt to actually reach the network during this cache-only
    re-join must fail loudly, not silently succeed (which would make the
    comparison meaningless -- it must be running strictly off the cache)."""
    def refuse(*args, **kwargs):
        raise NetworkCallAttempted(
            'a network call was attempted during a cache-only reconciliation run; '
            f'args={args!r} kwargs={kwargs!r}')
    return mock.patch.multiple(
        'subprocess', run=refuse), mock.patch('urllib.request.urlopen', side_effect=refuse)


# ---- Step 1: verify every manifest hash this reconciliation depends on ----

def verify_manifest_hashes(out: Path) -> list[dict]:
    """Fails hard (raises) on any mismatch -- this is a data-integrity
    precondition, not a soft check to record as False and continue past."""
    checks = []

    def check(label: str, path: Path, expected_sha: str):
        if not path.exists():
            raise FileNotFoundError(f'{label}: {path} does not exist')
        actual = digest(path)
        if actual != expected_sha:
            raise ValueError(f'{label}: {path} sha256 {actual} does not match recorded {expected_sha}')
        checks.append({'label': label, 'path': str(path.relative_to(ROOT)), 'sha256': actual})

    fm21 = json.loads((out / f'{SAMPLE21_RUN}_fetch_manifest.json').read_text())
    for e in fm21['requests']:
        check(f'sample21 cache {e["station"]}/{e["day"]}', ROOT / e['cache_file'], e['sha256'])
    jm21 = json.loads((out / f'{SAMPLE21_RUN}_join_manifest.json').read_text())
    check('sample21 original joined_sample.csv', ROOT / jm21['row_evidence'], jm21['row_sha256'])

    fm300 = json.loads((out / f'{EXPANDED300_RUN}_fetch_manifest.json').read_text())
    for e in fm300['requests']:
        check(f'expanded300 cache {e["station"]}/{e["day"]}', ROOT / e['cache_file'], e['sha256'])
    jm300 = json.loads((out / f'{EXPANDED300_RUN}_join_manifest.json').read_text())
    for lat in LATENCIES_MINUTES:
        check(f'expanded300 original joined_latency{lat}min.csv',
             ROOT / jm300['row_evidence_by_scenario'][str(lat)], jm300['row_sha256_by_scenario'][str(lat)])

    return checks


def code_fingerprints() -> dict:
    """SHA-256 of every code file this reconciliation actually imports and executes -- not just this
    script's own __file__. Anything not listed in CODE_FILES_USED_IN_RECOMBINATION is OUTSIDE what
    this manifest's fingerprint claims to cover (e.g. pandas/numpy internals, or unrelated notebooks
    modules that happen to live alongside these)."""
    return {rel: hashlib.sha256((ROOT / rel).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
           for rel in CODE_FILES_USED_IN_RECOMBINATION}


def verify_input_provenance(out: Path) -> list[dict]:
    """verify_manifest_hashes above covers the fetch caches and the two ORIGINAL joined outputs; it
    never touches the actual selection/mapping inputs this reconciliation reads to rebuild the join
    requests (the two runs' selection_with_prediction_at.csv, and the 300-row run's mapping_table.csv)
    -- 579 files that never included those. This records SHA-256 for those three and cross-checks the
    ones that have a genuine PRIOR recorded expectation to check against.

    mapping_table.csv and the 300-row run's PRE-prediction_at selection.csv are BOTH embedded in that
    fetch run's own plan_fingerprint mechanism (fetch_weather_sample_expanded.compute_plan_fingerprint)
    -- but that mechanism (and the caps fields it needs to recompute from) was only added in a LATER
    fix; the actual 300-row fetch_manifest.json on disk predates it and has no recorded plan_fingerprint
    at all. When a usable one IS recorded (plan_fingerprint present and every caps field
    compute_plan_fingerprint needs), recomputing it now from the CURRENT files and comparing against
    the recorded value is a genuine check against a previously recorded expectation (it does assume
    LOOKBACK_HOURS/LOOKAHEAD_HOURS/MAX_ATTEMPTS_PER_GROUP are unchanged since that run -- current code
    constants, not independently recorded in the manifest; that assumption is stated, not hidden).
    When no usable prior fingerprint exists (the real case for this run today), this never fabricates
    one to compare against -- it only records the CURRENT hash and says so explicitly.

    Neither run's selection_with_prediction_at.csv -- the exact file actually read by rejoin_sample21/
    rejoin_expanded300 -- was ever independently hashed anywhere before now: fetch_weather_sample.py
    (the 21-row run) has no fingerprinting mechanism at all, and even where the 300-row plan_fingerprint
    exists it covers the selection file BEFORE prediction_at is added, not this one. For both, this only
    records the file's CURRENT hash; it is never reported as checked against a prior expectation, and
    past identity is never claimed retroactively."""
    entries = []

    def record(label, path, *, checked_against_prior, detail):
        entries.append({'label': label, 'path': str(path.relative_to(ROOT)), 'sha256': digest(path),
                        'checked_against_prior_recorded_expectation': checked_against_prior,
                        'detail': detail})

    mapping_path = out / f'{EXPANDED300_MAPPING_RUN}_mapping_table.csv'
    raw_selection_path = out / f'{EXPANDED300_RUN}_selection.csv'
    fm300 = json.loads((out / f'{EXPANDED300_RUN}_fetch_manifest.json').read_text())
    recorded_fp = fm300.get('plan_fingerprint')
    caps = fm300.get('caps', {})
    required_caps = ('max_requests', 'max_bytes', 'max_seconds', 'subprocess_termination_grace_seconds',
                     'success_pause_seconds')
    can_cross_check = recorded_fp is not None and all(k in caps for k in required_caps)

    if can_cross_check:
        mapping = pd.read_csv(mapping_path).set_index('iata')
        raw_selection = pd.read_csv(raw_selection_path)
        raw_selection['prediction_at'] = compute_prediction_at(raw_selection)
        collectible = raw_selection.loc[raw_selection.collectible & raw_selection.prediction_at.notna()].copy()
        collectible['origin_station'] = collectible.Origin_Airport.map(mapping.candidate_sid)
        collectible['destination_station'] = collectible.Destination_Airport.map(mapping.candidate_sid)
        groups = build_station_day_groups(collectible)
        options = {'lookback_hours': LOOKBACK_HOURS, 'lookahead_hours': LOOKAHEAD_HOURS,
                  'max_requests': caps['max_requests'], 'max_bytes': caps['max_bytes'],
                  'max_seconds': caps['max_seconds'], 'max_attempts_per_group': MAX_ATTEMPTS_PER_GROUP,
                  'subprocess_termination_grace_seconds': caps['subprocess_termination_grace_seconds'],
                  'success_pause_seconds': caps['success_pause_seconds']}
        recomputed_fp = compute_plan_fingerprint(digest(raw_selection_path), digest(mapping_path), groups, options)
        fp_matches = recomputed_fp == recorded_fp
        detail = (f'recomputed plan_fingerprint {"matches" if fp_matches else "DOES NOT MATCH"} the value '
                 f'recorded in {EXPANDED300_RUN}_fetch_manifest.json (assumes LOOKBACK_HOURS/LOOKAHEAD_HOURS/'
                 'MAX_ATTEMPTS_PER_GROUP unchanged since that run)')
        if not fp_matches:
            raise ValueError(
                f'input provenance check failed for {EXPANDED300_MAPPING_RUN}/{EXPANDED300_RUN}: {detail}')
        record('expanded300 mapping_table.csv', mapping_path, checked_against_prior=True, detail=detail)
        record('expanded300 selection.csv (pre-prediction_at)', raw_selection_path,
              checked_against_prior=True, detail=detail)
    else:
        reason = ('no plan_fingerprint was recorded in the fetch manifest' if recorded_fp is None else
                  f'the fetch manifest\'s caps is missing field(s) {[k for k in required_caps if k not in caps]} '
                  'needed to recompute the fingerprint')
        no_prior_detail = (f'no usable prior recorded expectation exists for this file ({reason}); this '
                          'records its CURRENT hash only -- past identity is not claimed')
        record('expanded300 mapping_table.csv', mapping_path, checked_against_prior=False, detail=no_prior_detail)
        record('expanded300 selection.csv (pre-prediction_at)', raw_selection_path,
              checked_against_prior=False, detail=no_prior_detail)

    for run_name in (SAMPLE21_RUN, EXPANDED300_RUN):
        wpa_path = out / f'{run_name}_selection_with_prediction_at.csv'
        record(f'{run_name} selection_with_prediction_at.csv', wpa_path, checked_against_prior=False,
              detail='no prior run recorded an independent hash of this exact (post-prediction_at) '
                     'file; this records its CURRENT hash only -- past identity is not claimed')
    return entries


# ---- Step 2: re-run the CURRENT join code from cache only ----

def rejoin_sample21(out: Path) -> tuple[pd.DataFrame, int]:
    selection = pd.read_csv(out / f'{SAMPLE21_RUN}_selection_with_prediction_at.csv')
    selection['prediction_at'] = pd.to_datetime(selection.prediction_at, utc=True)
    fetch_manifest = json.loads((out / f'{SAMPLE21_RUN}_fetch_manifest.json').read_text())
    obs, dropped = load_observations(fetch_manifest)

    origin_req = selection[['ID', 'prediction_at']].copy()
    origin_req['station'] = selection.Origin_Airport.map(lambda a: AIRPORTS[a]['station'])
    dest_req = selection[['ID', 'prediction_at']].copy()
    dest_req['station'] = selection.Destination_Airport.map(lambda a: AIRPORTS[a]['station'])

    origin = run_join_sample21(origin_req, obs, 'origin').drop(columns='origin_prediction_at')
    dest = run_join_sample21(dest_req, obs, 'destination').drop(columns='destination_prediction_at')
    result = pd.concat([selection.reset_index(drop=True), origin.reset_index(drop=True),
                        dest.reset_index(drop=True)], axis=1)

    # required invariants, re-verified on the FRESH join (not assumed from the original run)
    assert result.ID.tolist() == selection.ID.tolist(), 'row identity/order must be preserved'
    assert len(result) == len(selection)
    unresolved = result.prediction_at.isna()
    for role, airport_col in (('origin', 'Origin_Airport'), ('destination', 'Destination_Airport')):
        matched = result[f'{role}_observed_at'].notna()
        assert (result.loc[matched, f'{role}_station']
                == result.loc[matched, airport_col].map(lambda a: AIRPORTS[a]['station'])).all()
        obsd = pd.to_datetime(result.loc[matched, f'{role}_observed_at'])
        avail = pd.to_datetime(result.loc[matched, f'{role}_available_at'])
        pred = pd.to_datetime(result.loc[matched, 'prediction_at'])
        assert (obsd <= pred).all(), 'observed_at <= prediction_at required'
        assert (avail <= pred).all(), 'available_at <= prediction_at required'
        age = result.loc[matched, f'{role}_weather_age_minutes']
        assert (age >= 0).all() and (age <= 90).all()
        assert result.loc[unresolved, f'{role}_observed_at'].isna().all(), \
            'a row with no valid prediction_at must join nothing'
    return result, dropped


def rejoin_expanded300(out: Path) -> tuple[dict[int, pd.DataFrame], int]:
    selection = pd.read_csv(out / f'{EXPANDED300_RUN}_selection_with_prediction_at.csv')
    selection['prediction_at'] = pd.to_datetime(selection.prediction_at, utc=True)
    mapping = pd.read_csv(out / f'{EXPANDED300_MAPPING_RUN}_mapping_table.csv').set_index('iata')
    fetch_manifest = json.loads((out / f'{EXPANDED300_RUN}_fetch_manifest.json').read_text())
    raw_obs, dropped = load_observations(fetch_manifest)
    origin_req, dest_req = build_requests(selection, mapping)

    results = {}
    for latency in LATENCIES_MINUTES:
        obs = raw_obs.copy()
        obs['available_at'] = obs.observed_at + pd.Timedelta(minutes=latency)
        origin = run_join_expanded300(origin_req, obs, 'origin').drop(columns='origin_prediction_at')
        dest = run_join_expanded300(dest_req, obs, 'destination').drop(columns='destination_prediction_at')
        result = pd.concat([selection.reset_index(drop=True), origin.reset_index(drop=True),
                            dest.reset_index(drop=True)], axis=1)

        assert result.ID.tolist() == selection.ID.tolist()
        assert len(result) == len(selection)
        for role, airport_col in (('origin', 'Origin_Airport'), ('destination', 'Destination_Airport')):
            matched = result[f'{role}_observed_at'].notna()
            expected_station = result[airport_col].map(mapping.candidate_sid)
            assert (result.loc[matched, f'{role}_station'] == expected_station[matched]).all()
            obsd = pd.to_datetime(result.loc[matched, f'{role}_observed_at'])
            avail = pd.to_datetime(result.loc[matched, f'{role}_available_at'])
            pred = pd.to_datetime(result.loc[matched, 'prediction_at'])
            assert (obsd <= pred).all() and (avail <= pred).all()
            age = result.loc[matched, f'{role}_weather_age_minutes']
            assert (age >= 0).all() and (age <= 90).all()
            not_collectible = ~result.collectible
            assert result.loc[not_collectible, f'{role}_observed_at'].isna().all()
            unresolved = result.prediction_at.isna()
            assert result.loc[unresolved, f'{role}_observed_at'].isna().all()
        results[latency] = result
    return results, dropped


# ---- Step 3: compare fresh output to the ORIGINAL saved result ----

def _column_kind(col: str) -> str:
    if col == 'prediction_at' or col.endswith(DATETIME_SUFFIXES):
        return 'datetime'
    if col.endswith(NUMERIC_SUFFIXES):
        return 'numeric'
    return 'string'


def _semantic_mismatch(old: pd.Series, new: pd.Series, kind: str) -> pd.Series:
    if kind == 'datetime':
        o = pd.to_datetime(old, utc=True, errors='coerce')
        n = pd.to_datetime(new, utc=True, errors='coerce')
        return ~(o.isna() & n.isna()) & (o != n)
    if kind == 'numeric':
        o = pd.to_numeric(old, errors='coerce')
        n = pd.to_numeric(new, errors='coerce')
        close = np.isclose(o.to_numpy(dtype=float), n.to_numpy(dtype=float), atol=1e-6, equal_nan=True)
        return pd.Series(~close, index=old.index)
    o, n = old.astype(str), new.astype(str)
    both_na = old.isna() & new.isna()
    return ~both_na & (o != n)


def _role_all_missing(frame: pd.DataFrame, role: str) -> pd.Series:
    """True for rows where EVERY weather-value column for this role
    (observed_at/available_at/all measurements) is missing in `frame`."""
    cols = [f'{role}{suf}' for suf in ROLE_WEATHER_SUFFIXES if f'{role}{suf}' in frame.columns]
    if not cols:
        return pd.Series(True, index=frame.index)
    return frame[cols].isna().all(axis=1)


def compare_to_original(new_result: pd.DataFrame, original_path: Path, label: str,
                        persist_dir: Path) -> dict:
    """Round-trips the FRESH result through the exact same to_csv/read_csv
    cycle as the original so that in-memory-dtype-vs-CSV-text is never itself
    counted as a difference. The fresh result is PERSISTED (not written to a
    throwaway temp file and deleted) so this run's own re-combined evidence
    stays on disk and is linked into the manifest by path and hash.

    Returns a report with an explicit `regression_pass` verdict: a structural
    mismatch (row count, ID uniqueness/order, or column set) is ALWAYS a
    failure with no per-cell diff attempted; otherwise every meaning-level
    cell mismatch is classified as either the ONE approved policy change
    (not_collectible row, station column value -> missing, with that row's
    weather columns missing on BOTH sides) or an unexpected regression.
    `any_meaning_level_difference_found` in the caller's manifest is computed
    from expected_policy_changes + unexpected_meaning_diffs together, so an
    approved change is never hidden by reporting it as "no difference".

    The structural check (ID present/unique on both sides, row count, ID
    order, column set) always runs first, REGARDLESS of whether the two
    files are byte-identical -- e.g. two files that happen to be byte-for-
    byte identical but both contain a duplicate ID are still a structural
    failure, not a silent pass. Only once the structural check has actually
    passed does a byte-identical result skip the (redundant) per-cell
    semantic comparison."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    rejoined_path = persist_dir / f'{label}.csv'
    new_result.to_csv(rejoined_path, index=False)
    raw_identical = digest(rejoined_path) == digest(original_path)
    try:
        rejoined_file = str(rejoined_path.relative_to(ROOT))
    except ValueError:
        rejoined_file = str(rejoined_path)  # outside ROOT, e.g. a test's tmp_path
    report = {'label': label, 'raw_byte_identical': raw_identical,
             'rejoined_file': rejoined_file, 'rejoined_sha256': digest(rejoined_path),
             'structural_pass': True, 'structural_failures': [], 'column_diffs': [],
             'expected_policy_change_cells': 0, 'unexpected_meaning_mismatch_cells': 0}

    old = pd.read_csv(original_path)
    new = pd.read_csv(rejoined_path)

    failures = []
    same_row_count = len(old) == len(new)
    if not same_row_count:
        failures.append(f'row count differs: old={len(old)} new={len(new)}')
    id_in_both = 'ID' in old.columns and 'ID' in new.columns
    if not id_in_both:
        failures.append('ID column missing from the original or the fresh result')
    old_id_unique = id_in_both and old.ID.is_unique
    new_id_unique = id_in_both and new.ID.is_unique
    if id_in_both and not old_id_unique:
        failures.append('duplicate ID values in the ORIGINAL result')
    if id_in_both and not new_id_unique:
        failures.append('duplicate ID values in the FRESH result')
    same_id_order = bool(id_in_both and same_row_count and old_id_unique and new_id_unique
                         and old.ID.tolist() == new.ID.tolist())
    if id_in_both and same_row_count and old_id_unique and new_id_unique and not same_id_order:
        failures.append('ID order differs between the original and the fresh result')
    old_cols, new_cols = set(old.columns), set(new.columns)
    missing_in_new = sorted(old_cols - new_cols)
    added_in_new = sorted(new_cols - old_cols)
    if missing_in_new:
        failures.append(f'columns present in the original but missing from the fresh result: {missing_in_new}')
    if added_in_new:
        failures.append(f'columns present in the fresh result but absent from the original: {added_in_new}')

    report['same_row_count'] = same_row_count
    report['same_id_order'] = same_id_order
    report['structural_pass'] = not failures
    report['structural_failures'] = failures
    if failures:
        # A structural mismatch makes per-column comparison meaningless (rows/columns cannot be
        # aligned); it is itself an unexpected regression, never silently skipped or treated as pass.
        report['unexpected_meaning_mismatch_cells'] = 1
        report['regression_pass'] = False
        return report

    if raw_identical:
        # Structure already verified above (this is not a bypass of it); a byte-identical file
        # cannot contain a semantic difference, so the per-cell comparison below is redundant here.
        report['regression_pass'] = True
        return report

    not_collectible = (~old['collectible'].astype(bool) if 'collectible' in old.columns
                       else pd.Series(False, index=old.index))
    role_all_missing_old = {role: _role_all_missing(old, role) for role in set(STATION_COLUMN_ROLE.values())}
    role_all_missing_new = {role: _role_all_missing(new, role) for role in set(STATION_COLUMN_ROLE.values())}

    for col in sorted(old_cols):
        kind = _column_kind(col)
        raw_diff = old[col].astype(str) != new[col].astype(str)
        raw_n = int(raw_diff.sum())
        if raw_n == 0:
            continue
        semantic_diff = _semantic_mismatch(old[col], new[col], kind)
        semantic_n = int(semantic_diff.sum())
        expected_n = 0
        if semantic_n and col in STATION_COLUMN_ROLE:
            role = STATION_COLUMN_ROLE[col]
            allowed_mask = (semantic_diff & not_collectible & old[col].notna() & new[col].isna()
                           & role_all_missing_old[role] & role_all_missing_new[role])
            expected_n = int(allowed_mask.sum())
        unexpected_n = semantic_n - expected_n
        report['expected_policy_change_cells'] += expected_n
        report['unexpected_meaning_mismatch_cells'] += unexpected_n
        report['column_diffs'].append({
            'column': col, 'kind': kind, 'raw_mismatch_cells': raw_n,
            'meaning_mismatch_cells': semantic_n, 'expected_policy_change_cells': expected_n,
            'unexpected_meaning_mismatch_cells': unexpected_n,
            'format_only_mismatch_cells': raw_n - semantic_n,
            'example_old': old.loc[raw_diff, col].head(3).tolist(),
            'example_new': new.loc[raw_diff, col].head(3).tolist(),
        })
    report['regression_pass'] = report['unexpected_meaning_mismatch_cells'] == 0
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    args = ap.parse_args()
    if not args.name.startswith('baseline_recovery_v2') or Path(args.name).name != args.name:
        raise ValueError('Invalid output name')
    out = ROOT / 'output'
    if list(out.glob(args.name + '_reconciliation*')):
        raise FileExistsError('Use a fresh --name; existing evidence is preserved')

    hash_checks = verify_manifest_hashes(out)
    input_provenance = verify_input_provenance(out)

    guard_subprocess, guard_urlopen = network_guard()
    with guard_subprocess, guard_urlopen:
        sample21_result, sample21_dropped = rejoin_sample21(out)
        expanded300_results, expanded300_dropped = rejoin_expanded300(out)

    # The fresh re-combined result for EVERY scenario is persisted here (not written to a throwaway
    # temp file and deleted), so this run's own evidence stays on disk and is linked by path/hash below.
    persist_dir = ROOT / 'data/weather_probe' / f'{args.name}_rejoined'
    comparisons = [compare_to_original(
        sample21_result, ROOT / 'data/weather_probe' / f'{SAMPLE21_RUN}_joined/joined_sample.csv',
        'sample21', persist_dir)]
    for lat in LATENCIES_MINUTES:
        comparisons.append(compare_to_original(
            expanded300_results[lat],
            ROOT / 'data/weather_probe' / f'{EXPANDED300_RUN}_joined' / f'joined_latency{lat}min.csv',
            f'expanded300_latency{lat}min', persist_dir))

    all_identical = all(c['raw_byte_identical'] for c in comparisons)
    structural_pass_all = all(c.get('structural_pass', False) for c in comparisons)
    expected_total = sum(c.get('expected_policy_change_cells', 0) for c in comparisons)
    unexpected_total = sum(c.get('unexpected_meaning_mismatch_cells', 0) for c in comparisons)
    # Any meaning-level difference at all -- approved policy change OR unexpected regression -- is
    # reported here; an approved change is never hidden by folding it into "no difference found".
    any_meaning_diff = (expected_total + unexpected_total) > 0
    overall_regression_pass = all(c.get('regression_pass', False) for c in comparisons)

    pd.DataFrame(comparisons).to_json(out / f'{args.name}_reconciliation_comparisons.json',
                                      orient='records', indent=2)
    manifest = {
        'name': args.name, 'sample21_run': SAMPLE21_RUN, 'expanded300_run': EXPANDED300_RUN,
        'expanded300_mapping_run': EXPANDED300_MAPPING_RUN,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'code_fingerprints_scope': (
            'code_sha256 above is this script\'s own file only; code_fingerprints lists every module '
            'this reconciliation actually imports and executes (the join/build_requests/select code '
            'plus src/weather.py) -- code outside that list (pandas/numpy, unrelated notebooks) is not '
            'covered by either hash.'),
        'code_fingerprints': code_fingerprints(),
        'fits_any_model': False, 'uses_external_data': False, 'target_columns_used': [],
        'network_guard_active': True,
        'manifest_hash_checks_passed': len(hash_checks),
        'input_provenance_checks': input_provenance,
        'input_provenance_checks_passed': len(input_provenance),
        'sample21_dropped_exact_duplicate_reports': sample21_dropped,
        'expanded300_dropped_exact_duplicate_reports': expanded300_dropped,
        'comparisons_table': str((out / f'{args.name}_reconciliation_comparisons.json').relative_to(ROOT)),
        'rejoined_results_dir': str(persist_dir.relative_to(ROOT)),
        'all_scenarios_byte_identical_to_original': all_identical,
        'structural_comparison_passed': structural_pass_all,
        'expected_policy_change_cells_total': expected_total,
        'unexpected_meaning_mismatch_cells_total': unexpected_total,
        'any_meaning_level_difference_found': any_meaning_diff,
        'final_regression_verdict': 'PASS' if overall_regression_pass else 'FAIL',
        'characterization': (
            'byte-identical to the original' if all_identical else
            'structural comparison FAILED (row count, ID uniqueness/order, or column set mismatch); '
            'see comparisons_table for structural_failures' if not structural_pass_all else
            'differs from the original at the byte level with no meaning-level difference (format-only) '
            '-- see comparisons_table' if not any_meaning_diff else
            f'differs from the original with {expected_total} approved policy-change cell(s) and 0 '
            'unexpected meaning-level differences -- see comparisons_table' if unexpected_total == 0 else
            f'differs from the original with {unexpected_total} UNEXPECTED meaning-level difference '
            'cell(s) beyond the approved policy change -- see comparisons_table'),
        'limitations': [
            'This re-runs the CURRENT join code (src/weather.py, join_weather_sample[_expanded].py) '
            'against the SAME cached observations already on disk; it proves the fixed code reproduces '
            'the previously reported result from real cached data, not that a full re-collection would '
            'match (no new network collection is performed here).',
            'A network guard monkeypatches subprocess.run and urllib.request.urlopen for the duration '
            'of the re-join; a NetworkCallAttempted error means the cache-only assumption was violated '
            'and the comparison is invalid, not that new data was collected.',
            'The per-column format-vs-meaning split is only computed for a column whose CSV-text '
            'representation actually differs between the fresh and original result; a byte-identical '
            'file skips this entirely and is reported as such.',
            'The ONLY approved semantic change is a not_collectible row\'s origin_station/'
            'destination_station flipping from a filled value to correctly-missing, with that row\'s '
            'weather columns missing on BOTH sides; the SAME station change on a collectible row, any '
            'actual weather value/observed-time/station change, or a structural mismatch is always an '
            'unexpected regression regardless of any_meaning_level_difference_found.',
            'manifest_hash_checks_passed (the fetch caches and the two ORIGINAL joined outputs) and '
            'input_provenance_checks (the selection/mapping files actually read to rebuild the join '
            'requests) are separate counts over disjoint file sets -- neither includes the other. Within '
            'input_provenance_checks, only the entries with checked_against_prior_recorded_expectation='
            'true were verified against an independently pre-recorded expectation (the 300-row fetch '
            'run\'s own plan_fingerprint); the two selection_with_prediction_at.csv entries record only '
            'a CURRENT hash because no run ever recorded an independent expectation for that exact file, '
            'and past identity for those two is not claimed.',
        ],
    }
    (out / f'{args.name}_reconciliation_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'limitations'}, indent=2, default=str))

    if not overall_regression_pass:
        raise RuntimeError(
            f'{args.name}: reconciliation regression FAILED (structural_comparison_passed='
            f'{structural_pass_all}, unexpected_meaning_mismatch_cells_total={unexpected_total}); '
            f'the failure report was saved to output/{args.name}_reconciliation_manifest.json and '
            f'{args.name}_reconciliation_comparisons.json for inspection.')


if __name__ == '__main__':
    main()
