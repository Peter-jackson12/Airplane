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

DATETIME_SUFFIXES = ('_observed_at', '_available_at')
NUMERIC_SUFFIXES = ('_weather_age_minutes', '_tmpf', '_dwpf', '_relh', '_sknt', '_gust', '_vsby',
                    '_p01i', '_snowdepth')


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


def compare_to_original(new_result: pd.DataFrame, original_path: Path, label: str,
                        work_dir: Path) -> dict:
    """Round-trips the FRESH result through the exact same to_csv/read_csv
    cycle as the original so that in-memory-dtype-vs-CSV-text is never itself
    counted as a difference; only mismatches surviving that round trip are
    inspected further."""
    tmp_csv = work_dir / f'_tmp_{label}.csv'
    new_result.to_csv(tmp_csv, index=False)
    raw_identical = digest(tmp_csv) == digest(original_path)
    report = {'label': label, 'raw_byte_identical': raw_identical, 'column_diffs': []}
    if raw_identical:
        tmp_csv.unlink()
        return report

    old = pd.read_csv(original_path)
    new = pd.read_csv(tmp_csv)
    tmp_csv.unlink()
    report['same_row_count'] = len(old) == len(new)
    report['same_id_order'] = ('ID' in old and 'ID' in new
                               and old.ID.tolist() == new.ID.tolist())
    if not (report['same_row_count'] and report['same_id_order']):
        report['column_diffs'] = 'row count or ID order differs; per-column diff skipped'
        return report

    for col in old.columns:
        if col not in new.columns:
            report['column_diffs'].append({'column': col, 'kind': 'missing_in_new'})
            continue
        kind = _column_kind(col)
        raw_diff = old[col].astype(str) != new[col].astype(str)
        raw_n = int(raw_diff.sum())
        if raw_n == 0:
            continue
        semantic_diff = _semantic_mismatch(old[col], new[col], kind)
        semantic_n = int(semantic_diff.sum())
        report['column_diffs'].append({
            'column': col, 'kind': kind, 'raw_mismatch_cells': raw_n,
            'meaning_mismatch_cells': semantic_n,
            'format_only_mismatch_cells': raw_n - semantic_n,
            'example_old': old.loc[raw_diff, col].head(3).tolist(),
            'example_new': new.loc[raw_diff, col].head(3).tolist(),
        })
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

    guard_subprocess, guard_urlopen = network_guard()
    with guard_subprocess, guard_urlopen:
        sample21_result, sample21_dropped = rejoin_sample21(out)
        expanded300_results, expanded300_dropped = rejoin_expanded300(out)

    comparisons = [compare_to_original(
        sample21_result, ROOT / 'data/weather_probe' / f'{SAMPLE21_RUN}_joined/joined_sample.csv',
        'sample21', out)]
    for lat in LATENCIES_MINUTES:
        comparisons.append(compare_to_original(
            expanded300_results[lat],
            ROOT / 'data/weather_probe' / f'{EXPANDED300_RUN}_joined' / f'joined_latency{lat}min.csv',
            f'expanded300_latency{lat}min', out))

    all_identical = all(c['raw_byte_identical'] for c in comparisons)
    any_meaning_diff = any(
        isinstance(c.get('column_diffs'), list)
        and any(d.get('meaning_mismatch_cells', 0) > 0 or d.get('kind') == 'missing_in_new'
               for d in c['column_diffs'])
        for c in comparisons)

    pd.DataFrame(comparisons).to_json(out / f'{args.name}_reconciliation_comparisons.json',
                                      orient='records', indent=2)
    manifest = {
        'name': args.name, 'sample21_run': SAMPLE21_RUN, 'expanded300_run': EXPANDED300_RUN,
        'expanded300_mapping_run': EXPANDED300_MAPPING_RUN,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'fits_any_model': False, 'uses_external_data': False, 'target_columns_used': [],
        'network_guard_active': True,
        'manifest_hash_checks_passed': len(hash_checks),
        'sample21_dropped_exact_duplicate_reports': sample21_dropped,
        'expanded300_dropped_exact_duplicate_reports': expanded300_dropped,
        'comparisons_table': str((out / f'{args.name}_reconciliation_comparisons.json').relative_to(ROOT)),
        'all_scenarios_byte_identical_to_original': all_identical,
        'any_meaning_level_difference_found': any_meaning_diff,
        'characterization': (
            'byte-identical to the original' if all_identical else
            'differs from the original at the byte level; see comparisons_table for the format-vs-meaning '
            'breakdown per column' if not any_meaning_diff else
            'differs from the original with at least one MEANING-level (not merely formatting) difference '
            '-- see comparisons_table'),
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
        ],
    }
    (out / f'{args.name}_reconciliation_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'limitations'}, indent=2, default=str))


if __name__ == '__main__':
    main()
