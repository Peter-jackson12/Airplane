"""Full adopted-row weather join; cache-only and no model fitting.

Run only after Git-only tests pass. Original evidence is read-only. The final
join manifest is written LAST; files in an interrupted run directory are not
successful evidence. Use a fresh baseline_recovery_v2_* name after failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess

import pandas as pd

from notebooks import scope_weather_collection_refined as scope
from notebooks.fetch_weather_full_sharded import build_bulk_month_plan, request_from_row
from notebooks.join_weather_sample import WEATHER_FIELDS as SAMPLE_FIELDS
from src.weather_full import (LATENCIES, WEATHER_FIELDS, digest, prepare_pool,
                              relative_path, require, run_partitioned, validate_entries)

ROOT = Path(__file__).resolve().parents[1]
DENOMINATORS = ['total_adopted_rows', 'mapping_eligible_rows_confirmed_period_both_ends',
                'utc_time_resolved_rows', 'mapping_eligible_but_utc_unresolved_rows',
                'not_mapping_eligible_rows']
CODE_FILES = ['src/weather.py', 'src/weather_full.py', 'notebooks/join_weather_full.py',
              'notebooks/scope_weather_collection_refined.py',
              'notebooks/fetch_weather_full_sharded.py', 'notebooks/join_weather_sample.py']


def validate_name(name: str) -> None:
    require(bool(re.fullmatch(r'baseline_recovery_v2_[A-Za-z0-9_]+', name)), 'invalid run name')


def load_inputs(transport_name: str, mapping_name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    for name in (transport_name, mapping_name):
        validate_name(name)
    out = ROOT / 'output'
    paths = {
        'plan_manifest': out / f'{transport_name}_full_weather_plan_manifest.json',
        'final_fetch_manifest': out / f'{transport_name}_full_weather_fetch_manifest.json',
        'corrected_attempt_audit': out / f'{transport_name}_full_weather_fetch_audit.json',
        'mapping': out / f'{mapping_name}_mapping_table.csv',
        'raw_train': ROOT / 'data/train.csv',
        'attribution_manifest': out / f'{scope.ATTRIBUTION_RUN}_manifest.json',
    }
    for path in paths.values():
        require(path.is_file(), f'required local input missing: {path}')
    plan, final, audit, attribution = [json.loads(paths[k].read_text(encoding='utf-8')) for k in
                                     ['plan_manifest', 'final_fetch_manifest', 'corrected_attempt_audit', 'attribution_manifest']]
    require(plan.get('schema_version') == 2 and final.get('schema_version') == 3
            and audit.get('schema_version') == 1, 'unsupported source manifest schema')
    require(plan['name'] == final['name'] == audit['name'] == transport_name, 'source run mismatch')
    require(plan['mapping_name'] == final['mapping_name'] == mapping_name, 'source mapping mismatch')
    require(plan['attribution_run'] == scope.ATTRIBUTION_RUN, 'attribution run mismatch')
    require(final.get('all_shards_complete') is True and final.get('failed_groups') == 0
            and final.get('cache_hashes_reverified') is True, 'transport is not finalized')
    require(digest(paths['plan_manifest']) == final['plan_manifest_sha256'], 'plan manifest SHA mismatch')
    require(relative_path(ROOT, final['plan_manifest']) == paths['plan_manifest'].resolve(), 'plan manifest path mismatch')
    require(digest(paths['final_fetch_manifest']) == audit['source_final_manifest_sha256'], 'immutable final SHA mismatch')
    require(relative_path(ROOT, audit['source_final_manifest']) == paths['final_fetch_manifest'].resolve(), 'audit source path mismatch')
    require(paths['final_fetch_manifest'].stat().st_size == audit['source_final_manifest_bytes'], 'final manifest size mismatch')
    require(digest(paths['mapping']) == plan['mapping_sha256'], 'mapping SHA mismatch')
    require(digest(paths['raw_train']) == plan['raw_train_sha256'], 'raw train SHA mismatch')
    paths['attribution_rows'] = relative_path(ROOT, attribution['row_evidence'])
    require(attribution['row_sha256'] == plan['attribution_row_sha256'], 'attribution identity mismatch')
    require(digest(paths['attribution_rows']) == plan['attribution_row_sha256'], 'attribution rows SHA mismatch')
    paths['bulk_request_plan'] = relative_path(ROOT, plan['plan_file'])
    require(final['plan_sha256'] == plan['plan_sha256'] == digest(paths['bulk_request_plan']), 'bulk plan SHA mismatch')
    require(WEATHER_FIELDS == SAMPLE_FIELDS == plan['transport_contract']['weather_fields'], 'weather schema changed')

    # EXACT loader used by the transport planner: same adopted order, HHMM,
    # IANA timezone, DST NaT policy, and UTC scheduled departure minus 60 min.
    require(scope.ROOT.resolve() == ROOT.resolve(), 'pool loader root mismatch')
    pool = scope.load_adopted_pool(mapping_name)
    mapping = pd.read_csv(paths['mapping'])
    prepare_pool(pool, mapping)
    raw_ids = pd.read_csv(paths['raw_train'], usecols=['ID'])['ID']
    require(raw_ids.notna().all() and raw_ids.is_unique, 'raw IDs not unique/nonmissing')
    pos = pool.row_position.astype('int64')
    require(pos.ge(0).all() and pos.lt(len(raw_ids)).all(), 'row_position outside raw input')
    require(raw_ids.iloc[pos].tolist() == pool.ID.tolist(), 'attribution/raw positional identity mismatch')

    regenerated, counts = build_bulk_month_plan(
        pool, mapping, shard_size=int(plan['denominators']['shard_size']),
        max_stations_per_request=int(plan['denominators']['max_stations_per_request']))
    for key in DENOMINATORS + ['bulk_request_groups', 'shard_count', 'distinct_stations',
                               'distinct_networks', 'station_month_pairs']:
        require(counts[key] == plan['denominators'][key], f'plan denominator drift: {key}')
    # Keep station_years' binary float on CSV round-trip. The default parser
    # may round its last bit; request identities/timestamps remain exact.
    recorded = pd.read_csv(paths['bulk_request_plan'], float_precision='round_trip')
    require(list(recorded.columns) == list(regenerated.columns), 'bulk plan schema drift')
    require(recorded[['ordinal', 'shard_index']].to_dict('list') ==
            regenerated[['ordinal', 'shard_index']].to_dict('list'), 'bulk plan order drift')
    expected = [request_from_row(r) for r in regenerated.itertuples(index=False)]
    require([request_from_row(r) for r in recorded.itertuples(index=False)] == expected,
            'recomputed prediction-time demand differs from immutable plan')
    requests = final['requests']
    validate_entries(requests, ROOT)
    require(len(requests) == len(expected) == final['bulk_request_groups'] == final['cache_file_count'],
            'final request count mismatch')
    require(final['shard_count'] == counts['shard_count'] and
            final['shard_size'] == plan['denominators']['shard_size'], 'final shard identity mismatch')
    actual = {e['request_id']: e for e in requests}
    require(set(actual) == {e['request_id'] for e in expected}, 'final request identities differ from plan')
    for exp in expected:
        rec = actual[exp['request_id']]
        for key in ['network', 'month', 'stations']:
            require(rec[key] == exp[key], f'final request {key} differs from plan')
        for key in ['window_start_utc', 'window_end_utc']:
            require(pd.Timestamp(rec[key]) == exp[key], 'final request window differs from plan')
    require(sum(int(e['rows']) for e in requests) == final['total_cache_rows'], 'final cache row total mismatch')
    sources = {key: {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path),
                     'bytes': path.stat().st_size} for key, path in paths.items()}
    meta = {'sources': sources, 'denominators': {key: counts[key] for key in DENOMINATORS},
            'all_prediction_at_resolved_rows': int(pool.prediction_at.notna().sum())}
    return pool, mapping, final, meta


def write_new_json(path: Path, value: dict) -> None:
    require(not path.exists(), f'output already exists: {path}')
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--transport-name', required=True)
    parser.add_argument('--mapping-name', required=True)
    parser.add_argument('--validate-inputs-only', action='store_true')
    args = parser.parse_args()
    validate_name(args.name)
    require(args.name != args.transport_name, 'join must use a NEW run name')
    local = ROOT / 'data/weather_probe' / f'{args.name}_full_join'
    out = ROOT / 'output'
    require(not local.exists() and not list(out.glob(f'{args.name}_full_weather_join*')),
            'use a fresh name; existing or interrupted evidence is preserved')
    pool, mapping, final, meta = load_inputs(args.transport_name, args.mapping_name)
    print(json.dumps({'input_validation': 'passed', **meta['denominators']}), flush=True)
    if args.validate_inputs_only:
        return
    code = {path: hashlib.sha256((ROOT / path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for path in CODE_FILES}
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                        cwd=ROOT, text=True).strip(), 'tracked worktree must be clean')
    result = run_partitioned(pool, mapping, final['requests'], root=ROOT, local=local)
    stats = result['cache_validation']
    require(stats['cache_rows_read'] == final['total_cache_rows'], 'validated cache rows differ from final')
    require(stats['cache_bytes_read'] == final['total_cache_bytes_on_disk'], 'validated cache bytes differ from final')
    summary_path = out / f'{args.name}_full_weather_join_summary.json'
    manifest_path = out / f'{args.name}_full_weather_join_manifest.json'
    write_new_json(summary_path, {'schema_version': 1, 'name': args.name,
                                 **meta, 'scenarios': result.pop('scenarios')})
    manifest = {
        'schema_version': 1, 'name': args.name, 'successful': True,
        'transport_name': args.transport_name, 'mapping_name': args.mapping_name,
        'git_sha': git_sha, 'code_sha256_lf': code, 'uv_lock_sha256': digest(ROOT / 'uv.lock'),
        'python_version': platform.python_version(), 'pandas_version': pd.__version__,
        **meta, **result,
        'summary': {'path': summary_path.relative_to(ROOT).as_posix(), 'sha256': digest(summary_path)},
        'fits_any_model': False, 'target_columns_used': [], 'network_requests': 0,
        'prediction_offset_minutes': 60, 'max_observation_age_minutes': 90,
        'latency_scenarios_minutes': list(LATENCIES), 'measured_publication_latency': False,
        'row_output_format': 'deterministic gzip CSV; input ID order; one full-denominator file per latency',
        'raw_weather_fields': WEATHER_FIELDS,
        'raw_feature_contract': 'IEM M -> missing; numeric blanks -> missing; categorical blanks retained and counted separately; metar is audit text, not automatically a model feature',
        'model_feature_selection_frozen': False,
        'limitations': [
            'Historical join under ASSUMED 0/10/30/60-minute publication latency, not verified real-time replay.',
            'confirmed_period retains the existing mapping evidence grade, not a new historical station certification.',
            'No weather-on training or feature selection occurred. Any later comparison must keep identical labels, seeds, outer folds, nested selection and metrics.',
            'All adopted rows remain in every output and denominator, including ineligible and unresolved rows.',
            'Source-window boundaries are inclusive, matching the finalized transport validator. Cross-month duplicate conflicts fail closed.',
            'Stale/absent reasons refer only to collected windows, not proof of absence from the provider archive.',
            'Partitions remain in the ignored local run directory. No resume or overwrite mode is provided.',
        ],
    }
    write_new_json(manifest_path, manifest)
    print(json.dumps({'successful': True, 'manifest': str(manifest_path),
                      'rows_per_scenario': len(pool), 'cache_validation': stats}), flush=True)


if __name__ == '__main__':
    main()
