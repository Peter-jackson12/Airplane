"""Synthetic source lineage and CLI integration, never real flight evidence."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from notebooks import join_weather_full as driver
from notebooks import scope_weather_collection_refined as scope
from notebooks.fetch_weather_full_sharded import build_bulk_month_plan, request_from_row
from src.weather_full import WEATHER_FIELDS, digest

TRANSPORT = 'baseline_recovery_v2_synthetic_transport'
MAPPING = 'baseline_recovery_v2_synthetic_mapping'
JOIN = 'baseline_recovery_v2_synthetic_join'


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')


@pytest.fixture
def source(tmp_path, monkeypatch):
    original_root = driver.ROOT
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    monkeypatch.setattr(scope, 'ROOT', tmp_path)
    out, data = tmp_path / 'output', tmp_path / 'data'
    out.mkdir()
    data.mkdir()
    raw = pd.DataFrame({'ID': ['first', 'second', 'held'],
                        'Origin_Airport': ['AAA', 'AAA', 'CCC'], 'Destination_Airport': ['BBB'] * 3,
                        'Estimated_Departure_Time': [1200, 100, 1300],
                        'Delay': ['Delayed', 'Not_Delayed', None]})
    raw.to_csv(data / 'train.csv', index=False)
    rows = data / 'attributed.csv'
    pd.DataFrame({'ID': raw.ID, 'row_position': [0, 1, 2],
                  'status': ['complete_single_candidate_year'] * 3,
                  'attributed_date': ['2019-01-15', '2019-02-01', '2019-01-15']}).to_csv(rows, index=False)
    attribution_path = out / f'{scope.ATTRIBUTION_RUN}_manifest.json'
    write_json(attribution_path, {'row_evidence': 'data/attributed.csv', 'row_sha256': digest(rows)})
    mapping = pd.DataFrame({'iata': ['AAA', 'BBB', 'CCC'], 'candidate_sid': ['A', 'B', 'A'],
                            'matched_network': ['TEST_ASOS'] * 3, 'mwgg_tz': ['UTC'] * 3,
                            'verification_tier': ['confirmed_period', 'confirmed_period', 'unconfirmed']})
    mapping_path = out / f'{MAPPING}_mapping_table.csv'
    mapping.to_csv(mapping_path, index=False)
    pool = scope.load_adopted_pool(MAPPING)
    plan, counts = build_bulk_month_plan(pool, mapping, shard_size=50, max_stations_per_request=20)
    local = data / 'weather_probe' / TRANSPORT
    local.mkdir(parents=True)
    plan_path = local / 'bulk_request_plan.csv'
    plan.to_csv(plan_path, index=False)
    requests = []
    for row in plan.itertuples(index=False):
        e = request_from_row(row)
        cache = local / f'{e["request_id"]}.csv'
        pd.DataFrame(columns=['station', 'valid', *WEATHER_FIELDS]).to_csv(cache, index=False)
        for key in ['window_start_utc', 'window_end_utc']:
            e[key] = e[key].isoformat()
        requests.append({**e, 'cache_file': cache.relative_to(tmp_path).as_posix(),
                         'rows': 0, 'sha256': digest(cache)})
    manifest_path = out / f'{TRANSPORT}_full_weather_plan_manifest.json'
    plan_manifest = {'schema_version': 2, 'name': TRANSPORT, 'mapping_name': MAPPING,
                     'attribution_run': scope.ATTRIBUTION_RUN, 'attribution_row_sha256': digest(rows),
                     'raw_train_sha256': digest(data / 'train.csv'), 'mapping_sha256': digest(mapping_path),
                     'plan_file': plan_path.relative_to(tmp_path).as_posix(), 'plan_sha256': digest(plan_path),
                     'denominators': counts, 'transport_contract': {'weather_fields': WEATHER_FIELDS}}
    write_json(manifest_path, plan_manifest)
    final_path = out / f'{TRANSPORT}_full_weather_fetch_manifest.json'
    final = {'schema_version': 3, 'name': TRANSPORT, 'mapping_name': MAPPING,
             'plan_manifest': manifest_path.relative_to(tmp_path).as_posix(),
             'plan_manifest_sha256': digest(manifest_path), 'plan_sha256': digest(plan_path),
             'all_shards_complete': True, 'failed_groups': 0, 'cache_hashes_reverified': True,
             'requests': requests, 'bulk_request_groups': len(requests), 'cache_file_count': len(requests),
             'shard_count': counts['shard_count'], 'shard_size': counts['shard_size'],
             'total_cache_rows': 0, 'total_cache_bytes_on_disk': sum((tmp_path / e['cache_file']).stat().st_size for e in requests)}
    write_json(final_path, final)
    audit_path = out / f'{TRANSPORT}_full_weather_fetch_audit.json'
    write_json(audit_path, {'schema_version': 1, 'name': TRANSPORT,
                            'source_final_manifest': final_path.relative_to(tmp_path).as_posix(),
                            'source_final_manifest_sha256': digest(final_path),
                            'source_final_manifest_bytes': final_path.stat().st_size})
    for relative in [*driver.CODE_FILES, 'uv.lock']:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original_root / relative).read_bytes())
    return {'root': tmp_path, 'plan': manifest_path, 'final': final_path, 'audit': audit_path,
            'mapping': mapping_path, 'raw': data / 'train.csv', 'rows': rows,
            'attribution': attribution_path}


def rebind(source):
    final = json.loads(source['final'].read_text())
    final['plan_manifest_sha256'] = digest(source['plan'])
    write_json(source['final'], final)
    audit = json.loads(source['audit'].read_text())
    audit['source_final_manifest_sha256'] = digest(source['final'])
    audit['source_final_manifest_bytes'] = source['final'].stat().st_size
    write_json(source['audit'], audit)


def test_full_source_lineage_and_original_prediction_loader(source):
    pool, mapping, final, meta = driver.load_inputs(TRANSPORT, MAPPING)
    assert pool.ID.tolist() == ['first', 'second', 'held']
    assert pool.prediction_at.iloc[0] == pd.Timestamp('2019-01-15 11:00Z')
    assert pool.prediction_at.iloc[1] == pd.Timestamp('2019-02-01 00:00Z')
    assert 'Delay' not in pool
    assert meta['denominators'] == {'total_adopted_rows': 3,
        'mapping_eligible_rows_confirmed_period_both_ends': 2, 'utc_time_resolved_rows': 2,
        'mapping_eligible_but_utc_unresolved_rows': 0, 'not_mapping_eligible_rows': 1}


@pytest.mark.parametrize('which,match', [('raw', 'raw train SHA'), ('mapping', 'mapping SHA'),
                                        ('final', 'immutable final SHA'), ('rows', 'attribution rows SHA'),
                                        ('plan', 'plan manifest SHA')])
def test_changed_source_fails_closed(source, which, match):
    path = source[which]
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match=match):
        driver.load_inputs(TRANSPORT, MAPPING)


def test_corrupt_final_station_identity_even_with_bound_audit(source):
    final = json.loads(source['final'].read_text())
    final['requests'][0]['stations'] = ['WRONG']
    write_json(source['final'], final)
    rebind(source)
    with pytest.raises(ValueError, match='stations differs from plan'):
        driver.load_inputs(TRANSPORT, MAPPING)


def test_plan_denominator_drift_is_not_silently_accepted(source):
    plan = json.loads(source['plan'].read_text())
    plan['denominators']['total_adopted_rows'] += 1
    write_json(source['plan'], plan)
    rebind(source)
    with pytest.raises(ValueError, match='denominator drift'):
        driver.load_inputs(TRANSPORT, MAPPING)


def test_positional_identity_rechecked_independently_of_source_hashes(source):
    rows = pd.read_csv(source['rows'])
    rows['row_position'] = [1, 0, 2]
    rows.to_csv(source['rows'], index=False)
    attribution = json.loads(source['attribution'].read_text())
    attribution['row_sha256'] = digest(source['rows'])
    write_json(source['attribution'], attribution)
    plan = json.loads(source['plan'].read_text())
    plan['attribution_row_sha256'] = digest(source['rows'])
    write_json(source['plan'], plan)
    rebind(source)
    with pytest.raises(ValueError, match='positional identity'):
        driver.load_inputs(TRANSPORT, MAPPING)


def test_missing_raw_never_replaced_with_synthetic_or_partial_input(source):
    source['raw'].unlink()
    with pytest.raises(ValueError, match='required local input missing'):
        driver.load_inputs(TRANSPORT, MAPPING)


def test_synthetic_cli_writes_manifest_last_and_preserves_all_rows(source, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['join_weather_full', '--name', JOIN,
                                     '--transport-name', TRANSPORT, '--mapping-name', MAPPING])
    monkeypatch.setattr(driver.subprocess, 'check_output', lambda args, **kwargs:
                        'f' * 40 if args[1] == 'rev-parse' else '')
    before = {key: digest(path) for key, path in source.items() if path.is_file()}
    driver.main()
    path = source['root'] / 'output' / f'{JOIN}_full_weather_join_manifest.json'
    manifest = json.loads(path.read_text())
    assert manifest['successful'] is True and manifest['fits_any_model'] is False
    assert manifest['network_requests'] == 0 and manifest['target_columns_used'] == []
    assert manifest['measured_publication_latency'] is False
    assert manifest['denominators']['total_adopted_rows'] == 3
    for entry in manifest['outputs'].values():
        got = pd.read_csv(source['root'] / entry['path'])
        assert got.ID.tolist() == ['first', 'second', 'held']
        assert got.origin_observed_at.isna().all()
        assert digest(source['root'] / entry['path']) == entry['sha256']
    assert before == {key: digest(path) for key, path in source.items() if path.is_file()}
    with pytest.raises(ValueError, match='fresh name'):
        driver.main()


def test_validate_inputs_only_does_not_write_join(source, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['join_weather_full', '--name', JOIN,
                                     '--transport-name', TRANSPORT, '--mapping-name', MAPPING,
                                     '--validate-inputs-only'])
    driver.main()
    assert not list((source['root'] / 'output').glob(f'{JOIN}_*'))
    assert not (source['root'] / 'data/weather_probe' / f'{JOIN}_full_join').exists()


@pytest.mark.parametrize('name', ['../../baseline_recovery_v2_foo', 'baseline_recovery_v2_', 'new',
                                 'baseline_recovery_v2_test/name'])
def test_invalid_run_names_rejected(name):
    with pytest.raises(ValueError, match='invalid run name'):
        driver.validate_name(name)


def test_json_publication_is_exclusive(tmp_path):
    path = tmp_path / 'manifest.json'
    driver.write_new_json(path, {'successful': True})
    with pytest.raises(ValueError, match='already exists'):
        driver.write_new_json(path, {'successful': False})
    assert json.loads(path.read_text())['successful'] is True
