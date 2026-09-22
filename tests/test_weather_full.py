"""Synthetic contracts only; no local archive and no model evidence."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src import weather_full as full
from src.weather import join_weather_asof
from notebooks.scope_weather_collection_refined import compute_prediction_at
from notebooks.join_weather_sample import WEATHER_FIELDS


def mapping():
    return pd.DataFrame({'iata': ['AAA', 'BBB', 'CCC'], 'candidate_sid': ['A', 'B', 'A'],
                         'matched_network': ['TEST_ASOS'] * 3,
                         'verification_tier': ['confirmed_period', 'confirmed_period', 'unconfirmed'],
                         'mwgg_tz': ['UTC'] * 3})


def pool():
    return pd.DataFrame({'ID': ['feb', 'jan', 'dst', 'held'], 'row_position': [20, 3, 99, 2],
                         'Origin_Airport': ['AAA', 'AAA', 'AAA', 'CCC'],
                         'Destination_Airport': ['BBB'] * 4,
                         'row_tier': ['confirmed_period'] * 3 + ['unconfirmed'],
                         'prediction_at': pd.to_datetime(['2019-02-01 00:10Z', '2019-01-31 23:55Z', None,
                                                         '2019-02-01 00:10Z'], utc=True)}, index=[7, 2, 9, 4])


def cache(root: Path, month: str, rows: list[tuple], *, suffix='', stations=None):
    path = root / 'data/weather_probe/cache' / f'{month}{suffix}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    values = []
    for sid, valid, temperature, metar in rows:
        value = dict.fromkeys(WEATHER_FIELDS, 'M')
        value.update(station=sid, valid=valid, tmpf=temperature, metar=metar, wxcodes='')
        values.append(value)
    pd.DataFrame(values, columns=['station', 'valid', *WEATHER_FIELDS]).to_csv(path, index=False)
    start, end = full.month_bounds(month)
    return {'request_id': month + suffix, 'month': month, 'network': 'TEST_ASOS',
            'stations': stations or ['A', 'B'], 'window_start_utc': start.isoformat(),
            'window_end_utc': end.isoformat(), 'cache_file': path.relative_to(root).as_posix(),
            'sha256': full.digest(path), 'rows': len(values)}


@pytest.fixture
def archive(tmp_path):
    january = cache(tmp_path, '2019-01', [
        ('A', '2019-01-31 23:00', 1, 'A OLD'), ('A', '2019-01-31 23:50', 2, 'A RECENT'),
        ('B', '2019-01-31 21:00', 3, 'B STALE'), ('B', '2019-01-31 23:55', 4, 'B RECENT'),
        ('A', '2019-02-01 00:00', 5, 'A BOUNDARY')])
    february = cache(tmp_path, '2019-02', [
        ('A', '2019-02-01 00:00', 5, 'A BOUNDARY'), ('A', '2019-02-01 01:00', 99, 'A FUTURE'),
        ('B', '2019-02-01 00:05', 6, 'B NEW'), ('B', '2019-02-01 01:00', 99, 'B FUTURE')])
    return [january, february]


def test_partitioned_matches_unpartitioned_reference_and_reads_once(tmp_path, archive, monkeypatch):
    assert full.WEATHER_FIELDS == WEATHER_FIELDS
    original = full.read_cache
    loaded = [original(e, tmp_path)[0] for e in archive]
    raw, _ = full.deduplicate(pd.concat(loaded, ignore_index=True))
    calls = []
    def counted(e, root):
        calls.append(e['request_id'])
        return original(e, root)
    monkeypatch.setattr(full, 'read_cache', counted)
    result = full.run_partitioned(pool(), mapping(), list(reversed(archive)),
                                  root=tmp_path, local=tmp_path / 'run')
    assert sorted(calls) == sorted(e['request_id'] for e in archive)
    assert result['cache_validation']['cache_rows_read'] == 9
    assert result['cache_validation']['dropped_exact_duplicate_reports'] == 1
    prepared, lookup = full.prepare_pool(pool(), mapping())
    for lat in full.LATENCIES:
        got = pd.read_csv(tmp_path / result['outputs'][str(lat)]['path'])
        assert got.ID.tolist() == ['feb', 'jan', 'dst', 'held']
        assert got.ID.is_unique and got.row_position.tolist() == [20, 3, 99, 2]
        assert len(got) == 4
        for role, airport in full.ROLES.items():
            eligible = prepared.row_tier.eq('confirmed_period') & prepared.prediction_at.notna()
            req = prepared[['prediction_at']].assign(station=prepared[airport].map(lookup.candidate_sid).where(eligible))
            obs = raw.assign(available_at=raw.observed_at + pd.Timedelta(minutes=lat))
            expected = join_weather_asof(req, obs, max_age='90min')
            pd.testing.assert_series_equal(pd.to_datetime(got[f'{role}_observed_at'], utc=True, format='mixed'),
                                           expected.observed_at, check_names=False, check_dtype=False)
            pd.testing.assert_series_equal(got[f'{role}_tmpf'], expected.tmpf,
                                           check_names=False, check_dtype=False)
            assert got.loc[2:, f'{role}_observed_at'].isna().all()
            assert got.loc[2:, f'{role}_station'].isna().all()
        assert result['outputs'][str(lat)]['rows'] == 4
    headline = pd.read_csv(tmp_path / result['outputs']['10']['path'])
    assert headline.loc[0, 'origin_tmpf'] == 5  # availability equality at 00:10
    assert headline.loc[0, 'destination_tmpf'] == 4  # 00:05 report not yet available
    later = pd.read_csv(tmp_path / result['outputs']['30']['path'])
    assert later.loc[0, 'origin_tmpf'] == 1  # previous-month available fallback
    for scenario in result['scenarios']:
        assert scenario['rows'] == 4
        assert scenario['both_matched'] + scenario['one_matched'] + scenario['none_matched'] == 4
        for role in full.ROLES:
            assert sum(scenario['roles'][role]['reasons'].values()) == 4
    assert set(result['invariants'].values()) == {0}


def test_deterministic_output_and_refuse_existing_run(tmp_path, archive):
    first = full.run_partitioned(pool(), mapping(), archive, root=tmp_path, local=tmp_path / 'first')
    second = full.run_partitioned(pool(), mapping(), archive[::-1], root=tmp_path, local=tmp_path / 'second')
    assert first['ordered_id_sha256'] == second['ordered_id_sha256']
    assert [x['sha256'] for x in first['outputs'].values()] == [x['sha256'] for x in second['outputs'].values()]
    with pytest.raises(FileExistsError):
        full.run_partitioned(pool(), mapping(), archive, root=tmp_path, local=tmp_path / 'first')


@pytest.mark.parametrize('age,matched_latencies', [(90, [0, 10, 30, 60]), (90 + 1/60, []),
                                                  (10, [0, 10]), (60, [0, 10, 30, 60]),
                                                  (0, [0]), (-1/60, [])])
def test_exact_age_and_availability_boundaries(tmp_path, age, matched_latencies):
    p = pool().iloc[:1].copy()
    p['prediction_at'] = pd.to_datetime(['2019-01-15 12:00Z'], utc=True)
    when = p.prediction_at.iloc[0] - pd.Timedelta(seconds=round(age * 60))
    entry = cache(tmp_path, '2019-01', [('A', when.isoformat(), 10, 'A'), ('B', when.isoformat(), 20, 'B')])
    obs, _ = full.read_cache(entry, tmp_path)
    prepared, lookup = full.prepare_pool(p, mapping())
    for latency, joined in full.join_block(prepared, lookup, obs):
        assert joined.origin_observed_at.notna().all() == (latency in matched_latencies)
        assert joined.destination_observed_at.notna().all() == (latency in matched_latencies)
        if latency in matched_latencies:
            assert joined.origin_weather_age_minutes.iloc[0] == pytest.approx(age)


def test_dst_2400_use_existing_transport_conversion():
    dates = pd.Series(pd.to_datetime(['2019-11-03', '2019-03-10', '2019-01-31']))
    times = pd.Series([130, 230, 2400])
    tz = pd.Series(['America/New_York'] * 3)
    prediction = compute_prediction_at(dates, times, tz) - pd.Timedelta(minutes=60)
    assert prediction.iloc[:2].isna().all()
    assert prediction.iloc[2] == pd.Timestamp('2019-02-01 04:00Z')


def test_all_ineligible_and_unresolved_preserve_denominator(tmp_path):
    p = pool().iloc[2:].copy()
    result = full.run_partitioned(p, mapping(), [], root=tmp_path, local=tmp_path / 'run')
    for output in result['outputs'].values():
        got = pd.read_csv(tmp_path / output['path'])
        assert got.ID.tolist() == ['dst', 'held']
        assert got.origin_observed_at.isna().all()
        assert got.origin_reason.tolist() == ['unresolved_prediction_at', 'not_mapping_eligible']
    assert result['cache_validation']['cache_files_read'] == 0


def test_empty_pool_is_well_formed(tmp_path):
    result = full.run_partitioned(pool().iloc[:0], mapping(), [], root=tmp_path, local=tmp_path / 'run')
    assert all(e['rows'] == 0 for e in result['outputs'].values())
    assert all(s['rows'] == 0 for s in result['scenarios'])


@pytest.mark.parametrize('field,value,match', [
    ('station', 'WRONG', 'station mismatch'), ('valid', '2019-03-01 00:00', 'out-of-window'),
    ('valid', 'M', 'missing observation'), ('tmpf', 'infinity', 'nonfinite'),
    ('tmpf', 'not-a-number', None)])
def test_semantically_invalid_cache_rejected(tmp_path, field, value, match):
    entry = cache(tmp_path, '2019-01', [('A', '2019-01-15 12:00', 10, 'METAR')])
    path = tmp_path / entry['cache_file']
    raw = pd.read_csv(path, keep_default_na=False)
    raw[field] = value
    raw.to_csv(path, index=False)
    entry['sha256'] = full.digest(path)
    with pytest.raises(ValueError, match=match):
        full.read_cache(entry, tmp_path)


def test_hash_and_row_count_validation(tmp_path):
    entry = cache(tmp_path, '2019-01', [])
    original = copy.deepcopy(entry)
    entry['rows'] = 1
    with pytest.raises(ValueError, match='row count'):
        full.read_cache(entry, tmp_path)
    path = tmp_path / entry['cache_file']
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='SHA mismatch'):
        full.read_cache(original, tmp_path)


@pytest.mark.parametrize('conflict', ['metar', 'tmpf'])
def test_aabb_conflicts_never_silently_resolved(tmp_path, conflict):
    entry = cache(tmp_path, '2019-01', [('A', '2019-01-15 12:00', 10, 'METAR')])
    obs, _ = full.read_cache(entry, tmp_path)
    other = obs.copy()
    other[conflict] = 'DIFFERENT' if conflict == 'metar' else 99
    with pytest.raises(ValueError, match='conflicting reports'):
        full.deduplicate(pd.concat([obs, obs, other, other], ignore_index=True))
    out, removed = full.deduplicate(pd.concat([obs] * 4, ignore_index=True))
    assert len(out) == 1 and removed == 3


def test_cross_month_conflict_fails_no_success_outputs(tmp_path, archive):
    path = tmp_path / archive[1]['cache_file']
    raw = pd.read_csv(path, keep_default_na=False)
    raw.loc[0, 'tmpf'] = 100
    raw.to_csv(path, index=False)
    archive[1]['sha256'] = full.digest(path)
    with pytest.raises(ValueError, match='conflicting reports'):
        full.run_partitioned(pool(), mapping(), archive, root=tmp_path, local=tmp_path / 'run')
    assert not list((tmp_path / 'run').glob('joined*.gz'))


@pytest.mark.parametrize('issue', ['id', 'position', 'target', 'naive', 'tier', 'mapping_duplicate'])
def test_pool_contract_rejects_ambiguous_input(issue):
    p, m = pool(), mapping()
    if issue == 'id': p.loc[2, 'ID'] = p.loc[7, 'ID']
    if issue == 'position': p.loc[2, 'row_position'] = p.loc[7, 'row_position']
    if issue == 'target': p['Delay'] = 'Delayed'
    if issue == 'naive': p['prediction_at'] = p.prediction_at.dt.tz_localize(None)
    if issue == 'tier': p.loc[4, 'row_tier'] = 'confirmed_period'
    if issue == 'mapping_duplicate': m = pd.concat([m, m.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError):
        full.prepare_pool(p, m)


def test_missing_previous_month_and_duplicate_entry_rejected(tmp_path, archive):
    with pytest.raises(ValueError, match='station-month demand'):
        full.run_partitioned(pool(), mapping(), archive[1:], root=tmp_path, local=tmp_path / 'run')
    with pytest.raises(ValueError, match='duplicate request'):
        full.validate_entries([archive[0], archive[0]], tmp_path)
    assert not (tmp_path / 'run').exists()


@pytest.mark.parametrize('path', ['../outside.csv', '/tmp/outside.csv', 'C:\\outside.csv'])
def test_cache_path_traversal_rejected(tmp_path, path):
    with pytest.raises(ValueError, match='path'):
        full.relative_path(tmp_path, path)


def test_wrong_network_rejected(tmp_path, archive):
    entries = copy.deepcopy(archive)
    entries[1]['network'] = 'OTHER_ASOS'
    with pytest.raises(ValueError, match='multiple networks'):
        full.validate_entries(entries, tmp_path)


def test_heap_merge_detects_duplicate_or_missing_ordinals(tmp_path):
    first, second = tmp_path / 'a.csv', tmp_path / 'b.csv'
    first.write_text('_join_ordinal,ID\n0,A\n')
    second.write_text('_join_ordinal,ID\n0,B\n')
    with pytest.raises(ValueError, match='identity/order'):
        full.merge_parts([first, second], tmp_path / 'bad.csv.gz', ['A', 'B'])
    assert not (tmp_path / 'bad.csv.gz').exists()


def test_numeric_missing_and_categorical_blank_are_distinct(tmp_path):
    entry = cache(tmp_path, '2019-01', [('A', '2019-01-15 12:00', '', 'METAR')])
    obs, _ = full.read_cache(entry, tmp_path)
    assert pd.isna(obs.tmpf.iloc[0]) and pd.isna(obs.sknt.iloc[0])
    assert obs.wxcodes.iloc[0] == ''
