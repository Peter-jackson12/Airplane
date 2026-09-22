"""Disk round-trip contracts independent of pandas' default NA inference."""
from __future__ import annotations

import pandas as pd
import pytest

from src import weather_full as full


def fixture(root):
    mapping = pd.DataFrame({'iata': ['AAA', 'BBB'], 'candidate_sid': ['A', 'B'],
                            'matched_network': ['TEST_ASOS'] * 2,
                            'verification_tier': ['confirmed_period'] * 2})
    pool = pd.DataFrame({'ID': ['001'], 'row_position': [7], 'Origin_Airport': ['AAA'],
                         'Destination_Airport': ['BBB'], 'row_tier': ['confirmed_period'],
                         'prediction_at': pd.to_datetime(['2019-01-15 12:00Z'], utc=True)})
    rows = []
    for station in ['A', 'B']:
        rec = dict.fromkeys(full.WEATHER_FIELDS, 'M')
        rec.update(station=station, valid='2019-01-15 11:00', metar=station + ' METAR',
                   skyc1='M' if station == 'A' else '', wxcodes='' if station == 'A' else 'M')
        rows.append(rec)
    path = root / 'cache.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    entry = {'request_id': 'TEST_ASOS_201901', 'month': '2019-01', 'network': 'TEST_ASOS',
             'stations': ['A', 'B'], 'window_start_utc': '2019-01-01T00:00:00+00:00',
             'window_end_utc': '2019-02-01T00:00:00+00:00', 'cache_file': 'cache.csv',
             'sha256': full.digest(path), 'rows': 2}
    return pool, mapping, [entry]


def test_nulls_and_blanks_survive_gzip_merge_and_chunked_read(tmp_path):
    pool, mapping, entries = fixture(tmp_path)
    result = full.run_partitioned(pool, mapping, entries, root=tmp_path, local=tmp_path / 'run')
    assert result['csv_serialization']['null_token'] == '<NA>'
    assert result['csv_serialization']['empty_string_is_null'] is False
    for latency, entry in result['outputs'].items():
        path = tmp_path / entry['path']
        got = full.read_joined_csv(path)
        assert got.ID.tolist() == ['001']
        assert pd.isna(got.origin_skyc1.iloc[0])
        assert got.origin_wxcodes.iloc[0] == ''
        assert got.destination_skyc1.iloc[0] == ''
        assert pd.isna(got.destination_wxcodes.iloc[0])
        assert pd.isna(got.origin_tmpf.iloc[0])
        chunks = list(full.read_joined_csv(path, chunksize=1))
        pd.testing.assert_frame_equal(pd.concat(chunks, ignore_index=True), got)
        subset = full.read_joined_csv(path, usecols=['origin_wxcodes'])
        assert subset.origin_wxcodes.iloc[0] == ''
        scenario = next(s for s in result['scenarios'] if s['latency_minutes'] == int(latency))
        for role in full.ROLES:
            stats = scenario['roles'][role]
            for field in full.WEATHER_FIELDS:
                values = got[f'{role}_{field}']
                assert int(values.isna().sum()) == stats['feature_missing_all_rows'][field]
                assert int(values.eq('').sum()) == stats['feature_blank_matched_rows'][field]


@pytest.mark.parametrize('dtype', [object, 'string'])
def test_reserved_null_token_cannot_silently_corrupt_literal_id(tmp_path, dtype):
    frame = pd.DataFrame({'_join_ordinal': [0], 'ID': pd.Series(['<NA>'], dtype=dtype)})
    path = tmp_path / 'part.csv'
    with pytest.raises(ValueError, match='null token collides'):
        full.write_partition(frame, path)
    assert not path.exists()


def test_reader_keeps_literal_na_and_null_words(tmp_path):
    path = tmp_path / 'part.csv'
    pd.DataFrame({'_join_ordinal': [0, 1], 'ID': ['001', '002'],
                  'weather': ['NA', 'null']}).pipe(full.write_partition, path)
    got = full.read_joined_csv(path)
    assert got.ID.tolist() == ['001', '002']
    assert got.weather.tolist() == ['NA', 'null']
