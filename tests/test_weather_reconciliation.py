import pandas as pd

from notebooks.reconcile_weather_cache_recombination import compare_to_original


COLUMNS = ['ID', 'collectible', 'origin_station', 'origin_observed_at', 'origin_available_at',
          'origin_tmpf', 'destination_station', 'destination_observed_at', 'destination_available_at',
          'destination_tmpf']


def base_rows():
    """R1: collectible, unchanged. R2: not_collectible, station correctly
    masked to missing on both weather sides -- the ONE approved change.
    R3: not_collectible but weather is NOT missing on the old side, so the
    station-masking exception must NOT apply to it."""
    return [
        {'ID': 'R1', 'collectible': True, 'origin_station': 'ATL', 'origin_observed_at': '2019-01-01T00:00:00Z',
         'origin_available_at': '2019-01-01T00:05:00Z', 'origin_tmpf': 50.0,
         'destination_station': 'ORD', 'destination_observed_at': '2019-01-01T00:00:00Z',
         'destination_available_at': '2019-01-01T00:05:00Z', 'destination_tmpf': 40.0},
        {'ID': 'R2', 'collectible': False, 'origin_station': 'PGSN', 'origin_observed_at': None,
         'origin_available_at': None, 'origin_tmpf': None,
         'destination_station': 'PGUM', 'destination_observed_at': None,
         'destination_available_at': None, 'destination_tmpf': None},
        {'ID': 'R3', 'collectible': False, 'origin_station': 'YUM', 'origin_observed_at': '2019-02-01T00:00:00Z',
         'origin_available_at': '2019-02-01T00:05:00Z', 'origin_tmpf': 70.0,
         'destination_station': 'DCA', 'destination_observed_at': None,
         'destination_available_at': None, 'destination_tmpf': None},
    ]


def old_df():
    return pd.DataFrame(base_rows(), columns=COLUMNS)


def new_df_with(overrides_by_id: dict) -> pd.DataFrame:
    rows = base_rows()
    by_id = {r['ID']: r for r in rows}
    for rid, changes in overrides_by_id.items():
        by_id[rid].update(changes)
    return pd.DataFrame(rows, columns=COLUMNS)


def write_original(tmp_path, df):
    path = tmp_path / 'original.csv'
    df.to_csv(path, index=False)
    return path


# ---- the approved policy change: not_collectible station -> missing, weather missing both sides ----

def test_approved_station_masking_passes_regression(tmp_path):
    original_path = write_original(tmp_path, old_df())
    new = new_df_with({'R2': {'origin_station': None, 'destination_station': None}})
    report = compare_to_original(new, original_path, 'case', tmp_path / 'persist')
    assert report['structural_pass'] is True
    assert report['unexpected_meaning_mismatch_cells'] == 0
    assert report['expected_policy_change_cells'] == 2  # origin_station + destination_station
    assert report['regression_pass'] is True


# ---- disallowed changes ----

def test_station_change_on_a_collectible_row_is_unexpected(tmp_path):
    """The SAME station -> missing flip on a COLLECTIBLE row (R1) must never
    be treated as the approved policy change."""
    original_path = write_original(tmp_path, old_df())
    new = new_df_with({'R1': {'origin_station': None}})
    report = compare_to_original(new, original_path, 'case', tmp_path / 'persist')
    assert report['structural_pass'] is True
    assert report['unexpected_meaning_mismatch_cells'] >= 1
    assert report['regression_pass'] is False


def test_station_masking_without_weather_missing_on_old_side_is_unexpected(tmp_path):
    """R3 is not_collectible, but the OLD row still has real weather values
    (observed_at/tmpf not missing) -- the exception requires weather missing
    on BOTH sides, so masking its station must count as unexpected."""
    original_path = write_original(tmp_path, old_df())
    new = new_df_with({'R3': {'origin_station': None}})
    report = compare_to_original(new, original_path, 'case', tmp_path / 'persist')
    assert report['expected_policy_change_cells'] == 0
    assert report['unexpected_meaning_mismatch_cells'] >= 1
    assert report['regression_pass'] is False


def test_actual_weather_value_change_is_unexpected(tmp_path):
    original_path = write_original(tmp_path, old_df())
    new = new_df_with({'R1': {'origin_tmpf': 999.0}})
    report = compare_to_original(new, original_path, 'case', tmp_path / 'persist')
    assert report['unexpected_meaning_mismatch_cells'] == 1
    assert report['regression_pass'] is False


def test_observed_time_change_is_unexpected(tmp_path):
    original_path = write_original(tmp_path, old_df())
    new = new_df_with({'R1': {'origin_observed_at': '2019-01-01T09:00:00Z'}})
    report = compare_to_original(new, original_path, 'case', tmp_path / 'persist')
    assert report['unexpected_meaning_mismatch_cells'] == 1
    assert report['regression_pass'] is False


# ---- structural failures: row deletion, ID reorder, duplicate ID, column add/remove ----

def test_row_deletion_is_a_structural_failure(tmp_path):
    original_path = write_original(tmp_path, old_df())
    new = old_df().iloc[:-1]  # drop the last row
    report = compare_to_original(new, original_path, 'case', tmp_path / 'persist')
    assert report['structural_pass'] is False
    assert any('row count differs' in f for f in report['structural_failures'])
    assert report['regression_pass'] is False


def test_id_reorder_is_a_structural_failure(tmp_path):
    original_path = write_original(tmp_path, old_df())
    new = old_df().iloc[::-1].reset_index(drop=True)  # same rows, reversed order
    report = compare_to_original(new, original_path, 'case', tmp_path / 'persist')
    assert report['structural_pass'] is False
    assert any('order differs' in f for f in report['structural_failures'])
    assert report['regression_pass'] is False


def test_duplicate_id_is_a_structural_failure(tmp_path):
    original_path = write_original(tmp_path, old_df())
    new = old_df().copy()
    new.loc[1, 'ID'] = 'R1'  # R1 now duplicated, R2's ID gone
    report = compare_to_original(new, original_path, 'case', tmp_path / 'persist')
    assert report['structural_pass'] is False
    assert any('duplicate ID' in f for f in report['structural_failures'])
    assert report['regression_pass'] is False


def test_added_column_is_a_structural_failure(tmp_path):
    original_path = write_original(tmp_path, old_df())
    new = old_df().copy()
    new['extra_column'] = 'x'
    report = compare_to_original(new, original_path, 'case', tmp_path / 'persist')
    assert report['structural_pass'] is False
    assert any('absent from the original' in f for f in report['structural_failures'])
    assert report['regression_pass'] is False


def test_removed_column_is_a_structural_failure(tmp_path):
    original_path = write_original(tmp_path, old_df())
    new = old_df().drop(columns=['origin_tmpf'])
    report = compare_to_original(new, original_path, 'case', tmp_path / 'persist')
    assert report['structural_pass'] is False
    assert any('missing from the fresh result' in f for f in report['structural_failures'])
    assert report['regression_pass'] is False


# ---- byte-identical short-circuit and persistence of the fresh result ----

def test_byte_identical_result_passes_without_reading_back(tmp_path):
    original_path = write_original(tmp_path, old_df())
    report = compare_to_original(old_df(), original_path, 'case', tmp_path / 'persist')
    assert report['raw_byte_identical'] is True
    assert report['regression_pass'] is True
    assert report['structural_pass'] is True


def test_fresh_result_is_persisted_to_disk_not_a_deleted_temp_file(tmp_path):
    original_path = write_original(tmp_path, old_df())
    new = new_df_with({'R2': {'origin_station': None, 'destination_station': None}})
    persist_dir = tmp_path / 'persist'
    report = compare_to_original(new, original_path, 'case', persist_dir)
    assert (persist_dir / 'case.csv').exists()  # kept on disk, not written-and-deleted
    assert 'rejoined_sha256' in report
