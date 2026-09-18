import json

import pandas as pd
import pytest

import notebooks.reconcile_weather_cache_recombination as reconcile_mod
from notebooks.reconcile_weather_cache_recombination import (code_fingerprints, compare_to_original,
                                                              verify_input_provenance)


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


# ---- byte-identical short-circuit only skips the SEMANTIC comparison, never the structural one ----

def test_byte_identical_result_passes_structural_check(tmp_path):
    original_path = write_original(tmp_path, old_df())
    report = compare_to_original(old_df(), original_path, 'case', tmp_path / 'persist')
    assert report['raw_byte_identical'] is True
    assert report['regression_pass'] is True
    assert report['structural_pass'] is True


def test_byte_identical_but_both_sides_have_the_same_duplicate_id_still_fails(tmp_path):
    """Two files can be byte-for-byte identical and still be structurally
    broken (e.g. a duplicate ID present on BOTH sides from a shared upstream
    bug). raw_identical must never bypass the structural check -- this must
    fail, not silently pass because the bytes match."""
    dup = old_df().copy()
    dup.loc[1, 'ID'] = 'R1'  # R1 duplicated on both sides; R2's ID is gone from both
    original_path = write_original(tmp_path, dup)
    report = compare_to_original(dup.copy(), original_path, 'case', tmp_path / 'persist')
    assert report['raw_byte_identical'] is True
    assert report['structural_pass'] is False
    assert any('duplicate ID' in f for f in report['structural_failures'])
    assert report['regression_pass'] is False


def test_byte_identical_but_both_sides_missing_id_column_still_fails(tmp_path):
    """Same idea for a missing ID column on both sides -- byte-identical
    bytes must not paper over the structural check."""
    no_id = old_df().drop(columns=['ID'])
    original_path = write_original(tmp_path, no_id)
    report = compare_to_original(no_id.copy(), original_path, 'case', tmp_path / 'persist')
    assert report['raw_byte_identical'] is True
    assert report['structural_pass'] is False
    assert any('ID column missing' in f for f in report['structural_failures'])
    assert report['regression_pass'] is False


def test_fresh_result_is_persisted_to_disk_not_a_deleted_temp_file(tmp_path):
    original_path = write_original(tmp_path, old_df())
    new = new_df_with({'R2': {'origin_station': None, 'destination_station': None}})
    persist_dir = tmp_path / 'persist'
    report = compare_to_original(new, original_path, 'case', persist_dir)
    assert (persist_dir / 'case.csv').exists()  # kept on disk, not written-and-deleted
    assert 'rejoined_sha256' in report


# ---- code_fingerprints: scope of what the reconciliation's code hash actually covers ----

def test_code_fingerprints_covers_every_module_actually_imported_and_executed():
    fps = code_fingerprints()
    assert set(fps) == set(reconcile_mod.CODE_FILES_USED_IN_RECOMBINATION)
    assert all(len(sha) == 64 for sha in fps.values())  # hex sha256


# ---- verify_input_provenance: hashes for the actual selection/mapping inputs, not just caches ----

def _selection_row(rid, collectible=True):
    return {'ID': rid, 'attributed_date': '2019-06-15', 'Estimated_Departure_Time': 900,
           'origin_timezone': 'America/New_York', 'Origin_Airport': 'ATL', 'Destination_Airport': 'ORD',
           'collectible': collectible}


def _write_provenance_fixture(out, monkeypatch, *, mismatched_fingerprint=False, no_plan_fingerprint=False):
    monkeypatch.setattr(reconcile_mod, 'ROOT', out.parent)
    monkeypatch.setattr(reconcile_mod, 'SAMPLE21_RUN', 'sample21run')
    monkeypatch.setattr(reconcile_mod, 'EXPANDED300_RUN', 'expanded300run')
    monkeypatch.setattr(reconcile_mod, 'EXPANDED300_MAPPING_RUN', 'mappingrun')
    out.mkdir(parents=True, exist_ok=True)

    mapping = pd.DataFrame({'iata': ['ATL', 'ORD'], 'candidate_sid': ['ATL', 'ORD']})
    mapping_path = out / 'mappingrun_mapping_table.csv'
    mapping.to_csv(mapping_path, index=False)

    raw_selection = pd.DataFrame([_selection_row('R1')])
    raw_selection_path = out / 'expanded300run_selection.csv'
    raw_selection.to_csv(raw_selection_path, index=False)

    from notebooks.fetch_weather_sample import compute_prediction_at
    from notebooks.fetch_weather_sample_expanded import (LOOKAHEAD_HOURS, LOOKBACK_HOURS,
                                                          MAX_ATTEMPTS_PER_GROUP, build_station_day_groups,
                                                          compute_plan_fingerprint)
    sel = raw_selection.copy()
    sel['prediction_at'] = compute_prediction_at(sel)
    mapping_indexed = mapping.set_index('iata')
    collectible = sel.loc[sel.collectible & sel.prediction_at.notna()].copy()
    collectible['origin_station'] = collectible.Origin_Airport.map(mapping_indexed.candidate_sid)
    collectible['destination_station'] = collectible.Destination_Airport.map(mapping_indexed.candidate_sid)
    groups = build_station_day_groups(collectible)
    caps = {'max_requests': 10, 'max_bytes': 1000, 'max_seconds': 60.0,
           'subprocess_termination_grace_seconds': 1.0, 'success_pause_seconds': 0.0}
    options = {'lookback_hours': LOOKBACK_HOURS, 'lookahead_hours': LOOKAHEAD_HOURS,
              'max_requests': caps['max_requests'], 'max_bytes': caps['max_bytes'],
              'max_seconds': caps['max_seconds'], 'max_attempts_per_group': MAX_ATTEMPTS_PER_GROUP,
              'subprocess_termination_grace_seconds': caps['subprocess_termination_grace_seconds'],
              'success_pause_seconds': caps['success_pause_seconds']}
    from notebooks.reconcile_weather_cache_recombination import digest
    fp = compute_plan_fingerprint(digest(raw_selection_path), digest(mapping_path), groups, options)
    if mismatched_fingerprint:
        fp = 'deliberately-wrong-fingerprint'
    manifest = {'caps': caps, 'plan_fingerprint': fp}
    if no_plan_fingerprint:
        # mirrors the REAL on-disk fetch_manifest.json for this project's actual 300-row run: it
        # predates the plan_fingerprint mechanism entirely, so plan_fingerprint is simply absent/None.
        manifest = {'caps': {'max_requests': caps['max_requests'], 'max_bytes': caps['max_bytes'],
                             'max_seconds': caps['max_seconds']}}
    (out / 'expanded300run_fetch_manifest.json').write_text(json.dumps(manifest))

    sel.to_csv(out / 'expanded300run_selection_with_prediction_at.csv', index=False)
    pd.DataFrame([_selection_row('R2')]).assign(
        prediction_at=lambda d: compute_prediction_at(d)).to_csv(
        out / 'sample21run_selection_with_prediction_at.csv', index=False)


def test_verify_input_provenance_records_current_and_cross_checked_entries(tmp_path, monkeypatch):
    out = tmp_path / 'output'
    _write_provenance_fixture(out, monkeypatch)
    entries = verify_input_provenance(out)
    by_label = {e['label']: e for e in entries}
    assert by_label['expanded300 mapping_table.csv']['checked_against_prior_recorded_expectation'] is True
    assert by_label['expanded300 selection.csv (pre-prediction_at)'][
        'checked_against_prior_recorded_expectation'] is True
    assert by_label['sample21run selection_with_prediction_at.csv'][
        'checked_against_prior_recorded_expectation'] is False
    assert by_label['expanded300run selection_with_prediction_at.csv'][
        'checked_against_prior_recorded_expectation'] is False
    assert all('sha256' in e and len(e['sha256']) == 64 for e in entries)


def test_verify_input_provenance_raises_when_mapping_table_no_longer_matches_recorded_fingerprint(
        tmp_path, monkeypatch):
    out = tmp_path / 'output'
    _write_provenance_fixture(out, monkeypatch, mismatched_fingerprint=True)
    with pytest.raises(ValueError, match='input provenance check failed'):
        verify_input_provenance(out)


def test_verify_input_provenance_never_fabricates_a_check_when_no_fingerprint_was_ever_recorded(
        tmp_path, monkeypatch):
    """Mirrors the real project state: the actual 300-row fetch_manifest.json predates the
    plan_fingerprint mechanism, so there is no prior recorded expectation for mapping_table.csv or the
    pre-prediction_at selection.csv either. This must record their current hashes and say so plainly --
    never raise, and never claim a cross-check that did not happen."""
    out = tmp_path / 'output'
    _write_provenance_fixture(out, monkeypatch, no_plan_fingerprint=True)
    entries = verify_input_provenance(out)
    by_label = {e['label']: e for e in entries}
    assert by_label['expanded300 mapping_table.csv']['checked_against_prior_recorded_expectation'] is False
    assert 'no plan_fingerprint was recorded' in by_label['expanded300 mapping_table.csv']['detail']
    assert by_label['expanded300 selection.csv (pre-prediction_at)'][
        'checked_against_prior_recorded_expectation'] is False
