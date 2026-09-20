from pathlib import Path

import pandas as pd
import pytest

from notebooks.assign_row_dates import (ADOPTED_STATUSES, KEYS, STATUS_COMPLETE_MULTI,
                                        STATUS_COMPLETE_NONE, STATUS_COMPLETE_SINGLE,
                                        STATUS_MISSING_MULTI, STATUS_MISSING_NONE,
                                        STATUS_MISSING_SINGLE, STATUS_NOT_CHECKED,
                                        VOLATILE_KEYS, classify_complete, classify_missing,
                                        group_stats, masking_diagnostic, required_zip_status)

ROOT = Path(__file__).resolve().parents[1]


def sample_key_row(**overrides):
    row = dict(zip(KEYS, [11, 1, 'N1', 'ATL', 'JFK', 760, 19790, 900, 1100]))
    row.update(overrides)
    return row


def test_only_complete_single_candidate_year_is_ever_adopted():
    """The prohibition list is explicit: a missing-key unique candidate is not
    equivalent to a full-fingerprint match, so it must never be an adopted status."""
    assert ADOPTED_STATUSES == {STATUS_COMPLETE_SINGLE}


def test_volatile_keys_are_the_only_keys_ever_missing_in_the_source():
    """Month, Day_of_Month, Tail_Number, Origin_Airport, Destination_Airport and
    Distance have 0 missing values across the full 1,000,000-row file, so the
    calendar date is always known; only these three keys can ever be absent."""
    assert set(VOLATILE_KEYS) < set(KEYS)
    assert len(VOLATILE_KEYS) == 3


@pytest.mark.local_data
def test_volatile_keys_match_the_local_source_schema():
    """Validate the original-data profile explicitly, never silently omit it."""
    schema_path = ROOT / 'output/label_coverage/schema.csv'
    schema = pd.read_csv(schema_path).set_index('column')['missing_n']
    never_missing = [k for k in KEYS if k not in VOLATILE_KEYS]
    for column in never_missing:
        assert schema.loc[column] == 0, column
    for column in VOLATILE_KEYS:
        assert schema.loc[column] > 0, column


def test_classify_complete_distinguishes_zero_one_two_candidate_years():
    frame = pd.DataFrame({'2018_marketing': [0, 1, 1], '2019_marketing': [0, 0, 1]})
    result = classify_complete(frame)
    assert result.status.tolist() == [STATUS_COMPLETE_NONE, STATUS_COMPLETE_SINGLE, STATUS_COMPLETE_MULTI]
    assert result.attributed_year.iloc[0] is pd.NA
    assert result.attributed_year.iloc[1] == 2018
    assert result.attributed_year.iloc[2] is pd.NA


def test_classify_missing_uses_total_row_count_not_candidate_year_count():
    """A reduced-key match with two BTS rows in the SAME year must NOT be treated
    as a unique candidate: unlike the complete-fingerprint case (where all 9 keys
    pin the physical flight and marketing-code agreement confirms codeshare),
    fewer keys give no such guarantee that a repeat is a harmless duplicate."""
    rows = pd.DataFrame({
        'candidate_bts_rows_total': [0, 1, 2],
        'candidate_years_count': [0, 1, 1],  # the count=2 row still hits only ONE year
        '2018_n_distinct_flight_numbers': [float('nan'), 1.0, 1.0],
        '2019_n_distinct_flight_numbers': [float('nan'), float('nan'), float('nan')],
    })
    result = classify_missing(rows)
    assert result.status.tolist() == [STATUS_MISSING_NONE, STATUS_MISSING_SINGLE, STATUS_MISSING_MULTI]


def test_group_stats_counts_rows_and_distinct_operating_flight_numbers():
    frame = pd.DataFrame([sample_key_row(), sample_key_row(), sample_key_row()])
    frame['Flight_Number_Operating_Airline'] = [123, 123, 123]
    stats = group_stats(frame, KEYS)
    assert stats['n_bts_rows'].iloc[0] == 3
    assert stats['n_distinct_flight_numbers'].iloc[0] == 1, 'codeshare rows share one operating flight number'


def test_group_stats_flags_genuinely_different_flights_sharing_the_key_set():
    frame = pd.DataFrame([sample_key_row(), sample_key_row()])
    frame['Flight_Number_Operating_Airline'] = [123, 456]
    stats = group_stats(frame, KEYS)
    assert stats['n_distinct_flight_numbers'].iloc[0] == 2, 'a real collision must not look like codeshare'


def test_masking_never_flips_a_known_year_to_the_wrong_one():
    """The row that produced the truth always still matches itself with fewer
    keys, so masking can only create ambiguity (more candidate years), never a
    confidently wrong single answer, for rows tested against themselves."""
    truth_row = sample_key_row()
    truth = pd.DataFrame([truth_row])
    truth['attributed_year'] = 2018
    bts_2018 = pd.DataFrame([truth_row])
    bts_2019 = pd.DataFrame([dict(truth_row, Estimated_Arrival_Time=1105)])
    result = masking_diagnostic({11: truth}, {(11, 2018): bts_2018, (11, 2019): bts_2019})
    assert (result.resolved_single_year_wrong == 0).all()
    ambiguous = result[result.masked_keys == 'Estimated_Arrival_Time'].iloc[0]
    assert ambiguous.became_multi_year == 1
    assert ambiguous.resolved_single_year_correct == 0


def test_masking_diagnostic_covers_every_non_empty_subset_of_the_volatile_keys():
    truth = pd.DataFrame([sample_key_row()])
    truth['attributed_year'] = 2018
    bts = pd.DataFrame([sample_key_row()])
    result = masking_diagnostic({11: truth}, {(11, 2018): bts, (11, 2019): bts.iloc[0:0]})
    assert len(result) == 2 ** len(VOLATILE_KEYS) - 1


def test_required_zip_status_reports_both_candidate_years_for_a_month():
    status = required_zip_status(7)
    assert len(status) == 2
    assert all('2018_7.zip' in s['file'] or '2019_7.zip' in s['file'] for s in status)
    assert all(isinstance(s['downloaded'], bool) for s in status)


def test_not_checked_status_is_never_in_the_adopted_set():
    assert STATUS_NOT_CHECKED not in ADOPTED_STATUSES
    assert STATUS_MISSING_SINGLE not in ADOPTED_STATUSES
    assert STATUS_MISSING_MULTI not in ADOPTED_STATUSES
    assert STATUS_MISSING_NONE not in ADOPTED_STATUSES
    assert STATUS_COMPLETE_MULTI not in ADOPTED_STATUSES
    assert STATUS_COMPLETE_NONE not in ADOPTED_STATUSES
