import pandas as pd

from notebooks.verify_bts_november_marketing import (KEYS, RENAMES, USECOLS, match_counts,
                                                     normalize)

DELAY_COLUMNS = {'DepDelay', 'ArrDelay', 'DepDel15', 'ArrDel15', 'CarrierDelay',
                 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay',
                 'Cancelled', 'Diverted', 'Delay'}


def sample(**overrides):
    row = dict(zip(KEYS, [11, 22, 'N123AB', 'ATL', 'JFK', 760, 19790, 900, 1100]))
    row.update(overrides)
    return pd.DataFrame([row], columns=KEYS)


def test_source_carrier_id_maps_to_the_operating_airline_not_the_marketing_one():
    """The source's Carrier_ID(DOT) is the operating carrier; swapping this silently
    changes what every match means."""
    assert RENAMES['DOT_ID_Operating_Airline'] == 'Carrier_ID(DOT)'
    assert 'DOT_ID_Marketing_Airline' not in RENAMES
    assert RENAMES['CRSDepTime'] == 'Estimated_Departure_Time'
    assert RENAMES['CRSArrTime'] == 'Estimated_Arrival_Time'


def test_no_delay_or_target_column_is_read_from_the_marketing_file():
    assert not (set(USECOLS) & DELAY_COLUMNS)
    assert not (set(KEYS) & DELAY_COLUMNS)


def test_marketing_columns_needed_for_the_semantics_check_are_requested():
    assert 'IATA_Code_Marketing_Airline' in USECOLS
    assert 'DOT_ID_Operating_Airline' in USECOLS


def test_codeshare_duplicates_are_counted_not_silently_deduplicated():
    """One operated flight sold by two carriers appears twice; the count must show it."""
    left = normalize(sample())
    right = normalize(pd.concat([sample(), sample()], ignore_index=True))
    assert match_counts(left, right, KEYS).tolist() == [2]


def test_a_different_operating_carrier_is_not_a_match():
    left = normalize(sample())
    right = normalize(sample(**{'Carrier_ID(DOT)': 20304}))
    assert match_counts(left, right, KEYS).tolist() == [0]


def test_tail_normalization_keeps_the_prefix_and_ignores_case_and_spacing():
    right = sample(Tail_Number=' n123ab ')
    assert match_counts(normalize(sample()), normalize(right), KEYS).tolist() == [1]
    assert match_counts(normalize(sample()), normalize(sample(Tail_Number='123AB')),
                        KEYS).tolist() == [0]
