import pandas as pd

from notebooks.verify_bts_november import KEYS, match_counts, normalize


def sample():
    return pd.DataFrame([[11, 22, 'N123AB', 'ATL', 'JFK', 760, 19790, 900, 1100]], columns=KEYS)


def test_exact_match_preserves_left_rows_and_counts_source_duplicates():
    left = normalize(pd.concat([sample(), sample()], ignore_index=True))
    right = normalize(pd.concat([sample(), sample(), sample()], ignore_index=True))
    assert match_counts(left, right, KEYS).tolist() == [3, 3]


def test_wrong_day_time_or_dot_cannot_be_exact_match():
    left = normalize(sample())
    for column in ['Day_of_Month', 'Estimated_Arrival_Time', 'Carrier_ID(DOT)']:
        right = sample()
        right[column] += 1
        assert match_counts(left, normalize(right), KEYS).tolist() == [0]


def test_missing_source_keys_are_not_matches():
    right = sample()
    right['Tail_Number'] = None
    assert match_counts(normalize(sample()), normalize(right), KEYS).tolist() == [0]


def test_text_normalization_keeps_aircraft_prefix():
    right = sample()
    right['Tail_Number'] = ' n123ab '
    assert match_counts(normalize(sample()), normalize(right), KEYS).tolist() == [1]
    right['Tail_Number'] = '123AB'
    assert match_counts(normalize(sample()), normalize(right), KEYS).tolist() == [0]
