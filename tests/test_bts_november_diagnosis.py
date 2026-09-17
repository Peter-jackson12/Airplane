import pandas as pd

from notebooks.assess_bts_november_recovery import BASE as RECOVERY_BASE
from notebooks.assess_bts_november_recovery import EXTRA
from notebooks.diagnose_bts_november_mismatch import (BASE, LADDER, field_disagreement,
                                                      match_counts, normalize)

TARGET_COLUMNS = {'Delay', 'DepDelay', 'ArrDelay', 'DepDel15', 'ArrDel15',
                  'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay',
                  'LateAircraftDelay', 'Cancelled', 'Diverted'}


def sample(**overrides):
    row = dict(zip(BASE, [11, 22, 'N123AB', 'ATL', 'JFK', 760, 19790, 900, 1100]))
    row.update(overrides)
    return pd.DataFrame([row], columns=BASE)


def test_no_ladder_level_uses_a_target_or_delay_column():
    for _, keys in LADDER:
        assert not (set(keys) & TARGET_COLUMNS)
        assert set(keys) <= set(BASE)


def test_ladder_levels_are_uniquely_named_and_start_from_the_full_key():
    names = [name for name, _ in LADDER]
    assert len(names) == len(set(names))
    assert LADDER[0] == ('L0_full', BASE)


def test_relaxing_a_key_never_lowers_the_candidate_count():
    """A subset key can only admit more BTS rows, never fewer."""
    left = normalize(sample())
    right = normalize(pd.concat([sample(), sample(Distance=761),
                                 sample(**{'Carrier_ID(DOT)': 19805})], ignore_index=True))
    full = match_counts(left, right, BASE)[0]
    for name, keys in LADDER:
        if set(keys) < set(BASE):
            assert match_counts(left, right, keys)[0] >= full, name


def test_dropping_distance_recovers_a_row_that_differs_only_in_distance():
    left = normalize(sample())
    right = normalize(sample(Distance=761))
    assert match_counts(left, right, BASE)[0] == 0
    relaxed = dict(LADDER)['L1_drop_distance']
    assert match_counts(left, right, relaxed)[0] == 1


def test_field_disagreement_reports_the_differing_field_and_keeps_the_closest_candidate():
    left = normalize(sample()).assign(ID='TRAIN_1')
    right = normalize(pd.concat([sample(Estimated_Departure_Time=1500),
                                 sample(Estimated_Departure_Time=905)], ignore_index=True))
    right['Reporting_Airline'] = 'DL'
    result = field_disagreement(left, right, 2018)
    assert len(result) == 1
    assert result.Distance_delta.iloc[0] == 0
    assert result.Estimated_Departure_Time_delta.iloc[0] == 5


def test_recovery_tiebreak_fields_exclude_the_target():
    assert not (set(EXTRA) & {'Delay'})
    assert set(RECOVERY_BASE) == set(BASE)


def test_missing_key_pattern_lists_exactly_the_observed_keys():
    frame = normalize(sample(**{'Carrier_ID(DOT)': None, 'Estimated_Arrival_Time': None}))
    present = frame[BASE].notna()
    pattern = present.apply(lambda r: '+'.join(c for c in BASE if r[c]), axis=1).iloc[0]
    assert 'Carrier_ID(DOT)' not in pattern.split('+')
    assert 'Estimated_Arrival_Time' not in pattern.split('+')
    assert 'Tail_Number' in pattern.split('+')
